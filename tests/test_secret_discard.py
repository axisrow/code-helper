"""The guard against silently discarding a wrapper's only copy of a secret.

A ``secret``-auth wrapper embeds its token nowhere but the generated script —
there is no config file, and the ``*_API_KEY`` env var is a possible source,
not a guaranteed copy. So replacing such a file with a wrapper that never
asked for a token destroys the only copy, with nothing to restore it from.

The predicate is deliberately narrow. It keys on the OUTGOING file's auth mode
and on whether the INCOMING wrapper prompted for a token of its own — not on
whether the body changed. Prompting on any content change would fire on every
model bump and every token rotation (``add``-as-update, the primary use), and
a prompt users see constantly is one they learn to dismiss.
"""

from __future__ import annotations

import pathlib

import pytest

from codehelper.errors import CodeHelperError
from codehelper.services.model import get_provider, with_base_url
from codehelper.services.paths import Paths
from codehelper.services.spec import build_spec
from codehelper.services.wrappers import install_wrapper

pytestmark = pytest.mark.unit


def _bin(tmp_path: pathlib.Path) -> Paths:
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _secret(alias: str = "w"):
    return build_spec(agent="claude", provider="zai", model="glm-x", alias=alias)


def _literal(alias: str = "w", model: str = "qwen3"):
    return build_spec(agent="claude", provider="ollama", model=model, alias=alias)


def _secret_openai_toml(alias: str = "w"):
    """A secret-auth OPENAI_TOML wrapper — its token travels in an ``export``
    line in the wrapper script itself, same as the ANTHROPIC_ENV case, but
    the shape also writes a sibling ``.codex/<alias>.config.toml`` profile.
    """
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    return build_spec(agent="codex", provider=provider, model="gpt-4o", alias=alias)


def _boom(_path):  # pragma: no cover - reached only on a regression
    raise AssertionError("must not ask")


# --- the hazard ------------------------------------------------------------


def test_discarding_the_only_copy_of_a_secret_is_refused(tmp_path):
    """No TTY means no way to ask, so it must fail fast rather than destroy."""
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="ONLY-COPY")

    with pytest.raises(CodeHelperError, match="only copy of its token"):
        install_wrapper(paths, _literal(), confirm=None)

    assert "ONLY-COPY" in (paths.bin_dir / "w").read_text(), "token was destroyed"


def test_confirming_allows_the_replacement(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="ONLY-COPY")

    assert install_wrapper(paths, _literal(), confirm=lambda _p: True) is True
    assert "ONLY-COPY" not in (paths.bin_dir / "w").read_text()


def test_declining_leaves_the_wrapper_untouched(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="ONLY-COPY")

    with pytest.raises(CodeHelperError):
        install_wrapper(paths, _literal(), confirm=lambda _p: False)

    assert "ONLY-COPY" in (paths.bin_dir / "w").read_text()


def test_force_overrides_without_asking(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="ONLY-COPY")

    assert install_wrapper(paths, _literal(), force=True, confirm=_boom) is True


def test_dry_run_never_prompts_and_never_writes(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="ONLY-COPY")

    assert install_wrapper(paths, _literal(), dry_run=True, confirm=_boom) is True
    assert "ONLY-COPY" in (paths.bin_dir / "w").read_text()


# --- what must stay silent -------------------------------------------------


def test_a_model_bump_never_prompts(tmp_path):
    """``add``-as-update is the primary use and must not acquire a prompt."""
    paths = _bin(tmp_path)
    install_wrapper(paths, _literal(model="m1"))

    assert install_wrapper(paths, _literal(model="m2"), confirm=_boom) is True


def test_rotating_a_secret_never_prompts(tmp_path):
    """secret -> secret replaces a token the user was just asked to supply.

    The old token does go away, but not by surprise: the incoming wrapper
    prompted for one, so this is a deliberate replacement, not a silent loss.
    """
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="TOKEN-A")

    assert install_wrapper(paths, _secret(), token="TOKEN-B", confirm=_boom) is True
    assert "TOKEN-B" in (paths.bin_dir / "w").read_text()


def test_replacing_a_literal_wrapper_never_prompts(tmp_path):
    """Nothing irrecoverable is lost when the outgoing token was a constant."""
    paths = _bin(tmp_path)
    install_wrapper(paths, _literal())

    assert install_wrapper(paths, _secret(), token="tok", confirm=_boom) is True


def test_an_identical_reinstall_is_still_a_no_op(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret(), token="tok")

    assert install_wrapper(paths, _secret(), token="tok", confirm=_boom) is False


# --- OPENAI_TOML secret wrapper with a missing/corrupt profile sibling -----
#
# Cycle-review round 3, finding C1: spec_from_installed needs the sibling
# .codex/<alias>.config.toml to recover the MODEL for an OPENAI_TOML wrapper.
# If that profile is missing or corrupt, spec_from_installed correctly
# returns None (per its own "cannot fully reconstruct" contract) — but
# _discards_only_secret was reading that None as "not a secret wrapper,
# nothing at stake" and letting a replacement through silently, with no
# --force and no confirm prompt, destroying the token embedded in the
# wrapper script even though the profile never carried it in the first
# place. The marker line alone (unaffected by the profile's state) already
# names the provider as secret-auth; the guard must consult that.


def test_discarding_a_toml_secret_survives_a_missing_profile_sibling(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret_openai_toml(), token="ONLY-COPY")
    paths.codex_config_for("w").unlink()  # the sibling profile is gone

    with pytest.raises(CodeHelperError, match="only copy of its token"):
        install_wrapper(paths, _literal(), confirm=None)

    assert "ONLY-COPY" in (paths.bin_dir / "w").read_text(), "token was destroyed"


def test_discarding_a_toml_secret_survives_a_corrupted_profile_sibling(tmp_path):
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret_openai_toml(), token="ONLY-COPY")
    paths.codex_config_for("w").write_text("not valid toml {{{", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="only copy of its token"):
        install_wrapper(paths, _literal(), confirm=None)

    assert "ONLY-COPY" in (paths.bin_dir / "w").read_text(), "token was destroyed"


def test_toml_secret_with_a_missing_profile_still_honours_force(tmp_path):
    """--force still overrides, same as every other secret-discard case —
    this fallback path must not become a NEW, stricter guard than the one it
    patches a gap in.
    """
    paths = _bin(tmp_path)
    install_wrapper(paths, _secret_openai_toml(), token="ONLY-COPY")
    paths.codex_config_for("w").unlink()

    assert install_wrapper(paths, _literal(), force=True, confirm=_boom) is True
