"""Regression tests for the findings of the first local review round on PR #6.

Each test here encodes a claim that was reproduced against ``1d3ca18`` before
any fix existed. They are grouped by the surface they pin rather than by
reviewer, since two reviewers independently landed on ``install_wrapper``'s
pre-guard read for different reasons.
"""

from __future__ import annotations

import http.client
import pathlib

import pytest

from codehelper.__main__ import main
from codehelper.errors import CodeHelperError
from codehelper.services.model import get_provider
from codehelper.services.models_api import list_models
from codehelper.services.paths import Paths
from codehelper.services.render import render_legacy_script
from codehelper.services.spec import build_spec
from codehelper.services.wrappers import get_spec, install_wrapper

pytestmark = pytest.mark.unit


def _bin(tmp_path: pathlib.Path) -> Paths:
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    return paths


# --- install_wrapper: the pre-guard read -----------------------------------


def test_non_utf8_file_at_target_does_not_crash_the_guard(tmp_path):
    """A binary at the target path must reach the ownership guard, not a traceback.

    ``is_managed`` deliberately catches ``UnicodeDecodeError`` so an unreadable
    file counts as "not ours". The idempotence read that runs BEFORE it had no
    such protection, so the guard it was meant to feed never ran.
    """
    paths = _bin(tmp_path)
    spec = build_spec(agent="claude", provider="ollama", model="m", alias="binfile")
    (paths.bin_dir / "binfile").write_bytes(b"\x7fELF\x02\x01\x01\x00\xff\xfe\xfd")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, spec, confirm=None)


def test_force_overwrites_a_non_utf8_file(tmp_path):
    """``--force`` must be able to rescue the binary-in-the-way case."""
    paths = _bin(tmp_path)
    spec = build_spec(agent="claude", provider="ollama", model="m", alias="binfile")
    (paths.bin_dir / "binfile").write_bytes(b"\xff\xfe\xfd")

    assert install_wrapper(paths, spec, force=True) is True
    assert (paths.bin_dir / "binfile").read_text().startswith("#!/bin/bash")


def test_a_legacy_markerless_wrapper_is_claimed_not_refused(tmp_path):
    """A wrapper written by the previous release must stay updatable.

    The marker did not exist before this branch, so every already-installed
    wrapper lacks it. Treating those as third-party files breaks the upgrade
    path for exactly the users who already run the tool.
    """
    paths = _bin(tmp_path)
    # Exactly what the pre-marker release wrote for this preset: the current
    # body minus the marker line.
    legacy = render_legacy_script(get_spec("glm-ollama"), "")
    assert "codehelper: managed wrapper" not in legacy
    (paths.bin_dir / "glm-ollama").write_text(legacy)

    # confirm=None means "no way to ask" — a legacy file must not need asking.
    assert install_wrapper(paths, "glm-ollama", confirm=None) is True
    assert "codehelper: managed wrapper" in (paths.bin_dir / "glm-ollama").read_text()


def test_a_genuinely_foreign_file_is_still_refused(tmp_path):
    """The legacy carve-out must not become a blanket bypass of the guard."""
    paths = _bin(tmp_path)
    (paths.bin_dir / "glm-ollama").write_text("#!/bin/bash\necho not ours\n")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, "glm-ollama", confirm=None)


def test_an_unreadable_file_is_never_mistaken_for_ours(tmp_path):
    """An undecodable file must read as "not ours", never as a match.

    Pins the None-vs-empty-string distinction in the pre-guard read: were an
    unreadable file to come back as ``""``, it would compare equal to a spec
    whose legacy render is empty and slip through the guard unasked. Uses a
    shape with no renderer, so the rendered body IS empty — the exact case
    that makes the distinction observable rather than academic.
    """
    from codehelper.backends._atomic import read_text_or_none
    from codehelper.services.wrappers import _ownership_full_match

    paths = _bin(tmp_path)
    spec = build_spec(agent="claude", provider="ollama", model="m", alias="weird")
    (paths.bin_dir / "weird").write_bytes(b"\xff\xfe\xfd")

    assert read_text_or_none(paths.bin_dir / "weird") is None
    assert _ownership_full_match(paths, spec, "") is False


# --- CLI contract ----------------------------------------------------------


def test_bogus_shape_is_a_clean_error_not_a_traceback(capsys):
    """``--shape <bogus>`` must fail the way every other bad input fails."""
    rc = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "ollama",
            "--model",
            "m",
            "--shape",
            "bogus",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "bogus" in err
    assert "Traceback" not in err


def test_unknown_name_still_lists_the_known_names(capsys):
    """The recovery hint must survive the bare-agent-name special case."""
    rc = main(["add", "definitely-not-a-preset"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "glm-ollama" in err


# --- edit-token ------------------------------------------------------------


def test_edit_token_preserves_a_customized_model(tmp_path, monkeypatch):
    """Rotating a credential must not silently revert an unrelated setting."""
    paths = _bin(tmp_path)
    install_wrapper(paths, "glm", token="tok1", model_override="custommodel")
    assert "custommodel" in (paths.bin_dir / "glm").read_text()

    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "tok2")
    assert main(["edit-token", "glm"]) == 0

    body = (paths.bin_dir / "glm").read_text()
    assert "custommodel" in body, "edit-token re-expanded the preset defaults"
    assert "tok2" in body


def test_edit_token_preserves_non_uniform_tier_models(tmp_path, monkeypatch):
    """Rotation must not flatten a preset whose tiers genuinely differ.

    ``glm`` is the reason presets still exist: haiku/sonnet/opus are three
    DIFFERENT models, which no single ``--model`` can express. Reconstructing
    the spec from one recovered model and letting ``build_spec`` synthesize
    uniform tiers turned the cycle-1 fix into a wider version of the bug it
    was closing — and the test written for that fix only covered a uniform
    override, so it could not see this.
    """
    paths = _bin(tmp_path)
    install_wrapper(paths, "glm", token="tok1")
    before = (paths.bin_dir / "glm").read_text()
    assert "'glm-4.7'" in before and "'glm-5.1'" in before

    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "tok2")
    assert main(["edit-token", "glm"]) == 0

    after = (paths.bin_dir / "glm").read_text()
    tiers = [ln for ln in after.splitlines() if "_MODEL=" in ln]
    assert tiers == [ln for ln in before.splitlines() if "_MODEL=" in ln], (
        "edit-token rewrote tier models it was not asked to touch"
    )
    assert "tok2" in after


def test_edit_token_rotates_a_legacy_customized_install(tmp_path, monkeypatch):
    """A markerless install customized with ``--model`` must still be rotatable.

    The guard's message tells the user to pass ``--force``, but ``edit-token``
    accepts no such flag — so this was a dead end with a hint pointing at
    something that does not exist, on a file this tool wrote.
    """
    from codehelper.services.spec import get_preset, spec_from_preset

    paths = _bin(tmp_path)
    legacy = spec_from_preset(get_preset("glm"), model_override="customX")
    (paths.bin_dir / "glm").write_text(render_legacy_script(legacy, "oldtok"))

    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "newtok")
    assert main(["edit-token", "glm"]) == 0

    after = (paths.bin_dir / "glm").read_text()
    assert "newtok" in after
    assert "customX" in after, "rotation discarded the legacy model choice"


def test_a_foreign_file_named_after_a_preset_is_not_adopted(tmp_path):
    """Sharing a preset's NAME must not be enough to be recognised as ours.

    The legacy lookup takes its axes from a same-named preset, so the only
    thing separating "our old output" from "someone else's file called glm"
    is the re-render proof: the reconstructed spec must reproduce the file's
    every non-token byte. Without that check this path would adopt — and then
    silently overwrite — any file whose name happened to match a preset.
    """
    from codehelper.services.wrappers import spec_from_installed

    paths = _bin(tmp_path)
    (paths.bin_dir / "glm").write_text(
        "#!/bin/bash\n"
        "(\n"
        "export ANTHROPIC_DEFAULT_HAIKU_MODEL='x'\n"
        "export ANTHROPIC_DEFAULT_SONNET_MODEL='y'\n"
        "export ANTHROPIC_DEFAULT_OPUS_MODEL='z'\n"
        "rm -rf /\n"
        ")\n"
    )

    assert spec_from_installed(paths, "glm") is None
    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, "glm", token="t", confirm=None)


def test_edit_token_reaches_an_axes_built_wrapper(tmp_path, monkeypatch):
    """A secret wrapper from the constructor is a first-class product of this PR."""
    paths = _bin(tmp_path)
    monkeypatch.setenv("ZAI_API_KEY", "tok1")
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "zai",
                "--model",
                "glm-x",
                "--alias",
                "mytok",
            ]
        )
        == 0
    )

    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "tok2")
    assert main(["edit-token", "mytok"]) == 0
    assert "tok2" in (paths.bin_dir / "mytok").read_text()


# --- cycle 3 ---------------------------------------------------------------


def test_spec_from_installed_keeps_the_recorded_shape(tmp_path):
    """The marker records the mechanism; re-deriving it can pick a different one.

    ``claude × ollama`` is compatible with BOTH shapes, and ``resolve_shape``
    prefers ``ANTHROPIC_ENV``. Dropping the marker's ``shape=`` therefore turns
    a wrapper installed as ``ollama launch`` into an ``ANTHROPIC_*`` env block
    — not a changed model but a changed mechanism.
    """
    from codehelper.services.model import ConfigShape
    from codehelper.services.render import render_script
    from codehelper.services.wrappers import spec_from_installed

    paths = _bin(tmp_path)
    install_wrapper(paths, "glm-ollama")
    before = (paths.bin_dir / "glm-ollama").read_text()

    spec = spec_from_installed(paths, "glm-ollama")
    assert spec is not None
    assert spec.shape is ConfigShape.OLLAMA_LAUNCH
    assert render_script(spec, "") == before, "re-render changed the mechanism"


def test_hard_cancel_at_the_picker_is_not_a_raw_traceback(monkeypatch, capsys):
    """Ctrl-C must not reach the terminal as a ``MenuCancelled`` dump.

    The exception still has to escape ``main`` — that is how a TUI caller
    tells "go back" from "leave the whole TUI", pinned by
    ``test_edit_token_hard_cancel_propagates``. So the translation belongs at
    the process boundary (``cli``), which is what the console script runs.
    """
    from codehelper.__main__ import cli
    from codehelper.cli.menu import MenuCancelled

    def _hard(*a, **k):
        raise MenuCancelled(hard=True)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _hard)
    monkeypatch.setattr("sys.argv", ["codehelper", "edit-token"])

    assert cli() == 130
    assert "Traceback" not in capsys.readouterr().err

    # ...and main() still lets it out, so the TUI contract is intact.
    with pytest.raises(MenuCancelled):
        main(["edit-token"])


def test_model_override_refreshes_the_description(tmp_path):
    """``list`` is the only place a user sees what a wrapper points at."""
    from codehelper.services.spec import get_preset, spec_from_preset

    spec = spec_from_preset(get_preset("deepseek"), model_override="qwen3")
    assert "deepseek-v4-flash" not in spec.description
    assert "qwen3" in spec.description


def test_discover_managed_skips_structurally_invalid_names(tmp_path):
    """Don't advertise a wrapper no command in the tool can act on — a name
    that is not a valid alias shape (whitespace, a leading dash) is skipped.
    A RESERVED name is still listed: it is a real, removable wrapper, and
    hiding it would strand it on PATH (see
    test_reserved_named_managed_wrapper_stays_discoverable_and_removable)."""
    from codehelper.services.render import MARKER_PREFIX
    from codehelper.services.wrappers import discover_managed

    paths = _bin(tmp_path)
    for bad in ("has space", "-leading-dash"):
        (paths.bin_dir / bad).write_text(
            f"#!/bin/bash\n{MARKER_PREFIX} "
            f"(agent=claude, provider=ollama, shape=anthropic-env)\n"
            f'claude "$@"\n'
        )

    assert discover_managed(paths) == []


# --- models_api ------------------------------------------------------------


def test_http_protocol_error_does_not_escape_list_models():
    """``HTTPException`` is not an ``OSError`` — the never-raises contract leaked.

    A server closing mid-body raises ``IncompleteRead``, which every call site
    is documented not to have to catch.
    """

    def fetch(url, timeout, token):
        raise http.client.IncompleteRead(b"partial")

    result = list_models(get_provider("ollama"), fetch=fetch)
    assert not result.ok
    assert result.models == ()
