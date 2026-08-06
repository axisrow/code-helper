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

from code_helper.__main__ import main
from code_helper.errors import CodeHelperError
from code_helper.services.model import get_provider
from code_helper.services.models_api import list_models
from code_helper.services.paths import Paths
from code_helper.services.render import render_legacy_script
from code_helper.services.spec import build_spec
from code_helper.services.wrappers import get_spec, install_wrapper

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
    assert "code-helper: managed wrapper" not in legacy
    (paths.bin_dir / "glm-ollama").write_text(legacy)

    # confirm=None means "no way to ask" — a legacy file must not need asking.
    assert install_wrapper(paths, "glm-ollama", confirm=None) is True
    assert "code-helper: managed wrapper" in (paths.bin_dir / "glm-ollama").read_text()


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
    from code_helper.services.wrappers import _is_ours, _read_text_or_none

    paths = _bin(tmp_path)
    spec = build_spec(agent="claude", provider="ollama", model="m", alias="weird")
    (paths.bin_dir / "weird").write_bytes(b"\xff\xfe\xfd")

    assert _read_text_or_none(paths.bin_dir / "weird") is None
    assert _is_ours(paths, spec, "") is False


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
