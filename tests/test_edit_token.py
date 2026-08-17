"""Tests for the ``edit-token`` CLI subcommand.

Follows the ``tests/test_wrappers.py`` "CLI through main([...])" pattern:
HOME is patched by the autouse ``_isolate_home`` fixture, so
``Paths.from_home(tmp_path)`` resolves to the same directory the handler
writes to via ``Paths.default()``.
"""

from __future__ import annotations

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths

_OLD_TOKEN = "00000000000000000000000000000000.aaaaaaaaaaaaaaaa"
_NEW_TOKEN = "11111111111111111111111111111111.bbbbbbbbbbbbbbbb"


@pytest.mark.integration
def test_edit_token_with_name_rewrites_token(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", _OLD_TOKEN)
    assert main(["add", "glm"]) == 0
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)
    assert main(["edit-token", "glm"]) == 0

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert _NEW_TOKEN in body
    assert _OLD_TOKEN not in body


@pytest.mark.integration
def test_edit_token_can_rotate_a_named_profile(tmp_path, monkeypatch):
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", _OLD_TOKEN, "work")
    assert main(["add", "glm", "--profile", "work"]) == 0

    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)
    assert main(["edit-token", "glm", "--profile", "work"]) == 0

    assert secrets.credential_for(paths, "zai", "work") == _NEW_TOKEN
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert _NEW_TOKEN in body


@pytest.mark.integration
def test_edit_token_unknown_name_exits_1(tmp_path, capsys):
    code = main(["edit-token", "nope"])
    assert code == 1
    err = capsys.readouterr().err
    assert "unknown wrapper" in err


@pytest.mark.integration
def test_edit_token_literal_auth_wrapper_rejected(tmp_path, capsys):
    code = main(["edit-token", "deepseek"])
    assert code == 1
    err = capsys.readouterr().err
    assert "no editable token" in err
    assert "auth=literal" in err


@pytest.mark.integration
def test_edit_token_empty_input_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "")
    code = main(["edit-token", "glm"])
    assert code == 1


@pytest.mark.integration
def test_edit_token_dry_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)
    assert main(["--dry-run", "edit-token", "glm"]) == 0

    paths = Paths.from_home(tmp_path)
    assert not paths.script_for("glm").exists()


@pytest.mark.integration
def test_edit_token_no_name_uses_menu(tmp_path, monkeypatch):
    # select_from_menu's `read_key` default is bound at def-time, so patching
    # `_read_key_raw` after import has no effect; patch select_from_menu
    # itself instead (its own key-parsing behavior is covered by test_menu.py).
    monkeypatch.setattr(
        "codehelper.cli.menu.select_from_menu", lambda items, **_kw: "glm"
    )
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)

    assert main(["edit-token"]) == 0

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert _NEW_TOKEN in body


@pytest.mark.integration
def test_edit_token_menu_cancelled_writes_nothing(tmp_path, monkeypatch, capsys):
    # Esc/q — a soft cancel — prints "cancelled" and returns 0, same as ever.
    from codehelper.cli.menu import MenuCancelled

    def _cancel(items, **_kw):
        raise MenuCancelled(hard=False)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _cancel)

    assert main(["edit-token"]) == 0

    out = capsys.readouterr().out
    assert "cancelled" in out
    paths = Paths.from_home(tmp_path)
    assert not paths.script_for("glm").exists()


@pytest.mark.integration
def test_edit_token_hard_cancel_propagates(tmp_path, monkeypatch):
    # Ctrl-C (a hard cancel) is NOT swallowed into "cancelled" + exit 0 — it
    # propagates like any other uncaught KeyboardInterrupt in the plain CLI,
    # so a TUI caller can distinguish "back" from "leave the whole TUI".
    from codehelper.cli.menu import MenuCancelled

    def _cancel(items, **_kw):
        raise MenuCancelled(hard=True)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _cancel)

    with pytest.raises(MenuCancelled):
        main(["edit-token"])

    paths = Paths.from_home(tmp_path)
    assert not paths.script_for("glm").exists()
