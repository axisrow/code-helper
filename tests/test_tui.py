"""Tests for the ``tui`` CLI subcommand (and the bare ``code-helper`` default).

Follows the ``tests/test_edit_token.py`` "CLI through main([...])" pattern:
HOME is patched by the autouse ``_isolate_home`` fixture, so
``Paths.from_home(tmp_path)`` resolves to the same directory the handler
writes to via ``Paths.default()``. The fake ``select_from_menu`` is patched
on ``code_helper.cli.menu`` (not imported at module scope by ``tui.py`` /
``parser.py``) so the patch lands — same trick as
``test_edit_token_no_name_uses_menu``.
"""

from __future__ import annotations

import pytest

from code_helper.__main__ import main
from code_helper.services.paths import Paths

_NEW_TOKEN = "11111111111111111111111111111111.bbbbbbbbbbbbbbbb"


def _menu_sequence(monkeypatch, answers):
    """Patch select_from_menu to return successive `answers` on each call."""
    it = iter(answers)

    def _fake(items, **_kw):
        return next(it)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _fake)


@pytest.mark.integration
def test_tui_list_dispatches_to_handler(tmp_path, monkeypatch, capsys):
    _menu_sequence(monkeypatch, ["list", "quit"])

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out
    assert "glm" in out


@pytest.mark.integration
def test_bare_invocation_opens_tui(tmp_path, monkeypatch, capsys):
    _menu_sequence(monkeypatch, ["list", "quit"])

    assert main([]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out
    assert "glm" in out


@pytest.mark.integration
def test_tui_add_installs_same_as_cli(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert main(["tui"]) == 0

    tui_body = (
        Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    )

    # Compare against a clean CLI install in a second, separate HOME.
    other_home = tmp_path.parent / f"{tmp_path.name}-cli-control"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    assert main(["add", "deepseek"]) == 0
    cli_body = (
        Paths.from_home(other_home).script_for("deepseek").read_text(encoding="utf-8")
    )

    assert tui_body == cli_body


@pytest.mark.integration
def test_tui_add_model_override_reaches_handler(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "my-model")

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    assert "my-model" in body


@pytest.mark.integration
def test_tui_add_empty_model_means_default(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "   ")

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    assert "deepseek-v4-flash:0731-cloud" in body


@pytest.mark.integration
def test_tui_edit_token_delegates(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ZAI_API_KEY", "00000000000000000000000000000000.aaaaaaaaaaaaaaaa"
    )
    assert main(["add", "glm"]) == 0
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    _menu_sequence(monkeypatch, ["edit-token", "glm", "quit"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert _NEW_TOKEN in body


@pytest.mark.integration
def test_tui_dry_run_toggle_prevents_write(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["dry-run: off", "add", "deepseek", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_loops_after_command(tmp_path, monkeypatch, capsys):
    # After `list` the menu reappears and `add` runs — proof of the loop.
    _menu_sequence(monkeypatch, ["list", "add", "deepseek", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out  # `list` output
    assert Paths.from_home(tmp_path).script_for("deepseek").exists()  # `add` ran


@pytest.mark.integration
def test_tui_error_returns_to_menu(tmp_path, monkeypatch, capsys):
    from code_helper.errors import CodeHelperError

    def _boom(**_kw):
        raise CodeHelperError("simulated failure")

    monkeypatch.setattr("code_helper.services.secrets.resolve_token", _boom)
    _menu_sequence(monkeypatch, ["add", "glm", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert main(["tui"]) == 0

    err = capsys.readouterr().err
    assert "error:" in err
    assert "simulated failure" in err
    # The error did not exit the loop: `quit` was reached, and nothing was written.
    assert not Paths.from_home(tmp_path).script_for("glm").exists()


@pytest.mark.integration
def test_tui_cancel_writes_nothing(tmp_path, monkeypatch, capsys):
    from code_helper.cli.menu import MenuCancelled

    def _cancel(items, **_kw):
        raise MenuCancelled()

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _cancel)

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "cancelled" in out
    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_quit_returns_zero(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["quit"])

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()
