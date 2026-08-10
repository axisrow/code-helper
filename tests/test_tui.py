"""Integration coverage for the provider-first interactive TUI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_helper.__main__ import main
from code_helper.services.paths import Paths


def _menu_sequence(monkeypatch, answers):
    iterator = iter(answers)

    def _select(_items, **_kwargs):
        return next(iterator)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)


def _real_menu_keys(monkeypatch, keys):
    import code_helper.cli.menu as menu

    iterator = iter(keys)
    real_select = menu.select_from_menu

    def _select(items, **kwargs):
        return real_select(items, read_key=lambda: next(iterator), **kwargs)

    monkeypatch.setattr(menu, "select_from_menu", _select)


@pytest.mark.integration
def test_main_menu_is_english_and_list_is_a_back_navigable_second_level(
    monkeypatch, capsys
):
    # Main -> List; wrapper browser -> Back; main -> Quit.
    _real_menu_keys(monkeypatch, ["ENTER", "END", "ENTER", "END", "ENTER"])

    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    assert "List" in output
    assert "Wrappers:" in output
    assert "Back" in output
    assert "Press any key" not in output
    assert "назад" not in output
    assert "New" not in output
    assert "edit-token" not in output


@pytest.mark.integration
def test_add_order_is_provider_model_agent_alias(monkeypatch):
    seen: list[str] = []
    answers = iter(["add", "ollama", "model-x", "codex", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        seen.append(prompt)
        return next(answers)

    import code_helper.services.models_api as api

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "my-codex")

    assert main(["tui"]) == 0
    assert seen[:4] == [
        "code-helper",
        "Select a provider:",
        "Select a model for ollama:",
        "Select an agent:",
    ]
    assert Paths.default().script_for("my-codex").exists()


@pytest.mark.integration
def test_literal_provider_skips_profile_screen(monkeypatch):
    seen: list[str] = []
    answers = iter(["add", "ollama", "model-x", "claude", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        seen.append(prompt)
        return next(answers)

    import code_helper.services.models_api as api

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "local")

    assert main(["tui"]) == 0
    assert not any("Token profile" in prompt for prompt in seen)


@pytest.mark.integration
def test_escape_from_alias_returns_to_agent_without_creating_wrapper(monkeypatch):
    import code_helper.services.models_api as api
    from code_helper.cli.menu import MenuCancelled

    answers = iter(["add", "ollama", "__custom__", "claude", "claude", "quit"])

    def _select(_items, **_kwargs):
        return next(answers)

    text = iter(["model-x", "__escape__", "final-wrapper"])

    def _read_line(_prompt, **_kwargs):
        value = next(text)
        if value == "__escape__":
            raise MenuCancelled(hard=False)
        return value

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("code_helper.cli.menu.read_line", _read_line)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )

    assert main(["tui"]) == 0
    assert Paths.default().script_for("final-wrapper").exists()


@pytest.mark.integration
def test_ctrl_c_from_tui_propagates_as_hard_cancel(monkeypatch):
    from code_helper.cli.menu import MenuCancelled

    def _hard(*_args, **_kwargs):
        raise MenuCancelled(hard=True)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _hard)
    with pytest.raises(MenuCancelled) as exc_info:
        main(["tui"])
    assert exc_info.value.hard is True


@pytest.mark.integration
def test_first_secret_token_creates_default_before_model(monkeypatch):
    _menu_sequence(monkeypatch, ["add", "zai", "__custom__", "claude", "quit"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-first")
    typed = iter(["glm-5", "glm-work"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))

    assert main(["tui"]) == 0

    import code_helper.services.secrets as secrets

    paths = Paths.default()
    assert secrets.credential_for(paths, "zai", "default") == "sk-first"
    assert "sk-first" in paths.script_for("glm-work").read_text()


@pytest.mark.integration
def test_second_profile_names_both_keys_and_preserves_them(monkeypatch):
    _menu_sequence(
        monkeypatch,
        [
            "add",
            "zai",
            "__custom__",
            "claude",
            "add",
            "zai",
            "__new_profile__",
            "__custom__",
            "claude",
            "quit",
        ],
    )
    tokens = iter(["sk-work", "sk-personal"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(tokens))
    typed = iter(
        ["glm-5", "work-wrapper", "work", "personal", "glm-5", "personal-wrapper"]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))

    assert main(["tui"]) == 0

    import code_helper.services.secrets as secrets

    assert secrets.load_credentials(Paths.default())["zai"] == {
        "personal": "sk-personal",
        "work": "sk-work",
    }


@pytest.mark.integration
def test_replacing_selected_profile_changes_only_that_profile(monkeypatch):
    import code_helper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-work", "work")
    secrets.save_credential(paths, "zai", "sk-personal", "personal")
    _menu_sequence(
        monkeypatch,
        ["add", "zai", "work", "__replace_token__", "__custom__", "claude", "quit"],
    )
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-work-new")
    typed = iter(["glm-5", "work-wrapper"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))

    assert main(["tui"]) == 0
    assert secrets.load_credentials(paths)["zai"] == {
        "personal": "sk-personal",
        "work": "sk-work-new",
    }


@pytest.mark.integration
def test_litellm_uses_selected_profile_for_model_discovery(monkeypatch):
    """A named profile's cached token IS trusted for model discovery, even
    for a REQUIRED-policy provider like ``litellm`` — the user explicitly
    chose this profile for this invocation, the same deliberate choice that
    already lets a profile win over the environment in ``resolve_token``.
    """
    import code_helper.services.models_api as api
    import code_helper.services.secrets as secrets

    secrets.save_credential(Paths.default(), "litellm", "sk-work", "work")
    captured: list[tuple[str, str]] = []

    def _models(provider, *, token=""):
        captured.append((provider.base_url, token))
        return api.ModelListResult(("gpt-test",), "fake")

    _menu_sequence(
        monkeypatch,
        ["add", "litellm", "work", "__use_profile__", "gpt-test", "codex", "quit"],
    )
    monkeypatch.setattr(api, "list_models", _models)
    typed = iter(["http://proxy.example/v1", "litellm-wrapper"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))

    assert main(["tui"]) == 0
    assert captured == [("http://proxy.example/v1", "sk-work")]


@pytest.mark.integration
def test_list_selects_secret_wrapper_for_token_edit(monkeypatch):
    import code_helper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-old", "default")
    assert main(["add", "glm", "--profile", "default"]) == 0
    _menu_sequence(monkeypatch, ["list", "glm", "default", "__back__", "quit"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-new")

    assert main(["tui"]) == 0
    assert secrets.credential_for(paths, "zai", "default") == "sk-new"
    assert "sk-new" in paths.script_for("glm").read_text()


@pytest.mark.integration
def test_list_handles_a_managed_wrapper_whose_marker_names_an_unknown_provider(
    monkeypatch, capsys
):
    """A managed-but-unrecognized wrapper must not crash the TUI's List loop.

    ``discover_managed`` lists any marker-carrying file, but
    ``spec_from_installed`` returns ``None`` when the marker's ``provider=``
    field names a provider this build doesn't recognise (e.g. one dropped
    from the registry after install). Before this fix, the ``_run_list``
    fallback to ``get_spec`` then raised an uncaught ``CodeHelperError`` for
    such a file, since it also isn't a preset name — crashing the whole TUI
    instead of reporting the error and returning to the menu.
    """
    paths = Paths.default()
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = paths.script_for("orphaned-wrapper")
    wrapper.write_text(
        "#!/bin/sh\n"
        "# code-helper: managed wrapper (agent=claude, provider=defunct, "
        "shape=anthropic-env)\n"
        "exit 0\n"
    )
    wrapper.chmod(0o755)

    _menu_sequence(monkeypatch, ["list", "orphaned-wrapper", "__back__", "quit"])

    assert main(["tui"]) == 0
    assert "error" in capsys.readouterr().err.lower()


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="PTY tests require POSIX")
def test_provider_first_flow_through_a_real_pty(tmp_path):
    """Smoke-test raw keys and the new second-level List navigation."""
    import errno
    import pty
    import select
    import subprocess
    import sys
    import termios
    import time

    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment.pop("ZAI_API_KEY", None)
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_dir, environment.get("PYTHONPATH")) if part
    )
    master_fd, slave_fd = pty.openpty()
    child = subprocess.Popen(
        [sys.executable, "-m", "code_helper", "tui"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    output = bytearray()
    search_from = 0

    def read_until(marker: str) -> None:
        nonlocal search_from
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            found = output.find(marker.encode(), search_from)
            if found >= 0:
                search_from = found + len(marker)
                return
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
        raise AssertionError(output.decode(errors="replace")[-3000:])

    def menu_key(value: str) -> None:
        # The frame is printed before the child switches the terminal into
        # raw mode. Wait until the read side is ready before sending a key.
        deadline = time.monotonic() + 5
        while termios.tcgetattr(master_fd)[3] & termios.ICANON:
            if time.monotonic() >= deadline:
                raise AssertionError("TUI did not enter raw mode")
            time.sleep(0.01)
        os.write(master_fd, value.encode())

    try:
        read_until("Esc quit")
        menu_key("1")
        read_until("Esc back")
        menu_key("\x1b")
        read_until("Esc quit")
        assert b"Press any key" not in output
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        os.close(master_fd)
        os.close(slave_fd)
