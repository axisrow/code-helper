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
def test_main_menu_lists_wrappers_grouped_by_agent_and_quit_exits(monkeypatch, capsys):
    # The wrapper list IS the main screen now (issue #29): no separate "List"
    # entry. END jumps to the last numbered item (Quit), Enter exits.
    _real_menu_keys(monkeypatch, ["END", "ENTER"])

    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    # Wrappers appear directly on the main screen, grouped by agent section.
    assert "claude" in output
    assert "Add" in output
    assert "Quit" in output
    # The old indirection is gone.
    assert "Wrappers:" not in output
    # No pause screens.
    assert "Press any key" not in output
    assert "назад" not in output
    assert "edit-token" not in output


@pytest.mark.integration
def test_add_order_is_provider_model_agent_alias(monkeypatch):
    seen: list[str] = []
    answers = iter(["add", "ollama", "model-x", "codex", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        # The main-menu header is a callable (live active-profile display).
        seen.append(prompt() if callable(prompt) else prompt)
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
        # The main-menu header is a callable (live active-profile display).
        seen.append(prompt() if callable(prompt) else prompt)
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
def test_t_rotates_token_for_secret_wrapper_from_main_screen(monkeypatch):
    """Issue #29: token rotation moved from the old List sub-screen to the `t`
    key on the main screen. With glm installed (secret), `t` on its row opens
    the profile picker and a new token is written — the same end-to-end result
    the old `list → glm → profile` flow produced, just without the sub-screen."""
    import code_helper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-old", "default")
    assert main(["add", "glm", "--profile", "default"]) == 0
    # Main screen: presets are [deepseek, glm, glm-ollama] under Section("claude").
    # DOWN moves to glm (index 1 among selectable wrappers), `t` opens its token
    # profile picker, `1` picks the first profile (default), the getpass stub
    # supplies the new token, then END + ENTER reaches Quit.
    _real_menu_keys(monkeypatch, ["DOWN", "TOKEN", "ENTER", "END", "ENTER"])
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
    assert "error" in capsys.readouterr().out.lower()


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
        # `1` is the first wrapper on the main screen (issue #29): Enter makes
        # it the default for its agent. Uninstalled wrappers now give a
        # visible instruction rather than creating a ghost default.
        menu_key("1")
        read_until("Wrapper not installed")
        menu_key(" ")
        read_until("Esc quit")
        assert b"Press any key" in output
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="PTY tests require POSIX")
def test_main_screen_default_wrapper_marker_through_a_real_pty(tmp_path):
    """Issue #29: Enter on a wrapper makes it the default (`●` marker moves),
    and the marker survives a TUI restart (state.json persistence). This is
    the manual-PTY verification the issue requires — the injected-`read_key`
    suite cannot see real ANSI redraw behavior."""
    import errno
    import fcntl
    import pty
    import select
    import struct
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

    def run_session(expect_marker_after_enter: bool) -> bytearray:
        """Spawn the TUI, press `1` (first wrapper = default), read the redraw,
        then Esc-quit. Returns the captured output."""
        master_fd, slave_fd = pty.openpty()
        # Force an 80-column terminal so a wrapper row is wide enough to show
        # the marker yet narrow enough that `_fit` truncation is exercised.
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
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
            deadline = time.monotonic() + 5
            while termios.tcgetattr(master_fd)[3] & termios.ICANON:
                if time.monotonic() >= deadline:
                    raise AssertionError("TUI did not enter raw mode")
                time.sleep(0.01)
            os.write(master_fd, value.encode())

        try:
            read_until("Esc quit")
            # Uninstalled wrappers cannot become defaults: the guidance stays
            # visible until acknowledged, then the menu redraws unchanged.
            menu_key("1")
            read_until("Wrapper not installed")
            menu_key(" ")
            read_until("Esc quit")
            if expect_marker_after_enter:
                assert b"\xe2\x97\x8f" not in output
            assert b"Press any key" in output
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
            os.close(master_fd)
            os.close(slave_fd)
        return output

    # One session is enough: fresh install, no default set, then Enter on the
    # first wrapper makes it the default and the `●` marker appears on the
    # redrawn frame. Persistence across restarts is covered by the round-trip
    # unit tests in test_state.py (set_default_wrapper/default_wrapper); this
    # test exists only to exercise the real ANSI redraw path that the
    # injected-`read_key` suite cannot see.
    run_session(expect_marker_after_enter=True)


# --- active token-profile pre-selection (issue #23) --------------------------


@pytest.mark.integration
def test_tui_profile_screen_sets_active_and_persists_across_runs(monkeypatch):
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")

    # Run 1: establish the active profile via the Profile screen.
    _menu_sequence(monkeypatch, ["profile", "litellm", "work", "quit"])
    assert main(["tui"]) == 0

    # Run 2: state.json survives — capture the main-menu header callable and
    # verify it now renders the profile the first run wrote.
    prompts = []

    def _select(_items, *, prompt, **_kwargs):
        prompts.append(prompt)
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert prompts and prompts[0]() == "code-helper — litellm/work"
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.integration
def test_tui_tab_cycles_the_active_profile(monkeypatch):
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection, set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    secrets.save_credential(paths, "litellm", "sk-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    # The main menu's on_tab handler cycles profiles; a single Tab advances
    # work -> personal (profile_names order: default, then alpha).
    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("litellm", "personal")


@pytest.mark.integration
def test_tui_tab_updates_profile_row_label_not_just_header(monkeypatch):
    """Issue #26: after Tab the header updated but the 'Profile: ...' row
    stayed frozen at its pre-Tab value, because the row label was a static
    string resolved before the menu opened while the header was a callable.
    The fix makes the row label a callable too — here we verify the row's
    callable reflects the post-Tab profile, not the pre-Tab one."""
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection, set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    secrets.save_credential(paths, "litellm", "sk-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    captured = {}

    def _select(items, *, on_tab=None, **_kwargs):
        # Find the Profile row — its label is now a callable that reads the
        # live active selection. Resolve it before Tab, then after. Skip
        # `Section` headers (issue #29 main screen) — they are not pairs.
        from code_helper.cli.menu import Section

        profile_label = next(
            entry[1]
            for entry in items
            if not isinstance(entry, Section) and entry[0] == "profile"
        )
        assert callable(profile_label), (
            "Profile row label must be a callable (issue #26)"
        )
        before = profile_label()
        if on_tab is not None:
            on_tab()
        after = profile_label()
        captured["before"] = before
        captured["after"] = after
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # Before Tab the row shows the pre-Tab profile; after Tab it tracks the
    # new active profile — the two must differ (the desync froze them equal).
    assert "work" in captured["before"]
    assert "personal" in captured["after"]
    assert captured["before"] != captured["after"]
    assert active_selection(paths) == ("litellm", "personal")


@pytest.mark.integration
def test_tui_tab_works_on_a_fresh_install_with_no_prior_profile_screen_visit(
    monkeypatch,
):
    """Reproduces the reported bug: Tab did nothing until the Profile screen
    had been visited once, because the stored selection is only ever written
    there. Deliberately does NOT call set_active_selection — only cached
    credentials exist, matching a real first-time install."""
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-one", "axisrow")
    secrets.save_credential(paths, "zai", "sk-two", "bemyownrobot")

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    selection = active_selection(paths)
    assert selection is not None
    provider, profile = selection
    assert provider == "zai"
    assert profile in ("axisrow", "bemyownrobot")


@pytest.mark.integration
def test_tui_tab_on_fresh_install_selects_the_first_profile(monkeypatch):
    """The stale/never-set case is "before the first profile": the first Tab
    must land on profile_names[0], not skip it to names[1]. Guards the
    off-by-one where the header shows names[0] but a real Tab selects
    names[1] (issue #23's _on_tab)."""
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-one", "axisrow")
    secrets.save_credential(paths, "zai", "sk-two", "bemyownrobot")
    # Deliberately no set_active_selection: fresh install, never-set state.
    names = list(secrets.profile_names(paths, "zai"))

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("zai", names[0])


@pytest.mark.integration
def test_tui_tab_is_a_silent_noop_with_no_cached_profiles(monkeypatch):
    from code_helper.services.state import load_state

    paths = Paths.default()

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert not paths.state_file().exists()
    assert load_state(paths) == {}


@pytest.mark.integration
def test_tui_tab_falls_through_a_stale_stored_provider(monkeypatch):
    """A stored active selection whose provider has no live profiles (e.g.
    credentials were cleared after the selection was made) must not wedge Tab
    silently — it should fall through to a provider that still has cached
    profiles."""
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection, set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-one", "work")
    # Stale: litellm has no cached profiles at all, unlike zai.
    set_active_selection(paths, "litellm", "ghost")

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("zai", "work")


@pytest.mark.integration
def test_tui_header_shows_active_profile_without_a_prior_profile_screen_visit(
    monkeypatch,
):
    import code_helper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-one", "work")

    prompts = []

    def _select(_items, *, prompt, **_kwargs):
        prompts.append(prompt)
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert prompts and prompts[0]() == "code-helper — zai/work"


@pytest.mark.integration
def test_tui_add_preselects_the_active_profile_first(monkeypatch):
    import code_helper.services.models_api as api
    import code_helper.services.secrets as secrets
    from code_helper.services.state import set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    secrets.save_credential(paths, "litellm", "sk-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    seen_items: list[list] = []
    answers = iter(
        ["add", "litellm", "work", "__use_profile__", "gpt-test", "codex", "quit"]
    )

    def _select(_items, **_kwargs):
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_a, **_k: api.ModelListResult(("gpt-test",), "fake"),
    )
    typed = iter(["http://proxy.example/v1", "litellm-wrapper"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-anything")

    assert main(["tui"]) == 0
    assert paths.script_for("litellm-wrapper").exists()
    # seen_items order: [0] main, [1] provider, [2] profile picker, ...
    profile_picker = seen_items[2]
    # The active profile is FIRST, pre-selected (cursor lands on it), marked.
    assert profile_picker[0] == ("work", "work (active)")
    assert "personal" in [value for value, _ in profile_picker]


@pytest.mark.integration
def test_tui_stale_active_profile_falls_back_without_crashing(monkeypatch):
    import code_helper.services.models_api as api
    import code_helper.services.secrets as secrets
    from code_helper.services.state import set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    # The stored active profile "ghost" no longer exists in the cache (it was
    # renamed/dropped) — the pre-selection must not install a wrapper under it.
    set_active_selection(paths, "litellm", "ghost")

    seen_items: list[list] = []
    answers = iter(
        ["add", "litellm", "work", "__use_profile__", "gpt-test", "codex", "quit"]
    )

    def _select(_items, **_kwargs):
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_a, **_k: api.ModelListResult(("gpt-test",), "fake"),
    )
    typed = iter(["http://proxy.example/v1", "litellm-wrapper"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-anything")

    assert main(["tui"]) == 0
    assert paths.script_for("litellm-wrapper").exists()
    profile_picker = seen_items[2]
    # No (active) marker, and the stale "ghost" was never offered.
    assert all("(active)" not in label for _v, label in profile_picker)
    assert "ghost" not in [value for value, _ in profile_picker]


# --- overridable provider auth (ollama can be literal OR secret) ------------


@pytest.mark.integration
def test_tui_provider_list_offers_ollama_with_a_token(monkeypatch):
    """An OVERRIDABLE provider gets a SECOND provider-list row rather than an
    extra interstitial screen — the common "just want ollama" path stays a
    single Enter (see cli/tui.py's _run_add)."""
    seen_items: list[list] = []
    answers = iter(["add", "__back__", "quit"])

    def _select(_items, **_kwargs):
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # seen_items order: [0] main menu, [1] provider list.
    provider_list = seen_items[1]
    values = [value for value, _label in provider_list]
    assert "ollama" in values
    assert "ollama:secret" in values
    # litellm/zai are FIXED — no second row for either.
    assert "litellm:secret" not in values
    assert "zai:secret" not in values


@pytest.mark.integration
def test_tui_add_ollama_with_token_installs_a_secret_wrapper(monkeypatch):
    import code_helper.services.models_api as api

    answers = iter(["add", "ollama:secret", "model-x", "claude", "quit"])

    def _select(_items, **_kwargs):
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_a, **_k: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "ollama-tok")
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-ollama-proxy")

    assert main(["tui"]) == 0
    paths = Paths.default()
    body = paths.script_for("ollama-tok").read_text(encoding="utf-8")
    assert "export ANTHROPIC_AUTH_TOKEN='sk-ollama-proxy'" in body
    mode = paths.script_for("ollama-tok").stat().st_mode & 0o777
    assert mode == 0o700


@pytest.mark.integration
def test_tui_profile_screen_finds_ollama_profiles_after_a_token_install(
    monkeypatch,
):
    """Cached profiles for an OVERRIDABLE provider (ollama) must be reachable
    from the Profile screen / Tab even though ollama's registry entry stays
    auth="literal" (with_auth returns a runtime copy, never mutates
    PROVIDERS) — see _secret_providers in cli/tui.py."""
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "ollama", "sk-ollama-proxy", "proxy")

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is not None:
            on_tab()
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("ollama", "proxy")


# --- main screen = wrapper list, Enter = default, t = token (issue #29) -----


@pytest.mark.integration
def test_main_screen_shows_wrappers_grouped_by_agent(monkeypatch):
    """The wrapper list IS the main screen now: no separate 'List' entry, and
    wrappers are grouped under non-selectable Section headers per agent."""
    from code_helper.cli.menu import Section

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    # Presets (all claude today) appear under a Section("claude") header.
    # Section defines no __eq__ (it is a render marker, not a value), so match
    # on `.text` rather than `in`.
    section_texts = [e.text for e in main_items if isinstance(e, Section)]
    assert "claude" in section_texts
    # The old indirection is gone.
    assert not any(not isinstance(e, Section) and e[0] == "list" for e in main_items)
    # The service rows are still present below the wrappers.
    values = [e[0] for e in main_items if not isinstance(e, Section)]
    assert "add" in values
    assert "quit" in values


@pytest.mark.integration
def test_main_screen_marker_on_default_wrapper(monkeypatch):
    """A wrapper that is the saved default_wrapper for its agent is rendered
    with the `●` marker; every other wrapper is not."""
    from code_helper.cli.menu import Section
    from code_helper.services.state import default_wrapper, set_default_wrapper
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    # A default must point to a real installed wrapper, not just a preset name.
    install_wrapper(paths, "glm", token="test-token")
    set_default_wrapper(paths, "claude", "glm")
    assert default_wrapper(paths, "claude") == "glm"

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    rows = {
        e[0]: (e[1]() if callable(e[1]) else e[1])
        for e in main_items
        if not isinstance(e, Section)
        and e[0] not in ("add", "profile", "settings", "quit")
    }
    assert "●" in rows["glm"]
    assert "●" not in rows["deepseek"]


@pytest.mark.integration
def test_main_screen_enter_sets_default_wrapper(monkeypatch):
    """Enter on a wrapper row makes it the default for its agent; the marker
    moves on the next redraw."""
    from code_helper.services.state import default_wrapper
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    assert default_wrapper(paths, "claude") is None
    install_wrapper(paths, "glm", token="test-token")

    # DOWN moves to glm (second selectable wrapper after deepseek), Enter sets
    # it as the default, then END + ENTER reaches Quit.
    _real_menu_keys(monkeypatch, ["DOWN", "ENTER", "END", "ENTER"])
    assert main(["tui"]) == 0
    assert default_wrapper(paths, "claude") == "glm"


@pytest.mark.integration
def test_main_screen_groups_colliding_managed_wrapper_under_installed_agent(
    monkeypatch,
):
    """A managed wrapper installed under a preset alias of a DIFFERENT agent
    (e.g. a codex wrapper named `glm`, which is a claude preset) is grouped
    under the installed wrapper's agent — display grouping must agree with
    what Enter resolves (installed-first), or the same row would sit under
    one agent's section but launch another."""
    from code_helper.cli.menu import Section
    from code_helper.services.spec import build_spec
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    # Install a codex wrapper named "glm" (collides with the claude preset).
    install_wrapper(
        paths,
        build_spec(agent="codex", provider="ollama", model="qwen3.5:9b", alias="glm"),
    )

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    section_of: dict[str, str] = {}
    current = None
    for e in main_items:
        if isinstance(e, Section):
            current = e.text
        elif e[0] not in ("add", "profile", "settings", "quit"):
            section_of[e[0]] = current
    assert section_of["glm"] == "codex"


@pytest.mark.integration
def test_main_screen_no_dead_end_for_non_secret_wrapper(monkeypatch, capsys):
    """`t` on a non-secret wrapper (deepseek is literal) is a silent no-op —
    the old dead-end 'has no editable token.' screen is gone (issue #29)."""
    # TOKEN on the first wrapper (deepseek, non-secret), then quit.
    _real_menu_keys(monkeypatch, ["TOKEN", "END", "ENTER"])
    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    assert "deepseek has no editable token." in output


@pytest.mark.integration
def test_main_screen_hint_advertises_token_key(monkeypatch):
    """The main screen's hint mentions `t: token` so the key is discoverable."""
    captured: dict = {}

    def _select(_items, *, hint, **_kwargs):
        captured["hint"] = hint() if callable(hint) else hint
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert "t: token" in captured["hint"]


# --- apply codex default into ~/.codex/config.toml (issue #30) --------------


@pytest.mark.integration
def test_main_screen_has_apply_codex_default_row(monkeypatch):
    """Issue #30: an explicit 'Apply codex default' row sits in the service
    block, separate from the Enter-on-wrapper action (#29) — applying the
    agent's native config is an intentional step, not a side effect."""
    from code_helper.cli.menu import Section

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    rows = {
        e[0]: (e[1]() if callable(e[1]) else e[1])
        for e in main_items
        if not isinstance(e, Section)
    }
    assert "set-default" in rows
    assert "codex" in rows["set-default"].lower()


@pytest.mark.integration
def test_apply_codex_default_dispatches_into_apply_set_default(monkeypatch):
    """Selecting the row calls ``apply_set_default`` with the codex default
    wrapper's ``(agent, provider, model)`` and preserves ``force=False`` so the
    CLI confirmation preview is shown before Codex configuration changes."""
    import code_helper.services.codex_default as codex_default
    from code_helper.services.spec import build_spec
    from code_helper.services.state import set_default_wrapper
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    # Install a real codex wrapper so valid_default_wrapper can resolve it.
    install_wrapper(
        paths, build_spec(agent="codex", provider="ollama", model="qwen3.5:9b")
    )
    set_default_wrapper(paths, "codex", "qwen3.5-codex")

    calls: list[dict] = []

    def _fake_apply(paths_arg, *, agent, provider, model, **_kw):
        calls.append(
            {
                "agent": agent.name,
                "provider": provider.name,
                "model": model,
                "force": _kw.get("force", False),
            }
        )
        return True

    monkeypatch.setattr(codex_default, "apply_set_default", _fake_apply)
    _menu_sequence(monkeypatch, ["set-default", "quit"])

    assert main(["tui"]) == 0
    assert len(calls) == 1
    assert calls[0]["agent"] == "codex"
    assert calls[0]["provider"] == "ollama"
    assert calls[0]["model"] == "qwen3.5:9b"
    assert calls[0]["force"] is False


@pytest.mark.unit
def test_run_shows_output_live_before_a_blocking_input_call(monkeypatch):
    """``_run`` must not hide a handler's output behind a confirmation prompt.

    ``_confirm_set_default`` prints a preview and then calls ``input()`` —
    exactly the shape reproduced here. Redirecting stdout for the whole
    handler (as ``_run`` used to) buffers the preview into a ``StringIO``:
    the preview is invisible on the real terminal at the moment ``input()``
    blocks, so the user answers blind and only sees the preview afterwards,
    too late to inform the decision.
    """
    import argparse
    import io

    from code_helper.cli.tui import TuiSession

    session = TuiSession(argparse.Namespace(debug=False))
    monkeypatch.setattr("code_helper.cli.menu.press_any_key", lambda *_a, **_kw: None)

    # A stand-in for the real terminal, so writes to it can be inspected
    # precisely at the moment input() blocks.
    real_terminal = io.StringIO()
    monkeypatch.setattr("sys.stdout", real_terminal)

    seen_before_input = {}

    def fake_input(prompt=""):
        # At the moment input() blocks, the preview line must already have
        # reached the real terminal — not be trapped in a redirect buffer
        # that is only replayed after the handler returns.
        seen_before_input["preview_visible"] = (
            "About to write /some/path" in real_terminal.getvalue()
        )
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)

    def handler(_args):
        print("About to write /some/path. Continue? [y/N]")
        input()

    session._run(handler)

    assert seen_before_input.get("preview_visible") is True


@pytest.mark.unit
def test_run_reraises_under_debug():
    """``_run`` must honor ``--debug`` like every other dispatch site
    (``emit_error``'s contract): re-raise ``CodeHelperError`` instead of
    swallowing it into a one-line buffered message, so ``--debug`` still
    surfaces the full traceback from the TUI."""
    import argparse

    from code_helper.cli.tui import TuiSession
    from code_helper.errors import CodeHelperError

    session = TuiSession(argparse.Namespace(debug=True))

    def handler(_args):
        raise CodeHelperError("boom")

    with pytest.raises(CodeHelperError, match="boom"):
        session._run(handler)


@pytest.mark.integration
def test_apply_codex_default_forwards_runtime_base_url(monkeypatch):
    """A codex default wrapper on a REQUIRED-base_url provider (e.g. litellm)
    must have its resolved base_url forwarded to ``apply_set_default`` — the
    row hardcoded ``args.base_url = None``, which makes ``_handle_set_default``
    re-derive the provider from the registry (empty base_url for REQUIRED)
    and crash with 'needs a base URL', discarding the exact endpoint the
    installed wrapper was built with (issue #30 follow-up)."""
    import code_helper.services.codex_default as codex_default
    from code_helper.services.model import get_provider, with_base_url
    from code_helper.services.spec import build_spec
    from code_helper.services.state import set_default_wrapper
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    litellm = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    install_wrapper(paths, build_spec(agent="codex", provider=litellm, model="gpt-4o"))
    set_default_wrapper(paths, "codex", "gpt-4o-codex")

    calls: list[dict] = []

    def _fake_apply(paths_arg, *, agent, provider, model, **_kw):
        calls.append({"base_url": provider.base_url})
        return True

    monkeypatch.setattr(codex_default, "apply_set_default", _fake_apply)
    _menu_sequence(monkeypatch, ["set-default", "quit"])

    assert main(["tui"]) == 0
    assert len(calls) == 1
    assert calls[0]["base_url"] == "http://h:4000/v1/"


@pytest.mark.integration
def test_apply_codex_default_respects_global_dry_run(monkeypatch):
    """``code-helper --dry-run tui`` selecting the row must not write
    ``~/.codex/config.toml`` — the row hardcoded ``args.dry_run = False``,
    overriding the already-parsed global ``--dry-run`` flag and breaking the
    CLAUDE.md invariant '--dry-run never writes any file' (issue #30
    follow-up)."""
    import code_helper.services.codex_default as codex_default
    from code_helper.services.spec import build_spec
    from code_helper.services.state import set_default_wrapper
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    install_wrapper(
        paths, build_spec(agent="codex", provider="ollama", model="qwen3.5:9b")
    )
    set_default_wrapper(paths, "codex", "qwen3.5-codex")

    calls: list[dict] = []
    real_apply = codex_default.apply_set_default

    def _spy_apply(paths_arg, *, dry_run, **kw):
        calls.append({"dry_run": dry_run})
        return real_apply(paths_arg, dry_run=dry_run, **kw)

    monkeypatch.setattr(codex_default, "apply_set_default", _spy_apply)
    _menu_sequence(monkeypatch, ["set-default", "quit"])

    assert main(["--dry-run", "tui"]) == 0
    assert len(calls) == 1
    assert calls[0]["dry_run"] is True


@pytest.mark.integration
def test_apply_codex_default_errors_when_no_codex_default_set(monkeypatch, capsys):
    """Without a codex default wrapper, the action reports a clear error
    naming the prerequisite (Enter on a codex wrapper row) rather than
    silently doing nothing or crashing."""
    import code_helper.services.codex_default as codex_default

    monkeypatch.setattr(
        codex_default,
        "apply_set_default",
        lambda *a, **kw: pytest.fail("must not be called without a codex default"),
    )
    _menu_sequence(monkeypatch, ["set-default", "quit"])

    assert main(["tui"]) == 0
    output = capsys.readouterr().out
    assert "no default wrapper set for codex" in output.lower()
