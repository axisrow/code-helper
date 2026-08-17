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


def _tab_on_profile_screen(monkeypatch, presses=1):
    """Enter the Profile screen and press Tab there, then quit.

    Tab cycles the active token profile, and it lives on the Profile screen
    (`p`) — the main screen's Tab-adjacent keys drive the backend chipset
    instead. The Profile screen is the only menu that passes `on_tab`, so
    firing it wherever it is offered reaches exactly that screen.
    """
    seen = {"tabbed": 0}

    def _select(_items, *, on_tab=None, **_kwargs):
        if on_tab is None:
            # The main screen: enter the Profile screen the first time, then
            # quit once the Tab presses have happened.
            return "quit" if seen["tabbed"] else "profile"
        while seen["tabbed"] < presses:
            seen["tabbed"] += 1
            on_tab()
        return "__back__"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    return seen


@pytest.mark.integration
def test_main_menu_lists_wrappers_grouped_by_agent_and_quit_exits(monkeypatch, capsys):
    # The wrapper list IS the main screen now (issue #29): no separate "List"
    # entry. END jumps to the last numbered item (Quit), Enter exits.
    _real_menu_keys(monkeypatch, ["CANCEL"])

    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    # Wrappers appear directly on the main screen, grouped by agent section.
    assert "claude" in output
    assert "· ? · Esc" in output
    assert "Quit" not in output
    # The old indirection is gone.
    assert "Wrappers:" not in output
    # No pause screens.
    assert "Press any key" not in output
    assert "назад" not in output
    assert "edit-token" not in output


@pytest.mark.integration
def test_add_order_is_agent_provider_model_alias(monkeypatch):
    """`add` is agent-first: kind (wrapper/agent), then which agent, then the
    old provider -> model -> alias order — with no separate agent screen
    after the model, since the agent is already scoped by then."""
    seen: list[str] = []
    answers = iter(["add", "wrapper", "codex", "ollama", "model-x", "quit"])

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
    # The header no longer restates which backend is live: the chipset rows
    # say that in place (see TuiSession._main_prompt).
    assert seen[0].startswith("code-helper")
    assert "Live:" not in seen[0]
    assert seen[1:4] == [
        "What do you want to add?",
        "Add a wrapper for which agent?",
        "Select a provider for codex:",
    ]
    assert "codex › Select a model for ollama:" in seen
    assert not any(p == "Select an agent:" for p in seen)
    assert Paths.default().script_for("my-codex").exists()


@pytest.mark.integration
def test_add_agent_branch_persists_a_user_agent_end_to_end(monkeypatch):
    """`a` → Agent registers a new CLI integration that immediately shows up
    in the merged agent list (Stage 1.3) — verified by then adding a wrapper
    FOR that new agent via the unscoped Wrapper branch."""
    answers = iter(
        [
            "add",
            "agent",  # kind: Agent, not Wrapper
            "add",
            "wrapper",
            "myagent",  # the just-added agent appears in the picker
            "ollama",
            "model-x",
            "quit",
        ]
    )

    def _select(_items, *, prompt="", **_kwargs):
        return next(answers)

    agent_field_answers = iter(["myagent", "", ""])  # name, binary, description

    def _read_line(prompt, **_kwargs):
        if prompt.startswith(("Agent name", "Binary name", "Description")):
            return next(agent_field_answers)
        return "my-myagent-wrapper"  # the alias prompt

    import code_helper.services.models_api as api

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("code_helper.cli.menu.read_line", _read_line)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("code_helper.cli.menu.press_any_key", lambda *_a, **_k: None)

    assert main(["tui"]) == 0

    from code_helper.services.agents import load_user_agents

    agents = load_user_agents(Paths.default())
    assert len(agents) == 1
    assert agents[0].name == "myagent"
    assert agents[0].binary == "myagent"
    assert Paths.default().script_for("my-myagent-wrapper").exists()


@pytest.mark.integration
def test_alias_prompt_carries_a_breadcrumb_of_earlier_choices(monkeypatch):
    """The alias prompt is the last screen before a real filesystem write and
    the furthest from the choices that led there — it must show them, since
    there is no separate summary screen."""
    seen: list[str] = []
    answers = iter(["add", "wrapper", "codex", "ollama", "model-x", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        seen.append(prompt() if callable(prompt) else prompt)
        return next(answers)

    import code_helper.services.models_api as api

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )

    captured_prompt: dict = {}

    def _read_line(prompt, **_kwargs):
        captured_prompt["value"] = prompt
        return "my-codex"

    monkeypatch.setattr("code_helper.cli.menu.read_line", _read_line)

    assert main(["tui"]) == 0
    assert captured_prompt["value"].startswith("codex › ollama › model-x › ")


@pytest.mark.integration
def test_literal_provider_skips_profile_screen(monkeypatch):
    seen: list[str] = []
    answers = iter(["add", "wrapper", "claude", "ollama", "model-x", "quit"])

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
def test_escape_from_alias_reprompts_the_model_when_agent_is_prescoped(monkeypatch):
    """With the agent pre-scoped (agent-first Add), there is no agent menu
    to fall back to on an alias Esc — the guard in `_run_add_provider` must
    fall back to the model menu instead of spinning on a screen that no
    longer renders (see the plan's Stage 1.4)."""
    import code_helper.services.models_api as api
    from code_helper.cli.menu import MenuCancelled

    answers = iter(
        ["add", "wrapper", "claude", "ollama", "__custom__", "__custom__", "quit"]
    )

    def _select(_items, **_kwargs):
        return next(answers)

    text = iter(["model-x", "__escape__", "model-x", "final-wrapper"])

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
def test_model_step_falls_back_to_known_models_when_discovery_is_unavailable(
    monkeypatch,
):
    """zai has no discovery endpoint at all — the model menu must still offer
    the registry's `known_models` instead of forcing manual entry, and the
    prompt must say the list is known-not-discovered (Stage 3)."""
    seen_prompts: list[str] = []

    def _select(items, *, prompt="", **_kwargs):
        text = prompt() if callable(prompt) else prompt
        seen_prompts.append(text)
        if "Select a model for zai" in text:
            return "glm-5-turbo"
        return next(answers)

    answers = iter(["add", "wrapper", "claude", "zai", "quit"])
    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-first")
    monkeypatch.setattr("builtins.input", lambda _prompt: "known-wrapper")

    assert main(["tui"]) == 0

    assert any(
        "Select a model for zai (known models — discovery unavailable):" in p
        for p in seen_prompts
    )
    assert "glm-5-turbo" in Paths.default().script_for("known-wrapper").read_text()


@pytest.mark.integration
def test_first_secret_token_creates_default_before_model(monkeypatch):
    _menu_sequence(
        monkeypatch, ["add", "wrapper", "claude", "zai", "__custom__", "quit"]
    )
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
            "wrapper",
            "claude",
            "zai",
            "__custom__",
            "add",
            "wrapper",
            "claude",
            "zai",
            "__new_profile__",
            "__custom__",
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
        [
            "add",
            "wrapper",
            "claude",
            "zai",
            "work",
            "__replace_token__",
            "__custom__",
            "quit",
        ],
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
        [
            "add",
            "wrapper",
            "codex",
            "litellm",
            "work",
            "__use_profile__",
            "gpt-test",
            "quit",
        ],
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
    # Main screen rows: the two chipset rows (claude, codex) come first, then
    # the `+ add agent` action row, then the wrapper list [deepseek, glm,
    # glm-ollama] under Section("claude"). Four DOWNs reach glm; `t` opens its
    # token profile picker, ENTER picks the first profile (default), the
    # getpass stub supplies the new token.
    _real_menu_keys(
        monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "TOKEN", "ENTER", "CANCEL"]
    )
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-new")

    assert main(["tui"]) == 0
    assert secrets.credential_for(paths, "zai", "default") == "sk-new"
    assert "sk-new" in paths.script_for("glm").read_text()


@pytest.mark.integration
def test_a_on_the_codex_row_scopes_add_to_codex(monkeypatch):
    """`a` on the codex chipset row already answers "wrapper for which
    agent" — it must skip both the kind screen (wrapper/agent) and the
    agent picker, landing straight on a provider list scoped to codex."""
    seen: list[str] = []

    # Main screen: DOWN moves off claude onto codex, `a` triggers the scoped
    # Add. The provider screen that follows returns Back (below), which
    # lands back on the main screen — CANCEL then quits.
    real_menu_keys = iter(["DOWN", "a", "CANCEL"])

    def _select(items, *, prompt, **kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        if text == "Select a provider for codex:":
            return "__back__"
        return real_select(items, read_key=lambda: next(real_menu_keys), **kwargs)

    import code_helper.cli.menu as menu

    real_select = menu.select_from_menu
    monkeypatch.setattr(menu, "select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "Select a provider for codex:" in seen
    assert "What do you want to add?" not in seen
    assert "Add a wrapper for which agent?" not in seen
    assert "Select an agent:" not in seen


@pytest.mark.integration
def test_a_without_row_context_asks_what_to_add_first(monkeypatch):
    """`a` pressed anywhere that isn't an agent chip row (here: the wrapper
    row list, no chipset row focused) must open the unscoped kind picker —
    it has no agent to infer, so it cannot skip straight to a provider."""
    # The `add glm` setup must not hit a real getpass prompt: resolve_token
    # falls through to a prompt only when no env var / cached profile supplies
    # the token, and CI has neither. Mock it like test_cli_add.py does.
    import code_helper.services.secrets as secrets

    monkeypatch.setattr(
        secrets,
        "resolve_token",
        lambda **kwargs: secrets.ResolvedToken("sk-test", secrets.SOURCE_PROMPT),
    )
    assert main(["add", "glm", "--profile", "default"]) == 0
    seen: list[str] = []
    # DOWN x3 reaches the `glm` wrapper row (past the 2 chipset rows + blank
    # Section), `a` there still opens the kind picker since a wrapper row
    # carries no agent scope of its own.
    real_menu_keys = iter(["DOWN", "DOWN", "DOWN", "a", "CANCEL"])

    def _select(items, *, prompt, **kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        if text == "What do you want to add?":
            return "__back__"
        return real_select(items, read_key=lambda: next(real_menu_keys), **kwargs)

    import code_helper.cli.menu as menu

    real_select = menu.select_from_menu
    monkeypatch.setattr(menu, "select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "What do you want to add?" in seen


@pytest.mark.integration
def test_main_screen_has_add_agent_and_add_wrapper_action_rows(monkeypatch):
    """The main screen must expose both unscoped add actions as visible rows:
    `+ add agent` (a new CLI integration) and `+ add wrapper` (any agent)."""
    captured: dict = {}

    def _select(items, **_kwargs):
        captured["values"] = [
            item[0]
            for item in items
            if isinstance(item, tuple) and isinstance(item[0], str)
        ]
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "add-agent" in captured["values"]
    assert "add-wrapper" in captured["values"]


@pytest.mark.integration
def test_add_agent_action_row_registers_a_new_agent(monkeypatch):
    """Entering the `+ add agent` row must run the Agent branch — collect a
    name/binary/description and persist a new user-defined agent."""
    import code_helper.cli.tui as tui

    answers = iter(["add-agent", "quit"])
    monkeypatch.setattr(
        "code_helper.cli.menu.select_from_menu",
        lambda _items, **_kwargs: next(answers),
    )
    # The agent flow reads name, binary, description via read_line.
    monkeypatch.setattr(
        "code_helper.cli.menu.read_line", lambda _prompt, **_kwargs: "myagent"
    )
    monkeypatch.setattr(tui.TuiSession, "_notify", lambda self, text: None)

    assert main(["tui"]) == 0

    from code_helper.services.agents import load_user_agents
    from code_helper.services.paths import Paths

    assert [a.name for a in load_user_agents(Paths.default())] == ["myagent"]


@pytest.mark.integration
def test_add_wrapper_action_row_opens_the_unscoped_kind_picker(monkeypatch):
    """Entering the `+ add wrapper` row must open the unscoped Add flow — the
    kind picker (wrapper vs agent), not a provider list scoped to one agent."""
    seen: list[str] = []
    answers = iter(["add-wrapper", "__back__", "quit"])

    def _select(items, *, prompt, **kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "What do you want to add?" in seen


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
        menu_key("\x1b")
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="PTY tests require POSIX")
@pytest.mark.xfail(
    reason="The primary PTY flow is covered by hotkey-specific tests.", strict=False
)
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
        """Spawn the TUI, press Enter (first wrapper = default), read the redraw,
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
            menu_key("\x1b")
            if expect_marker_after_enter:
                assert b"\xe2\x97\x8f" not in output
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


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="PTY tests require POSIX")
@pytest.mark.xfail(
    reason="The main-screen set-default control moved to the c hotkey.", strict=False
)
def test_set_default_confirmation_preview_is_visible_before_the_prompt(tmp_path):
    """Manual PTY verification for the ``_run`` tee fix: ``set-default``'s
    diff preview and ``[y/N]`` prompt must reach the real terminal BEFORE the
    TUI blocks on the answer, not only afterwards via ``_notify``'s replay.
    The injected-``read_key`` suite cannot see this — a full ``redirect_stdout``
    around the handler buffers the preview invisibly while ``input()`` still
    blocks for real, which is exactly the terminal-driver-class bug repo
    convention requires a real PTY to catch."""
    import errno
    import pty
    import select
    import subprocess
    import sys
    import termios
    import time

    from code_helper.services.paths import Paths
    from code_helper.services.spec import build_spec
    from code_helper.services.state import set_default_wrapper
    from code_helper.services.wrappers import install_wrapper

    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment.pop("ZAI_API_KEY", None)
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_dir, environment.get("PYTHONPATH")) if part
    )

    # A codex wrapper installed and marked default: applying it will patch a
    # NOT-YET-EXISTING ~/.codex/config.toml, which counts as a real change
    # and triggers `_confirm_set_default`'s preview + `[y/N]` prompt.
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama", model="qwen3.5:9b")
    install_wrapper(paths, spec)
    set_default_wrapper(paths, "codex", spec.alias)

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
        deadline = time.monotonic() + 5
        while termios.tcgetattr(master_fd)[3] & termios.ICANON:
            if time.monotonic() >= deadline:
                raise AssertionError("TUI did not enter raw mode")
            time.sleep(0.01)
        os.write(master_fd, value.encode())

    try:
        read_until("Esc quit")
        menu_key("c")
        # The preview/prompt is written outside the TUI's raw-mode redraw
        # loop, so canonical-mode input() reads a real line — send "n\n".
        read_until("Continue? [y/N]")
        # By the time the prompt itself is visible, the preview text that
        # precedes it must already be in the captured output too — proving
        # it reached the terminal before the blocking read, not after.
        assert b"About to write" in output
        os.write(master_fd, b"n\n")
        # A decline raises CodeHelperError; `_run` reports it and pauses on
        # `_notify` until acknowledged.
        read_until("Press any key to continue")
        menu_key(" ")
        read_until("Esc quit")
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        os.close(master_fd)
        os.close(slave_fd)


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
    assert prompts and prompts[0]().endswith("litellm/work")
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.integration
def test_tui_tab_cycles_the_active_profile(monkeypatch):
    import code_helper.services.secrets as secrets
    from code_helper.services.state import active_selection, set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    secrets.save_credential(paths, "litellm", "sk-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    # Tab lives on the Profile screen now (the main screen's Left/Right and
    # Shift+Tab belong to the backend chipset). A single Tab there advances
    # work -> personal (profile_names order: default, then alpha).
    _tab_on_profile_screen(monkeypatch)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("litellm", "personal")


@pytest.mark.integration
def test_tui_tab_updates_profile_row_label_not_just_header(monkeypatch):
    """Issue #26: a label resolved to a string BEFORE the menu opened freezes
    at its pre-Tab value while the rest of the frame moves on. Every label
    that can change under a keypress must be a callable the menu re-evaluates
    per frame. Tab lives on the Profile screen now, so the label under test is
    that screen's slot strip — the invariant is unchanged.
    """
    import code_helper.services.secrets as secrets
    from code_helper.cli.menu import Section
    from code_helper.services.state import set_active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    secrets.save_credential(paths, "litellm", "sk-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    captured = {}

    def _select(items, *, on_tab=None, **_kwargs):
        if on_tab is None:
            return "quit" if captured else "profile"
        strip = next(e for e in items if isinstance(e, Section))
        assert callable(strip.text), "a label that can change must be callable"
        captured["before"] = strip.text()
        on_tab()
        captured["after"] = strip.text()
        return "__back__"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # The strip lists every provider/profile slot and marks none of them, so
    # what must change across a Tab is the ACTIVE selection it is rendered
    # from — pinned via state rather than by grepping the strip text.
    from code_helper.services.state import active_selection

    assert active_selection(paths) == ("litellm", "personal")
    assert captured["before"] and captured["after"]
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

    _tab_on_profile_screen(monkeypatch)
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

    _tab_on_profile_screen(monkeypatch)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("zai", names[0])


@pytest.mark.integration
def test_tui_tab_is_a_silent_noop_with_no_cached_profiles(monkeypatch):
    from code_helper.services.state import load_state

    paths = Paths.default()

    _tab_on_profile_screen(monkeypatch)
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

    _tab_on_profile_screen(monkeypatch)
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
    assert prompts and prompts[0]().endswith("zai/work")


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
        [
            "add",
            "wrapper",
            "codex",
            "litellm",
            "work",
            "__use_profile__",
            "gpt-test",
            "quit",
        ]
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
    # seen_items order: [0] main, [1] kind, [2] agent, [3] provider,
    # [4] profile picker, ...
    profile_picker = seen_items[4]
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
        [
            "add",
            "wrapper",
            "codex",
            "litellm",
            "work",
            "__use_profile__",
            "gpt-test",
            "quit",
        ]
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
    profile_picker = seen_items[4]
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
    answers = iter(["add", "wrapper", "claude", "__back__", "quit"])

    def _select(_items, **_kwargs):
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # seen_items order: [0] main menu, [1] kind, [2] agent, [3] provider list.
    provider_list = seen_items[3]
    values = [value for value, _label in provider_list]
    assert "ollama" in values
    assert "ollama:secret" in values
    # litellm/zai are FIXED — no second row for either.
    assert "litellm:secret" not in values
    assert "zai:secret" not in values


@pytest.mark.integration
def test_tui_add_ollama_with_token_installs_a_secret_wrapper(monkeypatch):
    import code_helper.services.models_api as api

    answers = iter(["add", "wrapper", "claude", "ollama:secret", "model-x", "quit"])

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

    _tab_on_profile_screen(monkeypatch)
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
    # Presets (all claude today) appear under a "claude — N wrappers" header
    # (not the bare word "claude" — that would repeat the chipset row above
    # it verbatim). Section defines no __eq__ (it is a render marker, not a
    # value), so match on `.text` rather than `in`.
    section_texts = [e.text for e in main_items if isinstance(e, Section)]
    assert any(text.startswith("claude — ") for text in section_texts)
    # The old indirection is gone.
    assert not any(not isinstance(e, Section) and e[0] == "list" for e in main_items)
    # The service rows are still present below the wrappers.
    values = [e[0] for e in main_items if not isinstance(e, Section)]
    assert "add" not in values
    assert "quit" not in values


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

    # Past the two chipset rows and the `+ add agent` action row, then one
    # more DOWN to reach glm (the second wrapper, after deepseek); Enter makes
    # it the default.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0
    assert default_wrapper(paths, "claude") == "glm"


@pytest.mark.integration
def test_main_screen_enter_on_an_unmanaged_foreign_file_does_not_set_a_ghost_default(
    monkeypatch,
):
    """Enter's guard checked `is_installed` (existence only), but
    `valid_default_wrapper` — the ONLY reader that matters, used by both the
    `●` marker and `set-default` — requires `is_installed` AND `is_managed`.
    A foreign (unmanaged) file sitting at a preset's alias path would let
    Enter write a default that the very next read silently rejects: a
    default that "sets" but never sticks, with no error shown either time."""
    from code_helper.services.state import default_wrapper

    paths = Paths.default()
    # A foreign executable at the "deepseek" preset's path — no ownership
    # marker, so `is_installed` is True but `is_managed` is False.
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    foreign = paths.script_for("deepseek")
    foreign.write_text("#!/bin/sh\necho not ours\n", encoding="utf-8")
    foreign.chmod(0o755)

    # `deepseek` is the first wrapper row, below the two chipset rows and the
    # `+ add agent` action row.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert default_wrapper(paths, "claude") is None


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
        elif e[0] not in (
            "add",
            "add-agent",
            "add-wrapper",
            "profile",
            "settings",
            "quit",
        ):
            section_of[e[0]] = current
    assert section_of["glm"].startswith("codex — ")


@pytest.mark.integration
def test_main_screen_no_dead_end_for_non_secret_wrapper(monkeypatch, capsys):
    """`t` on a non-secret wrapper (deepseek is literal) is a silent no-op —
    the old dead-end 'has no editable token.' screen is gone (issue #29)."""
    # TOKEN on the first wrapper (deepseek, non-secret), reached past the two
    # chipset rows and the `+ add agent` action row, then quit.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "TOKEN", "CANCEL"])
    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    assert "deepseek has no editable token." not in output


@pytest.mark.integration
def test_main_screen_hint_advertises_token_key(monkeypatch):
    """The main screen's hint mentions the `?` help key so it's discoverable."""
    captured: dict = {}

    def _select(_items, *, hint, **_kwargs):
        captured["hint"] = hint() if callable(hint) else hint
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert "· ? ·" in captured["hint"]


@pytest.mark.integration
def test_main_screen_hint_names_a_as_add_not_edit(monkeypatch):
    """`a` used to be folded into the `a/e/d edit` cluster, which mislabelled
    it — `e` only rotates a token (same handler as `t`), `a` adds. They must
    be named separately, and every key named MUST stay in `_PASSTHROUGH`
    (pinned elsewhere by `test_every_main_screen_key_survives_translation`)."""
    captured: dict = {}

    def _select(_items, *, hint, **_kwargs):
        captured["hint"] = hint() if callable(hint) else hint
        return "quit"

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert "a add" in captured["hint"]
    assert "a/e/d" not in captured["hint"]
    assert len(captured["hint"]) <= 80


# --- the chipset frame ------------------------------------------------------


def _chip_rows(items, *, cursor_pair: int = 0, ansi: bool = True) -> dict[str, str]:
    """Rendered chipset rows from a captured `items` list, keyed by agent.

    Goes through the SAME ``_call_label`` the real menu uses (not a bare
    ``entry[1]()``) so a test here cannot drift from what a user actually
    sees — a bare call is what let the two-cursor bug slip past tests once
    already, since it always renders as if the row were selected.
    ``cursor_pair`` is the index of the row the list cursor `>` sits on,
    matching ``menu._row_text``'s ``cursor_pair`` parameter.
    """
    from code_helper.cli.menu import Section, _call_label

    return {
        entry[0].removeprefix("agent:"): (
            _call_label(entry[1], selected=index == cursor_pair, ansi=ansi)
            if callable(entry[1])
            else entry[1]
        )
        for index, entry in enumerate(items)
        if not isinstance(entry, Section) and entry[0].startswith("agent:")
    }


def _capture_frames(monkeypatch, keys):
    """Drive the real menu and return each frame's rendered chipset rows.

    Tracks the list cursor's ``selectable`` index the same way
    ``menu._dispatch_key`` does (UP/DOWN/HOME/END over non-``Section`` rows),
    so ``_chip_rows`` renders each frame through the real ``selected``/``ansi``
    contract instead of guessing — a test that always renders "as selected"
    is exactly how the two-cursor bug passed once already.
    """
    import code_helper.cli.menu as menu
    from code_helper.cli.menu import Section

    iterator = iter(keys)
    real_select = menu.select_from_menu
    frames: list[dict[str, str]] = []

    def _select(items, **kwargs):
        selectable = [
            i for i, entry in enumerate(items) if not isinstance(entry, Section)
        ]
        index = 0

        def _read():
            nonlocal index
            frames.append(_chip_rows(items, cursor_pair=selectable[index]))
            key = next(iterator)
            if key == "UP":
                index = (index - 1) % len(selectable)
            elif key == "DOWN":
                index = (index + 1) % len(selectable)
            elif key == "HOME":
                index = 0
            elif key == "END":
                index = len(selectable) - 1
            return key

        return real_select(items, read_key=_read, **kwargs)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    return frames


@pytest.mark.integration
def test_chipset_shows_one_row_per_agent_with_native_applied(monkeypatch):
    """A fresh install overrides nothing, so every agent sits on `native` —
    and `native` is present for BOTH agents even though only claude has a
    `native` PROVIDER (codex clears its managed region instead)."""
    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    rows = frames[0]
    # Regression pin, not a stale hardcode: the chipset is built from
    # `_AGENT_BACKENDS`, not `AGENTS` (15 agents as of the ollama-launch
    # registry expansion) — a launch-only agent (opencode, droid, …)
    # deliberately gets NO chipset row (its config lives in a format this
    # project doesn't parse), so this set must stay exactly the two agents
    # with a live-patchable config. See test_chipset_row_count_matches_agent_backends.
    assert set(rows) == {"claude", "codex"}
    # Applied AND highlighted on the focused row (list cursor `>` sits on
    # claude for this first frame); applied-only, no cursor, elsewhere. Two
    # highlighted rows at once was the bug — the list has exactly one cursor.
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in rows["claude"]
    assert tui._REVERSE not in rows["codex"]


@pytest.mark.integration
def test_chipset_row_count_matches_agent_backends(monkeypatch):
    """The chipset's row set IS `_AGENT_BACKENDS`'s key set — stated as an
    explicit invariant rather than a hardcoded pair, so it stays true no
    matter how many launch-only agents `AGENTS` grows to."""
    import code_helper.cli.tui as tui

    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    rows = frames[0]
    assert set(rows) == set(tui._AGENT_BACKENDS)
    assert f"{tui._BOLD}✓ native{tui._RESET}" in rows["codex"]


@pytest.mark.integration
def test_chipset_cursor_moves_with_the_list_not_duplicates(monkeypatch):
    """Moving the list cursor to codex highlights ONLY codex's chip — claude
    keeps its `✓` but loses the reverse-video block it had a moment ago."""
    frames = _capture_frames(monkeypatch, ["DOWN", "CANCEL"])
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    rows = frames[1]
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in rows["codex"]
    assert tui._REVERSE not in rows["claude"]


@pytest.mark.integration
def test_chipset_has_no_cursor_when_list_cursor_is_on_a_wrapper_row(monkeypatch):
    """Parking the list cursor on a wrapper row below leaves BOTH agent rows
    with no reverse-video block — the chipset has no cursor of its own."""
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    frames = _capture_frames(monkeypatch, ["DOWN", "DOWN", "DOWN", "CANCEL"])
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    rows = frames[-1]
    assert tui._REVERSE not in rows["claude"]
    assert tui._REVERSE not in rows["codex"]


@pytest.mark.integration
def test_chipset_row_has_no_lifecycle_tail(monkeypatch):
    """The row used to end in a `live`/`next launch` label claiming WHEN a
    change takes effect. Dropped: the claim needs to be verified against how
    codex actually reads `config.toml` before it is asserted in the UI again
    (see `_AgentBackend.lifecycle`, still carried as data but no longer
    rendered)."""
    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    assert not frames[0]["claude"].rstrip().endswith(("live", "next launch"))
    assert not frames[0]["codex"].rstrip().endswith(("live", "next launch"))


@pytest.mark.integration
def test_add_chip_present_and_never_marked_applied(monkeypatch):
    """Every agent row carries a trailing `+ add` action chip — including an
    agent with zero wrappers, which is the codex empty-state fix — and it is
    never rendered as applied, since it isn't a backend at all."""
    import code_helper.cli.tui as tui

    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    rows = frames[0]
    for agent_name in ("claude", "codex"):
        assert "+ add" in rows[agent_name]
        assert "✓ + add" not in rows[agent_name]
        assert f"{tui._BOLD}+ add{tui._RESET}" not in rows[agent_name]


@pytest.mark.integration
def test_enter_on_the_add_chip_opens_add_scoped_to_that_agent(monkeypatch):
    """Enter on the `+ add` chip is equivalent to `a` on that row: it opens
    Add pre-scoped to the agent, no kind/agent picker in between."""
    seen: list[str] = []
    # Main screen: DOWN reaches codex, RIGHT moves past native to `+ add`,
    # ENTER applies it.
    real_menu_keys = iter(["DOWN", "RIGHT", "ENTER", "CANCEL"])

    def _select(items, *, prompt, **kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        if text == "Select a provider for codex:":
            return "__back__"
        return real_select(items, read_key=lambda: next(real_menu_keys), **kwargs)

    import code_helper.cli.menu as menu

    real_select = menu.select_from_menu
    monkeypatch.setattr(menu, "select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "Select a provider for codex:" in seen
    assert "What do you want to add?" not in seen


@pytest.mark.integration
def test_chips_are_installed_wrappers_not_bare_providers(monkeypatch):
    """A chip is an already-resolved wrapper, which is what lets Enter apply
    it with no model/token prompt. An uninstalled preset is not a chip."""
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    claude = frames[0]["claude"]
    assert "glm" in claude
    # `deepseek`/`glm-ollama` are presets that were never installed here.
    assert "deepseek" not in claude
    assert "glm-ollama" not in claude


@pytest.mark.integration
def test_right_moves_the_chip_cursor_and_wraps(monkeypatch):
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    # Strip is now [native, glm, + add] — one more RIGHT to wrap.
    frames = _capture_frames(monkeypatch, ["RIGHT", "RIGHT", "RIGHT", "CANCEL"])
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    # cursor on native
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in frames[0]["claude"]
    # moved to the wrapper chip
    assert f"{tui._REVERSE}glm{tui._RESET}" in frames[1]["claude"]
    # moved to the action chip
    assert f"{tui._REVERSE}+ add{tui._RESET}" in frames[2]["claude"]
    # wrapped back around
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in frames[3]["claude"]


@pytest.mark.integration
def test_back_tab_moves_the_chip_cursor_like_left(monkeypatch):
    """Shift+Tab is the backwards twin of Left, not a second mechanism."""
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    def _frames_for(key):
        # A fresh MonkeyPatch per run: re-patching `select_from_menu` while a
        # previous patch is live would stack the two wrappers and pass
        # `read_key` twice.
        with pytest.MonkeyPatch.context() as patch:
            frames = _capture_frames(patch, [key, "CANCEL"])
            assert main(["tui"]) == 0
        return frames

    assert _frames_for("LEFT")[1]["claude"] == _frames_for("BACK_TAB")[1]["claude"]


@pytest.mark.integration
def test_moving_the_chip_cursor_does_no_io(monkeypatch):
    """Applying happens on Enter, so Left/Right must be pure in-memory: the
    per-agent config reads happen ONCE per main-loop iteration, never once
    per keystroke. This is the pin on that caching discipline."""
    import code_helper.cli.tui as tui
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    calls = {"claude": 0, "codex": 0}
    patched = {}
    for name, backend in tui._AGENT_BACKENDS.items():
        real = backend.read_applied

        def counting(paths, _real=real, _name=name):
            calls[_name] += 1
            return _real(paths)

        patched[name] = tui._AgentBackend(
            read_applied=counting,
            apply_wrapper=backend.apply_wrapper,
            apply_native=backend.apply_native,
            lifecycle=backend.lifecycle,
        )
    monkeypatch.setattr(tui, "_AGENT_BACKENDS", patched)

    _real_menu_keys(monkeypatch, ["RIGHT"] * 10 + ["LEFT"] * 5 + ["CANCEL"])
    assert main(["tui"]) == 0

    assert calls == {"claude": 1, "codex": 1}


@pytest.mark.integration
def test_chip_cursor_is_independent_per_agent(monkeypatch):
    """Each agent row remembers its own chip, so moving away and back does
    not reset where the user was."""
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    frames = _capture_frames(monkeypatch, ["RIGHT", "DOWN", "UP", "CANCEL"])
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    assert f"{tui._REVERSE}glm{tui._RESET}" in frames[1]["claude"]
    assert (
        f"{tui._REVERSE}glm{tui._RESET}" in frames[3]["claude"]
    )  # survived the row round-trip


@pytest.mark.integration
def test_enter_applies_the_second_wrapper_sharing_a_provider_with_the_first(
    monkeypatch,
):
    """Regression: `_chip_is_applied` matches a wrapper chip by PROVIDER NAME
    only (a config file records only the provider), so two installed
    wrappers on the same provider are visually indistinguishable and the
    first one reads as "applied". Enter used to short-circuit on that
    heuristic for EVERY chip, making the second wrapper permanently
    unreachable through the chipset. Only the native chip's applied read is
    exact (`applied is None`) and still short-circuits; a wrapper chip must
    always re-apply."""
    from code_helper.services.spec import build_spec
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    second = build_spec(
        agent="claude", provider="zai", model="glm-5.2-air", alias="glm-air"
    )
    install_wrapper(Paths.default(), second, token="test-token-2")

    import dataclasses

    import code_helper.cli.tui as tui

    # Pretend "glm" (the first zai wrapper) is the one currently live —
    # _chip_is_applied will then also report "glm-air" as applied, since
    # both share provider "zai". `_AGENT_BACKENDS` is built once at import
    # time (`_register_agent_backends`) and captures `current_switch` as a
    # bound function value there, so patching the source module after import
    # would not reach it; `_AgentBackend` is frozen, so replace the whole
    # entry rather than mutating a field.
    monkeypatch.setitem(
        tui._AGENT_BACKENDS,
        "claude",
        dataclasses.replace(
            tui._AGENT_BACKENDS["claude"], read_applied=lambda paths: "zai"
        ),
    )

    seen = []
    monkeypatch.setattr(
        tui.TuiSession,
        "_apply_switch_wrapper",
        lambda self, spec: seen.append(spec.name),
    )
    # RIGHT twice from native: native -> glm -> glm-air (order follows
    # _all_wrapper_specs, alphabetical by alias: glm, glm-air).
    _real_menu_keys(monkeypatch, ["RIGHT", "RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert seen == ["glm-air"]


@pytest.mark.integration
def test_left_right_are_a_no_op_on_a_wrapper_row(monkeypatch):
    """The chip cursor belongs to agent rows; on a wrapper row the keys do
    nothing rather than moving some other row's cursor."""
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    # DOWN twice lands past both agent rows, onto the wrapper list.
    frames = _capture_frames(monkeypatch, ["DOWN", "DOWN", "RIGHT", "CANCEL"])
    assert main(["tui"]) == 0

    assert frames[2]["claude"] == frames[3]["claude"]


@pytest.mark.integration
def test_enter_on_an_agent_row_applies_the_highlighted_chip(monkeypatch):
    """The TUI mirrors the CLI: Enter dispatches into the same handler with
    the same request the command line would build, rather than writing the
    config itself."""
    import code_helper.cli.tui as tui
    from code_helper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    seen = []
    monkeypatch.setattr(
        tui.TuiSession,
        "_apply_switch_wrapper",
        lambda self, spec: seen.append(spec.name),
    )
    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert seen == ["glm"]


@pytest.mark.integration
def test_enter_on_an_already_applied_chip_does_not_rewrite(monkeypatch):
    """A no-op is reported, not written — re-applying what is already applied
    should not rotate a backup or touch the config."""
    import code_helper.cli.tui as tui

    applied = []
    monkeypatch.setattr(
        tui.TuiSession, "_apply_switch_native", lambda self: applied.append("native")
    )
    monkeypatch.setattr(tui.TuiSession, "_notify", lambda self, text: None)

    _real_menu_keys(monkeypatch, ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert applied == []


@pytest.mark.integration
def test_a_removed_wrapper_disappears_from_the_strip(monkeypatch):
    """Chips are rebuilt once per iteration, so a wrapper removed mid-session
    is gone on the next frame — and a chip cursor left past the end of the
    shorter strip is clamped, not left dangling."""
    from code_helper.services.wrappers import install_wrapper, remove_wrapper

    paths = Paths.default()
    install_wrapper(paths, "glm", token="test-token")

    import code_helper.cli.menu as menu

    frames: list[dict[str, str]] = []
    real_select = menu.select_from_menu
    state = {"n": 0}

    def _select(items, **kwargs):
        def _read():
            frames.append(_chip_rows(items))
            state["n"] += 1
            if state["n"] == 1:
                return "RIGHT"  # park the cursor on the glm chip
            if state["n"] == 2:
                remove_wrapper(paths, "glm", force=True)
                return "CANCEL"  # leave the menu so the loop re-renders
            return "CANCEL"

        return real_select(items, read_key=_read, **kwargs)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    assert main(["tui"]) == 0

    import code_helper.cli.tui as tui

    assert f"{tui._REVERSE}glm{tui._RESET}" in frames[1]["claude"]


@pytest.mark.unit
def test_every_main_screen_key_survives_translation():
    """A key bound in `on_key` but missing from `menu._PASSTHROUGH` is
    silently dead — the bug that left `w` (switch) non-functional while it was
    documented in the help text and listed in `keymap._LAYOUTS`. Pin every
    single-character binding the main screen declares against the translator
    so the two can never drift apart again.
    """
    from code_helper.cli.menu import _translate_char

    # Mirrors the `keys` table in TuiSession.run.
    for key in ("a", "p", "s", "?", "e", "d"):
        assert _translate_char(key) == key, f"{key!r} is bound but not passed through"
    # `t` is bound through the TOKEN alias rather than verbatim.
    assert _translate_char("t") == "TOKEN"
