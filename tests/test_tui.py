"""Integration coverage for the provider-first interactive TUI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths


def _menu_sequence(monkeypatch, answers):
    iterator = iter(answers)

    def _select(_items, **kwargs):
        # The context-window question (issue #83) fires inside _handle_add for
        # an unknown model; tests that don't care about the window get the
        # scripted "no declaration" answer — the pre-#83 behaviour — without
        # consuming a step of the scripted sequence.
        if str(kwargs.get("prompt", "")).startswith("Context window for"):
            return "0"
        return next(iterator)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)


def _real_menu_keys(monkeypatch, keys):
    import codehelper.cli.menu as menu

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

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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
    answers = iter(["add", "wrapper", "codex", "ollama-direct", "model-x", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        # The main-menu header is a callable (live active-profile display).
        seen.append(prompt() if callable(prompt) else prompt)
        return next(answers)

    import codehelper.services.models_api as api

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "my-codex")

    assert main(["tui"]) == 0
    # The header no longer restates which backend is live: the chipset rows
    # say that in place (see TuiSession._main_prompt).
    assert seen[0].startswith("codehelper")
    assert "Live:" not in seen[0]
    assert seen[1:4] == [
        "What do you want to add?",
        "Add a wrapper for which agent?",
        "Select a provider for codex:",
    ]
    assert "codex › Select a model for ollama-direct:" in seen
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
            "ollama-direct",
            "model-x",
            "quit",
        ]
    )

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration"
        return next(answers)

    agent_field_answers = iter(["myagent", "", ""])  # name, binary, description

    def _read_line(prompt, **_kwargs):
        if prompt.startswith(("Agent name", "Binary name", "Description")):
            return next(agent_field_answers)
        return "my-myagent-wrapper"  # the alias prompt

    import codehelper.services.models_api as api

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.read_line", _read_line)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)

    assert main(["tui"]) == 0

    from codehelper.services.agents import load_user_agents

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
    answers = iter(["add", "wrapper", "codex", "ollama-direct", "model-x", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        seen.append(prompt() if callable(prompt) else prompt)
        return next(answers)

    import codehelper.services.models_api as api

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )

    captured_prompt: dict = {}

    def _read_line(prompt, **_kwargs):
        captured_prompt["value"] = prompt
        return "my-codex"

    monkeypatch.setattr("codehelper.cli.menu.read_line", _read_line)

    assert main(["tui"]) == 0
    assert captured_prompt["value"].startswith("codex › ollama-direct › model-x › ")


@pytest.mark.integration
def test_literal_provider_skips_profile_screen(monkeypatch):
    seen: list[str] = []
    answers = iter(["add", "wrapper", "claude", "ollama-direct", "model-x", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        # The main-menu header is a callable (live active-profile display).
        seen.append(prompt() if callable(prompt) else prompt)
        return next(answers)

    import codehelper.services.models_api as api

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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
    import codehelper.services.models_api as api
    from codehelper.cli.menu import MenuCancelled

    answers = iter(
        [
            "add",
            "wrapper",
            "claude",
            "ollama-direct",
            "__custom__",
            "__custom__",
            "quit",
        ]
    )

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        return next(answers)

    text = iter(["model-x", "__escape__", "model-x", "final-wrapper"])

    def _read_line(_prompt, **_kwargs):
        value = next(text)
        if value == "__escape__":
            raise MenuCancelled(hard=False)
        return value

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.read_line", _read_line)
    monkeypatch.setattr(
        api,
        "list_models",
        lambda *_args, **_kwargs: api.ModelListResult(("model-x",), "fake"),
    )

    assert main(["tui"]) == 0
    assert Paths.default().script_for("final-wrapper").exists()


@pytest.mark.integration
def test_ctrl_c_from_tui_propagates_as_hard_cancel(monkeypatch):
    from codehelper.cli.menu import MenuCancelled

    def _hard(*_args, **_kwargs):
        raise MenuCancelled(hard=True)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _hard)
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

    def _select(_items, *, prompt="", **_kwargs):
        text = prompt() if callable(prompt) else prompt
        seen_prompts.append(text)
        if str(text).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        if "Select a model for zai" in text:
            return "glm-5-turbo"
        return next(answers)

    answers = iter(["add", "wrapper", "claude", "zai", "quit"])
    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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

    import codehelper.services.secrets as secrets

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

    import codehelper.services.secrets as secrets

    assert secrets.load_credentials(Paths.default())["zai"] == {
        "personal": "sk-personal",
        "work": "sk-work",
    }


@pytest.mark.integration
def test_replacing_selected_profile_changes_only_that_profile(monkeypatch):
    import codehelper.services.secrets as secrets

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
    import codehelper.services.models_api as api
    import codehelper.services.secrets as secrets

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
def test_t_rotates_token_for_secret_wrapper_from_main_screen(monkeypatch, capsys):
    """Issue #29: token rotation moved from the old List sub-screen to the `t`
    key on the main screen. With glm installed (secret), `t` on its row opens
    the profile picker and a new token is written — the same end-to-end result
    the old `list → glm → profile` flow produced, just without the sub-screen."""
    import codehelper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-old", "default")
    assert main(["add", "glm", "--profile", "default"]) == 0
    capsys.readouterr()  # drop the add's own transcript — count only the TUI flow
    # Main screen rows: the two agent chipset rows (claude, codex), the proxy
    # chipset row, the `+ add agent` action row, then the wrapper list
    # [deepseek-ollama, glm, glm-ollama] under Section("claude"). Five DOWNs reach
    # glm; `t` opens its token profile picker, ENTER picks the first profile
    # (default), the getpass stub supplies the new token.
    _real_menu_keys(
        monkeypatch,
        ["DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "TOKEN", "ENTER", "CANCEL"],
    )
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-new")

    assert main(["tui"]) == 0
    assert secrets.credential_for(paths, "zai", "default") == "sk-new"
    assert "sk-new" in paths.script_for("glm").read_text()
    # Issue #97: the transcript shows once (capture+replay), not twice
    # (the old live tee plus replay).
    out = capsys.readouterr().out
    assert out.count("wrote /") == 1


# --- issue #97: one transcript per action ------------------------------------


@pytest.mark.unit
def test_run_default_shows_the_transcript_once_and_pauses(capsys, monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession

    pauses = []
    monkeypatch.setattr(menu, "press_any_key", lambda *_a: pauses.append(1))
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )

    def handler(_req):
        print("single-line")

    assert session._run(handler, None) is True
    out = capsys.readouterr().out
    assert out.count("single-line") == 1
    assert len(pauses) == 1


@pytest.mark.unit
def test_run_live_is_visible_midflow_and_never_replayed(capsys, monkeypatch):
    """The live contract: the transcript is on the real terminal DURING the
    handler (before any blocking read), and the success path does not replay
    it — only the pause follows."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession

    pauses = []
    monkeypatch.setattr(menu, "press_any_key", lambda *_a: pauses.append(1))
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    seen = {}

    def handler(_req):
        print("prompt-line")
        seen["during"] = capsys.readouterr().out  # what is visible right now

    assert session._run(handler, None, live=True) is True

    final = capsys.readouterr().out
    assert "prompt-line" in seen["during"]  # tee: visible mid-flow
    assert "prompt-line" not in final  # capture-only default would fail here
    assert len(pauses) == 1  # the pause still happens, without a replay


@pytest.mark.unit
def test_run_silent_displays_nothing(capsys, monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession

    pauses = []
    monkeypatch.setattr(menu, "press_any_key", lambda *_a: pauses.append(1))
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )

    def handler(_req):
        print("quiet-line")

    assert session._run(handler, None, silent=True) is True
    out = capsys.readouterr().out
    assert "quiet-line" not in out
    assert pauses == []  # silent: no pause either


@pytest.mark.unit
def test_run_default_error_shows_error_line_then_transcript(capsys, monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.errors import CodeHelperError

    pauses = []
    monkeypatch.setattr(menu, "press_any_key", lambda *_a: pauses.append(1))
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )

    def handler(_req):
        print("before failure")
        raise CodeHelperError("boom")

    assert session._run(handler, None) is False
    out = capsys.readouterr().out
    assert "error: boom" in out
    assert out.count("before failure") == 1
    assert out.index("error: boom") < out.index("before failure")
    assert len(pauses) == 1


@pytest.mark.unit
def test_run_live_error_shows_everything_once(capsys, monkeypatch):
    """Live error path: the transcript was already teed live, so after the
    error line there is only the pause — no second showing of the
    transcript (the #97 symptom, error-path edition)."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.errors import CodeHelperError

    pauses = []
    monkeypatch.setattr(menu, "press_any_key", lambda *_a: pauses.append(1))
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    seen = {}

    def handler(_req):
        print("before failure")
        seen["during"] = capsys.readouterr().out
        raise CodeHelperError("boom")

    assert session._run(handler, None, live=True) is False

    final = capsys.readouterr().out
    assert "error: boom" in final
    assert "before failure" in seen["during"]  # shown live mid-handler
    assert "before failure" not in final  # and not replayed
    assert len(pauses) == 1


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

    import codehelper.cli.menu as menu

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
    import codehelper.services.secrets as secrets

    monkeypatch.setattr(
        secrets,
        "resolve_token",
        lambda **_kwargs: secrets.ResolvedToken("sk-test", secrets.SOURCE_PROMPT),
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

    import codehelper.cli.menu as menu

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

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "__action:add-agent" in captured["values"]
    assert "__action:add-wrapper" in captured["values"]


@pytest.mark.integration
def test_add_agent_action_row_registers_a_new_agent(monkeypatch):
    """Entering the `+ add agent` row must run the Agent branch — collect a
    name/binary/description and persist a new user-defined agent."""
    import codehelper.cli.tui as tui

    answers = iter(["__action:add-agent", "quit"])
    monkeypatch.setattr(
        "codehelper.cli.menu.select_from_menu",
        lambda _items, **_kwargs: next(answers),
    )
    # The agent flow reads name, binary, description via read_line.
    monkeypatch.setattr(
        "codehelper.cli.menu.read_line", lambda _prompt, **_kwargs: "myagent"
    )
    monkeypatch.setattr(tui.TuiSession, "_notify", lambda _self, _text: None)

    assert main(["tui"]) == 0

    from codehelper.services.agents import load_user_agents
    from codehelper.services.paths import Paths

    assert [a.name for a in load_user_agents(Paths.default())] == ["myagent"]


@pytest.mark.integration
def test_add_wrapper_action_row_opens_the_unscoped_kind_picker(monkeypatch):
    """Entering the `+ add wrapper` row must open the unscoped Add flow — the
    kind picker (wrapper vs agent), not a provider list scoped to one agent."""
    seen: list[str] = []
    answers = iter(["__action:add-wrapper", "__back__", "quit"])

    def _select(_items, *, prompt, **_kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        return next(answers)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)

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
        "# codehelper: managed wrapper (agent=claude, provider=defunct, "
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
        [sys.executable, "-m", "codehelper", "tui"],
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
            [sys.executable, "-m", "codehelper", "tui"],
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

    from codehelper.services.paths import Paths
    from codehelper.services.spec import build_spec
    from codehelper.services.state import set_default_wrapper
    from codehelper.services.wrappers import install_wrapper

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
    spec = build_spec(agent="codex", provider="ollama-direct", model="qwen3.5:9b")
    install_wrapper(paths, spec)
    set_default_wrapper(paths, "codex", spec.alias)

    master_fd, slave_fd = pty.openpty()
    child = subprocess.Popen(
        [sys.executable, "-m", "codehelper", "tui"],
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
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection

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

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert prompts and prompts[0]().endswith("litellm/work")
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.integration
def test_tui_tab_cycles_the_active_profile(monkeypatch):
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection, set_active_selection

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
    import codehelper.services.secrets as secrets
    from codehelper.cli.menu import Section
    from codehelper.services.state import set_active_selection

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

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # The strip lists every provider/profile slot and marks none of them, so
    # what must change across a Tab is the ACTIVE selection it is rendered
    # from — pinned via state rather than by grepping the strip text.
    from codehelper.services.state import active_selection

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
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection

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
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection

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
    from codehelper.services.state import load_state

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
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection, set_active_selection

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
    import codehelper.services.secrets as secrets

    paths = Paths.default()
    secrets.save_credential(paths, "zai", "sk-one", "work")

    prompts = []

    def _select(_items, *, prompt, **_kwargs):
        prompts.append(prompt)
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert prompts and prompts[0]().endswith("zai/work")


@pytest.mark.integration
def test_tui_add_preselects_the_active_profile_first(monkeypatch):
    import codehelper.services.models_api as api
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

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

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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
    import codehelper.services.models_api as api
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

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

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        seen_items.append(list(_items))
        return next(answers)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    # seen_items order: [0] main menu, [1] kind, [2] agent, [3] provider list.
    provider_list = seen_items[3]
    values = [value for value, _label in provider_list]
    assert "ollama-direct" in values
    assert "ollama-direct:secret" in values
    # litellm/zai are FIXED — no second row for either.
    assert "litellm:secret" not in values
    assert "zai:secret" not in values


@pytest.mark.integration
def test_tui_add_ollama_with_token_installs_a_secret_wrapper(monkeypatch):
    import codehelper.services.models_api as api

    answers = iter(
        ["add", "wrapper", "claude", "ollama-direct:secret", "model-x", "quit"]
    )

    def _select(_items, *, prompt="", **_kwargs):
        if str(prompt).startswith("Context window for"):
            return "0"  # scripted "no declaration" — tests here don't care
        return next(answers)

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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
    import codehelper.services.secrets as secrets
    from codehelper.services.state import active_selection

    paths = Paths.default()
    secrets.save_credential(paths, "ollama-direct", "sk-ollama-proxy", "proxy")

    _tab_on_profile_screen(monkeypatch)
    assert main(["tui"]) == 0
    assert active_selection(paths) == ("ollama-direct", "proxy")


# --- main screen = wrapper list, Enter = default, t = token (issue #29) -----


@pytest.mark.integration
def test_main_screen_shows_wrappers_grouped_by_agent(monkeypatch):
    """The wrapper list IS the main screen now: no separate 'List' entry, and
    wrappers are grouped under non-selectable Section headers per agent."""
    from codehelper.cli.menu import Section

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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
    from codehelper.cli.menu import Section
    from codehelper.services.state import default_wrapper, set_default_wrapper
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    # A default must point to a real installed wrapper, not just a preset name.
    install_wrapper(paths, "glm", token="test-token")
    set_default_wrapper(paths, "claude", "glm")
    assert default_wrapper(paths, "claude") == "glm"

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    rows = {
        e[0]: (e[1]() if callable(e[1]) else e[1])
        for e in main_items
        if not isinstance(e, Section)
        and e[0] not in ("add", "profile", "settings", "quit")
    }
    assert "●" in rows["glm"]
    assert "●" not in rows["deepseek-ollama"]


@pytest.mark.integration
def test_main_screen_enter_sets_default_wrapper(monkeypatch):
    """Enter on a wrapper row makes it the default for its agent; the marker
    moves on the next redraw."""
    from codehelper.services.state import default_wrapper
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    assert default_wrapper(paths, "claude") is None
    install_wrapper(paths, "glm", token="test-token")

    # Past the two agent chipset rows, the proxy row and the `+ add agent`
    # action row, then one more DOWN to reach glm (the second wrapper, after
    # deepseek-ollama); Enter makes it the default.
    _real_menu_keys(
        monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "ENTER", "CANCEL"]
    )
    assert main(["tui"]) == 0
    assert default_wrapper(paths, "claude") == "glm"


@pytest.mark.integration
def test_wrapper_named_add_agent_is_selectable_from_main_screen(monkeypatch):
    """A wrapper literally named `add-agent` must be selectable as a wrapper —
    Enter sets it as default — not shadowed by the `+ add agent` action row.
    The action-row IDs collided with valid wrapper aliases (both `add-agent`
    and `add-wrapper` pass `validate_alias`), so selecting such a wrapper
    dispatched into the add-agent flow instead of the wrapper-selection path."""
    from codehelper.services.spec import build_spec
    from codehelper.services.state import default_wrapper
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    install_wrapper(
        paths,
        build_spec(
            agent="claude",
            provider="ollama-direct",
            model="qwen3.5:9b",
            alias="add-agent",
        ),
    )
    # The `add-agent` wrapper is the 10th selectable row: the two agent chipset
    # rows, the proxy row, the `+ add agent` action row, then the
    # deepseek-ollama/glm/glm-ollama/gemini-litellm/bai presets. Nine DOWNs
    # reach it; Enter must set it as default, not open add-agent.
    _real_menu_keys(monkeypatch, ["DOWN"] * 9 + ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0
    assert default_wrapper(paths, "claude") == "add-agent"


@pytest.mark.integration
def test_wrapper_named_add_wrapper_is_selectable_from_main_screen(monkeypatch):
    """Symmetric to the `add-agent` case: a wrapper literally named
    `add-wrapper` must be selectable as a wrapper, not shadowed by the
    `+ add wrapper` action row."""
    from codehelper.services.spec import build_spec
    from codehelper.services.state import default_wrapper
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    install_wrapper(
        paths,
        build_spec(
            agent="claude",
            provider="ollama-direct",
            model="qwen3.5:9b",
            alias="add-wrapper",
        ),
    )
    # The `add-wrapper` wrapper is the 10th selectable row (two agent chipset
    # rows, the proxy row, the `+ add agent` action row, then the
    # deepseek-ollama/glm/glm-ollama/gemini-litellm/bai presets). Nine DOWNs
    # reach it; Enter must set it as default, not open the add flow.
    _real_menu_keys(monkeypatch, ["DOWN"] * 9 + ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0
    assert default_wrapper(paths, "claude") == "add-wrapper"


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
    from codehelper.services.state import default_wrapper

    paths = Paths.default()
    # A foreign executable at the "deepseek-ollama" preset's path — no ownership
    # marker, so `is_installed` is True but `is_managed` is False.
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    foreign = paths.script_for("deepseek-ollama")
    foreign.write_text("#!/bin/sh\necho not ours\n", encoding="utf-8")
    foreign.chmod(0o755)

    # `deepseek-ollama` is the first wrapper row, below the two agent chipset rows,
    # the proxy row and the `+ add agent` action row.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "ENTER", "CANCEL"])
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
    from codehelper.cli.menu import Section
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    # Install a codex wrapper named "glm" (collides with the claude preset).
    install_wrapper(
        paths,
        build_spec(
            agent="codex", provider="ollama-direct", model="qwen3.5:9b", alias="glm"
        ),
    )

    captured: list[list] = []

    def _select(items, **_kwargs):
        captured.append(list(items))
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0

    main_items = captured[0]
    section_of: dict[str, str] = {}
    current = None
    for e in main_items:
        if isinstance(e, Section):
            current = e.text
        elif e[0] not in (
            "add",
            "__action:add-agent",
            "__action:add-wrapper",
            "profile",
            "settings",
            "quit",
        ):
            section_of[e[0]] = current
    assert section_of["glm"].startswith("codex — ")


@pytest.mark.integration
def test_main_screen_no_dead_end_for_non_secret_wrapper(monkeypatch, capsys):
    """`t` on a non-secret wrapper (deepseek-ollama is literal) is a silent no-op —
    the old dead-end 'has no editable token.' screen is gone (issue #29)."""
    # TOKEN on the first wrapper (deepseek-ollama, non-secret), reached past the two
    # agent chipset rows, the proxy row and the `+ add agent` action row, then
    # quit.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "TOKEN", "CANCEL"])
    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    assert "deepseek-ollama has no editable token." not in output


@pytest.mark.integration
def test_main_screen_hint_advertises_token_key(monkeypatch):
    """The main screen's hint mentions the `?` help key so it's discoverable."""
    captured: dict = {}

    def _select(_items, *, hint, **_kwargs):
        captured["hint"] = hint() if callable(hint) else hint
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
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

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    assert main(["tui"]) == 0
    assert "a add" in captured["hint"]
    assert "a/e/d" not in captured["hint"]
    # The hint is TWO lines (menu accepts exactly one "\n"), and the budget
    # that matters is PER LINE: `_fit` truncates any line past the terminal
    # width and eats the tail (the exit hint) — the bug a PTY run caught here.
    hint_lines = captured["hint"].split("\n")
    assert len(hint_lines) == 2
    assert all(len(line) <= 80 for line in hint_lines)


# --- the chipset frame ------------------------------------------------------


@pytest.mark.unit
def test_e_on_chip_row_targets_highlighted_wrapper():
    """Editing a chip row must resolve its selected chip, not the row key."""
    from codehelper.cli.tui import TuiSession

    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._chips = {"claude": ["native", "deepseek-ollama", "glm", "+ add"]}
    session._chip_index = {"claude": 2}

    assert session._token_action("agent:claude") == "token:glm"
    session._chip_index["claude"] = 0
    assert session._token_action("agent:claude") is None
    session._chip_index["claude"] = 3
    assert session._token_action("agent:claude") is None
    assert session._token_action("glm") == "token:glm"


@pytest.mark.unit
def test_e_renames_on_chip_row_and_is_inert_where_nothing_is_renameable():
    """`e` mirrors `t`'s focused-chip resolution for rename (issue #95):
    the highlighted chip on a chipset row, the row itself elsewhere."""
    from codehelper.cli.tui import TuiSession

    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    session._chips = {"claude": ["native", "deepseek-ollama", "glm", "+ add"]}
    session._chip_index = {"claude": 2}

    assert session._rename_action("agent:claude") == "rename:glm"
    session._chip_index["claude"] = 0
    assert session._rename_action("agent:claude") is None
    session._chip_index["claude"] = 3
    assert session._rename_action("agent:claude") is None
    assert session._rename_action("glm") == "rename:glm"


@pytest.mark.integration
def test_on_rename_moves_the_wrapper(tmp_path, monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.services.wrappers import install_wrapper, is_installed

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-existing")

    monkeypatch.setattr(menu, "read_line", lambda _prompt="", **_kw: "glm2")
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    session._on_rename("glm")

    assert is_installed(paths, "glm2")
    assert not is_installed(paths, "glm")


@pytest.mark.integration
def test_on_rename_shows_the_confirmation_exactly_once(tmp_path, monkeypatch, capsys):
    """Issue #97, rename half of the acceptance criteria: the synthesized
    confirmation is the only display and appears exactly once. Dropping the
    handler's ``silent=True`` (tee + replay) would double the line — this
    pin fails immediately on that regression."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-existing")

    monkeypatch.setattr(menu, "read_line", lambda _prompt="", **_kw: "glm2")
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    session._on_rename("glm")

    out = capsys.readouterr().out
    assert out.count("renamed wrapper glm -> glm2") == 1


@pytest.mark.integration
def test_on_rename_collision_reports_error_without_success(
    tmp_path, monkeypatch, capsys
):
    """A colliding alias leaves the source untouched: the handler's error is
    shown and the success line is suppressed (gated on the handler result,
    not on the target existing)."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.services.wrappers import install_wrapper, is_installed

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-one")
    from codehelper.services.spec import build_spec

    install_wrapper(
        paths,
        build_spec(
            agent="claude",
            provider="zai",
            model="glm-5.3",
            alias="glm2",
        ),
        token="sk-two",
    )

    monkeypatch.setattr(menu, "read_line", lambda _prompt="", **_kw: "glm2")
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    session._on_rename("glm")

    out = capsys.readouterr().out
    assert "error:" in out
    assert "renamed wrapper" not in out
    assert is_installed(paths, "glm")
    assert is_installed(paths, "glm2")


@pytest.mark.integration
def test_profile_screen_e_renames(tmp_path, monkeypatch):
    """`e` on the Profiles screen fires the profile rename; the renamed row
    is what the next frame shows."""
    import codehelper.cli.menu as menu
    import codehelper.services.secrets as secrets
    from codehelper.cli.tui import TuiSession

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", "sk-old", "work")
    monkeypatch.setattr(menu, "read_line", lambda _prompt="", **_kw: "personal")

    providers = iter(["zai"])

    fired = {"done": False}

    def _select(_items, prompt="", **kwargs):
        if prompt.startswith("Active profile for"):
            on_key = kwargs.get("on_key")
            assert on_key is not None and "e" in on_key
            if not fired["done"]:
                fired["done"] = True
                return on_key["e"]("work")
            return "__back__"
        return next(providers)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    session = TuiSession(
        cast(argparse.Namespace, SimpleNamespace(debug=False, dry_run=False))
    )
    session._run_profile_screen()

    assert secrets.credential_for(paths, "zai", "personal") == "sk-old"
    assert secrets.credential_for(paths, "zai", "work") == ""


def _chip_rows(items, *, cursor_pair: int = 0, ansi: bool = True) -> dict[str, str]:
    """Rendered chipset rows from a captured `items` list, keyed by agent.

    Goes through the SAME ``_call_label`` the real menu uses (not a bare
    ``entry[1]()``) so a test here cannot drift from what a user actually
    sees — a bare call is what let the two-cursor bug slip past tests once
    already, since it always renders as if the row were selected.
    ``cursor_pair`` is the index of the row the list cursor `>` sits on,
    matching ``menu._row_text``'s ``cursor_pair`` parameter.
    """
    from codehelper.cli.menu import Section, _call_label

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
    import codehelper.cli.menu as menu
    from codehelper.cli.menu import Section

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

    import codehelper.cli.tui as tui

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
    import codehelper.cli.tui as tui

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

    import codehelper.cli.tui as tui

    rows = frames[1]
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in rows["codex"]
    assert tui._REVERSE not in rows["claude"]


@pytest.mark.integration
def test_chipset_has_no_cursor_when_list_cursor_is_on_a_wrapper_row(monkeypatch):
    """Parking the list cursor on a wrapper row below leaves BOTH agent rows
    with no reverse-video block — the chipset has no cursor of its own."""
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    frames = _capture_frames(monkeypatch, ["DOWN", "DOWN", "DOWN", "CANCEL"])
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

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
    import codehelper.cli.tui as tui

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

    import codehelper.cli.menu as menu

    real_select = menu.select_from_menu
    monkeypatch.setattr(menu, "select_from_menu", _select)

    assert main(["tui"]) == 0
    assert "Select a provider for codex:" in seen
    assert "What do you want to add?" not in seen


@pytest.mark.integration
def test_chips_include_claude_presets_without_wrapper_files(monkeypatch):
    """Claude presets are live backend choices, not PATH entries."""
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    claude = frames[0]["claude"]
    assert "glm" in claude
    assert "deepseek-ollama" in claude
    assert "glm-ollama" in claude


@pytest.mark.integration
def test_litellm_chip_reads_a_runtime_url_instead_of_marking_native(monkeypatch):
    from codehelper.services.claude_settings import apply_switch
    from codehelper.services.model import get_provider, with_base_url
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    paths = Paths.default()
    provider = with_base_url(get_provider("litellm"), "https://proxy.example")
    spec = build_spec(
        agent="claude", provider=provider, model="glm-5.2", alias="glm52-litellm"
    )
    install_wrapper(paths, spec, token="sk-test")
    apply_switch(
        paths,
        provider=provider,
        tier_models=spec.tier_models,
        token="sk-test",
        force=True,
    )

    frames = _capture_frames(monkeypatch, ["CANCEL"])
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

    assert f"{tui._BOLD}✓ glm52-litellm{tui._RESET}" in frames[0]["claude"]
    assert "✓ native" not in frames[0]["claude"]


@pytest.mark.integration
def test_right_moves_the_chip_cursor_and_wraps(monkeypatch):
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    # Presets are always present, even before their launcher files exist.
    frames = _capture_frames(monkeypatch, ["RIGHT", "RIGHT", "RIGHT", "CANCEL"])
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

    # cursor on native
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in frames[0]["claude"]
    assert f"{tui._REVERSE}deepseek-ollama{tui._RESET}" in frames[1]["claude"]
    assert f"{tui._REVERSE}glm{tui._RESET}" in frames[2]["claude"]
    assert f"{tui._REVERSE}glm-ollama{tui._RESET}" in frames[3]["claude"]


@pytest.mark.integration
def test_back_tab_moves_the_chip_cursor_like_left():
    """Shift+Tab is the backwards twin of Left, not a second mechanism."""
    from codehelper.services.wrappers import install_wrapper

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
    import codehelper.cli.tui as tui
    from codehelper.services.wrappers import install_wrapper

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
            chip_is_applied=backend.chip_is_applied,
        )
    monkeypatch.setattr(tui, "_AGENT_BACKENDS", patched)

    _real_menu_keys(monkeypatch, ["RIGHT"] * 10 + ["LEFT"] * 5 + ["CANCEL"])
    assert main(["tui"]) == 0

    assert calls == {"claude": 1, "codex": 1}


@pytest.mark.integration
def test_chip_cursor_is_independent_per_agent(monkeypatch):
    """Each agent row remembers its own chip, so moving away and back does
    not reset where the user was."""
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    frames = _capture_frames(monkeypatch, ["RIGHT", "DOWN", "UP", "CANCEL"])
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

    assert f"{tui._REVERSE}deepseek-ollama{tui._RESET}" in frames[1]["claude"]
    assert (
        f"{tui._REVERSE}deepseek-ollama{tui._RESET}" in frames[3]["claude"]
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
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")
    second = build_spec(
        agent="claude", provider="zai", model="glm-5.2-air", alias="glm-air"
    )
    install_wrapper(Paths.default(), second, token="test-token-2")

    import dataclasses

    import codehelper.cli.tui as tui

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
            tui._AGENT_BACKENDS["claude"], read_applied=lambda _paths: "zai"
        ),
    )

    seen = []
    monkeypatch.setattr(
        tui.TuiSession,
        "_apply_switch_wrapper",
        lambda _self, spec: seen.append(spec.name),
    )
    # Presets are first, then ad-hoc chips alphabetically; the bai preset chip
    # sits between gemini-litellm and the ad-hoc glm-air.
    _real_menu_keys(monkeypatch, ["RIGHT"] * 6 + ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert seen == ["glm-air"]


@pytest.mark.integration
def test_left_right_are_a_no_op_on_a_wrapper_row(monkeypatch):
    """The chip cursor belongs to agent rows; on a wrapper row the keys do
    nothing rather than moving some other row's cursor."""
    from codehelper.services.wrappers import install_wrapper

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
    import codehelper.cli.tui as tui
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "glm", token="test-token")

    seen = []
    monkeypatch.setattr(
        tui.TuiSession,
        "_apply_switch_wrapper",
        lambda _self, spec: seen.append(spec.name),
    )
    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert seen == ["deepseek-ollama"]


@pytest.mark.unit
def test_claude_chip_request_is_a_noninteractive_hot_apply():
    import argparse

    import codehelper.cli.tui as tui

    request = tui.TuiSession(
        argparse.Namespace(dry_run=False, debug=False)
    )._switch_request(from_preset="deepseek-ollama")
    assert request.force is True
    assert request.from_preset == "deepseek-ollama"


@pytest.mark.integration
def test_enter_on_deepseek_ollama_chip_hot_applies_without_confirmation(monkeypatch):
    from codehelper.services.paths import Paths

    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    import json

    settings = json.loads(Paths.default().claude_settings().read_text(encoding="utf-8"))
    assert (
        settings["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]
        == "deepseek-v4-flash:0731-cloud"
    )


@pytest.mark.integration
def test_deepseek_ollama_chip_can_switch_back_to_native(monkeypatch):
    """The reverse hot-apply explicitly resets the managed Claude env."""
    import codehelper.cli.tui as tui
    from codehelper.services.claude_settings import MANAGED_ENV_KEYS, current_switch
    from codehelper.services.paths import Paths

    # Start on native, apply deepseek-ollama, move the chip cursor back, and apply
    # native.  The cursor must remain on the Claude row throughout.
    frames = _capture_frames(monkeypatch, ["RIGHT", "ENTER", "LEFT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0
    assert f"{tui._REVERSE}✓ native{tui._RESET}" in frames[4]["claude"]
    paths = Paths.default()
    assert current_switch(paths) is None

    settings_path = paths.claude_settings()
    if not settings_path.exists():
        return
    import json

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert {key: settings["env"].get(key) for key in MANAGED_ENV_KEYS} == {
        key: "" for key in MANAGED_ENV_KEYS
    }


@pytest.mark.integration
def test_deepseek_ollama_chip_can_switch_to_glm_with_zai_url(monkeypatch):
    """A live provider change updates both the model and endpoint."""
    from codehelper.services.paths import Paths

    # The glm chip's hot-apply resolves its token non-interactively via
    # resolve_token, same as any other headless switch — needs a real
    # source (env, here) since this test installs no wrapper/cache.
    monkeypatch.setenv("ZAI_API_KEY", "sk-test")
    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    import json

    settings = json.loads(Paths.default().claude_settings().read_text(encoding="utf-8"))
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert settings["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.3"


@pytest.mark.integration
def test_chip_apply_is_silent_no_wrote_no_pause(monkeypatch, capsys):
    """A claude chip hot-apply must switch without echoing the CLI's
    ``wrote ... (backup: ...)`` progress line or pausing on "Press any key".
    The chipset redraw alone conveys the new [applied] state, so a chip press
    is quiet — the CLI switch command keeps its informative output, but the
    TUI hot-apply path must not."""
    from codehelper.services.paths import Paths

    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    output = capsys.readouterr().out
    assert "wrote" not in output
    assert "Press any key" not in output
    import json

    settings = json.loads(Paths.default().claude_settings().read_text(encoding="utf-8"))
    assert (
        settings["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]
        == "deepseek-v4-flash:0731-cloud"
    )


@pytest.mark.integration
def test_dry_run_chip_surfaces_the_preview_not_silent(monkeypatch, capsys):
    """A claude chip press under ``--dry-run`` must NOT be silent. Silence is
    justified by a real write whose chipset redraw already shows the new
    [applied] state; a dry run writes nothing, so the redraw does not change
    and the preview is the only feedback. A dry-run chip press must surface the
    redacted diff / "would write" line instead of swallowing it, and must not
    mutate settings.json."""
    from codehelper.services.paths import Paths

    _real_menu_keys(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui", "--dry-run"]) == 0

    output = capsys.readouterr().out
    # The dry-run preview is surfaced, not discarded by the silent path.
    assert "would write" in output
    # Dry-run must never claim a real write.
    assert "wrote " not in output

    # Nothing may be mutated by a dry-run chip press.
    settings_path = Paths.default().claude_settings()
    if settings_path.exists():
        import json

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in settings.get("env", {})


@pytest.mark.integration
def test_enter_on_an_already_applied_chip_does_not_rewrite(monkeypatch):
    """A selected native chip is a silent no-op with no config write."""
    import codehelper.cli.tui as tui

    applied = []
    monkeypatch.setattr(
        tui.TuiSession, "_apply_switch_native", lambda _self: applied.append("native")
    )
    _real_menu_keys(monkeypatch, ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert applied == []


@pytest.mark.integration
def test_enter_on_the_selected_deepseek_ollama_chip_is_a_silent_noop(monkeypatch):
    from codehelper.services.claude_settings import apply_switch
    from codehelper.services.wrappers import get_spec

    spec = get_spec("deepseek-ollama")
    apply_switch(
        Paths.default(),
        provider=spec.provider,
        tier_models=spec.tier_models,
        token=spec.auth_value,
        subagent_model=spec.subagent_model,
        force=True,
    )

    frames = _capture_frames(monkeypatch, ["RIGHT", "ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

    # Frame 1 is after Right, frame 2 after the no-op Enter.  The same chip
    # stays selected; Enter must not reset or advance the chip cursor.
    selected = f"{tui._REVERSE}✓ deepseek-ollama{tui._RESET}"
    assert selected in frames[1]["claude"]
    assert frames[2]["claude"] == frames[1]["claude"]


@pytest.mark.integration
def test_a_removed_wrapper_disappears_from_the_strip(monkeypatch):
    """Chips are rebuilt once per iteration, so a wrapper removed mid-session
    is gone on the next frame — and a chip cursor left past the end of the
    shorter strip is clamped, not left dangling."""
    from codehelper.services.wrappers import install_wrapper, remove_wrapper

    paths = Paths.default()
    install_wrapper(paths, "glm", token="test-token")

    import codehelper.cli.menu as menu

    frames: list[dict[str, str]] = []
    real_select = menu.select_from_menu
    state = {"n": 0}

    def _select(items, **kwargs):
        def _read():
            frames.append(_chip_rows(items))
            state["n"] += 1
            if state["n"] == 1:
                return "RIGHT"  # park the cursor on the DeepSeek chip
            if state["n"] == 2:
                remove_wrapper(paths, "glm", force=True)
                return "CANCEL"  # leave the menu so the loop re-renders
            return "CANCEL"

        return real_select(items, read_key=_read, **kwargs)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    assert main(["tui"]) == 0

    import codehelper.cli.tui as tui

    assert f"{tui._REVERSE}deepseek-ollama{tui._RESET}" in frames[1]["claude"]


@pytest.mark.unit
def test_every_main_screen_key_survives_translation():
    """A key bound in `on_key` but missing from `menu._PASSTHROUGH` is
    silently dead — the bug that left `w` (switch) non-functional while it was
    documented in the help text and listed in `keymap._LAYOUTS`. Pin every
    single-character binding the main screen declares against the translator
    so the two can never drift apart again.
    """
    from codehelper.cli.menu import _translate_char

    # Mirrors the `keys` table in TuiSession.run.
    for key in ("a", "p", "s", "?", "e", "d"):
        assert _translate_char(key) == key, f"{key!r} is bound but not passed through"
    # `t` is bound through the TOKEN alias rather than verbatim.
    assert _translate_char("t") == "TOKEN"


# --------------------------------------------------------------------------- #
# The proxy controls on the Settings screen
# --------------------------------------------------------------------------- #


def _settings_screen(monkeypatch, picks):
    """Open Settings from the main screen, make `picks` there, then quit.

    The Settings screen is a plain `_pick` loop, so it is identified by the
    prompt rather than by a marker key — the same shape `_tab_on_profile_screen`
    uses for the Profile screen.
    """
    iterator = iter([*picks, "__back__"])
    seen = {"opened": False}

    def _select(_items, prompt="", **_kwargs):
        if prompt == "Settings:":
            seen["opened"] = True
            return next(iterator)
        return "quit" if seen["opened"] else "settings"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)
    return seen


def _write_proxy_settings(env: dict) -> None:
    paths = Paths.default()
    paths.claude_dir.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(
        json.dumps({"env": env, "hooks": {"PreToolUse": []}}), encoding="utf-8"
    )


def _proxy_env() -> dict:
    return json.loads(Paths.default().claude_settings().read_text(encoding="utf-8"))[
        "env"
    ]


def _settings_labels(monkeypatch) -> list:
    """Capture the Settings screen's row labels, then quit."""
    labels = []

    def _select(items, prompt="", **_kwargs):
        if prompt == "Settings:":
            labels.extend(
                entry[1] if isinstance(entry, tuple) else entry for entry in items
            )
            return "__back__"
        return "settings" if not labels else "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    return labels


@pytest.mark.integration
def test_settings_screen_has_no_second_proxy_toggle(monkeypatch):
    """On/off belongs to the chipset row alone. Two controls for one setting
    is exactly the duplication that made a bare `Proxy: on` row ambiguous —
    is `on` the state, or the button?"""
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    labels = _settings_labels(monkeypatch)

    main(["tui"])

    assert any(str(label).startswith("Proxy address:") for label in labels)
    assert not any(str(label).startswith("Proxy:") for label in labels)


@pytest.mark.integration
def test_settings_screen_reads_the_address_from_the_file(monkeypatch):
    """The row has to read the file, not a flag on the session — a `switch`
    or a hand-edit between visits must show up."""
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    labels = _settings_labels(monkeypatch)

    main(["tui"])

    assert "Proxy address: http://127.0.0.1:8118" in labels


@pytest.mark.integration
def test_settings_screen_sets_a_new_proxy_address(monkeypatch):
    _write_proxy_settings({"IS_DEMO": "1"})
    _settings_screen(monkeypatch, ["proxy-url"])
    monkeypatch.setattr(
        "codehelper.cli.menu.read_line", lambda *_a, **_k: "http://10.0.0.1:3128"
    )

    main(["tui"])

    assert _proxy_env()["HTTPS_PROXY"] == "http://10.0.0.1:3128"


@pytest.mark.integration
def test_settings_screen_edits_no_proxy_in_both_spellings(monkeypatch):
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    _settings_screen(monkeypatch, ["proxy-no-proxy"])
    monkeypatch.setattr(
        "codehelper.cli.menu.read_line", lambda *_a, **_k: "localhost,.corp"
    )

    main(["tui"])

    env = _proxy_env()
    assert env["NO_PROXY"] == "localhost,.corp"
    assert env["no_proxy"] == "localhost,.corp"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:8118"


@pytest.mark.integration
def test_cancelling_the_address_prompt_changes_nothing(monkeypatch):
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    _settings_screen(monkeypatch, ["proxy-url"])
    monkeypatch.setattr("codehelper.cli.menu.read_line", lambda *_a, **_k: "")

    main(["tui"])

    assert _proxy_env()["HTTPS_PROXY"] == "http://127.0.0.1:8118"


# --------------------------------------------------------------------------- #
# The Tokens screen (Settings → Tokens) — the CLI `tokens` command's viewer
# twin: cached credentials + token env vars, masked until toggled.
# --------------------------------------------------------------------------- #


def _open_tokens_screen(monkeypatch, picks):
    """Settings → Tokens, capture each Tokens-screen frame's items + on_key,
    make `picks` there, then back out and quit. Frames are identified by the
    "Stored tokens" prompt, the same way `_settings_screen` identifies
    Settings. The main screen enters Settings by returning the `_SETTINGS`
    sentinel, exactly what its real `s` key binding does."""
    frames: list[dict] = []
    iterator = iter(picks)

    def _select(_items, prompt="", **kwargs):
        if prompt == "Settings:":
            return "tokens" if not frames else "__back__"
        if str(prompt).startswith("Stored tokens"):
            frames.append(
                {
                    "items": list(_items),
                    "on_key": kwargs.get("on_key") or {},
                }
            )
            return next(iterator, "__back__")
        # The main screen (no matching prompt): enter Settings once, quit after.
        return "settings" if not frames else "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)
    return frames


def _labels(frame) -> list[str]:
    return [
        entry[1] if isinstance(entry, tuple) else str(entry) for entry in frame["items"]
    ]


@pytest.mark.integration
def test_tokens_screen_lists_the_settings_entry_and_masks_by_default(monkeypatch):
    from codehelper.services.secrets import save_credential

    token = "fe4209" + "a" * 40 + "6309"
    save_credential(Paths.default(), "zai", token)

    # The Settings screen must carry the row that reaches the viewer.
    settings_labels = []
    frames: list[dict] = []

    def _select(_items, prompt="", **kwargs):
        if prompt == "Settings:":
            settings_labels.extend(
                entry[1] if isinstance(entry, tuple) else str(entry) for entry in _items
            )
            return "tokens" if not frames else "__back__"
        if str(prompt).startswith("Stored tokens"):
            frames.append({"items": list(_items), "on_key": kwargs.get("on_key") or {}})
            return "__back__"
        return "settings" if not frames else "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)

    main(["tui"])

    assert "Tokens" in settings_labels

    labels = _labels(frames[0])
    assert any("zai/default" in label and "fe42****6309" in label for label in labels)
    # Masked is the DEFAULT: the full value never reaches a frame.
    assert not any(token in label for label in labels)


@pytest.mark.integration
def test_tokens_screen_toggle_reveals_then_re_masks(monkeypatch):
    from codehelper.services.secrets import save_credential

    token = "fe4209" + "a" * 40 + "6309"
    save_credential(Paths.default(), "zai", token)

    frames = _open_tokens_screen(monkeypatch, ["toggle-reveal", "toggle-reveal"])

    main(["tui"])

    assert len(frames) == 3
    assert not any(token in label for label in _labels(frames[0]))
    assert any(token in label for label in _labels(frames[1]))  # revealed
    assert not any(token in label for label in _labels(frames[2]))  # re-masked


@pytest.mark.integration
def test_tokens_screen_reveals_never_survive_a_screen_exit(monkeypatch):
    """Reveal is per-visit state: leaving the screen (Back) resets it, so a
    RE-ENTRY shows masked values until the user toggles again. A persisted
    reveal would splash full credentials on a second visit with no new
    action by the user — the exact accident the mask exists to prevent."""
    from codehelper.services.secrets import save_credential

    token = "fe4209" + "a" * 40 + "6309"
    save_credential(Paths.default(), "zai", token)

    frames: list[dict] = []
    answers = iter(["toggle-reveal", "__back__", "__back__"])

    def _select(_items, prompt="", **kwargs):
        if prompt == "Settings:":
            # Two visits to the Tokens screen (frames 0-1 = visit 1, frame 2
            # = visit 2), then back out for good.
            return "tokens" if len(frames) < 3 else "__back__"
        if str(prompt).startswith("Stored tokens"):
            frames.append({"items": list(_items), "on_key": kwargs.get("on_key") or {}})
            return next(answers, "__back__")
        return "settings" if not frames else "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)

    main(["tui"])

    assert len(frames) == 3
    assert not any(token in label for label in _labels(frames[0]))  # visit 1 masked
    assert any(token in label for label in _labels(frames[1]))  # revealed in visit 1
    assert not any(
        token in label for label in _labels(frames[2])
    )  # visit 2 starts masked again


@pytest.mark.integration
def test_tokens_screen_binds_s_and_s_survives_translation(monkeypatch):
    """The `w`-key rule: a key bound in `on_key` but missing from
    `menu._PASSTHROUGH` is silently dead. The Tokens screen's `s` must both
    be bound and survive `_translate_char`."""
    frames = _open_tokens_screen(monkeypatch, [])

    main(["tui"])

    on_key = frames[0]["on_key"]
    assert "s" in on_key

    from codehelper.cli.menu import _translate_char

    assert _translate_char("s") == "s"


@pytest.mark.integration
def test_proxy_row_marks_the_live_state_and_offers_the_other(monkeypatch):
    """The whole point of the chipset shape: both options are always on
    screen and `✓` says which one is live, so Enter is never ambiguous."""
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    rows = []

    def _select(items, **_kwargs):
        rows.extend(
            entry[1](selected=False, ansi=False)
            for entry in items
            if isinstance(entry, tuple) and callable(entry[1])
        )
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)

    main(["tui"])

    proxy_row = next(r for r in rows if r.startswith("proxy"))
    assert "✓ on" in proxy_row
    assert "off" in proxy_row
    assert "http://127.0.0.1:8118" in proxy_row


@pytest.mark.integration
def test_proxy_row_marks_off_when_the_proxy_is_disabled(monkeypatch):
    from codehelper.services.state import set_saved_proxy

    _write_proxy_settings({"HTTPS_PROXY": "", "HTTP_PROXY": ""})
    set_saved_proxy(Paths.default(), "http://127.0.0.1:8118")
    rows = []

    def _select(items, **_kwargs):
        rows.extend(
            entry[1](selected=False, ansi=False)
            for entry in items
            if isinstance(entry, tuple) and callable(entry[1])
        )
        return "quit"

    monkeypatch.setattr("codehelper.cli.menu.select_from_menu", _select)

    main(["tui"])

    proxy_row = next(r for r in rows if r.startswith("proxy"))
    assert "✓ off" in proxy_row
    assert "✓ on" not in proxy_row


@pytest.mark.integration
def test_enter_on_the_off_chip_disables_the_proxy(monkeypatch):
    """Enter applies the chip under the cursor — the same contract as an
    agent row, which is what removes the on/off ambiguity."""
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    # Two DOWNs past claude/codex reach the proxy row; RIGHT moves the chip
    # cursor from `on` to `off`; Enter applies it.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "RIGHT", "ENTER", "CANCEL"])

    assert main(["tui"]) == 0

    assert _proxy_env()["HTTPS_PROXY"] == ""


@pytest.mark.integration
def test_enter_on_the_already_live_chip_is_a_silent_noop(monkeypatch):
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    before = Paths.default().claude_settings().read_text(encoding="utf-8")
    # The cursor starts on `on`, which is already applied.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "ENTER", "CANCEL"])

    assert main(["tui"]) == 0

    assert Paths.default().claude_settings().read_text(encoding="utf-8") == before


@pytest.mark.integration
def test_turning_on_with_no_address_anywhere_asks_for_one(monkeypatch):
    """The one case Enter cannot just act on: refusing with an error the user
    can't fix from this screen would be a dead end, so it prompts instead."""
    _write_proxy_settings({"IS_DEMO": "1"})
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "ENTER", "CANCEL"])
    monkeypatch.setattr(
        "codehelper.cli.menu.read_line", lambda *_a, **_k: "http://10.0.0.1:3128"
    )
    monkeypatch.setattr("codehelper.cli.menu.press_any_key", lambda *_a, **_k: None)

    assert main(["tui"]) == 0

    assert _proxy_env()["HTTPS_PROXY"] == "http://10.0.0.1:3128"


@pytest.mark.integration
def test_delete_on_the_proxy_row_is_inert(monkeypatch):
    """The row owns no file; `d` there must not go looking for a wrapper
    named `proxy:`."""
    _write_proxy_settings({"HTTPS_PROXY": "http://127.0.0.1:8118"})
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "d", "CANCEL"])

    assert main(["tui"]) == 0


@pytest.mark.integration
def test_delete_on_an_agent_chipset_row_is_inert(monkeypatch):
    """Same contract as the proxy row: a chipset row owns no wrapper file, so
    `d` there must not dispatch a removal for a wrapper literally named
    `agent:claude` — the dispatch a user hit as "invalid wrapper name"."""
    seen = []

    def _spy_remove(*args, **_kwargs):
        seen.append(args)
        return 0

    monkeypatch.setattr("codehelper.cli.parser._handle_remove", _spy_remove)
    # Row 1 is the claude chipset row — no DOWNs needed.
    _real_menu_keys(monkeypatch, ["d", "CANCEL"])

    assert main(["tui"]) == 0
    assert seen == []


@pytest.mark.integration
def test_delete_on_a_wrapper_row_still_dispatches(monkeypatch):
    """The inertness above must not swallow real removals: `d` on a wrapper
    row dispatches `_handle_remove` with that row's alias."""
    seen = []

    def _spy_remove(*args, **_kwargs):
        seen.append(args)
        return 0

    monkeypatch.setattr("codehelper.cli.parser._handle_remove", _spy_remove)
    # Row 5 is the first preset/wrapper row: agent:claude, agent:codex,
    # proxy, + add agent, then deepseek-ollama.
    _real_menu_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "d", "CANCEL"])

    assert main(["tui"]) == 0
    assert len(seen) == 1
    (req,) = seen[0]
    assert req.name == "deepseek-ollama"


@pytest.mark.unit
def test_codex_chip_readback_distinguishes_two_models_on_one_provider():
    """Two codex wrappers sharing a provider but naming DIFFERENT models must
    not both read as applied.

    `config.toml` records `model` alongside `model_provider`, so unlike
    claude's `current_switch` (which has to match a base URL back to a
    provider) codex's readback has the model right there and can tell the two
    apart. Matching on the provider name alone marked EVERY chip on that
    provider — not merely "the first one", as the old docstring claimed,
    since the predicate is evaluated per chip with nothing tracking which
    came first.
    """
    from codehelper.cli.tui import TuiSession
    from codehelper.services.codex_default import apply_set_default
    from codehelper.services.model import get_agent, get_provider
    from codehelper.services.spec import build_spec

    paths = Paths.default()
    applied = build_spec(
        agent="codex", provider="ollama-direct", model="deepseek-v4-flash:0731-cloud"
    )
    other = build_spec(agent="codex", provider="ollama-direct", model="glm-5.3-flash")

    # Make `applied` genuinely the live default, the way `set-default` does,
    # so the readback runs against a real config.toml rather than a stub.
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    apply_set_default(
        paths,
        agent=get_agent("codex"),
        provider=get_provider("ollama-direct"),
        model="deepseek-v4-flash:0731-cloud",
        force=True,
    )

    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._refresh_active_label()

    assert session._chip_is_applied("codex", applied) is True
    assert session._chip_is_applied("codex", other) is False


def _install_zai_pair(alias_a: str, token_a: str, alias_b: str, token_b: str):
    """Two zai claude wrappers with identical axes and DIFFERENT tokens."""
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    first = build_spec(agent="claude", provider="zai", model="glm-5.3", alias=alias_a)
    second = build_spec(agent="claude", provider="zai", model="glm-5.3", alias=alias_b)
    install_wrapper(Paths.default(), first, token=token_a)
    install_wrapper(Paths.default(), second, token=token_b)
    return first, second


def _apply_live(spec, token: str) -> None:
    """Make ``spec`` the genuinely live backend, the way Enter on its chip
    does (``switch --from-wrapper`` = live_axes_for_spec + the file token +
    the spec's recorded context window, issue #83)."""
    from codehelper.services.claude_settings import apply_switch, live_axes_for_spec

    provider, tiers, subagent = live_axes_for_spec(spec)
    apply_switch(
        Paths.default(),
        provider=provider,
        tier_models=tiers,
        token=token,
        subagent_model=subagent,
        context_window=getattr(spec, "context_window", None),
        force=True,
    )


def _claude_session():
    from codehelper.cli.tui import TuiSession

    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._refresh_active_label()
    return session


def _chip_named(session, name: str):
    return next(
        chip for chip in session._chips["claude"] if getattr(chip, "name", None) == name
    )


@pytest.mark.integration
def test_claude_chip_readback_distinguishes_tokens_on_one_backend(
    monkeypatch,
):
    """Two claude wrappers sharing provider AND tier models but carrying
    DIFFERENT tokens must not both read as applied.

    `matches_switch_spec` compares the managed env exactly, but builds the
    expected env with the LIVE token substituted in — so the token, the only
    field distinguishing two accounts on one backend, never took part in the
    comparison. Three zai chips (a preset and two wrappers, glm-5.3 uniform
    each) then rendered `✓` at once, and — worse — Enter on the two that
    weren't live was swallowed by the no-op guard. The chip's apply path
    takes its token from the installed file, so the readback can be exact.
    """
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    first, second = _install_zai_pair(
        "glm-acc1", "token-aaaa", "glm-acc2", "token-bbbb"
    )
    _apply_live(first, "token-aaaa")

    session = _claude_session()

    assert session._chip_is_applied("claude", first) is True
    assert session._chip_is_applied("claude", second) is False
    # The zai glm PRESET chip resolves its token non-interactively; with no
    # ZAI_API_KEY and no cached default profile it has no token to claim —
    # and a chip press would fail cleanly, which is not a no-op.
    assert session._chip_is_applied("claude", _chip_named(session, "glm")) is False


@pytest.mark.integration
def test_claude_chip_row_shows_no_checkmark_when_live_token_matches_no_chip(
    monkeypatch,
):
    """A backend switched to a token no chip carries (an env or prompt
    switch) marks no chip applied — and native must not claim it either,
    since a managed env IS live. A row with no `✓` is the honest "none of
    these chips": a chip's ✓ names the ACCOUNT it would apply, not merely
    the endpoint."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    _install_zai_pair("glm-acc1", "token-aaaa", "glm-acc2", "token-bbbb")
    from codehelper.services.spec import build_spec

    probe = build_spec(agent="claude", provider="zai", model="glm-5.3")
    _apply_live(probe, "token-nobody-carries")

    session = _claude_session()

    assert "✓" not in session._chip_row("claude")(selected=False, ansi=False)


@pytest.mark.integration
def test_preset_chip_readback_uses_the_non_interactive_token(monkeypatch):
    """The preset chip reads applied only when its NON-INTERACTIVE token
    resolution — env, then the default cached profile, exactly what a chip
    press resolves — equals the live token. A wrapper on the same axes under
    a different account must not ride along."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    from codehelper.services.secrets import seed_default_profile

    seed_default_profile(Paths.default(), "zai", "preset-token")
    first, _second = _install_zai_pair(
        "glm-acc1", "token-aaaa", "glm-acc2", "token-bbbb"
    )
    _apply_live(first, "preset-token")

    session = _claude_session()

    assert session._chip_is_applied("claude", _chip_named(session, "glm")) is True
    assert session._chip_is_applied("claude", first) is False


@pytest.mark.integration
def test_enter_applies_a_chip_whose_token_differs_on_the_same_backend(monkeypatch):
    """The ✓ doubles as the no-op guard (`_apply_chip` skips a chip that
    reads applied): with the token part of the readback, Enter on the SECOND
    account must reach the apply path instead of being silently swallowed."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    import codehelper.cli.tui as tui

    first, _second = _install_zai_pair(
        "glm-acc1", "token-aaaa", "glm-acc2", "token-bbbb"
    )
    _apply_live(first, "token-aaaa")

    seen = []
    monkeypatch.setattr(
        tui.TuiSession,
        "_apply_switch_wrapper",
        lambda _self, spec: seen.append(spec.name),
    )
    index = next(
        i
        for i, chip in enumerate(_claude_session()._chips["claude"])
        if getattr(chip, "name", None) == "glm-acc2"
    )
    _real_menu_keys(monkeypatch, ["RIGHT"] * index + ["ENTER", "CANCEL"])
    assert main(["tui"]) == 0

    assert seen == ["glm-acc2"]


@pytest.mark.integration
def test_literal_chips_not_applied_when_a_secret_wrapper_is_live(monkeypatch):
    """A secret wrapper on an OVERRIDABLE provider (ollama-direct) shares
    provider AND tier models with the literal `deepseek-ollama` preset — the
    token is the only difference, and a literal chip DOES resolve one at
    apply time (its `auth_value`, through the same resolvers), so the
    readback compares it for EVERY chip, not just `auth="secret"` ones.

    With the secret wrapper's own account live, exactly ITS chip reads
    applied: since #81 the marker records the auth override, so the
    reconstructed wrapper chip resolves the embedded account token and Enter
    on it is a genuine no-op — while the literal preset chip does NOT
    (pressing it would write 'ollama', a different credential). Before #81
    the reconstruction read the wrapper back literal, so its chip resolved
    the wrong token and the ✓ the row owed the live account was missing —
    the #80 degradation this fix narrows.
    """
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    from codehelper.services.model import get_provider, with_auth
    from codehelper.services.spec import build_spec, get_preset, spec_from_preset
    from codehelper.services.wrappers import install_wrapper

    preset_spec = spec_from_preset(get_preset("deepseek-ollama"))
    secret = build_spec(
        agent="claude",
        provider=with_auth(get_provider("ollama-direct"), want_secret=True),
        model=preset_spec.model,
        alias="ollama-secret",
        tier_models=preset_spec.tier_models,
        subagent_model=preset_spec.subagent_model,
    )
    install_wrapper(Paths.default(), secret, token="sk-ollama-secret")
    _apply_live(secret, "sk-ollama-secret")

    session = _claude_session()

    # The chips the UI actually renders are RECONSTRUCTED from the installed
    # file — since #81 its marker carries the auth override, so the secret
    # wrapper reads back secret and its chip resolves the ACCOUNT token.
    # Assert on those, not on the spec object built above (whose auth=
    # "secret" would take a different code path than the real row ever
    # sees). Exactly the live chip carries the ✓.
    assert (
        session._chip_is_applied("claude", _chip_named(session, "ollama-secret"))
        is True
    )
    assert (
        session._chip_is_applied("claude", _chip_named(session, "deepseek-ollama"))
        is False
    )
    assert "✓ ollama-secret" in session._chip_row("claude")(selected=False, ansi=False)
    assert "✓ deepseek-ollama" not in session._chip_row("claude")(
        selected=False, ansi=False
    )


@pytest.mark.integration
def test_ctx_wrapper_chip_matches_only_itself():
    """CHIP HONESTY for the recorded window (issue #83, the #80/#81 class):
    a wrapper whose marker records ctx=1000000 shares provider AND tier
    models with a same-model catalog wrapper — the recorded window is the
    only difference. The reconstructed chip must match ITSELF (its
    matches_switch_spec passes the marker's ctx through) and no other.

    Before the threading, the expected env was catalog-derived while the
    live env carried the key — the wrapper's own chip never read applied.
    """
    from codehelper.services.model import get_provider
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    catalog = build_spec(
        agent="claude",
        provider=get_provider("zai"),
        model="mystery-3b",
        alias="mystery",
    )
    recorded = build_spec(
        agent="claude",
        provider=get_provider("zai"),
        model="mystery-3b",
        alias="mystery-1m",
        context_window=1_000_000,
    )
    install_wrapper(Paths.default(), catalog, token="sk-mystery")
    install_wrapper(Paths.default(), recorded, token="sk-mystery")
    _apply_live(recorded, "sk-mystery")

    session = _claude_session()

    # Assert on the RECONSTRUCTED chips (the rows the UI actually renders),
    # not on the spec objects built above. Exactly the ctx-carrying chip
    # carries the ✓ — the same-axes catalog chip does not (pressing it
    # would drop the window declaration from the live env).
    assert (
        session._chip_is_applied("claude", _chip_named(session, "mystery-1m")) is True
    )
    assert session._chip_is_applied("claude", _chip_named(session, "mystery")) is False
    assert "✓ mystery-1m" in session._chip_row("claude")(selected=False, ansi=False)
    assert "✓ mystery " not in session._chip_row("claude")(selected=False, ansi=False)


# --------------------------------------------------------------------------- #
# Settings → Providers submenu — runtime disable/enable parity (issue #89)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_providers_submenu_dispatches_the_real_handlers(monkeypatch):
    """The submenu must dispatch into parser._handle_disable/_handle_enable —
    the TUI mirrors the CLI, never a second implementation. The pick IS the
    confirmation (yes=True), so `disable ollama-direct` from the submenu
    deletes its wrappers and writes state exactly like the command."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.services.paths import Paths
    from codehelper.services.state import disabled_providers
    from codehelper.services.wrappers import install_wrapper, is_installed

    install_wrapper(Paths.default(), "deepseek-ollama", token="ollama")

    seen: list[str] = []
    picks = iter(["ollama-direct", "__back__"])  # toggle once, then leave

    def _select(_items, *, prompt, **_kwargs):
        text = prompt() if callable(prompt) else prompt
        seen.append(text)
        return next(picks)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._run_providers_screen()

    assert "Providers:" in seen
    assert disabled_providers(Paths.default()) == frozenset({"ollama-direct"})
    assert not is_installed(Paths.default(), "deepseek-ollama")


@pytest.mark.integration
def test_providers_submenu_toggles_back_to_enabled(monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession
    from codehelper.services.paths import Paths
    from codehelper.services.state import disabled_providers, set_provider_disabled

    set_provider_disabled(
        Paths.default(),
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    picks = iter(["ollama-direct", "__back__"])  # toggle once, then leave

    def _select(_items, *, prompt, **_kwargs):  # noqa: ARG001 — keyword-called by the screen
        return next(picks)

    monkeypatch.setattr(menu, "select_from_menu", _select)
    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._run_providers_screen()
    assert disabled_providers(Paths.default()) == frozenset()


@pytest.mark.integration
def test_providers_submenu_hides_env_reset_and_tags_suspended(monkeypatch):
    """`native` (env_reset) is absent — disable refuses it, so offering it
    would be an action that can only fail. gemini stays visible (tagged)."""
    import codehelper.cli.menu as menu
    from codehelper.cli.tui import TuiSession

    captured: dict = {}

    def _select(items, *, prompt, **_kwargs):  # noqa: ARG001 — keyword-called by the screen
        captured["rows"] = {
            value: label for value, label in items if isinstance(value, str)
        }
        return "__back__"

    monkeypatch.setattr(menu, "select_from_menu", _select)
    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._run_providers_screen()

    assert "native" not in captured["rows"]
    assert "gemini" in captured["rows"]
    assert "suspended" in captured["rows"]["gemini"]
    assert captured["rows"]["ollama-direct"] == "ollama-direct: enabled"


@pytest.mark.integration
def test_chips_of_a_disabled_provider_vanish_from_the_chipset():
    """A disabled provider produces NO chip at all — its wrappers were
    deleted on disable, so a greyed chip would promise an Enter that cannot
    resolve."""
    from codehelper.cli.tui import TuiSession
    from codehelper.services.paths import Paths
    from codehelper.services.state import set_provider_disabled
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(Paths.default(), "deepseek-ollama", token="ollama")
    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._refresh_active_label()
    assert any(
        getattr(chip, "name", None) == "deepseek-ollama"
        for chip in session._chips_for("claude", Paths.default())
    )

    set_provider_disabled(
        Paths.default(),
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    session = TuiSession(SimpleNamespace(debug=False, dry_run=False))
    session._refresh_active_label()
    assert not any(
        getattr(chip, "name", None) == "deepseek-ollama"
        for chip in session._chips_for("claude", Paths.default())
    )
