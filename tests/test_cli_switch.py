"""Tests for ``codehelper switch`` — live-patching ``~/.claude/settings.json``.

Everything runs through ``main([...])`` so the argparse wiring (positional
``provider`` vs ``--provider``, ``--restore``/``--slot`` validation) is
exercised too, not just the handler. Distinct from ``test_claude_settings.py``,
which tests the service layer directly — this file tests the CLI seam:
disambiguation, request building, and the ``--from-wrapper`` fast path's
integration with an installed wrapper.
"""

from __future__ import annotations

import json

import pytest

from codehelper.__main__ import main
from codehelper.cli.parser import _handle_switch
from codehelper.cli.requests import SwitchRequest
from codehelper.services.claude_settings import MANAGED_ENV_KEYS
from codehelper.services.paths import Paths
from codehelper.services.secrets import save_credential
from codehelper.services.spec import build_spec
from codehelper.services.wrappers import install_wrapper


def _settings(tmp_path) -> dict:
    return json.loads(
        Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")
    )


def _write_settings(tmp_path, data: dict) -> None:
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(json.dumps(data), encoding="utf-8")


def _preset_request(name: str) -> SwitchRequest:
    return SwitchRequest(
        provider=None,
        from_wrapper=None,
        model=None,
        haiku=None,
        sonnet=None,
        opus=None,
        subagent_model=None,
        base_url=None,
        auth=None,
        profile=None,
        restore=False,
        slot=None,
        status=False,
        dry_run=False,
        force=True,
        debug=False,
        from_preset=name,
    )


@pytest.mark.integration
def test_switch_positional_provider_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    code = main(["switch", "zai", "--model", "glm-5.2", "--force"])
    assert code == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-env"


@pytest.mark.integration
def test_switch_flag_provider_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    code = main(["switch", "--provider", "zai", "--model", "glm-5.2", "--force"])
    assert code == 0
    assert _settings(tmp_path)["env"]["ANTHROPIC_BASE_URL"] == (
        "https://api.z.ai/api/anthropic"
    )


@pytest.mark.integration
def test_switch_positional_and_flag_provider_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    code = main(["switch", "zai", "--provider", "zai", "--model", "x", "--force"])
    assert code != 0


@pytest.mark.integration
def test_switch_native_never_reads_env_or_prompts(tmp_path, monkeypatch):
    """switch native must not call secrets.resolve_token at all — a reset
    provider (auth='none', env_reset=True) has nothing to resolve."""

    def _explode(**_kwargs):
        raise AssertionError("resolve_token must not be called for switch native")

    monkeypatch.setattr("codehelper.cli.parser.secrets.resolve_token", _explode)
    _write_settings(
        tmp_path,
        {
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "sk-old",
            }
        },
    )
    code = main(["switch", "native", "--force"])
    assert code == 0
    assert {
        "ANTHROPIC_BASE_URL": _settings(tmp_path)["env"].get("ANTHROPIC_BASE_URL"),
        "ANTHROPIC_AUTH_TOKEN": _settings(tmp_path)["env"].get("ANTHROPIC_AUTH_TOKEN"),
    } == {
        "ANTHROPIC_BASE_URL": "",
        "ANTHROPIC_AUTH_TOKEN": "",
    }


@pytest.mark.integration
def test_switch_native_zai_native_round_trip(tmp_path, monkeypatch):
    """zai(glm) -> native must blank every managed key the zai switch itself
    set, without adding CLAUDE_CODE_SUBAGENT_MODEL — the glm preset never
    sets it, so there is no stale process value for native to reset — the
    same "don't add keys nobody set" invariant this cycle's own #61 review
    caught. Regression coverage for #60's audit scope beyond the ANTHROPIC_*
    keys the original deepseek->native bug was found in.

    Starts from a zai switch, not a pristine file: `switch native` on a
    settings.json with no managed override is a no-op (nothing to reset),
    so a leading `native --force` on a fresh tmp_path would write nothing
    and leave no settings.json to read."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-secret")

    assert _handle_switch(_preset_request("glm")) == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-zai-secret"
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env

    assert main(["switch", "native", "--force"]) == 0
    final_env = _settings(tmp_path)["env"]
    assert all(final_env[key] == "" for key in MANAGED_ENV_KEYS if key in env)
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in final_env


@pytest.mark.integration
def test_switch_warns_when_env_token_differs_from_cached(tmp_path, monkeypatch, capsys):
    """Issue #71: env wins the resolution silently — the warning names the
    env var and never leaks either full token. The env value is still what
    gets written (documented precedence; the warning is a diagnostic, not a
    prompt)."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env-stale-token")
    save_credential(Paths.from_home(tmp_path), "zai", "sk-cached-working")

    assert main(["switch", "zai", "--model", "glm-5.2", "--force"]) == 0

    err = capsys.readouterr().err
    assert "ZAI_API_KEY" in err
    assert "sk-env-stale-token" not in err
    assert "sk-cached-working" not in err
    assert _settings(tmp_path)["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-env-stale-token"


@pytest.mark.integration
def test_switch_dry_run_still_warns_on_env_cache_conflict(
    tmp_path, monkeypatch, capsys
):
    """The warning is emitted at resolution time, so a dry run — exactly the
    diagnostic context — shows it even though nothing is written."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env-stale-token")
    save_credential(Paths.from_home(tmp_path), "zai", "sk-cached-working")

    assert main(["switch", "zai", "--model", "glm-5.2", "--dry-run"]) == 0

    assert "ZAI_API_KEY" in capsys.readouterr().err


@pytest.mark.integration
def test_switch_silent_when_env_token_matches_cached(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-same-token")
    save_credential(Paths.from_home(tmp_path), "zai", "sk-same-token")

    assert main(["switch", "zai", "--model", "glm-5.2", "--force"]) == 0

    assert capsys.readouterr().err == ""


@pytest.mark.integration
def test_switch_deepseek_ollama_to_glm_cross_provider(tmp_path, monkeypatch):
    """deepseek-ollama -> glm (acceptance criteria in #60): both the model and
    ANTHROPIC_BASE_URL must change, and deepseek-ollama's CLAUDE_CODE_SUBAGENT_MODEL
    must not survive since the glm preset never sets one."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-secret")

    assert _handle_switch(_preset_request("deepseek-ollama")) == 0
    assert _settings(tmp_path)["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == (
        "deepseek-v4-flash:0731-cloud"
    )

    assert _handle_switch(_preset_request("glm")) == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.3"
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env


@pytest.mark.integration
def test_chip_preset_deepseek_ollama_applies_without_a_wrapper_file(tmp_path):
    foreign = Paths.from_home(tmp_path).script_for("deepseek-ollama")
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\necho foreign\n", encoding="utf-8")
    assert _handle_switch(_preset_request("deepseek-ollama")) == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "deepseek-v4-flash:0731-cloud"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "deepseek-v4-flash:0731-cloud"


@pytest.mark.integration
def test_chip_preset_glm_ollama_applies_live_ollama_settings(tmp_path):
    assert _handle_switch(_preset_request("glm-ollama")) == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert {
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] for tier in ("HAIKU", "SONNET", "OPUS")
    } == {"glm-5.2:cloud"}
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2:cloud"


@pytest.mark.integration
def test_chip_preset_glm_never_prompts_for_a_token(tmp_path, monkeypatch):
    """A token-bearing preset chip (glm / Z.ai) must resolve its token
    non-interactively: a prompt inside the running menu would swallow the
    user's keystrokes, breaking the no-prompt hot-apply a chip promises."""
    import codehelper.services.secrets as secrets
    from codehelper.cli.parser import _switch_axes_from_preset
    from codehelper.errors import CodeHelperError

    seen = {}
    cached = {"value": "sk-cached", "source": "cache"}

    def fake_resolve_token(*, env_var, prompt, paths, provider_name, **kwargs):
        seen["env_var"] = env_var
        seen["getpass_fn"] = kwargs.get("getpass_fn")
        seen["prompt"] = prompt
        # Simulate a cache hit so the spy never actually prompts.
        return type("R", (), cached)()

    monkeypatch.setattr(secrets, "resolve_token", fake_resolve_token)

    _switch_axes_from_preset(_preset_request("glm"), Paths.from_home(tmp_path))
    assert seen["env_var"] == "ZAI_API_KEY"
    # A chip press must never block on a hidden token prompt: resolve_token
    # must be handed a getpass_fn that raises rather than reads stdin.
    assert seen["getpass_fn"] is not None
    with pytest.raises(CodeHelperError):
        seen["getpass_fn"](seen["prompt"])


@pytest.mark.integration
def test_switch_status_never_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    main(["switch", "zai", "--model", "glm-5.2", "--force"])
    before = Paths.from_home(tmp_path).claude_settings().read_bytes()

    code = main(["switch", "--status"])

    assert code == 0
    assert Paths.from_home(tmp_path).claude_settings().read_bytes() == before
    out = capsys.readouterr().out
    assert "zai" in out


@pytest.mark.integration
def test_switch_status_reports_native_when_no_override(tmp_path, capsys):
    code = main(["switch", "--status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "native" in out


@pytest.mark.integration
def test_switch_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    _write_settings(tmp_path, {"env": {"HTTPS_PROXY": "http://x"}})
    before = Paths.from_home(tmp_path).claude_settings().read_bytes()

    code = main(["switch", "zai", "--model", "glm-5.2", "--dry-run"])

    assert code == 0
    assert Paths.from_home(tmp_path).claude_settings().read_bytes() == before


@pytest.mark.integration
def test_switch_restore_without_flag_after_switch_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    _write_settings(tmp_path, {"env": {"HTTPS_PROXY": "http://x"}})
    main(["switch", "zai", "--model", "glm-5.2", "--force"])
    assert "ANTHROPIC_BASE_URL" in _settings(tmp_path)["env"]

    code = main(["switch", "--restore", "--force"])

    assert code == 0
    assert _settings(tmp_path) == {"env": {"HTTPS_PROXY": "http://x"}}


@pytest.mark.integration
def test_switch_slot_without_restore_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    code = main(["switch", "--slot", "2", "zai", "--model", "x", "--force"])
    assert code != 0


@pytest.mark.integration
def test_switch_restore_with_provider_rejected(tmp_path):
    code = main(["switch", "--restore", "--provider", "zai", "--force"])
    assert code != 0


@pytest.mark.integration
def test_switch_restore_with_profile_rejected(tmp_path):
    code = main(["switch", "--restore", "--profile", "named", "--force"])
    assert code != 0


# --------------------------------------------------------------------------- #
# --from-wrapper
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_switch_from_wrapper_lifts_tier_models_and_token(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-secret")
    assert main(["add", "glm"]) == 0

    code = main(["switch", "--from-wrapper", "glm", "--force"])

    assert code == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-zai-secret"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.3"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.3"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-5.3"


@pytest.mark.integration
def test_switch_from_wrapper_glm_has_no_subagent_model(tmp_path, monkeypatch):
    """The glm preset deliberately sets subagent_model=None (services/spec.py)
    — a switch built from its wrapper must not write a stale/empty
    CLAUDE_CODE_SUBAGENT_MODEL, and must remove one left by a prior switch."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-secret")
    _write_settings(
        tmp_path, {"env": {"CLAUDE_CODE_SUBAGENT_MODEL": "stale-from-deepseek"}}
    )
    assert main(["add", "glm"]) == 0

    code = main(["switch", "--from-wrapper", "glm", "--force"])

    assert code == 0
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in _settings(tmp_path)["env"]


@pytest.mark.integration
def test_switch_from_wrapper_unknown_name_fails(tmp_path):
    code = main(["switch", "--from-wrapper", "does-not-exist", "--force"])
    assert code != 0


@pytest.mark.integration
def test_switch_from_wrapper_lifts_an_ollama_launch_wrapper_to_live_settings(tmp_path):
    assert main(["add", "glm-ollama"]) == 0
    code = main(["switch", "--from-wrapper", "glm-ollama", "--force"])
    assert code == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5.2:cloud"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2:cloud"


@pytest.mark.integration
def test_switch_from_wrapper_rejects_a_non_claude_wrapper(tmp_path, monkeypatch):
    """switch retargets a LIVE CLAUDE session, so a wrapper that belongs to
    another agent (codex shares Ollama) must fail closed — it must never
    silently retarget claude to a foreign selection just because the provider
    happens to declare the ANTHROPIC_SETTINGS shape."""
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "x",
                "--context-window",
                "none",
                "--alias",
                "codex-ollama",
            ]
        )
        == 0
    )

    code = main(["switch", "--from-wrapper", "codex-ollama", "--force"])

    assert code != 0
    settings = Paths.from_home(tmp_path).claude_settings()
    if settings.exists():
        env = json.loads(settings.read_text(encoding="utf-8")).get("env", {})
        assert not (set(env) & {"ANTHROPIC_BASE_URL", "ANTHROPIC_DEFAULT_SONNET_MODEL"})


@pytest.mark.integration
def test_switch_from_wrapper_and_provider_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0
    code = main(["switch", "--from-wrapper", "glm", "--provider", "zai", "--force"])
    assert code != 0


@pytest.mark.integration
def test_switch_from_wrapper_applies_a_secret_override_wrappers_own_token(
    tmp_path, monkeypatch
):
    """issue #81: a wrapper installed with `--auth secret` on ollama-direct
    embeds the ACCOUNT token — switching from it must apply that token, not
    the registry's literal 'ollama' credential the un-fixed reconstruction
    falls back to."""
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-account-token")
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--auth",
                "secret",
                "--model",
                "glm-5:cloud",
                "--context-window",
                "none",
                "--alias",
                "ollama-secure",
            ]
        )
        == 0
    )

    code = main(["switch", "--from-wrapper", "ollama-secure", "--force"])

    assert code == 0
    env = _settings(tmp_path)["env"]
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-account-token"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"


# --------------------------------------------------------------------------- #
# list providers — switch-only tag
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_list_providers_tags_native_as_switch_only(capsys):
    code = main(["list", "providers"])
    assert code == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "(switch-only)" in lines["native"]
    assert "(switch-only)" not in lines["zai"]
    assert "(switch-only)" not in lines["ollama-direct"]
    assert "(switch-only)" not in lines["litellm"]


# --------------------------------------------------------------------------- #
# Issue #83: the explicit context_window axis on switch.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_switch_flags_context_window_applies(tmp_path, monkeypatch):
    """`switch --provider P --model M --context-window N` declares N even
    though the catalog is silent."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")  # CI has no ambient token
    paths = Paths.from_home(tmp_path)
    assert (
        main(
            [
                "switch",
                "--provider",
                "zai",
                "--model",
                "mystery-3b",
                "--context-window",
                "750000",
                "--force",
            ]
        )
        == 0
    )
    from codehelper.services.claude_settings import read_settings

    _, settings = read_settings(paths)
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "750000"


@pytest.mark.integration
def test_switch_from_wrapper_uses_the_markers_ctx_and_never_prompts(tmp_path):
    """--from-wrapper lifts the recorded ctx out of the marker — zero prompts,
    same answer the wrapper's own script carries."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(
        paths,
        build_spec(
            agent="claude",
            provider="zai",
            model="mystery-3b",
            alias="mystery",
            context_window=750_000,
        ),
        token="sk-mystery",
    )

    def _fail(_items, **_kwargs):
        raise AssertionError("--from-wrapper showed the window menu")

    import unittest.mock as mock

    with mock.patch("codehelper.cli.menu.select_from_menu", _fail):
        assert main(["switch", "--from-wrapper", "mystery", "--force"]) == 0

    from codehelper.services.claude_settings import read_settings

    _, settings = read_settings(paths)
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "750000"


@pytest.mark.integration
def test_switch_rejects_context_window_with_from_wrapper(tmp_path, capsys):
    """One source per axis: a wrapper carries its own recorded ctx — an
    explicit --context-window next to it is a contradiction, refused. The
    wrapper EXISTS here, so exit 1 can only come from the conflict check —
    not from a missing-wrapper error masking it (the mutation trap this
    test originally fell into)."""
    install_wrapper(
        paths=Paths.from_home(tmp_path),
        spec=build_spec(
            agent="claude", provider="zai", model="mystery-3b", alias="mystery"
        ),
        token="sk-mystery",
    )
    assert (
        main(
            [
                "switch",
                "--from-wrapper",
                "mystery",
                "--context-window",
                "1000",
                "--force",
            ]
        )
        == 1
    )
    assert "explicit-axes form" in capsys.readouterr().err


@pytest.mark.integration
def test_chip_preset_apply_never_prompts_for_the_window(tmp_path, monkeypatch):
    """The chip hot-apply path (from_preset, non-interactive token) never
    reaches the window menu either."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")  # CI has no ambient token

    def _fail(_items, **_kwargs):
        raise AssertionError("chip apply showed the window menu")

    import unittest.mock as mock

    with mock.patch("codehelper.cli.menu.select_from_menu", _fail):
        assert _handle_switch(_preset_request("glm")) == 0


# --------------------------------------------------------------------------- #
# Review round 1 (PR #84): the switch flag must be parsed + validated like
# add's — switch never builds a spec, so nothing downstream would catch a
# raw string, and it would land in the LIVE settings verbatim.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("raw", ["abc", "-5", "0", "99999999999"])
def test_switch_context_window_rejects_garbage(tmp_path, capsys, monkeypatch, raw):
    """A malformed --context-window is a clean domain error (exit 1) and the
    live env keeps the PREVIOUS window — never the raw text or nonsense."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")  # CI has no ambient token
    assert (
        main(
            [
                "switch",
                "--provider",
                "zai",
                "--model",
                "glm-5.2",
                "--context-window",
                "500000",
                "--force",
            ]
        )
        == 0
    )  # sanity: a valid value applies

    assert (
        main(
            [
                "switch",
                "--provider",
                "zai",
                "--model",
                "glm-5.2",
                "--context-window",
                raw,
                "--force",
            ]
        )
        == 1
    )
    assert "--context-window" in capsys.readouterr().err

    env = _settings(tmp_path)["env"]
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "500000"


@pytest.mark.integration
def test_switch_context_window_none_suppresses_the_catalog(tmp_path, monkeypatch):
    """`--context-window none` is the scripted explicit suppression: even a
    catalog-known model gets no declaration."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")  # CI has no ambient token
    assert (
        main(
            [
                "switch",
                "--provider",
                "zai",
                "--model",
                "glm-5.3",
                "--context-window",
                "none",
                "--force",
            ]
        )
        == 0
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in _settings(tmp_path)["env"]


@pytest.mark.integration
def test_switch_split_tier_question_names_and_records_the_unknown_model(
    tmp_path, monkeypatch
):
    """Split tiers with one unknown model: the menu asks about THAT model and
    the answer records under it — not under the selected --model name, or
    the question would re-fire and mis-key the answer (round 1, PR #84)."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    import codehelper.cli.menu as menu

    seen: list[str] = []

    def _select(_items, *, prompt="", **_kwargs):
        seen.append(str(prompt))
        return "2000000"

    monkeypatch.setattr(menu, "select_from_menu", _select)

    assert (
        main(
            [
                "switch",
                "--provider",
                "zai",
                "--haiku",
                "mystery-haiku",
                "--sonnet",
                "glm-5.3",
                "--opus",
                "glm-5.3",
                "--force",
            ]
        )
        == 0
    )
    assert seen == ["Context window for mystery-haiku:"]

    from codehelper.services.state import context_window

    paths = Paths.from_home(tmp_path)
    # The answer records for the asked model and is NOT lost — but it does
    # NOT declare a session-wide window: the catalog's 1M on the other tiers
    # disagrees, and one variable may not claim a mix (review round 2).
    assert context_window(paths, "mystery-haiku") == 2_000_000
    assert context_window(paths, "glm-5.3") is None  # catalog tier: untouched
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in _settings(tmp_path)["env"]
