"""Tests for ``codehelper add`` in both modes, and ``list agents|providers|matrix``.

Everything runs through ``main([...])`` so the argparse wiring is exercised too,
not just the handler.
"""

from __future__ import annotations

import stat

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths
from codehelper.services.secrets import save_credential


def _body(tmp_path, name: str) -> str:
    return Paths.from_home(tmp_path).script_for(name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Constructor mode
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_agent_provider_model_creates_wrapper(tmp_path):
    """codex × ollama resolves to OPENAI_TOML by priority now: the wrapper runs
    ``codex --profile <alias>``, and the TOML profile is written alongside it
    — no model catalog (an empty synthesized ``base_instructions`` would
    silently replace Codex's real system prompt; see
    ``render.openai_toml_body``).
    """
    code = main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"  # alias derived from model + agent
    assert f"exec codex --profile '{alias}' \"$@\"" in _body(tmp_path, alias)
    config = paths.codex_config_for(alias).read_text(encoding="utf-8")
    assert config.startswith("# codehelper: managed wrapper")
    assert 'model = "glm-5:cloud"' in config
    assert 'base_url = "http://127.0.0.1:11434/v1/"' in config
    assert 'wire_api = "responses"' in config
    assert not paths.codex_catalog_for(alias).exists()


@pytest.mark.integration
def test_add_codex_ollama_with_explicit_launcher_shape(tmp_path):
    """The launcher path is still reachable via --shape ollama-launch."""
    code = main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
            "--shape",
            "ollama-launch",
        ]
    )
    assert code == 0
    assert "exec ollama launch codex --model 'glm-5:cloud'" in _body(
        tmp_path, "glm-5-codex"
    )
    # The launcher shape writes ONLY the wrapper — no TOML/catalog siblings.
    paths = Paths.from_home(tmp_path)
    assert not paths.codex_config_for("glm-5-codex").exists()


@pytest.mark.integration
def test_add_explicit_alias_wins(tmp_path):
    main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
            "--alias",
            "mycodex",
        ]
    )
    assert Paths.from_home(tmp_path).script_for("mycodex").exists()


@pytest.mark.integration
def test_add_agent_requires_model(capsys):
    assert main(["add", "--agent", "codex", "--provider", "ollama-direct"]) == 1
    assert "--model is required" in capsys.readouterr().err


@pytest.mark.integration
def test_add_agent_and_provider_must_come_together(capsys):
    assert main(["add", "--agent", "codex", "--model", "m"]) == 1
    assert "must be given together" in capsys.readouterr().err


@pytest.mark.integration
def test_preset_and_axes_are_mutually_exclusive(capsys):
    assert main(["add", "glm", "--agent", "codex", "--provider", "ollama-direct"]) == 1
    assert "not both" in capsys.readouterr().err


@pytest.mark.integration
def test_unknown_agent_exits_1(capsys):
    assert (
        main(["add", "--agent", "nope", "--provider", "ollama-direct", "--model", "m"])
        == 1
    )
    assert "unknown agent" in capsys.readouterr().err


@pytest.mark.integration
def test_incompatible_pairing_is_refused(capsys):
    """codex cannot speak the Anthropic protocol z.ai serves."""
    assert (
        main(["add", "--agent", "codex", "--provider", "zai", "--model", "glm-5-turbo"])
        == 1
    )
    assert "no common configuration" in capsys.readouterr().err


@pytest.mark.integration
def test_launch_only_agent_incompatible_with_zai(capsys):
    """A launch-only agent has no shape in common with a non-ollama provider,
    and the error's hint names `ollama` as what it DOES work with."""
    assert (
        main(["add", "--agent", "opencode", "--provider", "zai", "--model", "x"]) == 1
    )
    err = capsys.readouterr().err
    assert "no common configuration" in err
    assert "ollama-direct" in err


@pytest.mark.integration
def test_incompatible_pairing_never_prompts_for_a_token(monkeypatch):
    """Validation must happen BEFORE any interactive secret prompt."""
    import codehelper.services.secrets as secrets

    def _explode(*_a, **_kw):  # pragma: no cover - must not run
        raise AssertionError("must not prompt for a token on an invalid combination")

    monkeypatch.setattr(secrets, "resolve_token", _explode)
    assert (
        main(["add", "--agent", "codex", "--provider", "zai", "--model", "glm-5-turbo"])
        == 1
    )


# --------------------------------------------------------------------------- #
# --base-url — litellm (runtime base_url provider)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_litellm_claude_requires_base_url(capsys):
    code = main(
        ["add", "--agent", "claude", "--provider", "litellm", "--model", "gpt-4o"]
    )
    assert code == 1
    assert "needs a base URL" in capsys.readouterr().err


@pytest.mark.integration
def test_add_litellm_claude_with_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    body = _body(tmp_path, "gpt-4o-claude")
    # anthropic_base_url strips the /v1 suffix: Claude Code appends
    # /v1/messages itself, so a stored /v1 would double it to /v1/v1/messages
    # and every request would 404 — this is the bug a live user hit.
    assert "export ANTHROPIC_BASE_URL='http://localhost:4000'" in body
    assert "/v1" not in body
    assert "export ANTHROPIC_AUTH_TOKEN='sk-test'" in body


# --------------------------------------------------------------------------- #
# presets that carry a base_url (issue #86 mechanism): a preset for a
# REQUIRED-policy provider ships its instance's address and --base-url
# retargets it. No shipped preset carries one since the gemini story was
# removed (issue #112), so these run against a virtual preset registered
# only for the test — the mechanism is generic, the coverage stays.
# --------------------------------------------------------------------------- #


def _register_virtual_preset(monkeypatch):
    """Register a ``relay`` preset: claude × litellm with a carried URL."""
    import codehelper.services.spec as spec_module
    from codehelper.services.model import ConfigShape
    from codehelper.services.spec import Preset, TierModels

    virtual = Preset(
        alias="relay",
        agent="claude",
        provider="litellm",
        shape=ConfigShape.ANTHROPIC_ENV,
        model="relay-model",
        tier_models=TierModels.uniform("relay-model"),
        base_url="https://relay.example.com",
        description="Claude Code → relay-model via the relay proxy",
    )
    monkeypatch.setattr(spec_module, "PRESETS", (*spec_module.PRESETS, virtual))


@pytest.mark.integration
def test_add_virtual_preset_uses_the_carried_url(tmp_path, monkeypatch):
    _register_virtual_preset(monkeypatch)
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    assert main(["add", "relay", "--context-window", "none"]) == 0
    body = _body(tmp_path, "relay")
    assert "export ANTHROPIC_BASE_URL='https://relay.example.com'" in body
    assert "relay-model" in body


@pytest.mark.integration
def test_add_bai_uses_the_registry_url_and_preset_model(tmp_path, monkeypatch):
    """The bai preset carries no URL (the provider's address is FIXED registry
    data): the rendered wrapper points at the documented host verbatim."""
    monkeypatch.setenv("BAI_API_KEY", "sk-test")
    assert main(["add", "bai", "--context-window", "none"]) == 0
    body = _body(tmp_path, "bai")
    assert "export ANTHROPIC_BASE_URL='https://api.b.ai'" in body
    assert "qwen3.8-flash" in body
    # The env-sourced token lands embedded (the ANTHROPIC_ENV renderer resolves
    # it at build time) — the BAI_API_KEY name itself only ships in codex
    # profiles, where the TOML env_key needs the variable.
    assert "export ANTHROPIC_AUTH_TOKEN='sk-test'" in body


# --------------------------------------------------------------------------- #
# agy-native — the Antigravity preset (issue #112): native OAuth, no token
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_agy_native_installs_without_any_token_step(tmp_path, monkeypatch):
    """`add agy-native` must reach a completed install with NO credential:
    the native backend is OAuth — if the token chain were ever consulted it
    would hang on a hidden prompt (asserted by resolve_token exploding)."""
    import codehelper.services.secrets as secrets

    def _explode(*_a, **_kw):  # pragma: no cover - must not run
        raise AssertionError("native OAuth wrapper must not resolve a token")

    monkeypatch.setattr(secrets, "resolve_token", _explode)
    assert main(["add", "agy-native", "--context-window", "none"]) == 0
    body = _body(tmp_path, "agy-native")
    assert "exec agy --model 'gemini-3.8-flash-medium' \"$@\"" in body
    # No env block, no settings payload — the honest bare dispatch.
    assert "export " not in body
    assert "ANTHROPIC" not in body


@pytest.mark.integration
def test_add_agy_constructor_form(tmp_path):
    code = main(
        [
            "add",
            "--agent",
            "agy",
            "--provider",
            "antigravity",
            "--model",
            "gemini-3.1-pro-high",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    assert "exec agy --model 'gemini-3.1-pro-high' \"$@\"" in _body(
        tmp_path, "gemini-3.1-pro-high-agy"
    )


@pytest.mark.integration
def test_add_agy_with_a_foreign_provider_refuses_honestly(capsys):
    """agy has no documented OpenAI/Anthropic surface — every non-native
    pairing must refuse with the normal incompatibility error (whose hint
    names antigravity), never install a wrapper that cannot work."""
    code = main(["add", "--agent", "agy", "--provider", "zai", "--model", "m"])
    assert code == 1
    err = capsys.readouterr().err
    assert "no common configuration" in err
    assert "antigravity" in err


@pytest.mark.integration
def test_add_bare_agy_gets_the_agent_teaching_message(capsys):
    """`add agy` names an AGENT, not the preset (the preset alias is
    "agy-native" — a wrapper named after an agent's own binary is reserved).
    The fallback must teach the constructor form, not say "unknown"."""
    code = main(["add", "agy", "--context-window", "none"])
    assert code == 1
    err = capsys.readouterr().err
    assert "agy is an agent" in err
    assert "--agent agy" in err


@pytest.mark.integration
def test_list_matrix_agy_row(capsys):
    """agy's row: agent-native under antigravity, a gap everywhere else."""
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header_cols = lines[0].split()
    antigravity_idx = header_cols.index("antigravity")
    zai_idx = header_cols.index("zai")
    row = next(line for line in lines if line.startswith("agy")).split()
    # +1: each data row's first word is the agent name, not a provider column.
    assert row[antigravity_idx + 1] == "agent-native"
    assert row[zai_idx + 1] == "—"


@pytest.mark.integration
def test_add_preset_base_url_never_injects_the_active_profile(tmp_path, monkeypatch):
    """The implicit active-profile injection is for the provider's OWN default
    endpoint only — the exact rule the constructor path already enforces
    (see the guard at _handle_add). A preset retargeted with --base-url
    points at a host the stored profile was never authorized for: it must
    PROMPT, never silently embed the cached token of another host."""
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    _register_virtual_preset(monkeypatch)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-cached-for-host-a", "work")
    set_active_selection(paths, "litellm", "work")

    # resolve_token's getpass_fn default is bound at def time, so the prompt
    # is injected through the same seam the host-B test below uses.
    real_resolve_token = secrets.resolve_token

    def _resolve_with_fresh_prompt(**kwargs):
        return real_resolve_token(
            **kwargs, getpass_fn=lambda _p: "sk-typed-fresh-for-host-b"
        )

    monkeypatch.setattr(secrets, "resolve_token", _resolve_with_fresh_prompt)

    assert (
        main(
            [
                "add",
                "relay",
                "--base-url",
                "https://host-b.example",
                "--context-window",
                "none",
            ]
        )
        == 0
    )
    body = _body(tmp_path, "relay")
    # The freshly typed token is what got installed — never the host-A cache.
    assert "export ANTHROPIC_AUTH_TOKEN='sk-typed-fresh-for-host-b'" in body
    assert "sk-cached-for-host-a" not in body


@pytest.mark.integration
def test_add_preset_base_url_gate_accepts_an_overridable_provider(
    tmp_path, monkeypatch
):
    """The preset --base-url gate must be exactly as wide as the spec layer:
    spec_from_preset/with_base_url honour any provider whose address the
    caller may set (REQUIRED or OVERRIDABLE), so a hypothetical OVERRIDABLE
    preset provider must not be refused with the constructor-form message.
    Simulated by re-registering the virtual relay preset's provider as
    OVERRIDABLE — no shipped provider uses that policy today, which is why
    the real registry is untouched."""
    from dataclasses import replace

    import codehelper.cli.parser as parser_module
    from codehelper.services.model import BaseUrlPolicy, get_provider

    _register_virtual_preset(monkeypatch)
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setattr(
        parser_module,
        "get_provider",
        lambda name: replace(
            get_provider(name), base_url_policy=BaseUrlPolicy.OVERRIDABLE
        ),
    )
    assert (
        main(
            [
                "add",
                "relay",
                "--base-url",
                "https://overridable.example.com",
                "--context-window",
                "none",
            ]
        )
        == 0
    )
    assert "export ANTHROPIC_BASE_URL='https://overridable.example.com'" in _body(
        tmp_path, "relay"
    )


@pytest.mark.integration
def test_add_preset_with_fixed_provider_still_rejects_base_url(capsys):
    """The one-preset-override exception (issue #86) must not generalize:
    a preset whose provider carries its own registry address still rejects
    --base-url with the teaching message."""
    assert main(["add", "glm", "--base-url", "http://x"]) == 1
    assert "applies to the constructor form only" in capsys.readouterr().err


@pytest.mark.integration
def test_add_base_url_with_an_unknown_preset_name_fails_clean(capsys):
    """The override exception's lookup is defensive: a name that is not a
    preset at all must fail with a domain error (either the flag message or
    the unknown-wrapper hint) — never a crash, never a silent accept."""
    code = main(["add", "nope", "--base-url", "http://x"])
    assert code == 1
    err = capsys.readouterr().err
    assert "applies to the constructor form only" in err or "unknown wrapper" in err


@pytest.mark.integration
def test_add_warns_when_env_token_differs_from_cached(tmp_path, monkeypatch, capsys):
    """Issue #71 on the add path: the env token still wins and lands in the
    wrapper (documented precedence), but the disagreement is named on stderr
    with both sides redacted."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-env-stale-token")
    save_credential(Paths.from_home(tmp_path), "zai", "sk-cached-working")

    assert main(["add", "glm", "--force"]) == 0

    err = capsys.readouterr().err
    assert "ZAI_API_KEY" in err
    assert "sk-env-stale-token" not in err
    assert "sk-cached-working" not in err
    body = _body(tmp_path, "glm")
    assert "sk-env-stale-token" in body  # env value is what got installed


@pytest.mark.integration
def test_add_litellm_codex_writes_env_key_and_export(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    code = main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
            "--alias",
            "lm-codex",
        ]
    )
    assert code == 0
    body = _body(tmp_path, "lm-codex")
    assert "export LITELLM_API_KEY='sk-test'" in body
    assert body.index("export LITELLM_API_KEY") < body.index("exec codex")
    paths = Paths.from_home(tmp_path)
    config = paths.codex_config_for("lm-codex").read_text(encoding="utf-8")
    assert 'env_key = "LITELLM_API_KEY"' in config
    assert 'base_url = "http://localhost:4000/v1/"' in config


@pytest.mark.integration
def test_add_base_url_on_a_fixed_provider_is_refused(capsys):
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "ollama-direct",
            "--base-url",
            "http://x/v1",
            "--model",
            "m",
            "--context-window",
            "none",
        ]
    )
    assert code == 1
    assert "fixed base URL" in capsys.readouterr().err


@pytest.mark.integration
def test_add_base_url_with_a_preset_is_refused(capsys):
    code = main(["add", "deepseek-ollama", "--base-url", "http://x/v1"])
    assert code == 1
    assert "constructor form only" in capsys.readouterr().err


@pytest.mark.integration
def test_add_rejects_an_invalid_base_url_scheme(capsys):
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "ftp://x",
            "--model",
            "m",
            "--context-window",
            "none",
        ]
    )
    assert code == 1
    assert "http:// or https://" in capsys.readouterr().err


@pytest.mark.integration
def test_add_rejects_an_empty_base_url(capsys):
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "",
            "--model",
            "m",
            "--context-window",
            "none",
        ]
    )
    assert code == 1
    assert "needs a base URL" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --auth (AuthPolicy.OVERRIDABLE runtime override)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_ollama_with_auth_secret_writes_a_secret_mode_wrapper(
    tmp_path, monkeypatch
):
    """--auth secret overrides ollama's default literal token to a real secret
    — the wrapper carries the token, not the registry's "ollama" literal, and
    is written 0o700 like any other secret-backed wrapper."""
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-ollama-proxy")
    code = main(
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
    assert code == 0
    body = _body(tmp_path, "ollama-secure")
    assert "export ANTHROPIC_AUTH_TOKEN='sk-ollama-proxy'" in body
    mode = stat.S_IMODE(
        Paths.from_home(tmp_path).script_for("ollama-secure").stat().st_mode
    )
    assert mode == 0o700


@pytest.mark.integration
def test_add_ollama_without_auth_flag_keeps_the_literal_default(tmp_path):
    """The default (no --auth) is completely unchanged: ollama's literal
    "ollama" token, no prompt, 0o755."""
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
            "--alias",
            "ollama-default",
        ]
    )
    assert code == 0
    body = _body(tmp_path, "ollama-default")
    assert "export ANTHROPIC_AUTH_TOKEN='ollama'" in body
    mode = stat.S_IMODE(
        Paths.from_home(tmp_path).script_for("ollama-default").stat().st_mode
    )
    assert mode == 0o755


@pytest.mark.integration
def test_add_auth_secret_on_an_already_secret_provider_is_a_no_op(monkeypatch):
    """zai is already auth='secret' and FIXED — --auth secret must not refuse
    there, since with_auth treats "already secret" as nothing to override
    (see test_fixed_auth_provider_already_secret_ignores_override_request in
    test_model.py). The FIXED+not-yet-secret refusal itself has no shipped
    provider to exercise it against and is pinned at the model layer only,
    via an out-of-registry provider
    (test_fixed_auth_provider_refuses_an_override_when_not_already_secret)."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-test")
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "zai",
            "--auth",
            "secret",
            "--model",
            "glm-5",
            "--context-window",
            "none",
        ]
    )
    assert code == 0


@pytest.mark.integration
def test_add_auth_applies_to_the_constructor_form_only(capsys):
    code = main(["add", "deepseek-ollama", "--auth", "secret"])
    assert code == 1
    assert "constructor form only" in capsys.readouterr().err


@pytest.mark.integration
def test_add_auth_rejects_an_unknown_value(capsys):
    # argparse's own `choices` gate rejects this before codehelper ever sees
    # it — SystemExit(2), the same as any other invalid-choice flag, not a
    # CodeHelperError main() would turn into exit code 1.
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--auth",
                "bogus",
                "--model",
                "m",
                "--context-window",
                "none",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.integration
def test_bad_base_url_never_prompts_for_a_token(monkeypatch):
    """validate-before-prompt, pinned for the NEW validation channel."""
    import codehelper.services.secrets as secrets

    def _explode(*_a, **_kw):  # pragma: no cover - must not run
        raise AssertionError("must not prompt for a token on an invalid base URL")

    monkeypatch.setattr(secrets, "resolve_token", _explode)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "ftp://x",
            "--model",
            "m",
            "--context-window",
            "none",
        ]
    )
    assert code == 1


@pytest.mark.integration
def test_list_models_uses_the_runtime_base_url(monkeypatch):
    """The substitution happens BEFORE --list-models, not just before build_spec.

    Also pins that the discovery token reaches ``list_models`` (from env here)
    — the bug #15 point 1: discovery never sent a token at all."""
    import codehelper.services.models_api as models_api

    seen = {}

    def _fake_list_models(provider, *, token=""):
        seen["base_url"] = provider.base_url
        seen["token"] = token
        return models_api.ModelListResult(models=("m1",), source="fake")

    monkeypatch.setattr(models_api, "list_models", _fake_list_models)
    monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://h:4000/v1",
            "--list-models",
        ]
    )
    assert code == 0
    assert seen["base_url"] == "http://h:4000/v1"
    assert seen["token"] == "sk-from-env"


@pytest.mark.integration
def test_bad_alias_is_rejected(capsys):
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "m",
                "--context-window",
                "none",
                "--alias",
                "../../etc/passwd",
            ]
        )
        == 1
    )
    assert "path separator" in capsys.readouterr().err


@pytest.mark.integration
def test_alias_may_not_shadow_an_agent_binary(capsys):
    """`--alias claude` would exec itself forever and clobber a real symlink."""
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "m",
                "--context-window",
                "none",
                "--alias",
                "claude",
            ]
        )
        == 1
    )
    assert "reserved name" in capsys.readouterr().err


@pytest.mark.integration
def test_dry_run_in_constructor_mode_writes_nothing(tmp_path):
    main(
        [
            "--dry-run",
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
        ]
    )
    assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_two_agents_on_one_model_coexist(tmp_path):
    """The point of the feature: same model, different agents, no collision.

    The codex wrapper forces ``--shape ollama-launch`` so the assertion stays
    about coexistence (a shape-agnostic property) rather than which shape the
    default priority picked — that is covered by
    ``test_add_agent_provider_model_creates_wrapper``.
    """
    main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
            "--shape",
            "ollama-launch",
        ]
    )
    main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "ollama-direct",
            "--model",
            "glm-5:cloud",
            "--context-window",
            "none",
            "--shape",
            "ollama-launch",
        ]
    )
    assert "launch codex" in _body(tmp_path, "glm-5-codex")
    assert "launch claude" in _body(tmp_path, "glm-5-claude")


# --------------------------------------------------------------------------- #
# Preset mode — unchanged behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_preset_still_installs_by_bare_name(tmp_path):
    assert main(["add", "glm-ollama"]) == 0
    assert "exec ollama launch claude --model 'glm-5.2:cloud'" in _body(
        tmp_path, "glm-ollama"
    )


@pytest.mark.integration
def test_bare_agent_name_suggests_the_constructor(capsys):
    """`add codex` is a likely mistake — teach the right form, don't just refuse."""
    assert main(["add", "codex"]) == 1
    err = capsys.readouterr().err
    assert "is an agent" in err
    assert "--agent codex" in err


@pytest.mark.integration
def test_unknown_name_that_is_not_an_agent(capsys):
    assert main(["add", "nope"]) == 1
    assert "unknown wrapper name" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Collision guard through the CLI
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_foreign_file_refused_without_force(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    script = paths.script_for("glm-ollama")
    script.write_text("#!/bin/bash\necho mine\n", encoding="utf-8")

    assert main(["add", "glm-ollama"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert script.read_text(encoding="utf-8") == "#!/bin/bash\necho mine\n"


@pytest.mark.integration
def test_force_overwrites_foreign_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    paths.script_for("glm-ollama").write_text("#!/bin/bash\necho mine\n")

    assert main(["add", "glm-ollama", "--force"]) == 0
    assert "ollama launch claude" in _body(tmp_path, "glm-ollama")


# --------------------------------------------------------------------------- #
# list agents / providers / matrix
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_list_agents(capsys):
    assert main(["list", "agents"]) == 0
    out = capsys.readouterr().out
    assert "claude" in out and "codex" in out


@pytest.mark.integration
def test_list_providers(capsys):
    assert main(["list", "providers"]) == 0
    out = capsys.readouterr().out
    assert "ollama-direct" in out and "zai" in out and "litellm" in out
    assert "deepseek" in out and "deepseek-openai" in out


@pytest.mark.integration
def test_list_matrix_shows_a_gap_for_incompatible_pairs(capsys):
    """The matrix is rendered via resolve_shape, so it cannot lie about `add`."""
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    codex_row = next(line for line in lines if line.startswith("codex"))
    assert "—" in codex_row  # codex × zai is impossible
    # codex × ollama now resolves to the TOML profile by priority.
    assert "openai-toml" in codex_row


@pytest.mark.integration
def test_list_matrix_includes_litellm_with_no_gap(capsys):
    """litellm declares both shapes, so neither agent's row has a gap for it."""
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header_cols = lines[0].split()
    litellm_idx = header_cols.index("litellm")
    claude_row = next(line for line in lines if line.startswith("claude")).split()
    codex_row = next(line for line in lines if line.startswith("codex")).split()
    # +1: each data row's first word is the agent name, not a provider column.
    assert claude_row[litellm_idx + 1] == "anthropic-env"
    assert codex_row[litellm_idx + 1] == "openai-toml"


@pytest.mark.integration
def test_list_agents_includes_every_ollama_launch_integration(capsys):
    """`list agents` shows all 15 registered agents, not just claude/codex."""
    assert main(["list", "agents"]) == 0
    out = capsys.readouterr().out
    for name in ("opencode", "copilot", "droid", "cline", "qwen"):
        assert name in out


@pytest.mark.integration
def test_list_matrix_launch_only_agent_row(capsys):
    """A launch-only agent's row: ollama-launch for ollama, a gap elsewhere.

    `resolve_shape` renders this row, so it cannot disagree with what `add`
    actually accepts.
    """
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header_cols = lines[0].split()
    ollama_idx = header_cols.index("ollama-direct")
    zai_idx = header_cols.index("zai")
    row = next(line for line in lines if line.startswith("opencode")).split()
    # +1: each data row's first word is the agent name, not a provider column.
    assert row[ollama_idx + 1] == "ollama-launch"
    assert row[zai_idx + 1] == "—"


@pytest.mark.integration
def test_list_shows_ad_hoc_wrappers(capsys):
    main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "ollama-direct",
            "--model",
            "qwen3.5:9b",
            "--context-window",
            "none",
        ]
    )
    capsys.readouterr()

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ad-hoc wrappers" in out
    assert "qwen3.5-codex" in out


@pytest.mark.integration
def test_list_models_prints_and_writes_nothing(tmp_path, monkeypatch):
    import codehelper.services.models_api as api
    from codehelper.services.models_api import ModelListResult

    monkeypatch.setattr(
        api, "list_models", lambda _p, **_kw: ModelListResult(("a:1", "b:2"), "url")
    )
    assert (
        main(
            ["add", "--agent", "codex", "--provider", "ollama-direct", "--list-models"]
        )
        == 0
    )
    assert not Paths.from_home(tmp_path).bin_dir.exists()


# --------------------------------------------------------------------------- #
# Credential cache (issue #15) — a token typed at a prompt is cached, so the
# next add/--list-models does not ask again. Never under --dry-run.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_caches_a_prompt_typed_token(tmp_path, monkeypatch):
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    assert secrets.credential_for(paths, "litellm") == "sk-typed"


@pytest.mark.integration
def test_add_token_stdin_installs_and_caches(tmp_path, monkeypatch):
    """Issue #92: a piped token rides the pre-typed path — installed AND
    cached, with the hidden prompt never reached."""
    import io
    import sys

    import codehelper.services.secrets as secrets

    monkeypatch.delenv("BAI_API_KEY", raising=False)

    def _explode(_prompt: str) -> str:
        raise AssertionError("must not prompt when --token-stdin supplies the token")

    monkeypatch.setattr("getpass.getpass", _explode)
    monkeypatch.setattr(sys, "stdin", io.StringIO("sk-stdin\n"))

    assert main(["add", "bai", "--context-window", "none", "--token-stdin"]) == 0

    assert "sk-stdin" in _body(tmp_path, "bai")
    assert secrets.credential_for(Paths.from_home(tmp_path), "bai") == "sk-stdin"


@pytest.mark.integration
def test_add_token_stdin_refuses_a_tty(monkeypatch, capsys):
    """On a terminal the flag must fail fast, never echo-read: the interactive
    path is the hidden prompt (the _confirm-family isatty precedent)."""
    import sys

    class _Tty:
        def isatty(self):
            return True

        def readline(self):
            raise AssertionError("stdin must not be read on a TTY")

    monkeypatch.setattr(sys, "stdin", _Tty())
    code = main(["add", "bai", "--context-window", "none", "--token-stdin"])
    assert code == 1
    assert "--token-stdin" in capsys.readouterr().err


@pytest.mark.integration
def test_add_token_stdin_empty_stdin_rejected(monkeypatch):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    code = main(["add", "bai", "--context-window", "none", "--token-stdin"])
    assert code == 1


@pytest.mark.integration
def test_add_token_stdin_dry_run_writes_nothing(tmp_path, monkeypatch):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("sk-stdin\n"))
    code = main(
        ["--dry-run", "add", "bai", "--context-window", "none", "--token-stdin"]
    )
    assert code == 0
    assert not Paths.from_home(tmp_path).credentials_file().exists()


@pytest.mark.integration
def test_add_uses_the_selected_profile_even_when_env_has_another_token(
    tmp_path, monkeypatch
):
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", "sk-work", "work")
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")

    assert main(["add", "glm", "--profile", "work"]) == 0
    assert "sk-work" in _body(tmp_path, "glm")
    assert "sk-env" not in _body(tmp_path, "glm")


@pytest.mark.integration
def test_add_profile_caches_a_prompt_typed_token(tmp_path, monkeypatch):
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-personal", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    assert main(["add", "glm", "--profile", "personal"]) == 0

    assert (
        secrets.credential_for(Paths.from_home(tmp_path), "zai", "personal")
        == "sk-personal"
    )


@pytest.mark.integration
def test_profile_selects_the_default_alias_and_is_visible_in_list(tmp_path, capsys):
    """A profile is durable wrapper metadata, not an invisible install input."""
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", "sk-axisrow", "axisrow")

    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "zai",
                "--model",
                "glm-5",
                "--context-window",
                "none",
                "--profile",
                "axisrow",
            ]
        )
        == 0
    )

    wrapper = paths.script_for("glm-5-axisrow-claude")
    assert wrapper.exists()
    assert "profile=axisrow" in wrapper.read_text()
    assert main(["list"]) == 0
    assert "[profile: axisrow]" in capsys.readouterr().out


def test_suggest_alias_includes_the_agent_when_a_profile_is_set():
    """Two agents sharing a model+profile must not suggest the same alias.

    Before this fix, ``suggest_alias`` derived a profiled alias from
    model+profile only, dropping ``agent_name`` entirely — installing
    ``glm-5`` for both ``codex`` and ``claude`` with the same profile
    silently suggested the identical wrapper name for both.
    """
    from codehelper.services.spec import suggest_alias

    codex_alias = suggest_alias("glm-5:cloud", "codex", "axisrow")
    claude_alias = suggest_alias("glm-5:cloud", "claude", "axisrow")

    assert codex_alias != claude_alias
    assert codex_alias == "glm-5-axisrow-codex"
    assert claude_alias == "glm-5-axisrow-claude"


@pytest.mark.integration
def test_add_does_not_cache_an_env_resolved_token(tmp_path, monkeypatch):
    """The env value already outlives this process — caching it would just be
    a second, redundant copy, and the priority test in test_secrets.py already
    proves env beats the cache anyway."""

    monkeypatch.setenv("LITELLM_API_KEY", "sk-env")
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    assert not paths.credentials_file().exists()


@pytest.mark.integration
def test_add_env_sourced_install_invalidates_a_stale_cached_token(
    tmp_path, monkeypatch
):
    """An env-sourced install must not leave a now-stale cached token behind
    for a LATER, env-free run to silently resurrect.

    Sequence: (1) a prompt-typed token "old" gets cached for a FIXED
    provider; (2) a later ``add`` with the env var set to "new" installs
    successfully — env-sourced values are still never CACHED (an env value
    already outlives the process), but the stale "old" cache entry must not
    be left standing either, or (3) a still-later ``add`` run WITHOUT the env
    var would resolve "old" from the cache and silently reinstall over the
    wrapper that currently has "new" — reverting a rotated/revoked
    credential with no confirmation and no warning.
    """
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)

    # Step 1: cache "old" via a prompt-typed install.
    real_resolve_token = secrets.resolve_token

    def _fake_resolve_token_old(**_kwargs):
        return secrets.ResolvedToken("sk-old", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token_old)
    assert main(["add", "glm"]) == 0
    assert secrets.credential_for(paths, "zai") == "sk-old"
    # Restore the real resolve_token for steps 2-3 — NOT monkeypatch.undo(),
    # which would also roll back conftest.py's HOME-isolation fixture and
    # let this test escape into the real home directory.
    monkeypatch.setattr(secrets, "resolve_token", real_resolve_token)

    # Step 2: install with a NEW token via the env var. install_wrapper
    # actually changes the wrapper (different token -> not byte-identical).
    monkeypatch.setenv("ZAI_API_KEY", "sk-new")
    assert main(["add", "glm"]) == 0
    assert "sk-new" in _body(tmp_path, "glm")
    # The stale cache entry must be gone (or updated) — never left at "old".
    assert secrets.credential_for(paths, "zai") != "sk-old"

    # Step 3: a later run without the env var must NOT silently resolve "old"
    # from a stale cache and revert the wrapper. The cache was invalidated by
    # step 2's fix, so resolve_token falls through to a fresh prompt here —
    # simulate the user typing a new value; the point being proven is that
    # "sk-old" is never resolved from a stale cache.
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    def _fake_resolve_token_new(**_kwargs):
        return secrets.ResolvedToken("sk-freshly-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token_new)
    assert main(["add", "glm"]) == 0
    assert "sk-old" not in _body(tmp_path, "glm")
    assert "sk-freshly-typed" in _body(tmp_path, "glm")


@pytest.mark.integration
def test_add_invalidates_stale_cache_even_on_a_byte_identical_env_reinstall(
    tmp_path, monkeypatch
):
    """The stale-cache invalidation must fire even when install_wrapper
    no-ops (``wrote=False``) because the wrapper ALREADY has the env token
    baked in — not just when the env-sourced install actually changes
    something. Gating invalidation on ``wrote`` (the bug this test pins)
    means: install once with env token "new" (writes, invalidates "old" —
    covered by test_add_env_sourced_install_invalidates_a_stale_cached_token
    above); if a DIFFERENT stale token then reappears in the cache (e.g. an
    unrelated edit-token run for the same provider) and ``add`` is run AGAIN
    with the same env token "new" already installed, install_wrapper no-ops
    (byte-identical), so ``wrote=False`` — and the gate must still catch the
    staleness, or a later env-free run resurrects the stale value.
    """
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)

    # Install once with the env token — wrote=True, wrapper now has "sk-new".
    monkeypatch.setenv("ZAI_API_KEY", "sk-new")
    assert main(["add", "glm"]) == 0
    assert "sk-new" in _body(tmp_path, "glm")
    assert secrets.credential_for(paths, "zai") == ""  # nothing cached yet

    # A stale, DIFFERENT token reappears in the cache (independent of this
    # install — e.g. left over from an earlier prompt-driven install of a
    # different wrapper sharing the same provider).
    secrets.save_credential(paths, "zai", "sk-stale")

    # Re-run `add` with the SAME env token: install_wrapper no-ops
    # (byte-identical -> wrote=False), but the stale cache entry must still
    # be invalidated.
    assert main(["add", "glm"]) == 0
    assert secrets.credential_for(paths, "zai") != "sk-stale"


@pytest.mark.integration
def test_add_does_not_cache_a_prompt_typed_token_when_the_install_is_refused(
    tmp_path, capsys, monkeypatch
):
    """A token typed for an install that never happened must not be cached.

    A prompt-resolved token is only worth persisting once the install it was
    typed for actually landed — caching it unconditionally, before
    ``install_wrapper`` even runs, leaves a stale/orphaned token in
    ``credentials.json`` when the install is refused (foreign-file guard, no
    ``--force``, no TTY to confirm). ``edit-token`` already gets this right
    (caches only after ``wrote`` is truthy); ``add`` must match it.
    """
    import codehelper.services.secrets as secrets

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    # gpt-4o + claude -> suggested alias "gpt-4o-claude" (see suggest_alias).
    paths.script_for("gpt-4o-claude").write_text(
        "#!/bin/bash\necho mine\n", encoding="utf-8"
    )

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)

    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert secrets.credential_for(paths, "litellm") == ""


@pytest.mark.integration
def test_add_caches_a_prompt_typed_token_on_a_byte_identical_noop_reinstall(
    tmp_path, monkeypatch
):
    """A prompt-typed token IS still cached when install_wrapper no-ops because
    the re-typed token happens to match what's already installed byte-for-byte
    (``wrote=False``, not a refusal) — the fix must not overcorrect into never
    caching when ``wrote`` is falsy for this reason instead of a refusal."""
    import codehelper.services.secrets as secrets

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)

    args = [
        "add",
        "--agent",
        "claude",
        "--provider",
        "litellm",
        "--base-url",
        "http://localhost:4000/v1",
        "--model",
        "gpt-4o",
        "--context-window",
        "none",
    ]

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    assert main(args) == 0  # first install, writes the wrapper
    # Clear the cache so the second run must go through resolve_token again
    # (the fake always reports SOURCE_PROMPT) while install_wrapper itself
    # sees byte-identical content and no-ops.
    paths.credentials_file().unlink()
    assert not paths.credentials_file().exists()

    assert main(args) == 0  # second install: same token -> install_wrapper no-ops
    assert secrets.credential_for(paths, "litellm") == "sk-typed"


@pytest.mark.integration
def test_add_dry_run_never_writes_the_credentials_file(tmp_path, monkeypatch):
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    code = main(
        [
            "--dry-run",
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    assert not paths.credentials_file().exists()


@pytest.mark.integration
def test_add_warns_when_bin_dir_is_not_on_path(monkeypatch, capsys):
    """A real (non-dry-run) install warns on stderr when ``paths.bin_dir``
    isn't on ``PATH`` — otherwise the freshly-installed alias just gives
    ``command not found`` with no clue why."""
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    monkeypatch.setenv("PATH", "/usr/bin")

    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "not on PATH" in err
    assert "gpt-4o-claude" in err


@pytest.mark.integration
def test_add_dry_run_does_not_print_a_path_warning(monkeypatch, capsys):
    """``install_wrapper`` returns True under ``dry_run`` too ("would write"),
    so the PATH check must not fire off that truthy-but-nothing-written
    signal — a dry run must not warn about a file it never created."""
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**_kwargs):
        return secrets.ResolvedToken("sk-typed", secrets.SOURCE_PROMPT)

    monkeypatch.setattr(secrets, "resolve_token", _fake_resolve_token)
    monkeypatch.setenv("PATH", "/usr/bin")  # deliberately excludes bin_dir

    code = main(
        [
            "--dry-run",
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    assert "not on PATH" not in capsys.readouterr().err


@pytest.mark.integration
def test_add_reuses_a_cached_token_without_prompting(tmp_path, monkeypatch):
    """A token cached by a previous add is picked up silently on the next one
    — for a FIXED-base_url_policy provider (zai), where the address never
    varies and the cache is safe to reuse across installs."""
    import codehelper.services.secrets as secrets

    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", "sk-cached")

    def _explode(_prompt):  # pragma: no cover - must not run
        raise AssertionError("must not prompt when the cache already has the token")

    monkeypatch.setattr("codehelper.services.secrets.getpass.getpass", _explode)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "zai",
            "--model",
            "glm-5",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    body = _body(tmp_path, "glm-5-claude")
    assert "export ANTHROPIC_AUTH_TOKEN='sk-cached'" in body


@pytest.mark.integration
def test_add_ignores_a_cached_token_for_a_runtime_address_provider_on_install(
    tmp_path, monkeypatch
):
    """A token cached for litellm under one --base-url must not be silently
    reused (and baked into a wrapper) for a DIFFERENT --base-url given for
    the same provider name — the install-path twin of
    ``test_list_models_ignores_a_cached_token_for_a_runtime_address_provider``
    (discovery path). Falls through to the prompt instead of the stale cache."""
    import codehelper.services.secrets as secrets

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-cached-for-host-a")

    real_resolve_token = secrets.resolve_token

    def _resolve_token_with_fake_prompt(**kwargs):
        return real_resolve_token(**kwargs, getpass_fn=lambda _p: "sk-typed-for-host-b")

    monkeypatch.setattr(secrets, "resolve_token", _resolve_token_with_fake_prompt)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://host-b:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
        ]
    )
    assert code == 0
    body = _body(tmp_path, "gpt-4o-claude")
    assert "export ANTHROPIC_AUTH_TOKEN='sk-typed-for-host-b'" in body
    assert "sk-cached-for-host-a" not in body


@pytest.mark.integration
def test_add_reuses_a_named_profile_across_two_different_base_urls(
    tmp_path, monkeypatch
):
    """Same ``--profile``, two different ``--base-url`` values, one cached
    token — deliberate, not an oversight (issue #19).

    An explicit ``--profile`` is an informed choice, so a NAMED profile's
    cache is trusted regardless of ``base_url_policy``; that is what makes
    profiles usable for a REQUIRED-policy provider like ``litellm`` at all.
    Contrast ``test_add_ignores_a_cached_token_for_a_runtime_address_provider_on_install``
    directly above: the UNNAMED cache is not reused across a changed address,
    because nothing selected it for that invocation. Adding the
    ``base_url_policy`` guard to ``resolve_token``'s named branch too makes
    named profiles unusable for ``litellm`` — a regression this project
    shipped in review and reverted by explicit decision.
    """
    import codehelper.services.secrets as secrets

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-profile-work", "work")

    def _explode(_prompt):  # pragma: no cover - must not run
        raise AssertionError("must not prompt when the named profile's cache has it")

    monkeypatch.setattr("codehelper.services.secrets.getpass.getpass", _explode)

    code_a = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://host-a:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
            "--alias",
            "lite-a",
            "--profile",
            "work",
        ]
    )
    code_b = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://host-b:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
            "--alias",
            "lite-b",
            "--profile",
            "work",
        ]
    )
    assert code_a == 0
    assert code_b == 0

    body_a = _body(tmp_path, "lite-a")
    body_b = _body(tmp_path, "lite-b")
    assert "export ANTHROPIC_AUTH_TOKEN='sk-profile-work'" in body_a
    assert "export ANTHROPIC_AUTH_TOKEN='sk-profile-work'" in body_b
    assert "http://host-a:4000" in body_a
    assert "http://host-b:4000" in body_b


@pytest.mark.integration
def test_list_models_ignores_a_cached_token_for_a_runtime_address_provider(
    tmp_path, monkeypatch
):
    """--list-models discovery must NOT hand a cached token to a provider whose
    ``base_url`` the caller can point anywhere (litellm is REQUIRED-policy) —
    see ``token_for_discovery``'s ``base_url_policy`` gate in secrets.py.
    Through the real CLI path (not a fake ``token_for_discovery``), proving the
    substitution actually reaches discovery for a REQUIRED provider."""
    import codehelper.services.models_api as models_api
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-cached")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    seen = {}

    def _fake_list_models(_provider, *, token=""):
        seen["token"] = token
        return models_api.ModelListResult(models=("m1",), source="fake")

    monkeypatch.setattr(models_api, "list_models", _fake_list_models)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--list-models",
        ]
    )
    assert code == 0
    assert seen["token"] == ""


@pytest.mark.integration
def test_list_models_does_not_use_the_implicit_active_profile_for_a_custom_base_url(
    tmp_path, monkeypatch
):
    """Issue #23's CLI parity must not silently hand the persisted active
    profile's cached token to a caller-supplied ``--base-url``. The implicit
    injection (when ``--profile`` is omitted) applies only to the provider's
    own default endpoint; with a custom ``--base-url`` the caller must
    authorize explicitly (``--profile`` or env), so the discovery token stays
    empty. Guards the trust-boundary hole the active-profile injection opened
    (parser.py)."""
    import codehelper.services.models_api as models_api
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

    paths = Paths.from_home(tmp_path)
    # A named profile is cached AND is the persisted active selection.
    secrets.save_credential(paths, "litellm", "sk-active", "work")
    set_active_selection(paths, "litellm", "work")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    seen = {}

    def _fake_list_models(_provider, *, token=""):
        seen["token"] = token
        return models_api.ModelListResult(models=("m1",), source="fake")

    monkeypatch.setattr(models_api, "list_models", _fake_list_models)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--list-models",
        ]
    )
    assert code == 0
    assert seen["token"] == ""


@pytest.mark.integration
def test_list_models_uses_a_named_profile_for_a_runtime_address_provider(
    tmp_path, monkeypatch
):
    """The named-profile counterpart to the test above: ``--list-models``
    discovery DOES hand a cached token to a REQUIRED-policy provider when the
    caller named the profile explicitly — mirroring ``resolve_token``'s same
    named/unnamed asymmetry on the discovery path (``token_for_discovery``,
    secrets.py). Deliberate, per issue #19."""
    import codehelper.services.models_api as models_api
    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-profile-work", "work")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    seen = {}

    def _fake_list_models(_provider, *, token=""):
        seen["token"] = token
        return models_api.ModelListResult(models=("m1",), source="fake")

    monkeypatch.setattr(models_api, "list_models", _fake_list_models)
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://localhost:4000/v1",
            "--profile",
            "work",
            "--list-models",
        ]
    )
    assert code == 0
    assert seen["token"] == "sk-profile-work"


@pytest.mark.integration
def test_edit_token_updates_the_cache(tmp_path, monkeypatch):
    """Rotation must overwrite a stale cached value, not just the installed
    script — otherwise the next add hands out the OLD token."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-old")
    assert main(["add", "glm"]) == 0
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    import codehelper.services.secrets as secrets

    # _handle_edit_token does `import getpass` locally and calls
    # getpass.getpass directly (never resolve_token) — patch that exact path.
    monkeypatch.setattr("getpass.getpass", lambda _p="": "sk-new")
    assert main(["edit-token", "glm"]) == 0
    paths = Paths.from_home(tmp_path)
    assert secrets.credential_for(paths, "zai") == "sk-new"


@pytest.mark.integration
def test_edit_token_caches_even_on_a_byte_identical_noop(tmp_path, monkeypatch):
    """Retyping the token that's already installed no-ops install_wrapper
    (byte-identical), but the cache must still be refreshed/created — matching
    ``add``'s explicit rule that ``wrote=False`` is not a refusal and the
    token that produced the no-op is exactly the one worth having cached."""
    monkeypatch.setenv("ZAI_API_KEY", "sk-same")
    assert main(["add", "glm"]) == 0
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    import codehelper.services.secrets as secrets

    paths = Paths.from_home(tmp_path)
    # The install above resolved its token from ZAI_API_KEY (not a prompt), so
    # nothing is cached yet — the wrapper on disk has "sk-same" baked in, but
    # credentials.json doesn't exist. edit-token must still create the cache
    # entry even though install_wrapper no-ops (same token, byte-identical).
    assert not paths.credentials_file().exists()

    monkeypatch.setattr("getpass.getpass", lambda _p="": "sk-same")
    assert main(["edit-token", "glm"]) == 0  # no-op: same token, byte-identical
    assert secrets.credential_for(paths, "zai") == "sk-same"


# --------------------------------------------------------------------------- #
# Issue #23: CLI parity — the active profile is picked up by `add`/shown by `list`
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_without_profile_picks_up_the_active_profile(tmp_path, monkeypatch):
    """CLI ``add`` without ``--profile`` falls back to the stored active profile
    when the provider's own endpoint is used (no caller-supplied ``--base-url``).

    An explicit ``--profile`` always wins (next test); and a custom
    ``--base-url`` never silently reuses the persisted pointer (see the
    custom-base-url test) — the implicit fallback is only for the provider's
    default endpoint, where the cached profile token is legitimate. Proved by
    patching getpass to explode: if the active profile's cached token were NOT
    reused, ``add`` would have to prompt and the test would fail.
    """
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "zai", "sk-active-work", "work")
    set_active_selection(paths, "zai", "work")

    def _explode(_prompt):  # pragma: no cover - must not run
        raise AssertionError("must not prompt when the active profile has a token")

    monkeypatch.setattr("codehelper.services.secrets.getpass.getpass", _explode)

    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "zai",
            "--model",
            "glm-5",
            "--context-window",
            "none",
            "--alias",
            "lite-active",
        ]
    )
    assert code == 0
    assert "export ANTHROPIC_AUTH_TOKEN='sk-active-work'" in _body(
        tmp_path, "lite-active"
    )


@pytest.mark.integration
def test_add_explicit_profile_wins_over_the_active_profile(tmp_path, monkeypatch):
    """An explicit ``--profile`` overrides the stored active profile (unconditional).

    The active profile is invisible state; the flag must always win over it.
    """
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-active-work", "work")
    secrets.save_credential(paths, "litellm", "sk-explicit-personal", "personal")
    set_active_selection(paths, "litellm", "work")

    def _explode(_prompt):  # pragma: no cover - must not run
        raise AssertionError("must not prompt when the explicit profile has a token")

    monkeypatch.setattr("codehelper.services.secrets.getpass.getpass", _explode)

    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "litellm",
            "--base-url",
            "http://host:4000/v1",
            "--model",
            "gpt-4o",
            "--context-window",
            "none",
            "--alias",
            "lite-explicit",
            "--profile",
            "personal",
        ]
    )
    assert code == 0
    # The explicit --profile personal won, not the active "work".
    assert "export ANTHROPIC_AUTH_TOKEN='sk-explicit-personal'" in _body(
        tmp_path, "lite-explicit"
    )


@pytest.mark.integration
def test_list_shows_the_active_profile_header(tmp_path, capsys):
    """``list`` surfaces the stored active profile as a one-line header.

    A user coming from the TUI can see, from the CLI, which profile is active
    without re-entering the menu.
    """
    import codehelper.services.secrets as secrets
    from codehelper.services.state import set_active_selection

    paths = Paths.from_home(tmp_path)
    secrets.save_credential(paths, "litellm", "sk-work", "work")
    set_active_selection(paths, "litellm", "work")

    assert main(["list"]) == 0
    assert "Active profile: litellm/work" in capsys.readouterr().out


@pytest.mark.integration
def test_list_omits_the_active_profile_header_when_unset(capsys):
    assert main(["list"]) == 0
    assert "Active profile:" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Issue #83: the context-window question — asked once for an unknown model,
# remembered per model, never for --dry-run, overridable by the flag.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_unknown_model_prompts_once_and_records(tmp_path):
    """The window menu fires for a model the catalog doesn't know; the answer
    is remembered, so the SECOND identical add never prompts again."""
    paths = Paths.from_home(tmp_path)
    answers = iter(["1000000"])

    def _select(_items, *, prompt="", **_kwargs):
        assert str(prompt).startswith("Context window for mystery-3b")
        return next(answers)

    monkey_target = "codehelper.cli.menu.select_from_menu"

    # First add: the menu answers 1M.
    import unittest.mock as mock

    with mock.patch(monkey_target, _select):
        assert (
            main(
                [
                    "add",
                    "--agent",
                    "claude",
                    "--provider",
                    "ollama-direct",
                    "--model",
                    "mystery-3b",
                    "--alias",
                    "mystery",
                    "--force",
                ]
            )
            == 0
        )
    from codehelper.services.state import context_window

    assert context_window(paths, "mystery-3b") == 1_000_000
    body = paths.script_for("mystery").read_text(encoding="utf-8")
    assert "ctx=1000000" in body
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS='1000000'" in body

    # Second add: no menu reachable — a recorded model is never re-asked.
    called = {"menu": False}

    def _fail(_items, **_kwargs):
        called["menu"] = True
        return "0"

    with mock.patch(monkey_target, _fail):
        assert (
            main(
                [
                    "add",
                    "--agent",
                    "claude",
                    "--provider",
                    "ollama-direct",
                    "--model",
                    "mystery-3b",
                    "--alias",
                    "mystery",
                    "--force",
                ]
            )
            == 0
        )
    assert called["menu"] is False


@pytest.mark.integration
def test_add_known_model_never_prompts_for_the_window():
    """A catalog-known model is derived silently — no menu is reachable."""
    import unittest.mock as mock

    def _fail(_items, **_kwargs):
        raise AssertionError("menu shown for a catalog-known model")

    with mock.patch("codehelper.cli.menu.select_from_menu", _fail):
        assert main(["add", "deepseek-ollama", "--force"]) == 0


@pytest.mark.integration
def test_add_dry_run_never_prompts_for_the_window(tmp_path, capsys):
    """--dry-run never prompts NOR writes state — the model stays unrecorded."""
    paths = Paths.from_home(tmp_path)

    def _fail(_items, **_kwargs):
        raise AssertionError("dry run showed the window menu")

    import unittest.mock as mock

    with mock.patch("codehelper.cli.menu.select_from_menu", _fail):
        assert (
            main(
                [
                    "--dry-run",
                    "add",
                    "--agent",
                    "claude",
                    "--provider",
                    "ollama-direct",
                    "--model",
                    "mystery-3b",
                    "--alias",
                    "mystery",
                ]
            )
            == 0
        )
    from codehelper.services.state import context_window

    assert context_window(paths, "mystery-3b") is None
    assert "would write" in capsys.readouterr().out.lower()


@pytest.mark.integration
def test_context_window_flag_parses_int_and_none(tmp_path, capsys):
    """The flag takes a token count; 'none' is the explicit suppression;
    garbage is a clean domain error (exit 1), not a traceback."""
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--model",
                "mystery-3b",
                "--alias",
                "mystery",
                "--context-window",
                "750000",
            ]
        )
        == 0
    )
    body = paths_body(tmp_path, "mystery")
    assert "ctx=750000" in body

    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--model",
                "mystery-3b",
                "--alias",
                "mystery2",
                "--context-window",
                "none",
            ]
        )
        == 0
    )
    assert "ctx=0" in paths_body(tmp_path, "mystery2")

    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--model",
                "mystery-3b",
                "--alias",
                "mystery3",
                "--context-window",
                "bogus",
            ]
        )
        == 1
    )
    assert "--context-window" in capsys.readouterr().err


def paths_body(tmp_path, name: str) -> str:
    return Paths.from_home(tmp_path).script_for(name).read_text(encoding="utf-8")


@pytest.mark.integration
def test_context_window_none_flag_suppresses_a_catalog_declaration(tmp_path):
    """`--context-window none` on a KNOWN model pins the suppression — the
    catalog's 1M must not leak into the wrapper."""
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "ollama-direct",
                "--model",
                "glm-5.3",
                "--alias",
                "glm-none",
                "--context-window",
                "none",
            ]
        )
        == 0
    )
    body = paths_body(tmp_path, "glm-none")
    assert "ctx=0" in body
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in body


@pytest.mark.integration
def test_window_prompt_comes_after_spec_validation(capsys):
    """A bad alias must error BEFORE any window menu — the ordering
    invariant: validate everything, then go interactive."""
    import unittest.mock as mock

    def _fail(_items, **_kwargs):
        raise AssertionError("menu shown before validation completed")

    with mock.patch("codehelper.cli.menu.select_from_menu", _fail):
        assert (
            main(
                [
                    "add",
                    "--agent",
                    "claude",
                    "--provider",
                    "ollama-direct",
                    "--model",
                    "mystery-3b",
                    "--alias",
                    "bad alias!",
                ]
            )
            == 1
        )
    assert "alias" in capsys.readouterr().err.lower()
