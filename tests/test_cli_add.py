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
        ["add", "--agent", "codex", "--provider", "ollama", "--model", "glm-5:cloud"]
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
            "ollama",
            "--model",
            "glm-5:cloud",
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
            "ollama",
            "--model",
            "glm-5:cloud",
            "--alias",
            "mycodex",
        ]
    )
    assert Paths.from_home(tmp_path).script_for("mycodex").exists()


@pytest.mark.integration
def test_add_agent_requires_model(tmp_path, capsys):
    assert main(["add", "--agent", "codex", "--provider", "ollama"]) == 1
    assert "--model is required" in capsys.readouterr().err


@pytest.mark.integration
def test_add_agent_and_provider_must_come_together(tmp_path, capsys):
    assert main(["add", "--agent", "codex", "--model", "m"]) == 1
    assert "must be given together" in capsys.readouterr().err


@pytest.mark.integration
def test_preset_and_axes_are_mutually_exclusive(tmp_path, capsys):
    assert main(["add", "glm", "--agent", "codex", "--provider", "ollama"]) == 1
    assert "not both" in capsys.readouterr().err


@pytest.mark.integration
def test_unknown_agent_exits_1(tmp_path, capsys):
    assert main(["add", "--agent", "nope", "--provider", "ollama", "--model", "m"]) == 1
    assert "unknown agent" in capsys.readouterr().err


@pytest.mark.integration
def test_incompatible_pairing_is_refused(tmp_path, capsys):
    """codex cannot speak the Anthropic protocol z.ai serves."""
    assert (
        main(["add", "--agent", "codex", "--provider", "zai", "--model", "glm-5-turbo"])
        == 1
    )
    assert "no common configuration" in capsys.readouterr().err


@pytest.mark.integration
def test_launch_only_agent_incompatible_with_zai(tmp_path, capsys):
    """A launch-only agent has no shape in common with a non-ollama provider,
    and the error's hint names `ollama` as what it DOES work with."""
    assert (
        main(["add", "--agent", "opencode", "--provider", "zai", "--model", "x"]) == 1
    )
    err = capsys.readouterr().err
    assert "no common configuration" in err
    assert "ollama" in err


@pytest.mark.integration
def test_incompatible_pairing_never_prompts_for_a_token(tmp_path, monkeypatch):
    """Validation must happen BEFORE any interactive secret prompt."""
    import codehelper.services.secrets as secrets

    def _explode(*a, **kw):  # pragma: no cover - must not run
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
def test_add_litellm_claude_requires_base_url(tmp_path, capsys):
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
def test_add_base_url_on_a_fixed_provider_is_refused(tmp_path, capsys):
    code = main(
        [
            "add",
            "--agent",
            "claude",
            "--provider",
            "ollama",
            "--base-url",
            "http://x/v1",
            "--model",
            "m",
        ]
    )
    assert code == 1
    assert "fixed base URL" in capsys.readouterr().err


@pytest.mark.integration
def test_add_base_url_with_a_preset_is_refused(tmp_path, capsys):
    code = main(["add", "deepseek-ollama", "--base-url", "http://x/v1"])
    assert code == 1
    assert "constructor form only" in capsys.readouterr().err


@pytest.mark.integration
def test_add_rejects_an_invalid_base_url_scheme(tmp_path, capsys):
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
        ]
    )
    assert code == 1
    assert "http:// or https://" in capsys.readouterr().err


@pytest.mark.integration
def test_add_rejects_an_empty_base_url(tmp_path, capsys):
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
            "ollama",
            "--auth",
            "secret",
            "--model",
            "glm-5:cloud",
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
            "ollama",
            "--model",
            "glm-5:cloud",
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
def test_add_auth_secret_on_an_already_secret_provider_is_a_no_op(
    tmp_path, monkeypatch
):
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
        ]
    )
    assert code == 0


@pytest.mark.integration
def test_add_auth_applies_to_the_constructor_form_only(tmp_path, capsys):
    code = main(["add", "deepseek-ollama", "--auth", "secret"])
    assert code == 1
    assert "constructor form only" in capsys.readouterr().err


@pytest.mark.integration
def test_add_auth_rejects_an_unknown_value(tmp_path, capsys):
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
                "ollama",
                "--auth",
                "bogus",
                "--model",
                "m",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.integration
def test_bad_base_url_never_prompts_for_a_token(tmp_path, monkeypatch):
    """validate-before-prompt, pinned for the NEW validation channel."""
    import codehelper.services.secrets as secrets

    def _explode(*a, **kw):  # pragma: no cover - must not run
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
        ]
    )
    assert code == 1


@pytest.mark.integration
def test_list_models_uses_the_runtime_base_url(tmp_path, monkeypatch, capsys):
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
def test_bad_alias_is_rejected(tmp_path, capsys):
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama",
                "--model",
                "m",
                "--alias",
                "../../etc/passwd",
            ]
        )
        == 1
    )
    assert "path separator" in capsys.readouterr().err


@pytest.mark.integration
def test_alias_may_not_shadow_an_agent_binary(tmp_path, capsys):
    """`--alias claude` would exec itself forever and clobber a real symlink."""
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama",
                "--model",
                "m",
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
            "ollama",
            "--model",
            "glm-5:cloud",
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
            "ollama",
            "--model",
            "glm-5:cloud",
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
            "ollama",
            "--model",
            "glm-5:cloud",
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
def test_bare_agent_name_suggests_the_constructor(tmp_path, capsys):
    """`add codex` is a likely mistake — teach the right form, don't just refuse."""
    assert main(["add", "codex"]) == 1
    err = capsys.readouterr().err
    assert "is an agent" in err
    assert "--agent codex" in err


@pytest.mark.integration
def test_unknown_name_that_is_not_an_agent(tmp_path, capsys):
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
def test_list_agents(tmp_path, capsys):
    assert main(["list", "agents"]) == 0
    out = capsys.readouterr().out
    assert "claude" in out and "codex" in out


@pytest.mark.integration
def test_list_providers(tmp_path, capsys):
    assert main(["list", "providers"]) == 0
    out = capsys.readouterr().out
    assert "ollama" in out and "zai" in out and "litellm" in out
    assert "deepseek" in out and "deepseek-openai" in out


@pytest.mark.integration
def test_list_matrix_shows_a_gap_for_incompatible_pairs(tmp_path, capsys):
    """The matrix is rendered via resolve_shape, so it cannot lie about `add`."""
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    codex_row = next(line for line in lines if line.startswith("codex"))
    assert "—" in codex_row  # codex × zai is impossible
    # codex × ollama now resolves to the TOML profile by priority.
    assert "openai-toml" in codex_row


@pytest.mark.integration
def test_list_matrix_includes_litellm_with_no_gap(tmp_path, capsys):
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
def test_list_agents_includes_every_ollama_launch_integration(tmp_path, capsys):
    """`list agents` shows all 15 registered agents, not just claude/codex."""
    assert main(["list", "agents"]) == 0
    out = capsys.readouterr().out
    for name in ("opencode", "copilot", "droid", "cline", "qwen"):
        assert name in out


@pytest.mark.integration
def test_list_matrix_launch_only_agent_row(tmp_path, capsys):
    """A launch-only agent's row: ollama-launch for ollama, a gap elsewhere.

    `resolve_shape` renders this row, so it cannot disagree with what `add`
    actually accepts.
    """
    assert main(["list", "matrix"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header_cols = lines[0].split()
    ollama_idx = header_cols.index("ollama")
    zai_idx = header_cols.index("zai")
    row = next(line for line in lines if line.startswith("opencode")).split()
    # +1: each data row's first word is the agent name, not a provider column.
    assert row[ollama_idx + 1] == "ollama-launch"
    assert row[zai_idx + 1] == "—"


@pytest.mark.integration
def test_list_shows_ad_hoc_wrappers(tmp_path, capsys):
    main(["add", "--agent", "codex", "--provider", "ollama", "--model", "qwen3.5:9b"])
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
        api, "list_models", lambda p, **kw: ModelListResult(("a:1", "b:2"), "url")
    )
    assert (
        main(["add", "--agent", "codex", "--provider", "ollama", "--list-models"]) == 0
    )
    assert not Paths.from_home(tmp_path).bin_dir.exists()


# --------------------------------------------------------------------------- #
# Credential cache (issue #15) — a token typed at a prompt is cached, so the
# next add/--list-models does not ask again. Never under --dry-run.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_caches_a_prompt_typed_token(tmp_path, monkeypatch):
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**kwargs):
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
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    assert secrets.credential_for(paths, "litellm") == "sk-typed"


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
def test_profile_selects_the_default_alias_and_is_visible_in_list(
    tmp_path, monkeypatch, capsys
):
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

    def _fake_resolve_token_old(**kwargs):
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

    def _fake_resolve_token_new(**kwargs):
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

    def _fake_resolve_token(**kwargs):
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
    ]

    def _fake_resolve_token(**kwargs):
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

    def _fake_resolve_token(**kwargs):
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
        ]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    assert not paths.credentials_file().exists()


@pytest.mark.integration
def test_add_warns_when_bin_dir_is_not_on_path(tmp_path, monkeypatch, capsys):
    """A real (non-dry-run) install warns on stderr when ``paths.bin_dir``
    isn't on ``PATH`` — otherwise the freshly-installed alias just gives
    ``command not found`` with no clue why."""
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**kwargs):
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
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "not on PATH" in err
    assert "gpt-4o-claude" in err


@pytest.mark.integration
def test_add_dry_run_does_not_print_a_path_warning(tmp_path, monkeypatch, capsys):
    """``install_wrapper`` returns True under ``dry_run`` too ("would write"),
    so the PATH check must not fire off that truthy-but-nothing-written
    signal — a dry run must not warn about a file it never created."""
    import codehelper.services.secrets as secrets

    def _fake_resolve_token(**kwargs):
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

    def _fake_list_models(provider, *, token=""):
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

    def _fake_list_models(provider, *, token=""):
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

    def _fake_list_models(provider, *, token=""):
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
def test_list_omits_the_active_profile_header_when_unset(tmp_path, capsys):
    assert main(["list"]) == 0
    assert "Active profile:" not in capsys.readouterr().out
