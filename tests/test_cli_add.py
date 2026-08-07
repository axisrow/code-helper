"""Tests for ``code-helper add`` in both modes, and ``list agents|providers|matrix``.

Everything runs through ``main([...])`` so the argparse wiring is exercised too,
not just the handler.
"""

from __future__ import annotations

import pytest

from code_helper.__main__ import main
from code_helper.services.paths import Paths


def _body(tmp_path, name: str) -> str:
    return Paths.from_home(tmp_path).script_for(name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Constructor mode
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_agent_provider_model_creates_wrapper(tmp_path):
    """codex × ollama resolves to OPENAI_TOML by priority now: the wrapper runs
    ``codex --profile <alias>``, and the TOML profile + model catalog are
    written alongside it.
    """
    code = main(
        ["add", "--agent", "codex", "--provider", "ollama", "--model", "glm-5:cloud"]
    )
    assert code == 0
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"  # alias derived from model + agent
    assert f"exec codex --profile '{alias}' \"$@\"" in _body(tmp_path, alias)
    config = paths.codex_config_for(alias).read_text(encoding="utf-8")
    assert config.startswith("# code-helper: managed wrapper")
    assert 'model = "glm-5:cloud"' in config
    assert 'base_url = "http://127.0.0.1:11434/v1/"' in config
    assert 'wire_api = "responses"' in config
    assert paths.codex_catalog_for(alias).exists()


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
def test_incompatible_pairing_never_prompts_for_a_token(tmp_path, monkeypatch):
    """Validation must happen BEFORE any interactive secret prompt."""
    import code_helper.services.secrets as secrets

    def _explode(*a, **kw):  # pragma: no cover - must not run
        raise AssertionError("must not prompt for a token on an invalid combination")

    monkeypatch.setattr(secrets, "resolve_token", _explode)
    assert (
        main(["add", "--agent", "codex", "--provider", "zai", "--model", "glm-5-turbo"])
        == 1
    )


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
    assert "ollama" in out and "zai" in out


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
def test_list_shows_ad_hoc_wrappers(tmp_path, capsys):
    main(["add", "--agent", "codex", "--provider", "ollama", "--model", "qwen3.5:9b"])
    capsys.readouterr()

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ad-hoc wrappers" in out
    assert "qwen3.5-codex" in out


@pytest.mark.integration
def test_list_models_prints_and_writes_nothing(tmp_path, monkeypatch):
    import code_helper.services.models_api as api
    from code_helper.services.models_api import ModelListResult

    monkeypatch.setattr(
        api, "list_models", lambda p, **kw: ModelListResult(("a:1", "b:2"), "url")
    )
    assert (
        main(["add", "--agent", "codex", "--provider", "ollama", "--list-models"]) == 0
    )
    assert not Paths.from_home(tmp_path).bin_dir.exists()
