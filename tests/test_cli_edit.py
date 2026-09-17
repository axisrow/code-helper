"""Tests for the ``edit`` subcommand (issue #100): CLI parity for the TUI's
edit screen — model/tier/provider/effort changes on an installed wrapper,
with the three-state sentinel contract pinned end-to-end.

Follows the ``tests/test_cli_rename.py`` pattern: ``main([...])`` through the
real dispatch, HOME isolated by the autouse fixture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codehelper.__main__ import main
from codehelper.cli.requests import UNSET, EditWrapperRequest
from codehelper.services.paths import Paths


@pytest.mark.integration
def test_edit_model_in_place(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert main(["edit", "glm", "--model", "glm-5.2:cloud"]) == 0

    out = capsys.readouterr().out
    # the glm preset carries genuinely different tiers — the model edit
    # resets them to uniform, so both axes are named.
    assert "edited glm (model, tiers)" in out
    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert "export ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.2:cloud'" in body


@pytest.mark.integration
def test_edit_single_tier(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert (
        main(["edit", "glm", "--tier", "haiku=mystery-3b", "--context-window", "none"])
        == 0
    )

    out = capsys.readouterr().out
    assert "edited glm (tiers, context_window)" in out
    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert "export ANTHROPIC_DEFAULT_HAIKU_MODEL='mystery-3b'" in body
    assert "export ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.3'" in body


@pytest.mark.integration
def test_edit_model_and_tier_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert main(["edit", "glm", "--model", "glm-5.2:cloud", "--tier", "haiku=x"]) != 0


@pytest.mark.integration
def test_edit_effort_round_trip_on_a_codex_wrapper(tmp_path, capsys):
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "glm-5.2:cloud",
                "--alias",
                "codex-m",
            ]
        )
        == 0
    )

    assert main(["edit", "codex-m", "--effort", "high"]) == 0
    companion = (
        Paths.from_home(tmp_path)
        .codex_config_for("codex-m")
        .read_text(encoding="utf-8")
    )
    assert 'model_reasoning_effort = "high"' in companion

    assert main(["edit", "codex-m", "--effort", "none"]) == 0
    companion = (
        Paths.from_home(tmp_path)
        .codex_config_for("codex-m")
        .read_text(encoding="utf-8")
    )
    assert "model_reasoning_effort" not in companion


@pytest.mark.integration
def test_edit_model_only_preserves_a_recorded_effort(tmp_path, capsys):
    """The sentinel pin, end-to-end: a model edit without --effort must NOT
    strip the recorded effort (absent flag = keep, never clear)."""
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "glm-5.2:cloud",
                "--alias",
                "codex-m",
            ]
        )
        == 0
    )
    assert main(["edit", "codex-m", "--effort", "high"]) == 0

    assert main(["edit", "codex-m", "--model", "glm-5.3"]) == 0

    out = capsys.readouterr().out
    assert "edited codex-m (model)" in out
    companion = (
        Paths.from_home(tmp_path)
        .codex_config_for("codex-m")
        .read_text(encoding="utf-8")
    )
    assert 'model_reasoning_effort = "high"' in companion
    assert 'model = "glm-5.3"' in companion


@pytest.mark.integration
def test_edit_context_window_none_records_the_suppression(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "zai",
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

    assert main(["edit", "mystery", "--context-window", "none"]) == 0

    body = Paths.from_home(tmp_path).script_for("mystery").read_text(encoding="utf-8")
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in body
    assert "ctx=0)" in body


@pytest.mark.integration
def test_edit_subagent_set_then_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert (
        main(
            [
                "add",
                "--agent",
                "claude",
                "--provider",
                "zai",
                "--model",
                "glm-5.3",
                "--alias",
                "glm",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "edit",
                "glm",
                "--subagent-model",
                "sub-m",
                "--context-window",
                "500000",
            ]
        )
        == 0
    )
    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert "export CLAUDE_CODE_SUBAGENT_MODEL='sub-m'" in body

    assert (
        main(
            [
                "edit",
                "glm",
                "--subagent-model",
                "none",
                "--context-window",
                "500000",
            ]
        )
        == 0
    )
    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in body


@pytest.mark.integration
def test_edit_auth_override_marks_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-ollama")
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
                "glm-cl",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "edit",
                "glm-cl",
                "--provider",
                "ollama-direct",
                "--auth",
                "secret",
            ]
        )
        == 0
    )

    body = Paths.from_home(tmp_path).script_for("glm-cl").read_text(encoding="utf-8")
    assert "auth=secret" in body


@pytest.mark.integration
def test_edit_base_url_requires_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert main(["edit", "glm", "--base-url", "http://x.example"]) != 0


@pytest.mark.integration
def test_edit_dry_run_reports_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0
    before = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")

    assert main(["edit", "glm", "--model", "glm-5.2:cloud", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "edited glm (model, tiers)" in out
    assert "dry run — nothing written" in out
    after = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert after == before


@pytest.mark.integration
def test_edit_refuses_an_explicitly_disabled_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0
    monkeypatch.setattr(
        "codehelper.services.state.disabled_providers",
        lambda _paths: frozenset({"zai"}),
    )

    assert main(["edit", "glm", "--provider", "zai", "--model", "glm-5.2:cloud"]) != 0


@pytest.mark.unit
def test_edit_request_from_namespace_sentinel_contract():
    """The bridge maps an ABSENT flag to UNSET (keep) and the ``none``
    spelling to None (clear) — the two must never collapse."""
    req = EditWrapperRequest.from_namespace(SimpleNamespace(name="glm"))
    assert isinstance(req.model, type(UNSET))
    assert isinstance(req.effort, type(UNSET))
    assert isinstance(req.subagent_model, type(UNSET))
    assert isinstance(req.context_window, type(UNSET))
    assert req.tier_overrides is None or isinstance(req.tier_overrides, type(UNSET))

    req = EditWrapperRequest.from_namespace(
        SimpleNamespace(
            name="glm",
            effort="none",
            subagent_model="none",
            tier=["haiku=mystery-3b"],
        )
    )
    assert req.effort is None
    assert req.subagent_model is None
    assert req.tier_overrides == {"haiku": "mystery-3b"}


@pytest.mark.unit
def test_edit_request_rejects_a_malformed_tier():
    with pytest.raises(Exception, match="TIER=MODEL"):
        EditWrapperRequest.from_namespace(SimpleNamespace(name="glm", tier=["haiku"]))
