"""Tests for the ``codehelper tokens`` viewer and ``secrets.mask_token``.

Read-only surface: everything here asserts on printed output and file state,
never mutating the credential store. ``HOME`` isolation is conftest.py's
autouse ``_isolate_home`` fixture, like every other CLI test.
"""

from __future__ import annotations

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths
from codehelper.services.secrets import mask_token, save_credential
from codehelper.services.state import set_active_selection

_LONG_TOKEN = "fe4209" + "a" * 40 + "6309"  # 50 chars — a realistic head/tail


# --------------------------------------------------------------------------- #
# mask_token
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_mask_keeps_head_and_tail():
    assert mask_token(_LONG_TOKEN) == "fe42****6309"


@pytest.mark.unit
def test_short_token_is_all_asterisks():
    """Head+tail of a value of 8 characters or fewer would leave nothing
    hidden — the whole point of the mask is that the middle is unrecoverable.
    """
    assert mask_token("short") == "*****"
    assert mask_token("12345678") == "********"


# --------------------------------------------------------------------------- #
# `codehelper tokens` — the masked default
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_tokens_masks_cached_values(tmp_path, capsys):
    paths = Paths.default()
    save_credential(paths, "zai", _LONG_TOKEN)

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    assert "fe42****6309" in out
    assert _LONG_TOKEN not in out


@pytest.mark.integration
def test_tokens_lists_every_profile_with_the_active_marker(tmp_path, capsys):
    paths = Paths.default()
    save_credential(paths, "zai", _LONG_TOKEN)
    save_credential(paths, "zai", "644d" + "b" * 40 + "58f1", "bemyownrobot")
    save_credential(paths, "deepseek", "sk-97abcdef1234a12")
    set_active_selection(paths, "zai", "default")

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    assert "zai        default        fe42****6309  ← active" in out
    assert "bemyownrobot" in out
    assert "deepseek" in out
    # Exactly one active marker, even with three rows.
    assert out.count("← active") == 1


# --------------------------------------------------------------------------- #
# The environment section — env beats the cache in resolve_token's
# precedence, so "which key is live" is a question about BOTH stores.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_tokens_shows_set_and_unset_env_vars(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-zai-local")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    assert "LITELLM_API_KEY" in out
    assert "set      sk-l****ocal" in out
    assert "ZAI_API_KEY" in out
    assert "not set" in out
    # The env row is masked like any other.
    assert "sk-litellm-zai-local" not in out


@pytest.mark.integration
def test_tokens_env_row_printed_once_for_shared_env_vars(tmp_path, capsys):
    """deepseek and deepseek-openai share DEEPSEEK_API_KEY — the row is about
    the VARIABLE, so it prints once, not once per provider."""
    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    assert out.count("DEEPSEEK_API_KEY") == 1


@pytest.mark.integration
def test_tokens_with_an_empty_store_reports_nothing_saved(
    tmp_path, capsys, monkeypatch
):
    for var in ("LITELLM_API_KEY", "ZAI_API_KEY", "OLLAMA_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    assert "no saved tokens" in out


# --------------------------------------------------------------------------- #
# --reveal
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_tokens_reveal_prints_full_values_and_warns(tmp_path, capsys):
    paths = Paths.default()
    save_credential(paths, "zai", _LONG_TOKEN)
    capsys.readouterr()

    assert main(["tokens", "--reveal"]) == 0

    captured = capsys.readouterr()
    assert _LONG_TOKEN in captured.out
    assert "revealing stored credentials" in captured.err
