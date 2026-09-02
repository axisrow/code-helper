"""Tests for the ``codehelper tokens`` viewer and ``secrets.mask_token``.

Read-only surface: everything here asserts on printed output and file state,
never mutating the credential store. ``HOME`` isolation is conftest.py's
autouse ``_isolate_home`` fixture, like every other CLI test.
"""

from __future__ import annotations

import json

import pytest

from codehelper.__main__ import main
from codehelper.services.model import RETIRED_PROVIDER_NAMES
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


@pytest.mark.unit
def test_mask_escapes_control_characters():
    """The masked form is a TERMINAL-rendered string, and nothing constrains
    what a cached or environment-provided value may contain — a control
    character that survives into the readable head/tail could spoof the
    display (ANSI colouring, cursor moves, fake rows), so the mask escapes
    every non-printable as a visible \\xNN form.
    """
    masked = mask_token("\x1b[31m" + "a" * 40 + "ok\n")
    assert "\x1b" not in masked
    assert "\n" not in masked
    assert "****" in masked
    assert "\\x1b" in masked and "\\x0a" in masked


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


@pytest.mark.integration
def test_tokens_marks_a_legacy_keyed_profile_active(tmp_path, capsys):
    """The active-profile lookup understands RETIRED provider names (a token
    cached under the pre-rename key stays usable), so the viewer's marker
    must too — in BOTH rename combinations: the active selection may name
    the current provider while the cache key is legacy, or (mirrored) the
    stored selection may carry the legacy name while the cache key is
    current. Either way the honest "this profile is live" claim must not
    silently vanish."""
    paths = Paths.default()
    legacy_key = next(
        retired
        for retired, current in RETIRED_PROVIDER_NAMES.items()
        if current == "ollama-direct"
    )
    paths.credentials_file().parent.mkdir(parents=True, exist_ok=True)

    # Combination 1: legacy cache key, current-name selection.
    paths.credentials_file().write_text(
        json.dumps({legacy_key: {"default": _LONG_TOKEN}}), encoding="utf-8"
    )
    set_active_selection(paths, "ollama-direct", "default")

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    # The row renders under the provider's CURRENT name (the display, like
    # the active lookup, thinks in current names), and it carries the marker.
    marked = [line for line in out.splitlines() if "← active" in line]
    assert len(marked) == 1
    assert marked[0].startswith("ollama-direct")
    assert _LONG_TOKEN not in out

    # Combination 2 (mirrored): current cache key, legacy-name selection.
    paths.credentials_file().write_text(
        json.dumps({"ollama-direct": {"default": _LONG_TOKEN}}), encoding="utf-8"
    )
    set_active_selection(paths, legacy_key, "default")

    assert main(["tokens"]) == 0

    out = capsys.readouterr().out
    marked = [line for line in out.splitlines() if "← active" in line]
    assert len(marked) == 1
    assert marked[0].startswith("ollama-direct")


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
def test_tokens_env_row_printed_once_for_shared_env_vars(tmp_path, monkeypatch, capsys):
    """deepseek and deepseek-openai share DEEPSEEK_API_KEY — the row is about
    the VARIABLE, so it prints once, not once per provider. The env var is
    SET here: with neither a cache nor any env var the command exits early
    via "no saved tokens" and prints no env section at all (CI has none)."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shared")

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
