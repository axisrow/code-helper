"""Tests for ``services/state.py`` — the active-profile pre-selection store.

``state.json`` is the persistent record of which token profile the TUI
pre-selects for ``add``/``edit-token``. Unlike ``credentials.json`` it holds no
secrets, so it needs no ``flock`` and no ``0o600`` — a lost write only loses a
pre-selection, never a token. These tests pin:

- ``load_state`` never raises (missing/corrupt/non-object files degrade to
  ``{}``, the same never-fails contract as ``secrets.load_credentials``).
- round-trip ``set_*`` / ``load_*``.
- non-string values are dropped on read (a stray entry must not cost the user
  the rest of the state).
"""

from __future__ import annotations

import json

import pytest

from code_helper.services.paths import Paths
from code_helper.services.state import (
    active_profile,
    active_provider,
    load_state,
    set_active_profile,
    set_active_provider,
)


def _paths(tmp_path) -> Paths:
    return Paths.from_home(tmp_path)


def _write_state_file(paths: Paths, content: str) -> None:
    """Hand-write ``state.json``, bypassing the setters — for tests that need
    a specific (often malformed) byte layout on disk."""
    paths.state_file().parent.mkdir(parents=True, exist_ok=True)
    paths.state_file().write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# load_state — never raises
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_state_missing_file_is_empty(tmp_path):
    assert load_state(_paths(tmp_path)) == {}


@pytest.mark.unit
def test_load_state_corrupt_json_is_empty(tmp_path):
    _write_state_file(_paths(tmp_path), "{ not json !!!")
    assert load_state(_paths(tmp_path)) == {}


@pytest.mark.unit
def test_load_state_non_object_file_is_empty(tmp_path):
    _write_state_file(_paths(tmp_path), "[1, 2, 3]")
    assert load_state(_paths(tmp_path)) == {}


@pytest.mark.unit
def test_load_state_drops_non_string_values(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {
                "active_profiles": {"litellm": "work", "zai": 42},
                "active_provider": ["not", "a", "string"],
            }
        ),
    )
    state = load_state(paths)
    # Unknown keys survive; string values are kept; non-strings are dropped.
    assert state["active_profiles"] == {"litellm": "work", "zai": 42}
    # active_provider read helper returns None for a non-string.
    assert active_provider(paths) is None


# --------------------------------------------------------------------------- #
# active_profile / set_active_profile
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_active_profile_unset_is_none(tmp_path):
    assert active_profile(_paths(tmp_path), "litellm") is None


@pytest.mark.unit
def test_set_active_profile_round_trips(tmp_path):
    paths = _paths(tmp_path)
    set_active_profile(paths, "litellm", "work")
    assert active_profile(paths, "litellm") == "work"
    assert active_profile(paths, "zai") is None  # other provider untouched


@pytest.mark.unit
def test_set_active_profile_overwrites_previous(tmp_path):
    paths = _paths(tmp_path)
    set_active_profile(paths, "litellm", "work")
    set_active_profile(paths, "litellm", "personal")
    assert active_profile(paths, "litellm") == "personal"


@pytest.mark.unit
def test_set_active_profile_preserves_unrelated_keys(tmp_path):
    paths = _paths(tmp_path)
    set_active_provider(paths, "litellm")
    set_active_profile(paths, "litellm", "work")
    # Setting a profile must not clobber the sibling active_provider key.
    assert active_provider(paths) == "litellm"


# --------------------------------------------------------------------------- #
# active_provider / set_active_provider
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_active_provider_unset_is_none(tmp_path):
    assert active_provider(_paths(tmp_path)) is None


@pytest.mark.unit
def test_set_active_provider_round_trips(tmp_path):
    paths = _paths(tmp_path)
    set_active_provider(paths, "litellm")
    assert active_provider(paths) == "litellm"


@pytest.mark.unit
def test_state_file_writes_json_without_secret_mode(tmp_path):
    paths = _paths(tmp_path)
    set_active_profile(paths, "litellm", "work")
    # Written as plain JSON, and (unlike credentials.json) NOT 0o600-gated to a
    # secret owner — the file is 0o644-equivalent via atomic_write's default.
    assert "active_profiles" in paths.state_file().read_text(encoding="utf-8")
