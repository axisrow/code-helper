"""Tests for ``services/state.py`` — the active-profile pre-selection store.

``state.json`` is the persistent record of which (provider, profile) pair the
TUI pre-selects for ``add``/``edit-token``. Unlike ``credentials.json`` it
holds no secrets, so it needs no ``flock`` and no ``0o600`` — a lost write
only loses a pre-selection, never a token. These tests pin:

- ``load_state`` never raises (missing/corrupt/non-object files degrade to
  ``{}``, the same never-fails contract as ``secrets.load_credentials``).
- ``active_selection``/``set_active_selection`` round-trip through the single
  ``{"active": {"provider": ..., "profile": ...}}`` pointer.
- a pre-existing installation's OLD ``active_provider``/``active_profiles``
  shape is still read (never lost on upgrade), but every write emits only
  the new shape.
"""

from __future__ import annotations

import json

import pytest

from code_helper.services.paths import Paths
from code_helper.services.state import (
    active_selection,
    load_state,
    set_active_selection,
)


def _paths(tmp_path) -> Paths:
    return Paths.from_home(tmp_path)


def _write_state_file(paths: Paths, content: str) -> None:
    """Hand-write ``state.json``, bypassing the setters — for tests that need
    a specific (often malformed or legacy-shaped) byte layout on disk."""
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


# --------------------------------------------------------------------------- #
# active_selection / set_active_selection — current single-pointer shape
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_active_selection_unset_is_none(tmp_path):
    assert active_selection(_paths(tmp_path)) is None


@pytest.mark.unit
def test_set_active_selection_round_trips(tmp_path):
    paths = _paths(tmp_path)
    set_active_selection(paths, "litellm", "work")
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.unit
def test_set_active_selection_overwrites_previous(tmp_path):
    paths = _paths(tmp_path)
    set_active_selection(paths, "litellm", "work")
    set_active_selection(paths, "litellm", "personal")
    assert active_selection(paths) == ("litellm", "personal")


@pytest.mark.unit
def test_set_active_selection_switches_provider(tmp_path):
    """There is exactly one active selection — switching providers replaces
    it rather than remembering a separate profile per provider (the
    deliberate trade-off for collapsing the old two-key shape: no duplicated
    state, at the cost of not recalling a prior provider's profile)."""
    paths = _paths(tmp_path)
    set_active_selection(paths, "zai", "axisrow")
    set_active_selection(paths, "litellm", "work")
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.unit
def test_state_file_writes_json_without_secret_mode(tmp_path):
    paths = _paths(tmp_path)
    set_active_selection(paths, "litellm", "work")
    # Written as plain JSON, and (unlike credentials.json) NOT 0o600-gated to a
    # secret owner — the file is 0o644-equivalent via atomic_write's default.
    text = paths.state_file().read_text(encoding="utf-8")
    assert json.loads(text) == {"active": {"provider": "litellm", "profile": "work"}}


@pytest.mark.unit
def test_set_active_selection_preserves_unrelated_keys(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"future_field": "keep-me"}))
    set_active_selection(paths, "litellm", "work")
    state = load_state(paths)
    assert state["future_field"] == "keep-me"
    assert state["active"] == {"provider": "litellm", "profile": "work"}


@pytest.mark.unit
def test_active_selection_ignores_a_malformed_active_value(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"active": {"provider": "litellm"}}))
    assert active_selection(paths) is None  # profile missing — incomplete pointer

    _write_state_file(paths, json.dumps({"active": "not-a-dict"}))
    assert active_selection(paths) is None

    _write_state_file(
        paths, json.dumps({"active": {"provider": "litellm", "profile": 42}})
    )
    assert active_selection(paths) is None  # non-string profile


# --------------------------------------------------------------------------- #
# Legacy two-key shape — read-only, never written again
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_active_selection_reads_the_legacy_two_key_shape(tmp_path):
    """A state.json written by the earlier (pre-collapse) release must still
    be read after upgrading, or the user's existing selection is silently
    lost."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps({"active_provider": "zai", "active_profiles": {"zai": "axisrow"}}),
    )
    assert active_selection(paths) == ("zai", "axisrow")


@pytest.mark.unit
def test_active_selection_legacy_shape_requires_matching_provider_entry(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps({"active_provider": "zai", "active_profiles": {"litellm": "work"}}),
    )
    assert active_selection(paths) is None


@pytest.mark.unit
def test_active_selection_prefers_the_new_shape_over_legacy_keys(tmp_path):
    """If somehow both shapes are present (e.g. a hand-edited file), the new
    single pointer wins — it is the current contract."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {
                "active": {"provider": "litellm", "profile": "work"},
                "active_provider": "zai",
                "active_profiles": {"zai": "axisrow"},
            }
        ),
    )
    assert active_selection(paths) == ("litellm", "work")


@pytest.mark.unit
def test_set_active_selection_drops_legacy_keys_on_write(tmp_path):
    """A write must not try to keep the two shapes in sync — that duplication
    is exactly what the single pointer removes. The old keys are dropped, not
    updated, the next time anything is written."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps({"active_provider": "zai", "active_profiles": {"zai": "axisrow"}}),
    )
    set_active_selection(paths, "litellm", "work")
    state = load_state(paths)
    assert "active_provider" not in state
    assert "active_profiles" not in state
    assert state["active"] == {"provider": "litellm", "profile": "work"}
