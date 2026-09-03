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

from codehelper.services.paths import Paths
from codehelper.services.state import (
    active_selection,
    context_window,
    default_wrapper,
    disabled_providers,
    load_state,
    set_active_selection,
    set_context_window,
    set_default_wrapper,
    set_provider_disabled,
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


# --------------------------------------------------------------------------- #
# default_wrapper / set_default_wrapper — per-agent default pointer (issue #28)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_default_wrapper_unset_is_none(tmp_path):
    assert default_wrapper(_paths(tmp_path), "claude") is None


@pytest.mark.unit
def test_set_default_wrapper_round_trips(tmp_path):
    paths = _paths(tmp_path)
    set_default_wrapper(paths, "codex", "codex-pro")
    assert default_wrapper(paths, "codex") == "codex-pro"


@pytest.mark.unit
def test_set_default_wrapper_independent_per_agent(tmp_path):
    """Unlike ``active`` (a single pointer), ``default_wrapper`` is a MAP with
    one slot per agent — setting claude's default must not touch codex's slot,
    so a per-agent preference survives switching attention between agents."""
    paths = _paths(tmp_path)
    set_default_wrapper(paths, "claude", "glm")
    set_default_wrapper(paths, "codex", "codex-pro")
    assert default_wrapper(paths, "claude") == "glm"
    assert default_wrapper(paths, "codex") == "codex-pro"


@pytest.mark.unit
def test_set_default_wrapper_overwrites_same_agent(tmp_path):
    """A second set for the SAME agent replaces the alias (one default per
    agent), but leaves other agents untouched."""
    paths = _paths(tmp_path)
    set_default_wrapper(paths, "claude", "glm")
    set_default_wrapper(paths, "codex", "codex-pro")
    set_default_wrapper(paths, "claude", "glm-ollama")
    assert default_wrapper(paths, "claude") == "glm-ollama"
    assert default_wrapper(paths, "codex") == "codex-pro"


@pytest.mark.unit
def test_set_default_wrapper_independent_of_active_key(tmp_path):
    """``default_wrapper`` and ``active`` are separate top-level keys — writing
    one must never clobber the other, in either order."""
    paths = _paths(tmp_path)
    set_active_selection(paths, "litellm", "work")
    set_default_wrapper(paths, "claude", "glm")
    # Write active AGAIN after default_wrapper to prove the second key survives
    # a later write to the first.
    set_active_selection(paths, "litellm", "personal")
    state = load_state(paths)
    assert state["active"] == {"provider": "litellm", "profile": "personal"}
    assert state["default_wrapper"] == {"claude": "glm"}


@pytest.mark.unit
def test_default_wrapper_ignores_a_malformed_value(tmp_path):
    """A hand-edited or partially-corrupt ``default_wrapper`` value degrades to
    None, never raises — the same defensive read ``active_selection`` applies
    to its key."""
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"default_wrapper": {"claude": 42}}))
    assert default_wrapper(paths, "claude") is None  # non-string alias

    _write_state_file(paths, json.dumps({"default_wrapper": "not-a-dict"}))
    assert default_wrapper(paths, "claude") is None  # not a map at all

    _write_state_file(paths, json.dumps({"default_wrapper": {"claude": ""}}))
    assert default_wrapper(paths, "claude") is None  # empty string

    # A different agent's slot is unaffected by a malformed slot value.
    _write_state_file(
        paths, json.dumps({"default_wrapper": {"claude": 42, "codex": "codex-pro"}})
    )
    assert default_wrapper(paths, "claude") is None
    assert default_wrapper(paths, "codex") == "codex-pro"


@pytest.mark.unit
def test_set_default_wrapper_preserves_unrelated_keys(tmp_path):
    """A read-modify-write must keep every other top-level key intact — both
    ``active`` (the other pointer) and any future field."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {
                "active": {"provider": "litellm", "profile": "work"},
                "future_field": "keep-me",
            }
        ),
    )
    set_default_wrapper(paths, "claude", "glm")
    state = load_state(paths)
    assert state["active"] == {"provider": "litellm", "profile": "work"}
    assert state["future_field"] == "keep-me"
    assert state["default_wrapper"] == {"claude": "glm"}


@pytest.mark.unit
def test_context_window_round_trips_including_zero_sentinel(tmp_path):
    """The recorded answer for a model round-trips; ``0`` is a REAL answer
    ("no declaration"), never normalized to ``None`` — the sentinel is what
    keeps an answered model from being asked again."""
    paths = _paths(tmp_path)
    assert context_window(paths, "mystery-3b") is None  # unrecorded

    set_context_window(paths, "mystery-3b", 1_000_000)
    assert context_window(paths, "mystery-3b") == 1_000_000

    set_context_window(paths, "mystery-3b", 0)  # explicit "no declaration"
    assert context_window(paths, "mystery-3b") == 0


@pytest.mark.unit
def test_context_window_independent_per_model(tmp_path):
    """Slots are keyed by model name and don't interfere."""
    paths = _paths(tmp_path)
    set_context_window(paths, "mystery-3b", 1_000_000)
    set_context_window(paths, "other-model", 0)
    assert context_window(paths, "mystery-3b") == 1_000_000
    assert context_window(paths, "other-model") == 0
    assert context_window(paths, "unheard-of") is None


@pytest.mark.unit
def test_context_window_ignores_a_malformed_value(tmp_path):
    """Non-dict maps and non-int (bool included) slot values read as
    unrecorded — a corrupt slot asks again rather than lying."""
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"context_windows": "oops"}))
    assert context_window(paths, "mystery-3b") is None

    _write_state_file(paths, json.dumps({"context_windows": {"mystery-3b": "1M"}}))
    assert context_window(paths, "mystery-3b") is None

    _write_state_file(paths, json.dumps({"context_windows": {"mystery-3b": True}}))
    assert context_window(paths, "mystery-3b") is None

    _write_state_file(paths, json.dumps({"context_windows": {"mystery-3b": 1.5}}))
    assert context_window(paths, "mystery-3b") is None


@pytest.mark.unit
def test_set_context_window_preserves_unrelated_keys(tmp_path):
    """A read-modify-write must keep every other top-level key intact."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {
                "default_wrapper": {"claude": "glm"},
                "future_field": "keep-me",
            }
        ),
    )
    set_context_window(paths, "mystery-3b", 2_000_000)
    state = load_state(paths)
    assert state["default_wrapper"] == {"claude": "glm"}
    assert state["future_field"] == "keep-me"
    assert state["context_windows"] == {"mystery-3b": 2_000_000}


@pytest.mark.unit
def test_context_window_out_of_range_reads_as_unrecorded(tmp_path):
    """A hand-edited negative or oversized value would ride an explicit
    answer straight into a wrapper marker or the live settings — read it
    as unrecorded instead (review round 2, PR #84)."""
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"context_windows": {"m": -5}}))
    assert context_window(paths, "m") is None
    _write_state_file(paths, json.dumps({"context_windows": {"m": 99_999_999}}))
    assert context_window(paths, "m") is None


@pytest.mark.unit
def test_set_context_window_refuses_an_out_of_range_value(tmp_path):
    """The writer enforces the same range the reader accepts, so a record
    written here can always be read back."""
    paths = _paths(tmp_path)
    with pytest.raises(Exception, match="unusable context window"):
        set_context_window(paths, "m", -1)
    with pytest.raises(Exception, match="unusable context window"):
        set_context_window(paths, "m", 20_000_000)
    assert context_window(paths, "m") is None  # nothing recorded


# --------------------------------------------------------------------------- #
# disabled_providers / set_provider_disabled — runtime provider disable (issue #89)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_disabled_providers_unset_is_empty(tmp_path):
    assert disabled_providers(_paths(tmp_path)) == frozenset()


@pytest.mark.unit
def test_set_provider_disabled_round_trips(tmp_path):
    paths = _paths(tmp_path)
    ollama = frozenset({"ollama-direct"})
    set_provider_disabled(paths, "ollama-direct", True, storage_names=ollama)
    assert disabled_providers(paths) == frozenset({"ollama-direct"})
    set_provider_disabled(paths, "ollama-direct", False, storage_names=ollama)
    assert disabled_providers(paths) == frozenset()


@pytest.mark.unit
def test_disabled_providers_ignores_a_malformed_value(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"disabled_providers": "oops"}))
    assert disabled_providers(paths) == frozenset()
    _write_state_file(paths, json.dumps({"disabled_providers": [42, None, "zai"]}))
    assert disabled_providers(paths) == frozenset({"zai"})


@pytest.mark.unit
def test_set_provider_disabled_normalizes_legacy_spelling(tmp_path):
    """A previously stored legacy spelling of the same provider is rewritten
    to the canonical name on the next write — the set never holds two
    spellings of one provider."""
    paths = _paths(tmp_path)
    _write_state_file(paths, json.dumps({"disabled_providers": ["ollama"]}))
    set_provider_disabled(
        paths,
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    assert disabled_providers(paths) == frozenset({"ollama-direct"})


@pytest.mark.unit
def test_set_provider_disabled_preserves_unrelated_keys_and_providers(tmp_path):
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {
                "disabled_providers": ["deepseek"],
                "default_wrapper": {"claude": "glm"},
                "future_field": "keep-me",
            }
        ),
    )
    set_provider_disabled(
        paths,
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama-direct"}),
    )
    state = load_state(paths)
    assert state["disabled_providers"] == ["deepseek", "ollama-direct"]
    assert state["default_wrapper"] == {"claude": "glm"}
    assert state["future_field"] == "keep-me"


@pytest.mark.unit
def test_disable_clears_the_active_pointer_for_both_spellings(tmp_path):
    """A disabled provider must not survive as the active pre-selection — its
    wrappers are gone, so the pointer could never resolve. Both the legacy
    ("ollama") and canonical ("ollama-direct") spellings are matched."""
    paths = _paths(tmp_path)
    set_active_selection(paths, "ollama", "default")
    set_provider_disabled(
        paths,
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    assert active_selection(paths) is None
    # …and an active pointer naming a DIFFERENT provider survives.
    set_active_selection(paths, "zai", "axisrow")
    set_provider_disabled(
        paths,
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    assert active_selection(paths) == ("zai", "axisrow")


@pytest.mark.unit
def test_disable_clears_a_legacy_shape_active_pointer(tmp_path):
    """The pointer cleanup reads through ``active_selection``, so a pre-upgrade
    two-key ``state.json`` whose ``active_provider`` names the disabled
    provider is cleared too — the docstring promises a disabled provider never
    survives as the active pre-selection, in EITHER stored shape."""
    paths = _paths(tmp_path)
    _write_state_file(
        paths,
        json.dumps(
            {"active_provider": "ollama", "active_profiles": {"ollama": "default"}}
        ),
    )
    assert active_selection(paths) == ("ollama", "default")
    set_provider_disabled(
        paths,
        "ollama-direct",
        True,
        storage_names=frozenset({"ollama", "ollama-direct"}),
    )
    assert active_selection(paths) is None
    assert "active_provider" not in load_state(paths)
    assert "active_profiles" not in load_state(paths)


@pytest.mark.unit
def test_enable_leaves_the_active_pointer_alone(tmp_path):
    paths = _paths(tmp_path)
    set_active_selection(paths, "zai", "axisrow")
    ollama = frozenset({"ollama-direct"})
    set_provider_disabled(paths, "ollama-direct", True, storage_names=ollama)
    set_provider_disabled(paths, "ollama-direct", False, storage_names=ollama)
    assert active_selection(paths) == ("zai", "axisrow")
