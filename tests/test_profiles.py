"""Tests for the pure new-profile naming classifier (issue #37, P1.2).

``classify_new_profile`` and ``validate_new_profile_name`` own the SHARED
decision logic that the CLI (``_handle_edit_token``) and TUI
(``_new_profile``) previously duplicated with different error channels. The
contract here is the OUTCOME each decision yields — the CLI/TUI map those
outcomes to their own raises/prints, and their integration tests pin the
wording. These tests pin only the decision.

Pure by design: no IO, no fixtures beyond the arguments. Every case is a row.
"""

from __future__ import annotations

import pytest

from codehelper.services.profiles import (
    NewProfileOutcome,
    classify_new_profile,
    profile_slots,
    validate_new_profile_name,
)

# --- classify_new_profile -------------------------------------------------
#
# The single-profile branch: user provides a new name for the CURRENT profile
# AND a name for a brand-new one. Five outcomes are reachable.


@pytest.mark.unit
@pytest.mark.parametrize(
    "existing,current,new,expected",
    [
        # OK — neither name collides.
        (["default"], "work", "personal", NewProfileOutcome.OK),
        (["default"], "default", "personal", NewProfileOutcome.OK),
        # EMPTY — either name blank. Both call sites strip() the raw input
        # BEFORE calling the classifier, so a blank name arrives here as ""
        # (or None when the input reader was cancelled and the caller did not
        # short-circuit). A whitespace-only string is therefore NOT treated
        # as empty by this pure function — that is the IO layer's job.
        (["default"], "", "personal", NewProfileOutcome.EMPTY),
        (["default"], "work", "", NewProfileOutcome.EMPTY),
        (["default"], None, "personal", NewProfileOutcome.EMPTY),
        (["default"], "work", None, NewProfileOutcome.EMPTY),
        # SAME — the two names are identical (CLI merges this with
        # COLLISION_NEW; the classifier keeps them split).
        (["default"], "work", "work", NewProfileOutcome.SAME),
        # COLLISION_RENAMED — the renamed-old name already exists as a
        # DIFFERENT profile. current == existing[0] is the benign "renaming
        # the sole profile to itself" case and is NOT a collision.
        (["default", "work"], "default", "personal", NewProfileOutcome.OK),
        (["default", "work"], "work", "personal", NewProfileOutcome.COLLISION_RENAMED),
        # COLLISION_NEW — the new name is already taken.
        (["default"], "work", "default", NewProfileOutcome.COLLISION_NEW),
        (["work", "personal"], "default", "work", NewProfileOutcome.COLLISION_NEW),
    ],
    ids=[
        "ok_new_names",
        "ok_rename_to_self_then_new",
        "empty_current_blank",
        "empty_new_blank",
        "empty_current_none",
        "empty_new_none",
        "same_identical_names",
        "renamed_self_not_collision",
        "renamed_other_collision",
        "new_collision_single",
        "new_collision_multi",
    ],
)
def test_classify_new_profile_outcomes(existing, current, new, expected):
    assert classify_new_profile(existing, current, new) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "existing,current,new",
    [
        # Three profiles, current_name is one of them (not the first) — a
        # collision against the OTHER existing entry.
        (["a", "b", "c"], "b", "d"),
        # current_name collides with the third entry while existing[0] is
        # something else entirely.
        (["x", "y"], "y", "z"),
    ],
)
def test_classify_new_profile_collision_renamed_multi(existing, current, new):
    """COLLISION_RENAMED fires whenever current is in existing but not [0].

    The classifier inspects membership rather than ``len(existing) == 1``,
    so it stays correct if a future caller reaches the single-profile branch
    with a longer list.
    """
    assert (
        classify_new_profile(existing, current, new)
        is NewProfileOutcome.COLLISION_RENAMED
    )


@pytest.mark.unit
def test_classify_new_profile_empty_list_does_not_crash():
    """The single-profile branch is only ever called with len(existing)==1.

    A future caller that reaches it with an empty list must not crash: every
    membership check is simply false, so only EMPTY/SAME can fire.
    """
    assert classify_new_profile([], "work", "personal") is NewProfileOutcome.OK
    assert classify_new_profile([], "", "personal") is NewProfileOutcome.EMPTY
    assert classify_new_profile([], "work", "work") is NewProfileOutcome.SAME


@pytest.mark.unit
def test_classify_new_profile_current_equals_existing_zero_not_collision():
    """``current_name == existing[0]`` is normal, NOT a collision.

    The single-profile branch renames existing[0] (the sole profile) to
    itself when the user keeps its name — that is the rename-source, not a
    collision against a different entry.
    """
    assert (
        classify_new_profile(["default"], "default", "personal") is NewProfileOutcome.OK
    )


# --- validate_new_profile_name -------------------------------------------
#
# The multi-profile branch: only a brand-new name is collected. SAME and
# COLLISION_RENAMED do not apply (no second name, no rename in flight).


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,existing,expected",
    [
        ("personal", ["default"], NewProfileOutcome.OK),
        ("work", ["default", "personal"], NewProfileOutcome.OK),
        ("", ["default"], NewProfileOutcome.EMPTY),
        (None, ["default"], NewProfileOutcome.EMPTY),
        ("default", ["default"], NewProfileOutcome.COLLISION_NEW),
        ("work", ["default", "work"], NewProfileOutcome.COLLISION_NEW),
    ],
    ids=[
        "ok_single_existing",
        "ok_multi_existing",
        "empty_blank",
        "empty_none",
        "collision_single_existing",
        "collision_multi_existing",
    ],
)
def test_validate_new_profile_name_outcomes(name, existing, expected):
    assert validate_new_profile_name(name, existing) is expected


@pytest.mark.unit
def test_validate_new_profile_name_empty_existing_list():
    """No existing names → only EMPTY can fire, never COLLISION_NEW."""
    assert validate_new_profile_name("personal", []) is NewProfileOutcome.OK
    assert validate_new_profile_name("", []) is NewProfileOutcome.EMPTY


@pytest.mark.unit
def test_profile_slots_honours_a_preloaded_creds_snapshot(tmp_path, monkeypatch):
    """``creds=`` (issue #110) answers exactly as a fresh read and never
    re-reads ``credentials.json`` — the TUI feeds its per-iteration
    snapshot so the slot strip costs no file I/O per redraw frame."""
    from codehelper.services.paths import Paths
    from codehelper.services.secrets import load_credentials, save_credential

    paths = Paths.from_home(tmp_path)
    save_credential(paths, "zai", "sk-work", "work")
    save_credential(paths, "deepseek", "sk-ds")
    snapshot = load_credentials(paths)
    fresh = profile_slots(paths)
    assert ("zai", "work") in fresh
    assert ("deepseek", "default") in fresh

    def _forbidden(_paths):
        raise AssertionError("creds= call re-read the file")

    monkeypatch.setattr("codehelper.services.secrets.load_credentials", _forbidden)
    assert profile_slots(paths, creds=snapshot) == fresh
