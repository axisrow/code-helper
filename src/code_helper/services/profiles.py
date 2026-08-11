"""Pure validation for token-profile naming decisions (issue #37, P1.2).

CLI ``_handle_edit_token`` and TUI ``_new_profile`` both branch identically on
the count of existing profile names and apply the same uniqueness/emptiness
rules, but differ in how they report problems (raise vs print) and in message
wording. This module owns the SHARED decision logic and returns a structured
outcome, leaving each caller free to raise or print as fits its UI.

Pure by contract: no IO, no ``raise``, no ``print``. A caller maps an outcome
to whichever error channel its UI uses.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["NewProfileOutcome", "classify_new_profile", "validate_new_profile_name"]


class NewProfileOutcome(StrEnum):
    """Result of validating a new-profile naming decision."""

    OK = "ok"
    EMPTY = "empty"
    SAME = "same"  # current == new (single-profile rename branch)
    COLLISION_RENAMED = "collision_renamed"  # renamed-old name already exists
    COLLISION_NEW = "collision_new"  # new name already exists


def classify_new_profile(
    existing: list[str],
    current_name: str | None,
    new_name: str | None,
) -> NewProfileOutcome:
    """Validate the 'rename single + create new' decision.

    Encapsulates the single-profile branch (``len(existing) == 1``) where the
    user provides a new name for the CURRENT profile AND a name for a
    brand-new one. Both callers (CLI ``_handle_edit_token`` and TUI
    ``_new_profile``) apply the SAME rule chain here; only their error channel
    differs.

    Order matters and mirrors both pre-refactor call sites:

    1. ``EMPTY`` — either name is blank. Checked before collisions so the
       message points at the actual blocker (a blank name) rather than a
       downstream coincidence (the blank happening not to be in ``existing``).
    2. ``SAME`` — the two names are identical. CLI merges this with
       ``COLLISION_NEW`` into one "must be unique" message; TUI keeps it
       separate as "must be different". The classifier splits them so each
       caller can map them however it likes.
    3. ``COLLISION_RENAMED`` — the renamed-old name already exists as a
       DIFFERENT profile. ``current_name == existing[0]`` is the normal case
       (renaming the sole profile to itself), NOT a collision; the
       ``current_name != existing[0]`` guard excludes it.
    4. ``COLLISION_NEW`` — the new name is already taken.
    5. ``OK`` — neither name collides.

    The single-profile branch is only ever entered with ``len(existing) == 1``
    at the call sites, but this function does NOT enforce that: it inspects
    membership (``in existing``) rather than length, so it stays correct if a
    future caller reaches it with a longer list. ``existing == []`` is likewise
    safe — every membership check is simply false, so only ``EMPTY``/``SAME``
    can fire.
    """
    if not current_name or not new_name:
        return NewProfileOutcome.EMPTY
    if current_name == new_name:
        return NewProfileOutcome.SAME
    # current_name == existing[0] is the benign "renaming the sole profile to
    # itself" case (existing[0] IS the profile we are renaming FROM); only a
    # DIFFERENT existing entry constitutes a collision.
    if current_name in existing and current_name != existing[0]:
        return NewProfileOutcome.COLLISION_RENAMED
    if new_name in existing:
        return NewProfileOutcome.COLLISION_NEW
    return NewProfileOutcome.OK


def validate_new_profile_name(
    name: str | None, existing: list[str]
) -> NewProfileOutcome:
    """Validate a single new profile name against ``existing``.

    The multi-profile branch (``len(existing) >= 2``): only a brand-new name is
    collected, with no rename of an existing profile. ``SAME`` and
    ``COLLISION_RENAMED`` do not apply here — there is no second name to clash
    with and no rename in flight — so only three outcomes are possible.
    """
    if not name:
        return NewProfileOutcome.EMPTY
    if name in existing:
        return NewProfileOutcome.COLLISION_NEW
    return NewProfileOutcome.OK
