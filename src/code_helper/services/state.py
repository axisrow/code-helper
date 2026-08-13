"""Persistent UI pre-selection state (issue #23).

Holds the TUI's active token-profile pre-selection — which profile is selected
by default for ``add``/``edit-token`` — in ``~/.config/code-helper/state.json``.
This is deliberately a SEPARATE module and file from ``services/secrets.py``:

- ``secrets.py`` is built around ``credentials.json`` and holds TOKENS. Its
  schema ``{provider_name: {profile_name: token}}`` cannot grow without a
  versioned migration, because ``load_credentials`` treats every top-level key
  as a provider name (issue #19). ``state.json`` uses named top-level keys
  from the start, so future fields never need a migration.
- This module holds NO secrets, so no ``0o600`` and no ``flock``. A race here
  loses a pre-selection at worst, never a token — unlike
  :func:`secrets._locked_update`, whose serialization protects the only
  persistent copy of a credential. Write is a plain read-modify-write through
  :func:`code_helper.backends._atomic.atomic_write`.

**Schema: a single pointer, ``{"active": {"provider": ..., "profile": ...}}``.**
An earlier version stored ``active_provider`` and ``active_profiles`` (a
provider -> profile map) as two independent top-level keys — which could
disagree with each other (a stale ``active_provider`` pointing at a provider
whose ``active_profiles`` entry had since changed for a *different* provider).
A single pointer to one (provider, profile) pair makes that disagreement
unrepresentable: there is exactly one active selection, full stop. The
accepted trade-off is that switching to another provider and back does not
recall the profile that was active there before — the new provider's first
profile is used again. ``load_state`` transparently reads the OLD two-key
shape (never written again) so an existing installation's selection is not
silently lost on upgrade; every write emits only the new ``active`` shape.

A second, independent pointer, ``default_wrapper: {agent_name: alias}``,
holds which wrapper alias is the default for each agent — the marker #29
renders on the main screen and #30 applies in the agent's native config. It
is a MAP (one slot per agent), not a single pair, because unlike the active
profile there is a meaningful default per agent that should survive switching
attention between agents. It is a separate top-level key from ``active`` so
the two can never clobber each other, and neither's presence requires the
other.

A saved profile name can go stale (renamed via ``secrets.rename_profile`` or
dropped via ``secrets.invalidate_cached_credential``), so readers must
cross-check against the live ``secrets.profile_names``. That cross-check lives
in ``secrets.valid_active_profile`` (NOT here): this module must not depend on
``secrets.py``, while ``secrets.py`` importing ``state.py`` is a one-way edge
that creates no cycle. A saved ``default_wrapper`` alias can go stale too
(uninstalled/renamed), and its cross-check lives in
``wrappers.valid_default_wrapper`` for the same one-way-edge reason.

Every read here **never raises** — a missing, corrupt, or oddly-shaped file is
equivalent to "no pre-selection", mirroring ``secrets.load_credentials``.
"""

from __future__ import annotations

import json

from code_helper.backends._atomic import atomic_write
from code_helper.services.paths import Paths

__all__ = [
    "load_state",
    "active_selection",
    "set_active_selection",
    "default_wrapper",
    "set_default_wrapper",
    "clear_default_wrapper",
]


def load_state(paths: Paths) -> dict[str, object]:
    """Read ``state.json`` as a mapping of named top-level keys.

    **Never raises.** A missing, unreadable, malformed, or non-object file is
    equivalent to "no pre-selection" — the same never-fails-on-its-way-out
    contract ``secrets.load_credentials`` follows, because this is read on the
    optional UI path where absent state is a normal condition. The caller
    interprets known keys; unknown keys are preserved so future fields survive
    a round-trip. Does NOT migrate the old ``active_provider``/
    ``active_profiles`` shape — see :func:`active_selection` for that.
    """
    path = paths.state_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if isinstance(key, str)}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _write_state(paths: Paths, state: dict) -> None:
    """Persist ``state`` to the state file atomically."""
    atomic_write(paths.state_file(), json.dumps(state))


def active_selection(paths: Paths) -> tuple[str, str] | None:
    """The active ``(provider, profile)`` pair, or ``None`` when unset.

    Raw read — does NOT cross-check that the profile still exists for the
    provider (see ``secrets.valid_active_profile``, which is the sole reader
    expected to do that check). Reads the current ``{"active": {"provider":
    ..., "profile": ...}}`` shape; if that key is absent, falls back to the
    OLD ``active_provider`` + ``active_profiles`` shape so a pre-existing
    ``state.json`` is not silently ignored after an upgrade — that fallback
    is read-only, the old keys are never written again.
    """
    state = load_state(paths)
    active = state.get("active")
    if isinstance(active, dict):
        provider = _string_or_none(active.get("provider"))
        profile = _string_or_none(active.get("profile"))
        if provider and profile:
            return provider, profile

    # Legacy two-key shape, read-only: {"active_provider": p, "active_profiles":
    # {p: profile, ...}}.
    provider = _string_or_none(state.get("active_provider"))
    profiles = state.get("active_profiles")
    if provider and isinstance(profiles, dict):
        profile = _string_or_none(profiles.get(provider))
        if profile:
            return provider, profile
    return None


def set_active_selection(paths: Paths, provider_name: str, profile_name: str) -> None:
    """Record ``(provider_name, profile_name)`` as the active selection.

    Plain read-modify-write through ``atomic_write`` — no ``flock``, because
    losing a pre-selection race is harmless (see module docstring). Always
    writes the current single-pointer shape; any legacy ``active_provider``/
    ``active_profiles`` keys are dropped on the next write rather than kept in
    sync, since keeping two shapes consistent is exactly the duplication this
    schema removes.
    """
    state = load_state(paths)
    state.pop("active_provider", None)
    state.pop("active_profiles", None)
    state["active"] = {"provider": provider_name, "profile": profile_name}
    _write_state(paths, state)


def default_wrapper(paths: Paths, agent_name: str) -> str | None:
    """The saved default-wrapper alias for ``agent_name``, or ``None``.

    Raw read — no staleness check; that lives in ``wrappers.valid_default_wrapper``
    (see module docstring for the one-way-edge reason).
    """
    state = load_state(paths)
    wrappers = state.get("default_wrapper")
    if not isinstance(wrappers, dict):
        return None
    return _string_or_none(wrappers.get(agent_name))


def clear_default_wrapper(paths: Paths, alias: str) -> None:
    """Remove any default pointers to a deleted wrapper alias."""
    state = load_state(paths)
    wrappers = state.get("default_wrapper")
    if not isinstance(wrappers, dict):
        return
    remaining = {agent: value for agent, value in wrappers.items() if value != alias}
    if remaining == wrappers:
        return
    if remaining:
        state["default_wrapper"] = remaining
    else:
        state.pop("default_wrapper", None)
    _write_state(paths, state)


def set_default_wrapper(paths: Paths, agent_name: str, alias: str) -> None:
    """Record ``alias`` as the default wrapper for ``agent_name``.

    Plain read-modify-write through ``atomic_write`` — no ``flock`` (losing a
    pre-selection race is harmless). Sets only the ``agent_name`` slot, leaving
    other agents' slots and every other top-level key untouched. Raw store — no
    validation; staleness is ``wrappers.valid_default_wrapper``'s job.
    """
    state = load_state(paths)
    wrappers = state.get("default_wrapper")
    if not isinstance(wrappers, dict):
        wrappers = {}
    wrappers[agent_name] = alias
    state["default_wrapper"] = wrappers
    _write_state(paths, state)
