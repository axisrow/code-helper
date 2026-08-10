"""Persistent UI pre-selection state (issue #23).

Holds the TUI's active token-profile pre-selection — which profile is selected
by default for ``add``/``edit-token`` — in ``~/.config/code-helper/state.json``.
This is deliberately a SEPARATE module and file from ``services/secrets.py``:

- ``secrets.py`` is built around ``credentials.json`` and holds TOKENS. Its
  schema ``{provider_name: {profile_name: token}}`` cannot grow without a
  versioned migration, because ``load_credentials`` treats every top-level key
  as a provider name (issue #19). ``state.json`` uses named top-level keys
  (``active_profiles``, ``active_provider``) from the start, so future fields
  never need a migration.
- This module holds NO secrets, so no ``0o600`` and no ``flock``. A race here
  loses a pre-selection at worst, never a token — unlike
  :func:`secrets._locked_update`, whose serialization protects the only
  persistent copy of a credential. Write is a plain read-modify-write through
  :func:`code_helper.backends._atomic.atomic_write`.

A saved profile name can go stale (renamed via ``secrets.rename_profile`` or
dropped via ``secrets.invalidate_cached_credential``), so readers must
cross-check against the live ``secrets.profile_names``. That cross-check lives
in ``secrets.valid_active_profile`` (NOT here): this module must not depend on
``secrets.py``, while ``secrets.py`` importing ``state.py`` is a one-way edge
that creates no cycle.

Every read here **never raises** — a missing, corrupt, or oddly-shaped file is
equivalent to "no pre-selection", mirroring ``secrets.load_credentials``.
"""

from __future__ import annotations

import json

from code_helper.backends._atomic import atomic_write
from code_helper.services.paths import Paths

__all__ = [
    "load_state",
    "active_profile",
    "set_active_profile",
    "active_provider",
    "set_active_provider",
]


def load_state(paths: Paths) -> dict[str, object]:
    """Read ``state.json`` as a mapping of named top-level keys.

    **Never raises.** A missing, unreadable, malformed, or non-object file is
    equivalent to "no pre-selection" — the same never-fails-on-its-way-out
    contract ``secrets.load_credentials`` follows, because this is read on the
    optional UI path where absent state is a normal condition. Only string
    values are retained (a stray non-string entry must not cost the user the
    rest of the state). The caller interprets known keys; unknown keys are
    preserved so future fields survive a round-trip.
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


def active_profile(paths: Paths, provider_name: str) -> str | None:
    """The stored active profile for ``provider_name``, or ``None``.

    Raw read — does NOT cross-check that the profile still exists (see
    ``secrets.valid_active_profile``). ``None`` when unset or non-string.
    """
    profiles = load_state(paths).get("active_profiles")
    if not isinstance(profiles, dict):
        return None
    return _string_or_none(profiles.get(provider_name))


def set_active_profile(paths: Paths, provider_name: str, profile_name: str) -> None:
    """Record ``profile_name`` as the active profile for ``provider_name``.

    Plain read-modify-write through ``atomic_write`` — no ``flock``, because
    losing a pre-selection race is harmless (see module docstring).
    """
    state = load_state(paths)
    profiles = state.get("active_profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[provider_name] = profile_name
    state["active_profiles"] = profiles
    atomic_write(paths.state_file(), json.dumps(state))


def active_provider(paths: Paths) -> str | None:
    """The stored active provider name, or ``None`` when unset/non-string."""
    return _string_or_none(load_state(paths).get("active_provider"))


def set_active_provider(paths: Paths, name: str) -> None:
    """Record ``name`` as the active provider. Plain atomic read-modify-write."""
    state = load_state(paths)
    state["active_provider"] = name
    atomic_write(paths.state_file(), json.dumps(state))
