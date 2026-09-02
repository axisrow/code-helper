"""Persistent UI pre-selection state (issue #23).

Holds the TUI's active token-profile pre-selection — which profile is selected
by default for ``add``/``edit-token`` — in ``~/.config/codehelper/state.json``.
This is deliberately a SEPARATE module and file from ``services/secrets.py``:

- ``secrets.py`` is built around ``credentials.json`` and holds TOKENS. Its
  schema ``{provider_name: {profile_name: token}}`` cannot grow without a
  versioned migration, because ``load_credentials`` treats every top-level key
  as a provider name (issue #19). ``state.json`` uses named top-level keys
  from the start, so future fields never need a migration.
- ``secrets.py`` is the token store; this file is not. It DOES now hold one
  credential-bearing value (:func:`set_saved_proxy` — a proxy URL may carry
  basic-auth), so that one writer passes ``0o600``; the rest keep the file's
  existing mode.

This module originally shipped with no locking, on the reasoning that "a race
here loses a pre-selection at worst, never a token". That was true while the
file held only the active-profile pointer and the default-wrapper map — both
trivially re-set from the UI that wrote them. :func:`set_saved_proxy` broke
the premise: ``proxy off`` blanks the live values in ``settings.json`` and
banks the address HERE, so this entry is the only copy. Losing it to a
concurrent writer is not "re-pick a profile", it is the address gone for good
with ``proxy on`` left with nothing to restore. Every writer therefore goes
through :func:`_locked_update`, the same ``flock``-for-the-read-modify-write
window ``secrets._locked_update`` uses — see ``tests/test_state_concurrency.py``
for the reproduction.

What an UNAVAILABLE lock means differs per writer, so it is a per-call
decision rather than one module-wide policy. The pre-selection writers keep
the never-raises posture (an unlocked write beats crashing ``add``/
``edit-token`` over a pointer the user can re-pick); :func:`set_saved_proxy`
passes ``required=True`` and refuses, because an unserialized write of the
only copy of an address reports success while silently risking its loss.

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

import contextlib
import json

from codehelper.backends._atomic import atomic_write, file_lock
from codehelper.errors import CodeHelperError
from codehelper.services.paths import Paths

__all__ = [
    "load_state",
    "active_selection",
    "set_active_selection",
    "default_wrapper",
    "set_default_wrapper",
    "clear_default_wrapper",
    "saved_proxy",
    "set_saved_proxy",
    "context_window",
    "set_context_window",
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


@contextlib.contextmanager
def _locked_update(paths: Paths, *, required: bool = False):
    """Serialize one read-modify-write cycle against ``state.json``.

    Every writer here goes through this ONE context manager rather than each
    pairing :func:`load_state` with :func:`_write_state` independently — the
    same "one decision point" shape ``secrets._locked_update`` uses, so the
    writers cannot drift on how they serialize.

    Without it the module's read-modify-write window is open: a writer that
    loaded before another committed writes its stale snapshot back on top,
    silently dropping the other's update (reproduced in
    ``tests/test_state_concurrency.py``). ``atomic_write`` prevents a TORN
    file, never a lost update — those are different problems.

    What an unavailable lock means depends on WHAT is being written, so
    ``required`` is the caller's call rather than one module-wide policy:

    * ``required=False`` (default) — degrade to an unlocked write. The value
      is re-pickable UI state (the active profile, a default wrapper), and
      this module's never-raises contract matters more than serializing it:
      turning an unavailable lock into a crash would break ``add``/
      ``edit-token`` over a pointer the user can simply set again. Strictly
      no worse than the pre-lock behaviour this module shipped with.
    * ``required=True`` — refuse. :func:`set_saved_proxy` writes the ONLY
      copy of the proxy address, and an unserialized write of that is worse
      than no write at all: a silently-lost update still reports success, and
      the value is not one the user can reconstruct. Its caller
      (``services/proxy.py``) is already in a position to surface the error.

    Acquisition is the only best-effort part; exceptions raised by the
    caller's body always propagate untouched.

    Raises:
        CodeHelperError: ``required=True`` and the lock could not be taken.
    """
    # Acquire OUTSIDE the yield. Wrapping the yield in `try/except OSError`
    # would also swallow an OSError raised by the CALLER's body and then
    # yield a second time — a "generator didn't stop after throw()" crash,
    # and worse, it would run the caller's block twice.
    try:
        lock = file_lock(paths.state_file())
        lock.__enter__()
    except OSError as exc:
        if required:
            raise CodeHelperError(
                f"{paths.state_file()} could not be locked ({exc}) — refusing "
                f"to write the saved proxy address unserialized, because a "
                f"concurrent write would silently discard it and this is the "
                f"only copy; fix the permissions on that directory and retry"
            ) from exc
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.__exit__(None, None, None)


def _write_state(paths: Paths, state: dict, *, mode: int | None = None) -> None:
    """Persist ``state`` to the state file atomically.

    ``mode=None`` keeps this module's default posture — no secrets, so no
    explicit permissions (``atomic_write`` preserves an existing file's mode).
    A caller persisting something credential-bearing passes ``0o600``; see
    :func:`set_saved_proxy`, the only such caller.
    """
    atomic_write(paths.state_file(), json.dumps(state), mode=mode)


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
    with _locked_update(paths):
        state = load_state(paths)
        state.pop("active_provider", None)
        state.pop("active_profiles", None)
        state["active"] = {"provider": provider_name, "profile": profile_name}
        _write_state(paths, state)


def saved_proxy(paths: Paths) -> str | None:
    """The proxy address remembered for the next ``proxy on``, or ``None``.

    A third independent top-level pointer, alongside ``active`` and
    ``default_wrapper`` — the named-key schema this module adopted from the
    start exists precisely so a new field costs no migration.

    Turning the proxy off blanks the values in ``settings.json`` rather than
    deleting the keys (see ``services/proxy.py`` on why), which means the
    address itself would be gone and ``proxy on`` would have nothing to
    restore. This is where it survives the round trip. Raw read, no
    validation: ``proxy.validate_proxy_url`` is the one gate, and it runs
    before any write rather than on every read.
    """
    state = load_state(paths)
    proxy = state.get("proxy")
    if not isinstance(proxy, dict):
        return None
    return _string_or_none(proxy.get("url"))


def set_saved_proxy(paths: Paths, url: str) -> None:
    """Remember ``url`` as the address to restore on the next ``proxy on``.

    The one writer here that may persist a CREDENTIAL: Claude Code documents
    basic-auth inside the proxy URL (``http://user:pass@host:8118``), so this
    value can carry a password even though nothing else in ``state.json``
    does. It is therefore written ``0o600`` — the same mode
    ``claude_settings`` uses for exactly this reason.

    The explicit mode is load-bearing, not belt-and-braces: ``atomic_write``
    PRESERVES an existing file's mode, so a ``state.json`` that already sat at
    ``0o644`` (created before this field existed, or by a looser umask) would
    otherwise keep a proxy password group/world-readable. Passing the mode
    tightens the file on the write that introduces the secret.
    """
    with _locked_update(paths, required=True):
        state = load_state(paths)
        state["proxy"] = {"url": url}
        _write_state(paths, state, mode=0o600)


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
    with _locked_update(paths):
        state = load_state(paths)
        wrappers = state.get("default_wrapper")
        if not isinstance(wrappers, dict):
            return
        remaining = {
            agent: value for agent, value in wrappers.items() if value != alias
        }
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
    with _locked_update(paths):
        state = load_state(paths)
        wrappers = state.get("default_wrapper")
        if not isinstance(wrappers, dict):
            wrappers = {}
        wrappers[agent_name] = alias
        state["default_wrapper"] = wrappers
        _write_state(paths, state)


def context_window(paths: Paths, model: str) -> int | None:
    """The recorded context-window answer for ``model``, or ``None``.

    ``0`` is a REAL answer ("no declaration" — issue #83), never normalized
    to ``None``: the sentinel is what keeps an answered model from being
    asked again on the next add/switch. A malformed slot (non-dict map, a
    string, a float, ``True``) reads as unrecorded — a corrupt value asks
    again rather than lying.
    """
    state = load_state(paths)
    windows = state.get("context_windows")
    if not isinstance(windows, dict):
        return None
    value = windows.get(model)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def set_context_window(paths: Paths, model: str, value: int) -> None:
    """Record ``value`` as the context-window answer for ``model``.

    Same posture as :func:`set_default_wrapper`: a re-pickable pre-selection,
    so a lost write is harmless (the model is simply asked again). ``0``
    records an explicit "no declaration". Sets only the ``model`` slot,
    leaving other models' slots and every other top-level key untouched.
    """
    with _locked_update(paths):
        state = load_state(paths)
        windows = state.get("context_windows")
        if not isinstance(windows, dict):
            windows = {}
        windows[model] = value
        state["context_windows"] = windows
        _write_state(paths, state)
