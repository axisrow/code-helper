"""User-defined agents — CLI integrations beyond the built-in :data:`~code_helper.
services.model.AGENTS` registry.

``AGENTS`` is a frozen, read-only, module-level tuple (see ``model.py``) — by
design, since ``render.py`` relies on every ``agent.binary`` being a registry
constant safe to interpolate unquoted into a generated script. Letting a user
add "one more CLI integration ``ollama launch`` happens to support" therefore
needs a genuinely separate storage layer, not a mutation of that tuple.

Scope is deliberately narrow: a user-defined agent is always
:attr:`~code_helper.services.model.ConfigShape.OLLAMA_LAUNCH`-only — the shape
every non-``claude``/``codex`` CLI integration in the built-in registry uses
(see the 13 entries at the bottom of ``AGENTS``). ``ANTHROPIC_ENV``/
``OPENAI_TOML`` need a renderer contract this module has no way to supply, so
they stay out.

Persistence: ``~/.config/code-helper/agents.json``
(:meth:`~code_helper.services.paths.Paths.agents_file`), shape
``{"agents": [{"name": ..., "binary": ..., "description": ...}, ...]}``.
Written through :func:`~code_helper.backends._atomic.atomic_write`, the only
code in this project that touches the filesystem.

Contract that matters, all enforced by :func:`add_user_agent`:

- A user entry must NOT shadow a built-in agent name, nor an existing user
  agent name — rejected outright, never silently overridden. Silently
  overriding ``claude`` would change what every ``claude``-agent wrapper
  execs.
- ``name``/``binary`` are validated through
  :func:`~code_helper.services.model.validate_agent_binary` — the exact same
  gate the built-in registry itself is checked against at import time.
  ``binary`` is interpolated UNQUOTED into a generated script (see
  ``render.py``); skipping this validation is a shell-injection hole, not a
  cosmetic one.
- The chosen ``name``/``binary`` must not collide with
  :data:`~code_helper.services.naming.RESERVED_ALIASES` — the same hazard
  (infinite recursion / permanent shadowing on ``PATH``) that reserves every
  built-in agent's binary also applies to a newly-added one.

:func:`load_user_agents` never raises: a missing or corrupt file degrades to
"no user agents", the same never-fails-on-its-way-out contract
``services/secrets.py``'s ``load_credentials`` and
``services/models_api.py``'s ``list_models`` already follow.

Honest degradation: a user-added agent gets full ``add``/``list``/``remove``
support (it is a real :class:`~code_helper.services.model.Agent`), but NO
chipset row in the TUI — there is no live-patchable config this project knows
how to read for it (see ``cli/tui.py``'s ``_AgentBackend`` docstring). That is
correct, not a gap to fill.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass
from pathlib import Path

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.model import AGENTS, Agent, ConfigShape, validate_agent_binary
from code_helper.services.naming import RESERVED_ALIASES
from code_helper.services.paths import Paths

__all__ = [
    "load_user_agents",
    "load_user_agents_strict",
    "all_agents",
    "get_agent",
    "add_user_agent",
]


@dataclass(frozen=True)
class _RawUserAgent:
    name: str
    binary: str
    description: str


def _parse_entry(entry: object) -> _RawUserAgent | None:
    """Return a validated ``_RawUserAgent``, or ``None`` to skip a bad one.

    A single malformed entry (a stray non-dict, a missing field, a wrong
    type) must not cost the user every other agent in the file — the same
    per-entry-skip posture ``load_credentials`` takes for a bad credential.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    binary = entry.get("binary", name)
    description = entry.get("description", "")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(binary, str) or not binary:
        return None
    if not isinstance(description, str):
        return None
    return _RawUserAgent(name=name, binary=binary, description=description)


def _entries_to_agents(entries: list[object]) -> tuple[Agent, ...]:
    """Turn a validated ``agents`` list into a tuple of :class:`Agent`.

    Per-entry skip, shared by both readers: a stray bad entry (a non-dict, a
    missing field, a name/binary that fails the regex) must not cost the user
    every other agent in the file. Defense in depth: entries are validated
    again on the way IN by :func:`add_user_agent`, but a hand-edited file
    bypasses that gate, so a binary/name that fails the same regex checked at
    write time is skipped here too rather than trusted.
    """
    agents: list[Agent] = []
    seen_names: set[str] = set()
    for raw_entry in entries:
        parsed = _parse_entry(raw_entry)
        if parsed is None:
            continue
        try:
            validate_agent_binary(parsed.name)
            validate_agent_binary(parsed.binary)
        except CodeHelperError:
            continue
        if parsed.name in seen_names:
            continue
        seen_names.add(parsed.name)
        agents.append(
            Agent(
                name=parsed.name,
                binary=parsed.binary,
                shapes=frozenset({ConfigShape.OLLAMA_LAUNCH}),
                description=parsed.description,
            )
        )
    return tuple(agents)


def load_user_agents(paths: Paths) -> tuple[Agent, ...]:
    """Read ``agents.json`` as a tuple of :class:`Agent`.

    **Never raises.** A missing, unreadable, malformed, or oddly-shaped file
    degrades to "no user agents" — see the module docstring. Every returned
    entry is :attr:`ConfigShape.OLLAMA_LAUNCH`-only, by construction: this is
    the sole shape a user-defined agent can declare (see the module
    docstring's Scope section).
    """
    path = paths.agents_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    try:
        data = json.loads(raw)
    except ValueError:
        return ()
    if not isinstance(data, dict):
        return ()
    entries = data.get("agents")
    if not isinstance(entries, list):
        return ()
    return _entries_to_agents(entries)


def load_user_agents_strict(paths: Paths) -> tuple[Agent, ...]:
    """Read ``agents.json`` for MUTATION, failing closed on ANY corruption.

    The permissive :func:`load_user_agents` (never raises, degrades to empty,
    skips bad entries) is right for listing/UI, but wrong as the read side of
    a read-modify-write: a malformed file would read as "no agents" and the
    next add would overwrite every prior entry with just the new one — silent,
    irreversible deletion. This strict variant raises on a file that cannot be
    parsed AND on any individual invalid/duplicate entry, so a corrupt registry
    is surfaced instead of destroyed. A MISSING file is fine (a fresh
    registry); an unreadable one is not (we cannot know what it holds).

    The per-entry strictness matters as much as the file shape: a mutation
    that serializes only the surviving entries would silently drop a malformed
    entry on the next write — data loss dressed up as cleanup. Fail closed
    instead, and let the user repair the file.
    """
    path = paths.agents_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise CodeHelperError(f"cannot read agents registry: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CodeHelperError(f"agents registry is corrupt: {exc}") from exc
    if not isinstance(data, dict):
        raise CodeHelperError("agents registry is corrupt: expected an object")
    entries = data.get("agents")
    if not isinstance(entries, list):
        raise CodeHelperError("agents registry is corrupt: expected an 'agents' list")

    agents: list[Agent] = []
    seen_names: set[str] = set()
    for raw_entry in entries:
        parsed = _parse_entry(raw_entry)
        if parsed is None:
            raise CodeHelperError("agents registry contains an invalid entry")
        try:
            validate_agent_binary(parsed.name)
            validate_agent_binary(parsed.binary)
        except CodeHelperError:
            raise CodeHelperError(
                f"agents registry contains an invalid entry: {parsed.name!r}"
            ) from None
        if parsed.name in seen_names:
            raise CodeHelperError(
                f"agents registry contains a duplicate entry: {parsed.name!r}"
            )
        seen_names.add(parsed.name)
        agents.append(
            Agent(
                name=parsed.name,
                binary=parsed.binary,
                shapes=frozenset({ConfigShape.OLLAMA_LAUNCH}),
                description=parsed.description,
            )
        )
    return tuple(agents)


def all_agents(paths: Paths) -> tuple[Agent, ...]:
    """Built-in :data:`AGENTS` first, user-defined agents appended.

    Merge order matters: a user entry can never shadow a built-in one (see
    :func:`add_user_agent`), so appending is safe — a name lookup that walks
    this tuple in order always finds the built-in definition first.
    """
    return AGENTS + load_user_agents(paths)


def get_agent(paths: Paths, name: str) -> Agent:
    """Look up an agent by name across built-ins and user-defined agents.

    Raises:
        CodeHelperError: unknown name (lists the known ones).
    """
    agents = all_agents(paths)
    for agent in agents:
        if agent.name == name:
            return agent
    known = ", ".join(a.name for a in agents)
    raise CodeHelperError(f"unknown agent: {name} (known: {known})")


def _agents_lock(paths: Paths) -> Path:
    """Return the lock-file path guarding ``agents.json``.

    A SEPARATE file from ``agents.json`` itself: ``atomic_write`` replaces the
    destination inode, so a lock held on ``agents.json`` would silently point
    at the stale pre-replace inode after the first write. The lock file is
    never replaced, only flock-ed, so its inode is stable for the process's
    lifetime.
    """
    return paths.agents_file().with_suffix(".json.lock")


def add_user_agent(
    paths: Paths, name: str, binary: str | None = None, description: str = ""
) -> Agent:
    """Validate, then persist, a new user-defined agent.

    ``binary`` defaults to ``name`` — the common case, per
    :attr:`~code_helper.services.model.Agent.binary`'s docstring: for an
    ``OLLAMA_LAUNCH``-only agent the executable name and the ``ollama
    launch`` integration name are the same thing for every built-in entry
    today.

    The read-modify-write (load → validate → replace) runs under an exclusive
    ``fcntl.flock`` on a sibling lock file, so two concurrent invocations
    (two TUI/process instances) cannot each read the same registry, both pass
    the duplicate checks, and the later writer silently delete the earlier
    agent. ``atomic_write`` prevents torn JSON, not this lost-update race; the
    lock closes that gap.

    Raises:
        CodeHelperError: ``name``/``binary`` fails
            :func:`~code_helper.services.model.validate_agent_binary`, OR
            collides with a built-in agent, an existing user agent, or
            :data:`~code_helper.services.naming.RESERVED_ALIASES`.
    """
    binary = binary or name
    validate_agent_binary(name)
    validate_agent_binary(binary)

    lock_path = _agents_lock(paths)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Strict read: a corrupt registry must fail closed here, not read
            # as "no agents" and be overwritten with just the new entry.
            user_agents = load_user_agents_strict(paths)
            existing = AGENTS + user_agents
            existing_names = {a.name for a in existing}
            existing_binaries = {a.binary for a in existing}
            if name in existing_names:
                raise CodeHelperError(f"an agent named {name!r} already exists")
            # RESERVED_ALIASES already contains every built-in binary, but
            # checking it explicitly (rather than relying solely on
            # existing_binaries) also covers "code-helper" itself and stays
            # correct even if that set's derivation ever changes shape.
            if name in RESERVED_ALIASES or binary in RESERVED_ALIASES:
                raise CodeHelperError(
                    f"{name!r} collides with a reserved name — an agent named "
                    f"after an existing binary on PATH would either re-invoke "
                    f"itself forever or permanently shadow the real one"
                )
            if binary in existing_binaries:
                raise CodeHelperError(
                    f"an agent with binary {binary!r} already exists — two "
                    f"agents sharing a binary would make wrapper generation "
                    f"ambiguous"
                )

            agent = Agent(
                name=name,
                binary=binary,
                shapes=frozenset({ConfigShape.OLLAMA_LAUNCH}),
                description=description,
            )

            payload = {
                "agents": [
                    {"name": a.name, "binary": a.binary, "description": a.description}
                    for a in (*user_agents, agent)
                ]
            }
            atomic_write(
                paths.agents_file(),
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                mode=None,
            )
            return agent
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
