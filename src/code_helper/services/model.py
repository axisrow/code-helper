"""The three axes a wrapper is built from: **agent × provider × model**.

The old model was a flat ``WrapperSpec`` registry: each wrapper hard-coded both
*which agent* it launched and *how that agent reached a backend*. The three
shipped presets were already points in an agent×provider matrix — the axes just
had no names:

===========  =======  ==========================  =====================
preset       agent    provider                    mechanism
===========  =======  ==========================  =====================
deepseek     claude   ollama (direct HTTP)        ``ANTHROPIC_*`` env
glm          claude   z.ai                        ``ANTHROPIC_*`` env
glm-ollama   claude   ollama (via ``launch``)     ``ollama launch``
===========  =======  ==========================  =====================

Naming the axes is what makes new combinations expressible as *data* instead of
new code.

Compatibility is NOT a hand-written ``(agent, provider) -> bool`` table
-----------------------------------------------------------------------
The two axes are asymmetric: an **agent** can only be configured in certain
ways (Claude Code reads ``ANTHROPIC_*`` env vars; Codex reads a TOML profile),
and a **provider** can only present itself in certain ways (z.ai speaks the
Anthropic protocol; NVIDIA speaks OpenAI's). So both sides declare a set of
:class:`ConfigShape` they support, and compatibility is simply a non-empty
intersection.

That is worth the indirection because it makes the impossible combinations fall
out of the data rather than being enumerated:

- ``claude × ollama``  → ``{ANTHROPIC_ENV, OLLAMA_LAUNCH}`` — two ways, pick by priority
- ``codex  × ollama``  → ``{OLLAMA_LAUNCH, OPENAI_TOML}`` — priority picks the TOML
  profile (no extra binary on ``PATH``); the launcher remains reachable via
  ``--shape ollama-launch``.
- ``claude × z.ai``    → ``{ANTHROPIC_ENV}``
- ``claude × <openai-only provider>`` → ``∅`` → **rejected with no special-case code**

The last line is the one that matters: an OpenAI-compatible endpoint genuinely
cannot drive Claude Code (verified: such endpoints return 404 for
``/v1/messages``), and the model says so on its own. Adding a provider later
cannot silently create a broken combination.

The same :class:`ConfigShape` that answers "is this allowed?" also selects the
renderer (see ``services/render.py``), so the two can never disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from code_helper.errors import CodeHelperError

__all__ = [
    "ConfigShape",
    "ModelListAPI",
    "Agent",
    "Provider",
    "AGENTS",
    "PROVIDERS",
    "get_agent",
    "get_provider",
    "resolve_shape",
    "compatible_providers",
]


class ConfigShape(str, Enum):
    """How an agent is pointed at a backend — the hidden third axis.

    Both an :class:`Agent` (what it can consume) and a :class:`Provider` (what
    it can present) declare a set of these; their intersection is what makes a
    combination possible.
    """

    #: ``ANTHROPIC_BASE_URL``/``ANTHROPIC_AUTH_TOKEN``/``ANTHROPIC_DEFAULT_*``
    #: env vars around a ``claude`` invocation. Requires an endpoint speaking
    #: the Anthropic protocol (``/v1/messages``).
    ANTHROPIC_ENV = "anthropic-env"

    #: Delegate to ``ollama launch <agent> --model <model>``. Ollama sets the
    #: agent's own environment/config itself, so the wrapper stays one line.
    OLLAMA_LAUNCH = "ollama-launch"

    #: Codex's ``[model_providers.X]`` TOML profile (``base_url`` +
    #: ``wire_api`` + ``env_key``) plus ``-c`` overrides.
    #:
    #: Implemented for ``codex × ollama`` (see ``render.openai_toml_body`` /
    #: ``openai_catalog_body``): the renderer writes its own
    #: ``~/.codex/<alias>.config.toml`` and launches ``codex --profile <alias>``
    #: — ``~/.codex/config.toml`` is never read or modified. A provider
    #: declaring this shape but with no renderer entry still resolves here and
    #: surfaces as "not implemented yet" from ``render_script`` — distinct from
    #: "incompatible", a different, honest error.
    OPENAI_TOML = "openai-toml"


class ModelListAPI(str, Enum):
    """Which HTTP shape lists a provider's models (see ``services/models_api``)."""

    #: ``GET {base}/api/tags`` -> ``{"models": [{"name": ...}]}``
    OLLAMA_TAGS = "ollama-tags"

    #: ``GET {base}/models`` -> ``{"data": [{"id": ...}]}`` (OpenAI standard)
    OPENAI_V1 = "openai-v1"

    #: No machine-readable listing — the user types the model name.
    NONE = "none"


#: Preference order when an agent and provider share more than one shape.
#: Deterministic on purpose: a "smart" choice that varies by context would make
#: the generated script unpredictable. Direct HTTP beats delegating to another
#: launcher because it needs no extra binary on ``PATH``.
_SHAPE_PRIORITY: tuple[ConfigShape, ...] = (
    ConfigShape.ANTHROPIC_ENV,
    ConfigShape.OPENAI_TOML,
    ConfigShape.OLLAMA_LAUNCH,
)

#: An agent's binary name is interpolated into a generated script UNQUOTED (see
#: ``render.py``), so it must be a registry constant of a boring shape — not
#: user input. Enforced at import time by :func:`_validate_registries`.
_BINARY_RE = re.compile(r"\A[a-z][a-z0-9_-]*\Z")


@dataclass(frozen=True)
class Agent:
    """A coding agent (the thing that actually runs) — one axis."""

    name: str
    #: Executable name. Interpolated into generated scripts unquoted; see
    #: :data:`_BINARY_RE`.
    binary: str
    #: Config shapes this agent can CONSUME.
    shapes: frozenset[ConfigShape]
    description: str = ""


@dataclass(frozen=True)
class Provider:
    """A model backend (endpoint + auth) — the other axis."""

    name: str
    #: Config shapes this provider can PRESENT itself as.
    shapes: frozenset[ConfigShape]
    base_url: str = ""
    #: ``"literal"`` — a non-secret token baked into the script (Ollama's local
    #: daemon accepts ``"ollama"``); ``"secret"`` — resolved from env/prompt and
    #: the script is written 0o700; ``"none"`` — the launcher authenticates.
    auth: str = "none"
    auth_value: str = ""
    token_env_var: str = ""
    model_list_api: ModelListAPI = ModelListAPI.NONE
    #: Only when the listing lives somewhere other than ``base_url``.
    model_list_url: str = ""
    #: For :attr:`ConfigShape.OPENAI_TOML`: ``"responses"`` or ``"chat"``.
    wire_api: str = ""
    description: str = ""


AGENTS: tuple[Agent, ...] = (
    Agent(
        name="claude",
        binary="claude",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV, ConfigShape.OLLAMA_LAUNCH}),
        description="Claude Code",
    ),
    Agent(
        name="codex",
        binary="codex",
        # Codex cannot be configured through environment variables at all —
        # there is no OPENAI_BASE_URL equivalent; it needs a TOML profile. So
        # reaching a non-Ollama backend requires OPENAI_TOML, which is why
        # `codex × z.ai` is (correctly) impossible today.
        shapes=frozenset({ConfigShape.OPENAI_TOML, ConfigShape.OLLAMA_LAUNCH}),
        description="OpenAI Codex CLI",
    ),
)


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="ollama",
        # Three shapes: the daemon serves the Anthropic protocol directly (so
        # `claude` can point straight at it — this is the `deepseek` preset),
        # `ollama launch` can configure an agent for us (the `glm-ollama`
        # preset), and an OpenAI-compatible `/v1` endpoint feeds Codex's own
        # TOML profile (codex × ollama via OPENAI_TOML — no launcher binary
        # needed). Same provider, three connection mechanisms.
        shapes=frozenset(
            {
                ConfigShape.ANTHROPIC_ENV,
                ConfigShape.OLLAMA_LAUNCH,
                ConfigShape.OPENAI_TOML,
            }
        ),
        base_url="http://127.0.0.1:11434",
        auth="literal",
        auth_value="ollama",
        model_list_api=ModelListAPI.OLLAMA_TAGS,
        # Consumed by the OPENAI_TOML renderer (`wire_api` in the profile).
        wire_api="responses",
        description="local Ollama daemon",
    ),
    Provider(
        name="zai",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="https://api.z.ai/api/anthropic",
        auth="secret",
        token_env_var="ZAI_API_KEY",
        # The Anthropic-compatible path exposes no OpenAI-style model list.
        model_list_api=ModelListAPI.NONE,
        description="Z.ai (Anthropic-compatible)",
    ),
)


def _validate_registries() -> None:
    """Fail at import time on a malformed registry entry.

    Cheap, and it upholds the invariant ``render.py`` relies on: an agent's
    ``binary`` is safe to interpolate unquoted because it can only ever be a
    lowercase identifier from this file.
    """
    for agent in AGENTS:
        if not _BINARY_RE.match(agent.binary):
            raise CodeHelperError(f"invalid agent binary in registry: {agent.binary!r}")
        if not agent.shapes:
            raise CodeHelperError(f"agent {agent.name!r} declares no config shapes")
    for provider in PROVIDERS:
        if not provider.shapes:
            raise CodeHelperError(
                f"provider {provider.name!r} declares no config shapes"
            )
        if provider.auth not in ("none", "literal", "secret"):
            raise CodeHelperError(
                f"provider {provider.name!r} has invalid auth {provider.auth!r}"
            )
    for label, names in (
        ("agent", [a.name for a in AGENTS]),
        ("provider", [p.name for p in PROVIDERS]),
    ):
        if len(names) != len(set(names)):
            raise CodeHelperError(f"duplicate {label} name in registry")


_validate_registries()


def get_agent(name: str) -> Agent:
    """Look up an agent by name.

    Raises:
        CodeHelperError: unknown name (lists the known ones).
    """
    for agent in AGENTS:
        if agent.name == name:
            return agent
    known = ", ".join(a.name for a in AGENTS)
    raise CodeHelperError(f"unknown agent: {name} (known: {known})")


def get_provider(name: str) -> Provider:
    """Look up a provider by name.

    Raises:
        CodeHelperError: unknown name (lists the known ones).
    """
    for provider in PROVIDERS:
        if provider.name == name:
            return provider
    known = ", ".join(p.name for p in PROVIDERS)
    raise CodeHelperError(f"unknown provider: {name} (known: {known})")


def resolve_shape(
    agent: Agent, provider: Provider, *, preferred: ConfigShape | None = None
) -> ConfigShape:
    """Pick the config shape connecting ``agent`` to ``provider``.

    The single point where compatibility is decided. Pure, no IO — safe to call
    from a menu to filter choices.

    Args:
        agent: The agent to run.
        provider: The backend to reach.
        preferred: Force a specific shape when several are possible (the CLI's
            ``--shape``). Must be supported by both sides.

    Returns:
        The chosen shape. Note this may be a shape with no renderer yet
        (:attr:`ConfigShape.OPENAI_TOML`) — "possible in principle" and
        "implemented" are deliberately separate questions, answered here and in
        ``render.py`` respectively, with different error messages.

    Raises:
        CodeHelperError: no shared shape, or ``preferred`` is not shared.
    """
    common = agent.shapes & provider.shapes
    if not common:
        alternatives = [p.name for p in PROVIDERS if p.shapes & agent.shapes]
        hint = (
            f" — {agent.name} works with: {', '.join(alternatives)}"
            if alternatives
            else ""
        )
        raise CodeHelperError(
            f"{agent.name} cannot use provider {provider.name}: no common "
            f"configuration mechanism{hint}"
        )

    if preferred is not None:
        if preferred not in common:
            shared = ", ".join(sorted(s.value for s in common))
            raise CodeHelperError(
                f"{agent.name} + {provider.name} cannot use shape "
                f"{preferred.value} (available: {shared})"
            )
        return preferred

    for shape in _SHAPE_PRIORITY:
        if shape in common:
            return shape
    # Unreachable while _SHAPE_PRIORITY covers ConfigShape; guards a future
    # member added to the enum but forgotten in the priority tuple.
    raise CodeHelperError(
        f"no prioritised shape for {agent.name} + {provider.name}"
    )  # pragma: no cover


def compatible_providers(agent: Agent) -> list[Provider]:
    """Providers that ``agent`` can actually use — for menus and ``list matrix``.

    Read-only and IO-free, so the TUI may call it to avoid offering a
    combination that would only fail later.
    """
    return [p for p in PROVIDERS if p.shapes & agent.shapes]
