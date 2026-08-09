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
from dataclasses import dataclass, replace
from enum import StrEnum

from code_helper.errors import CodeHelperError

__all__ = [
    "ConfigShape",
    "ModelListAPI",
    "BaseUrlPolicy",
    "Agent",
    "Provider",
    "AGENTS",
    "PROVIDERS",
    "get_agent",
    "get_provider",
    "resolve_shape",
    "compatible_providers",
    "with_base_url",
]


class ConfigShape(StrEnum):
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
    #: — this per-alias profile never reads or modifies ``~/.codex/config.toml``
    #: itself. That file has a separate, explicit writer instead: the
    #: ``set-default`` command (``services/codex_default.py``), which patches
    #: only its own managed keys there and touches no per-alias profile. A
    #: provider declaring this shape but with no renderer entry still resolves
    #: here and surfaces as "not implemented yet" from ``render_script`` —
    #: distinct from "incompatible", a different, honest error.
    OPENAI_TOML = "openai-toml"


class ModelListAPI(StrEnum):
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

#: ``token_env_var`` is now interpolated into a generated script UNQUOTED too
#: (``export {token_env_var}={quoted token}`` — the shape's OPENAI_TOML
#: renderer, added for a runtime-``base_url`` provider that needs to carry a
#: secret). The name sits left of ``=``, where shell syntax has no quoting
#: form at all, so it needs the same structural gate as ``agent.binary``
#: rather than relying on registry authors to type a safe value.
_ENV_VAR_RE = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")


class BaseUrlPolicy(StrEnum):
    """Where a :class:`Provider`'s ``base_url`` comes from.

    Most providers (``ollama``, ``zai``) have one true address: it lives in
    the registry and nothing at runtime can override it. A provider whose
    endpoint is the *user's own* (a self-hosted LiteLLM proxy, say) cannot
    ship an address at all — the registry ``base_url`` would just be someone
    else's server. This is the third axis of that distinction, not a second
    meaning bolted onto an empty ``base_url`` string: an empty ``base_url``
    already means "this provider has no address" to ``models_api.list_models``
    (it reports ``has no base URL configured``), and reusing that same empty
    string to also mean "ask the user" would make a registry typo and a
    deliberate runtime-address provider indistinguishable.
    """

    #: The registry ``base_url`` is the only value — never overridable.
    #: ``base_url`` must be non-empty.
    FIXED = "fixed"

    #: No registry ``base_url`` at all; the caller MUST supply one at runtime
    #: (``--base-url`` / a TUI prompt) via :func:`with_base_url`. Registry
    #: ``base_url`` must be empty — a non-empty one here would be a second,
    #: silently-losing source of truth for the same value.
    REQUIRED = "required"

    #: The registry ``base_url`` is a default the caller MAY override at
    #: runtime. Registry ``base_url`` must be non-empty (an empty default is
    #: meaningless — that is what REQUIRED is for). No shipped provider uses
    #: this today; it exists so a future provider with a sensible default
    #: doesn't need a new axis, only this value.
    OVERRIDABLE = "overridable"


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
    #: See :class:`BaseUrlPolicy`. Constrains what ``base_url`` may hold —
    #: enforced at import time by :func:`_validate_provider`.
    base_url_policy: BaseUrlPolicy = BaseUrlPolicy.FIXED
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
    Provider(
        name="litellm",
        # A self-hosted LiteLLM proxy serves BOTH protocols off one host: the
        # Anthropic Messages passthrough (/v1/messages) and an OpenAI-style
        # /v1 endpoint — so it declares both shapes, and _SHAPE_PRIORITY
        # (ANTHROPIC_ENV > OPENAI_TOML) resolves `claude × litellm` to the
        # direct env shape and `codex × litellm` to the TOML profile, with no
        # special-case code needed for either.
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV, ConfigShape.OPENAI_TOML}),
        # REQUIRED, not FIXED: this is the user's own server, not an address
        # this project could ship a default for. base_url is supplied at
        # runtime (--base-url / a TUI prompt) via model.with_base_url — see
        # BaseUrlPolicy and cli/parser.py's --base-url handling.
        base_url="",
        base_url_policy=BaseUrlPolicy.REQUIRED,
        auth="secret",
        token_env_var="LITELLM_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        # "chat", not "responses": LiteLLM's proxy implements
        # /chat/completions across all its backends; /responses is not
        # proxied uniformly for every provider it fronts.
        wire_api="chat",
        description="LiteLLM proxy (user-supplied base URL)",
    ),
)


def _validate_provider(provider: Provider) -> None:
    """Fail on a malformed :class:`Provider` entry.

    Split out from :func:`_validate_registries` so a provider that is
    deliberately built *outside* ``PROVIDERS`` (a test proving
    :class:`BaseUrlPolicy` is enforced rather than merely followed by the two
    shipped providers) can be checked without monkeypatching the module-level
    registry.
    """
    if not _BINARY_RE.match(provider.name):
        raise CodeHelperError(f"invalid provider name in registry: {provider.name!r}")
    if not provider.shapes:
        raise CodeHelperError(f"provider {provider.name!r} declares no config shapes")
    if provider.auth not in ("none", "literal", "secret"):
        raise CodeHelperError(
            f"provider {provider.name!r} has invalid auth {provider.auth!r}"
        )
    # openai_toml_body (render.py) consumes wire_api unconditionally for
    # every provider that can resolve to OPENAI_TOML — an empty or
    # unrecognised value renders a profile Codex rejects at runtime rather
    # than a registry error at import time. The whole selling point of the
    # shape is "a second OpenAI-compatible provider is just a PROVIDERS
    # entry"; catching a missing wire_api here, not at `codex` runtime, is
    # what keeps that promise honest.
    if ConfigShape.OPENAI_TOML in provider.shapes and provider.wire_api not in (
        "responses",
        "chat",
    ):
        raise CodeHelperError(
            f"provider {provider.name!r} declares openai-toml but has "
            f"invalid wire_api {provider.wire_api!r} (must be 'responses' "
            f"or 'chat')"
        )
    # `export {token_env_var}=...` interpolates this name unquoted, left of
    # `=`, in the OPENAI_TOML wrapper for a secret provider — see
    # render.openai_env_key. A provider with `auth != "secret"` never reaches
    # that interpolation, but the gate is unconditional (whenever the field is
    # non-empty) rather than gated on auth, so a future auth-mode change can
    # never silently un-guard it.
    if provider.token_env_var and not _ENV_VAR_RE.match(provider.token_env_var):
        raise CodeHelperError(
            f"provider {provider.name!r} has invalid token_env_var "
            f"{provider.token_env_var!r} (must match [A-Z][A-Z0-9_]*) — it is "
            f"interpolated into a generated script unquoted"
        )
    # A secret-auth provider with an empty token_env_var is a silent
    # authentication hole for OPENAI_TOML specifically: openai_env_key
    # returns "" for an empty token_env_var, so neither the TOML profile's
    # env_key nor the wrapper's `export` line gets written — installation
    # still succeeds (a token can still be resolved/prompted for) but the
    # resulting wrapper starts codex with no credential wired in at all.
    # This is a registry-authoring mistake this import-time check exists to
    # catch (cycle-review re-review finding on PR #12) — no shipped provider
    # hits it today (zai/litellm both carry a real token_env_var), but
    # nothing else would catch a future secret provider added without one.
    if provider.auth == "secret" and not provider.token_env_var:
        raise CodeHelperError(
            f"provider {provider.name!r} declares auth='secret' but has no "
            f"token_env_var — OPENAI_TOML would silently install with no "
            f"credential wired in (see openai_env_key)"
        )
    # See BaseUrlPolicy: exactly one of "registry supplies base_url" / "caller
    # must supply it at runtime" holds per policy — never both, never neither.
    if provider.base_url_policy is BaseUrlPolicy.REQUIRED and provider.base_url:
        raise CodeHelperError(
            f"provider {provider.name!r} declares base_url_policy=REQUIRED but "
            f"also carries a registry base_url {provider.base_url!r} — the "
            f"runtime value passed to with_base_url would silently overwrite it"
        )
    if provider.base_url_policy is not BaseUrlPolicy.REQUIRED and not provider.base_url:
        raise CodeHelperError(
            f"provider {provider.name!r} declares base_url_policy="
            f"{provider.base_url_policy.value!r} but has no registry base_url "
            f"— FIXED has no other source, and an OVERRIDABLE default cannot "
            f"be empty (that is what REQUIRED is for)"
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
        _validate_provider(provider)
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


def with_base_url(provider: Provider, base_url: str | None) -> Provider:
    """The single substitution point for a runtime ``base_url``.

    Every one of ``base_url``'s four readers (``render._render_anthropic_env``,
    ``render.openai_base_url`` inside ``openai_toml_body``,
    ``models_api.list_models``, ``codex_default.resolve_default_patch``) reads
    it off a :class:`Provider` object, never off :class:`WrapperSpec`
    directly — so substituting the provider once, here, at the command's entry
    point (before ``build_spec``, before ``list_models``, before
    ``resolve_token``) is enough to make every one of them see the right
    value. This is deliberately DATA-driven: the branch is on
    ``provider.base_url_policy``, never on ``provider.name`` — the whole point
    of :class:`BaseUrlPolicy` is that a future runtime-address provider needs
    no new code here, only a registry entry.

    Args:
        provider: The provider as looked up from the registry.
        base_url: What the caller supplied (``--base-url`` / a TUI prompt),
            or ``None``/empty if nothing was supplied.

    Returns:
        ``provider`` unchanged (FIXED with nothing supplied, or OVERRIDABLE
        with nothing supplied — the registry default applies), or a copy with
        ``base_url`` replaced (REQUIRED or OVERRIDABLE with a value supplied,
        after :func:`~code_helper.services.naming.validate_base_url`).

    Raises:
        CodeHelperError: a URL was supplied for a FIXED provider; no URL was
            supplied for a REQUIRED provider; or the supplied URL fails
            validation.
    """
    from code_helper.services.naming import validate_base_url

    if provider.base_url_policy is BaseUrlPolicy.FIXED:
        if base_url:
            raise CodeHelperError(
                f"provider {provider.name!r} has a fixed base URL "
                f"({provider.base_url!r}) — --base-url only applies to: "
                + ", ".join(
                    p.name
                    for p in PROVIDERS
                    if p.base_url_policy is not BaseUrlPolicy.FIXED
                )
            )
        return provider

    if not base_url:
        if provider.base_url_policy is BaseUrlPolicy.REQUIRED:
            raise CodeHelperError(
                f"provider {provider.name!r} needs a base URL — pass "
                f"--base-url https://host:port/v1"
            )
        return provider  # OVERRIDABLE, nothing supplied: the default stands.

    validate_base_url(base_url)
    return replace(provider, base_url=base_url)
