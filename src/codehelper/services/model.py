"""The three axes a wrapper is built from: **agent × provider × model**.

The old model was a flat ``WrapperSpec`` registry: each wrapper hard-coded both
*which agent* it launched and *how that agent reached a backend*. The three
shipped presets were already points in an agent×provider matrix — the axes just
had no names:

===========  =======  ==========================  =====================
preset       agent    provider                    mechanism
===========  =======  ==========================  =====================
deepseek-ollama  claude   ollama (direct HTTP)    ``ANTHROPIC_*`` env
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

from codehelper.errors import CodeHelperError

__all__ = [
    "ConfigShape",
    "ModelListAPI",
    "BaseUrlPolicy",
    "AuthPolicy",
    "Agent",
    "Provider",
    "AGENTS",
    "PROVIDERS",
    "get_agent",
    "get_provider",
    "validate_agent_binary",
    "resolve_shape",
    "compatible_providers",
    "switchable_providers",
    "with_base_url",
    "with_auth",
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
    #: Implemented for ``codex × ollama`` (see ``render.openai_toml_body``):
    #: the renderer writes its own
    #: ``~/.codex/<alias>.config.toml`` and launches ``codex --profile <alias>``
    #: — this per-alias profile never reads or modifies ``~/.codex/config.toml``
    #: itself. That file has a separate, explicit writer instead: the
    #: ``set-default`` command (``services/codex_default.py``), which patches
    #: only its own managed keys there and touches no per-alias profile. A
    #: provider declaring this shape but with no renderer entry still resolves
    #: here and surfaces as "not implemented yet" from ``render_script`` —
    #: distinct from "incompatible", a different, honest error.
    OPENAI_TOML = "openai-toml"

    #: Patches the ``env`` block of Claude Code's OWN ``~/.claude/settings.json``
    #: in place (``services/claude_settings.py``, the ``switch`` command) —
    #: instead of generating a wrapper script, it changes what an ALREADY
    #: RUNNING ``claude`` process does on its next prompt, because Claude Code
    #: re-reads that file between prompts.
    #:
    #: Deliberately declared by no :class:`Agent`: this is not a mechanism an
    #: agent is launched *through* (the thing ``ConfigShape`` otherwise always
    #: means) — it is consumed by an agent re-reading its own settings file
    #: while already running. Keeping it off every ``Agent.shapes`` is what
    #: guarantees, structurally rather than by ``_SHAPE_PRIORITY`` ordering,
    #: that ``resolve_shape``/``compatible_providers``/wrapper generation are
    #: unaffected by this shape's existence — see
    #: ``test_anthropic_settings_shape_on_no_agent``. Compatibility for the
    #: ``switch`` command is instead computed by the sibling resolver
    #: ``claude_settings.switchable_providers``.
    ANTHROPIC_SETTINGS = "anthropic-settings"


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
    # Last, and inert in practice: no Agent declares ANTHROPIC_SETTINGS (see
    # its docstring), so `common` can never contain it and this priority slot
    # is never actually consulted by resolve_shape. Listed anyway so the
    # "unreachable" guard at the end of resolve_shape stays honest — every
    # ConfigShape member has a priority entry, none silently unprioritised.
    ConfigShape.ANTHROPIC_SETTINGS,
)

#: An agent's binary name is interpolated into a generated script UNQUOTED (see
#: ``render.py``), so it must be a registry constant of a boring shape — not
#: user input. Enforced at import time by :func:`_validate_registries`, and
#: reused (via :func:`validate_agent_binary`) by ``services/agents.py`` to
#: gate a user-supplied agent the exact same way — a hand-duplicated copy of
#: this pattern there would be one edit away from drifting out of sync with
#: the actual shell-injection guard.
_BINARY_RE = re.compile(r"\A[a-z][a-z0-9_-]*\Z")


def validate_agent_binary(binary: str) -> None:
    """Raise unless ``binary`` is safe to interpolate unquoted into a script.

    The single public gate on the pattern :data:`_BINARY_RE` encodes — lower-
    case alphanumerics, ``_``/``-``, starting with a letter. Any caller that
    accepts an agent name/binary from outside the registry (today: a
    user-defined agent in ``services/agents.py``) MUST run it through this
    before persisting or using it; skipping it is a shell-injection hole,
    since ``agent.binary`` is later interpolated unquoted (see ``render.py``).

    Raises:
        CodeHelperError: ``binary`` does not match the required pattern.
    """
    if not _BINARY_RE.match(binary):
        raise CodeHelperError(
            f"invalid agent binary {binary!r} — must start with a lowercase "
            f"letter and contain only lowercase letters, digits, '_' and '-'"
        )


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


class AuthPolicy(StrEnum):
    """Whether a :class:`Provider`'s ``auth`` mode is fixed or overridable.

    ``auth`` used to be treated as an immutable fact about a provider —
    ``ollama`` was always ``"literal"``, on the (wrong) assumption that a
    local daemon never sits behind auth. In reality Ollama is routinely
    fronted by a reverse proxy or run as Ollama Cloud, both of which want a
    real secret. ``auth`` describes how a *particular installation* is
    reached, not an immutable property of the provider — the same
    distinction :class:`BaseUrlPolicy` already draws for ``base_url``, and
    this is the matching axis for ``auth``.

    ``zai``/``litellm`` have exactly one true auth mode (``secret``) and stay
    FIXED. ``ollama`` defaults to a literal token but a caller MAY switch it
    to ``secret`` at runtime via :func:`with_auth` — mirroring
    ``OVERRIDABLE``'s ``base_url`` story: the registry default stands unless
    the caller explicitly overrides it.
    """

    #: The registry ``auth``/``auth_value``/``token_env_var`` are the only
    #: values — never overridable at runtime.
    FIXED = "fixed"

    #: The registry ``auth`` is a default the caller MAY override to
    #: ``"secret"`` at runtime (``--auth secret`` / a TUI prompt) via
    #: :func:`with_auth`. Requires a registry ``token_env_var`` even while the
    #: default is ``"literal"``, so the override is expressible without
    #: touching the registry entry (``_validate_provider`` enforces this).
    OVERRIDABLE = "overridable"


@dataclass(frozen=True)
class Agent:
    """A coding agent (the thing that actually runs) — one axis."""

    name: str
    #: Executable name. Interpolated into generated scripts unquoted; see
    #: :data:`_BINARY_RE`.
    #:
    #: For :attr:`ConfigShape.OLLAMA_LAUNCH` this value does DOUBLE DUTY: it is
    #: both the executable ``exec``'d directly by other shapes and the
    #: *integration name* passed as ``ollama launch <this>`` (see
    #: ``render._render_ollama_launch``). The two happen to coincide for every
    #: agent in the registry today. If a future agent's ``ollama launch``
    #: integration name ever diverges from its executable (e.g. VS Code: the
    #: binary is ``code``, the integration is ``vscode``), the fix is a
    #: separate ``launch_name`` field defaulting to ``binary`` — not overloading
    #: this one further.
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
    #: ``base_url`` is already the complete OpenAI-compatible root, so the
    #: OPENAI_TOML renderer must NOT append ``/v1/`` to it. Most OpenAI-only
    #: providers follow the ``/v1``-suffix convention (``render.openai_base_url``
    #: appends ``/v1/`` to a bare root and ``/`` to a ``/v1``-suffixed one), but
    #: a provider whose documented endpoint IS the full OpenAI surface (e.g.
    #: Gemini's ``https://generativelanguage.googleapis.com/v1beta/openai/``)
    #: would otherwise be rewritten to a nonexistent ``.../openai/v1/`` and
    #: every request 404s. Declared data, never a provider-name check — the
    #: renderer branches on this field, exactly like ``env_reset``.
    base_url_is_openai_root: bool = False
    #: ``"literal"`` — a non-secret token baked into the script (Ollama's local
    #: daemon accepts ``"ollama"``); ``"secret"`` — resolved from env/prompt and
    #: the script is written 0o700; ``"none"`` — the launcher authenticates.
    auth: str = "none"
    #: See :class:`AuthPolicy`. Constrains whether ``auth`` may be overridden
    #: at runtime — enforced at import time by :func:`_validate_provider`.
    auth_policy: AuthPolicy = AuthPolicy.FIXED
    auth_value: str = ""
    token_env_var: str = ""
    model_list_api: ModelListAPI = ModelListAPI.NONE
    #: Only when the listing lives somewhere other than ``base_url``.
    model_list_url: str = ""
    #: For :attr:`ConfigShape.OPENAI_TOML`: ``"responses"`` or ``"chat"``.
    wire_api: str = ""
    #: This provider is the agent's NATIVE backend: "switching to it" means
    #: REMOVING every managed key from the target config, restoring whatever
    #: the agent does with no redirection at all (for ``claude``: OAuth +
    #: native models). Declared, not inferred from an empty ``base_url`` +
    #: ``auth="none"`` — that combination is also what an unresolved
    #: ``BaseUrlPolicy.REQUIRED`` provider looks like before ``with_base_url``
    #: runs, and a reader (or ``claude_settings.resolve_switch_patch``) must
    #: be able to tell "deliberately no address" from "not yet resolved" from
    #: the registry alone. Enforced by ``_validate_provider``: ``env_reset``
    #: implies an empty ``base_url`` (policy FIXED), ``auth="none"``, an empty
    #: ``token_env_var``, and ``model_list_api is NONE`` — a reset provider
    #: that carried an address or a credential would be a contradiction in
    #: terms. Mirrors how ``BaseUrlPolicy``/``AuthPolicy`` already turn a
    #: provider axis into declared data instead of a name check: the only
    #: reader of this field branches on ``provider.env_reset``, never on
    #: ``provider.name == "anthropic"``.
    env_reset: bool = False
    #: Fallback model names offered when :func:`~codehelper.services.
    #: models_api.list_models` returns none — either because the provider
    #: structurally cannot publish a list (``model_list_api is NONE``, e.g.
    #: zai's Anthropic-compatible endpoint) or because discovery failed for
    #: this call (daemon down, no base URL yet). Discovery is always tried
    #: FIRST; this is what keeps the model step a menu instead of a bare
    #: prompt when discovery comes back empty. The TUI is responsible for
    #: labelling the source ("known models" vs "discovered") so a stale
    #: built-in entry is never presented as if it were live — see
    #: ``cli/tui.py``'s ``_choose_add_model``.
    known_models: tuple[str, ...] = ()
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
    # The remaining 13 are every OTHER *CLI* integration `ollama launch`
    # supports (verified against `ollama launch --help`, ollama 0.32.13) —
    # GUI/desktop integrations (chatgpt/codex-app, hermes-desktop, vscode)
    # are deliberately excluded: they have no CLI binary, so `"$@"` in a
    # generated wrapper would be meaningless for them.
    #
    # Every one of these declares ONLY OLLAMA_LAUNCH, never OPENAI_TOML: that
    # shape means "this agent reads Codex's own `--profile <alias>` TOML
    # convention" (see render._render_openai_toml), and none of them does —
    # each has its own native config format that only `ollama launch` itself
    # knows how to write (verified by hand against `--help` for opencode,
    # copilot, droid, cline, pi). Reaching a non-Ollama provider (z.ai,
    # litellm) is therefore correctly impossible for all 13, exactly like
    # `codex × z.ai` above — do not "fix" that by adding OPENAI_TOML here.
    #
    # binary == name for every one of these (there is no GUI/CLI name split
    # to account for yet — see Agent.binary's docstring), so each is just a
    # (name, description) pair rather than a hand-repeated Agent(...) call.
    *(
        Agent(
            name=name,
            binary=name,
            shapes=frozenset({ConfigShape.OLLAMA_LAUNCH}),
            description=description,
        )
        for name, description in (
            ("hermes", "Hermes Agent"),
            ("openclaw", "OpenClaw"),
            ("opencode", "OpenCode"),
            ("copilot", "GitHub Copilot CLI"),
            ("omp", "OMP"),
            ("droid", "Droid"),
            ("dsh", "DeepSeek Harness"),
            ("kimi", "Kimi Code CLI"),
            ("muse", "Muse Code"),
            ("pi", "Pi"),
            ("pool", "Pool"),
            ("cline", "Cline"),
            ("qwen", "Qwen Code"),
        )
    ),
)


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="ollama",
        # Three shapes: the daemon serves the Anthropic protocol directly (so
        # `claude` can point straight at it — this is the `deepseek-ollama`
        # preset),
        # `ollama launch` can configure an agent for us (the `glm-ollama`
        # preset), and an OpenAI-compatible `/v1` endpoint feeds Codex's own
        # TOML profile (codex × ollama via OPENAI_TOML — no launcher binary
        # needed). Same provider, three connection mechanisms.
        shapes=frozenset(
            {
                ConfigShape.ANTHROPIC_ENV,
                ConfigShape.OLLAMA_LAUNCH,
                ConfigShape.OPENAI_TOML,
                # Lets `switch --from-wrapper` retarget a live claude session
                # at this same daemon — see ConfigShape.ANTHROPIC_SETTINGS.
                ConfigShape.ANTHROPIC_SETTINGS,
            }
        ),
        base_url="http://127.0.0.1:11434",
        auth="literal",
        auth_value="ollama",
        # OVERRIDABLE, not FIXED: the registry default fits an unauthenticated
        # local daemon, but a real Ollama install is routinely fronted by a
        # reverse proxy or run as Ollama Cloud — both want a real secret.
        # token_env_var must be present even while the default is "literal"
        # (see AuthPolicy and _validate_provider) so `--auth secret` has a
        # variable name to export the moment a caller opts in.
        auth_policy=AuthPolicy.OVERRIDABLE,
        token_env_var="OLLAMA_API_KEY",
        model_list_api=ModelListAPI.OLLAMA_TAGS,
        # Consumed by the OPENAI_TOML renderer (`wire_api` in the profile).
        wire_api="responses",
        description="local Ollama daemon",
    ),
    Provider(
        name="zai",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV, ConfigShape.ANTHROPIC_SETTINGS}),
        base_url="https://api.z.ai/api/anthropic",
        auth="secret",
        token_env_var="ZAI_API_KEY",
        # The Anthropic-compatible path exposes no OpenAI-style model list.
        model_list_api=ModelListAPI.NONE,
        # Discovery is structurally unavailable (model_list_api is NONE
        # above), so these are the ONLY models the model step can ever offer
        # besides manual entry. glm-5.3 is the current flagship; the older
        # glm-5-turbo / glm-5.1 remain listed as known-good fallbacks.
        known_models=("glm-5.3", "glm-5-turbo", "glm-5.1"),
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
        shapes=frozenset(
            {
                ConfigShape.ANTHROPIC_ENV,
                ConfigShape.OPENAI_TOML,
                ConfigShape.ANTHROPIC_SETTINGS,
            }
        ),
        # REQUIRED, not FIXED: this is the user's own server, not an address
        # this project could ship a default for. base_url is supplied at
        # runtime (--base-url / a TUI prompt) via model.with_base_url — see
        # BaseUrlPolicy and cli/parser.py's --base-url handling.
        base_url="",
        base_url_policy=BaseUrlPolicy.REQUIRED,
        auth="secret",
        token_env_var="LITELLM_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        # "responses", the ONLY value current Codex accepts: chat/completions
        # support was removed entirely (hard startup error since ~Feb 2026 —
        # openai/codex discussion #7782). The old "chat, not responses" choice
        # is moot because "chat" is no longer a wire_api a profile can carry;
        # the LiteLLM proxy serves /v1/responses (the proxy itself must have
        # responses mode enabled for the upstream backend).
        wire_api="responses",
        description="LiteLLM proxy (user-supplied base URL)",
    ),
    Provider(
        name="gemini",
        # Gemini's documented compatibility surface is OpenAI chat
        # completions only.  It does not expose Claude's Anthropic Messages
        # protocol, so claude x gemini must remain incompatible by the normal
        # shape intersection rather than a provider-name special case.
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # The stored base_url IS the complete OpenAI-compatible root — the
        # renderer must not append /v1/ to it (that would 404). See the field
        # docstring on Provider.base_url_is_openai_root.
        base_url_is_openai_root=True,
        auth="secret",
        token_env_var="GEMINI_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="chat",
        description="Google Gemini (OpenAI-compatible)",
    ),
    Provider(
        name="deepseek",
        # DeepSeek's Anthropic-compatible surface (https://api.deepseek.com/anthropic)
        # drives Claude Code directly — the same two shapes as zai: a generated
        # wrapper (ANTHROPIC_ENV) and a live `switch` (ANTHROPIC_SETTINGS).
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV, ConfigShape.ANTHROPIC_SETTINGS}),
        base_url="https://api.deepseek.com/anthropic",
        auth="secret",
        token_env_var="DEEPSEEK_API_KEY",
        # The /anthropic path serves no OpenAI-style model list, so discovery is
        # pointed at the OpenAI surface explicitly (models_api.list_models uses
        # model_list_url over base_url, then normalizes to .../v1/models).
        model_list_api=ModelListAPI.OPENAI_V1,
        model_list_url="https://api.deepseek.com",
        # Discovery is available (OPENAI_V1 above), but these are the documented
        # models to fall back on when the listing call fails — same role as
        # zai's known_models, just not the only source.
        known_models=(
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        ),
        description="DeepSeek (Anthropic-compatible)",
    ),
    Provider(
        name="deepseek-openai",
        # DeepSeek's OpenAI-compatible surface (https://api.deepseek.com, /v1 is
        # a path alias) feeds Codex's TOML profile — the same single shape as
        # gemini. DeepSeek natively serves the Responses API (POST /responses,
        # added for Codex), so wire_api="responses" is the documented value.
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.deepseek.com",
        # A bare root, NOT is_openai_root: openai_base_url appends /v1/ (the
        # documented OpenAI surface), exactly like litellm's bare-root handling.
        auth="secret",
        token_env_var="DEEPSEEK_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        # No model_list_url needed: base_url is already the OpenAI root, so
        # discovery hits https://api.deepseek.com/v1/models directly.
        wire_api="responses",
        known_models=(
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        ),
        description="DeepSeek (OpenAI-compatible)",
    ),
    Provider(
        name="native",
        # "native", not "anthropic": this entry does not represent a backend
        # (an address to send requests to) — it represents the ABSENCE of
        # one. `env_reset=True` is what that means: switching to it CLEARS
        # the override rather than pointing at anything. The name says so
        # honestly, rather than implying "Anthropic" is one more provider
        # among equals with its own endpoint.
        #
        # Only the switch-only mechanism — see ConfigShape.ANTHROPIC_SETTINGS.
        # `resolve_shape(claude, native)` therefore always raises "no common
        # configuration mechanism": this is correct, not a gap — a wrapper
        # cannot mean "be normal claude", that is just `claude` with no
        # wrapper at all. `codehelper add`/`list matrix` show it as
        # switch-only rather than a usable pairing (see cli/parser.py's
        # `_handle_list_axes`).
        shapes=frozenset({ConfigShape.ANTHROPIC_SETTINGS}),
        env_reset=True,
        description="native — clears the override, restores OAuth + stock models",
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
    # See AuthPolicy: an OVERRIDABLE provider needs a token_env_var even while
    # its registry default is not "secret" — with_auth's whole job is to flip
    # auth to "secret" at runtime, and without a token_env_var already
    # present that override would hit the exact silent-hole case the check
    # above exists to catch, just one step later.
    if provider.auth_policy is AuthPolicy.OVERRIDABLE and not provider.token_env_var:
        raise CodeHelperError(
            f"provider {provider.name!r} declares auth_policy=OVERRIDABLE but "
            f"has no token_env_var — with_auth would have no variable to "
            f"export once a caller opts into auth='secret'"
        )
    # See BaseUrlPolicy: exactly one of "registry supplies base_url" / "caller
    # must supply it at runtime" holds per policy — never both, never neither.
    if provider.base_url_policy is BaseUrlPolicy.REQUIRED and provider.base_url:
        raise CodeHelperError(
            f"provider {provider.name!r} declares base_url_policy=REQUIRED but "
            f"also carries a registry base_url {provider.base_url!r} — the "
            f"runtime value passed to with_base_url would silently overwrite it"
        )
    # env_reset is exempt from this FIXED/OVERRIDABLE-must-carry-a-base_url
    # rule: it is checked, more specifically, by the env_reset block below —
    # an env_reset provider is SUPPOSED to have no base_url at all.
    if (
        not provider.env_reset
        and provider.base_url_policy is not BaseUrlPolicy.REQUIRED
        and not provider.base_url
    ):
        raise CodeHelperError(
            f"provider {provider.name!r} declares base_url_policy="
            f"{provider.base_url_policy.value!r} but has no registry base_url "
            f"— FIXED has no other source, and an OVERRIDABLE default cannot "
            f"be empty (that is what REQUIRED is for)"
        )
    # An env_reset provider is declared "no address, no credential" ON
    # PURPOSE — that is the whole meaning of "switching to it clears the
    # override". If it carried any of these, claude_settings.resolve_switch_patch
    # would have something to write, contradicting env_reset's own contract.
    if provider.env_reset:
        prefix = f"provider {provider.name!r} declares env_reset=True but also "
        if provider.base_url or provider.base_url_policy is not BaseUrlPolicy.FIXED:
            raise CodeHelperError(
                f"{prefix}a base_url {provider.base_url!r} / policy "
                f"{provider.base_url_policy.value!r} — a reset provider must "
                f"have no address at all"
            )
        if provider.auth != "none" or provider.token_env_var:
            raise CodeHelperError(
                f"{prefix}auth={provider.auth!r}/token_env_var="
                f"{provider.token_env_var!r} — a reset provider must carry no "
                f"credential"
            )
        if provider.model_list_api is not ModelListAPI.NONE:
            raise CodeHelperError(
                f"{prefix}model_list_api={provider.model_list_api.value!r} — "
                f"a reset provider has no endpoint to list models from"
            )
    # Same class of check as every other registry field: an empty/whitespace
    # entry would render as a blank, unselectable-looking menu row in the
    # fallback picker — catch it at import time, not when a user hits it.
    for known in provider.known_models:
        if not known or not known.strip():
            raise CodeHelperError(
                f"provider {provider.name!r} has an empty/blank known_models "
                f"entry: {known!r}"
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


def switchable_providers() -> list[Provider]:
    """Providers reachable by ``switch`` — the ANTHROPIC_SETTINGS mechanism.

    The ``switch`` analogue of :func:`compatible_providers`. Deliberately NOT
    routed through :func:`resolve_shape`: that function answers "can a
    wrapper be generated", and ``ANTHROPIC_SETTINGS`` is on no ``Agent``, so
    the intersection there is empty by construction (see the shape's
    docstring). Keeping the two resolvers separate is what guarantees adding
    this shape cannot perturb wrapper generation.
    """
    return [p for p in PROVIDERS if ConfigShape.ANTHROPIC_SETTINGS in p.shapes]


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
        after :func:`~codehelper.services.naming.validate_base_url`).

    Raises:
        CodeHelperError: a URL was supplied for a FIXED provider; no URL was
            supplied for a REQUIRED provider; or the supplied URL fails
            validation.
    """
    from codehelper.services.naming import normalize_base_url, validate_base_url

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
                f"--base-url https://host:port/v1 (or a bare host like "
                f"78.47.183.125, which is auto-completed)"
            )
        return provider  # OVERRIDABLE, nothing supplied: the default stands.

    # A bare host/IP (no scheme) is auto-completed to a full URL before
    # validation — the single substitution point, so every caller (CLI add,
    # set-default, TUI) gets the same convenience.
    base_url = normalize_base_url(base_url)
    validate_base_url(base_url)
    return replace(provider, base_url=base_url)


def with_auth(provider: Provider, want_secret: bool) -> Provider:
    """The single substitution point for a runtime ``auth`` override.

    Mirrors :func:`with_base_url`'s shape exactly: every one of ``auth``'s
    readers (``render.py``, ``models_api.list_models``, ``parser.py``,
    ``wrappers.py``, ``secrets.py`` — see the module-level list in
    ``CLAUDE.md``) reads it off a :class:`Provider`/``WrapperSpec`` object it
    was handed, never re-derives it — so substituting the provider once, here,
    before ``build_spec`` runs, is enough for all of them to see the right
    mode. Deliberately DATA-driven: the branch is on
    ``provider.auth_policy``, never on ``provider.name`` — a future provider
    that wants the same override needs only a registry entry, no new code
    here.

    Args:
        provider: The provider as looked up from the registry.
        want_secret: Whether the caller asked to override to ``auth="secret"``
            (``--auth secret`` / a TUI prompt). ``False`` means "nothing
            supplied" — the registry default stands, exactly like
            ``with_base_url``'s ``base_url=None``.

    Returns:
        ``provider`` unchanged when ``want_secret`` is ``False``, or (for an
        OVERRIDABLE provider) a copy with ``auth="secret"`` and
        ``auth_value=""`` — the literal value would otherwise linger,
        unused but misleading, on a provider now resolved as secret.

    Raises:
        CodeHelperError: ``want_secret`` was requested for a FIXED provider
            whose registry ``auth`` is not already ``"secret"``.
    """
    if not want_secret:
        return provider  # Registry default stands, for either policy.

    if provider.auth == "secret":
        return provider  # Already secret — no override needed either way.

    if provider.auth_policy is not AuthPolicy.OVERRIDABLE:
        raise CodeHelperError(
            f"provider {provider.name!r} has a fixed auth mode "
            f"({provider.auth!r}) — --auth secret only applies to: "
            + ", ".join(
                p.name for p in PROVIDERS if p.auth_policy is AuthPolicy.OVERRIDABLE
            )
        )

    return replace(provider, auth="secret", auth_value="")
