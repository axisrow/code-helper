"""``WrapperSpec`` — a resolved point in the agent × provider × model matrix.

This is the object everything downstream (render, install, describe) consumes.
It is deliberately *plain*: by the time one exists, compatibility has been
checked, the model has been resolved, and the alias has been validated. Neither
the renderers nor the install path should have to re-derive any of that.

Two ways to get one:

- :func:`build_spec` — from the three axes, i.e. the constructor path.
- :func:`spec_from_preset` — from a named :class:`Preset`.

Both funnel through the same construction, so a preset cannot behave
differently from the equivalent hand-specified combination.

Why presets still exist after the refactor
------------------------------------------
They are not a parallel mechanism — each one is just a named argument bundle
that expands into the same axes. They earn their place because:

- ``glm`` uses a *different model per tier* (``glm-4.7`` / ``glm-5-turbo`` /
  ``glm-5.1``); that is not expressible as a single ``--model``, and
  demanding three flags for the common case would be absurd;
- ``codehelper add glm`` must keep working as one word.
"""

from __future__ import annotations

from dataclasses import dataclass

from codehelper.errors import CodeHelperError
from codehelper.services.model import (
    Agent,
    BaseUrlPolicy,
    ConfigShape,
    Provider,
    get_agent,
    get_provider,
    resolve_shape,
    with_base_url,
)
from codehelper.services.naming import validate_alias

#: Reasoning-effort values current Codex accepts in ``model_reasoning_effort``
#: (issue #100). The ONLY extension point: a new value lands here, and the
#: marker regex (``[^,()\s]+``) plus the TOML renderer need no edits. Codex
#: hard-rejects unknown values at config deserialization (the ``wire_api``
#: precedent, openai/codex#7782), so an unlisted value must be refused here —
#: never rendered into a profile Codex would then refuse to load.
REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")

__all__ = [
    "WrapperSpec",
    "TierModels",
    "Preset",
    "PRESETS",
    "REASONING_EFFORTS",
    "build_spec",
    "spec_from_preset",
    "get_preset",
    "preset_names",
    "suggest_alias",
]


@dataclass(frozen=True)
class TierModels:
    """Per-tier models for the Anthropic env shape.

    Claude Code asks for a haiku/sonnet/opus model separately, and a provider
    may map them to genuinely different models (``glm``) or to one model
    repeated (``deepseek-ollama``). Collapsing this to a single field would lose
    the former.
    """

    haiku: str
    sonnet: str
    opus: str

    @classmethod
    def uniform(cls, model: str) -> TierModels:
        return cls(haiku=model, sonnet=model, opus=model)


@dataclass(frozen=True)
class WrapperSpec:
    """A validated, fully-resolved wrapper, ready to render and install."""

    alias: str
    agent: Agent
    provider: Provider
    #: Already chosen by :func:`resolve_shape` — the renderer dispatches on it.
    shape: ConfigShape
    #: The resolved model (any ``--model`` override already applied).
    model: str
    tier_models: TierModels | None = None
    subagent_model: str | None = None
    description: str = ""
    #: Named secret profile selected when this wrapper was installed.  It is
    #: metadata, not a runtime lookup: the rendered wrapper still carries the
    #: resolved token so it works even if the cache is later removed.
    profile_name: str | None = None
    #: EXPLICIT context window (issue #83): ``None`` means "derive from the
    #: catalog" (``render.uniform_context_window``); ``0`` means an explicit
    #: "no declaration" — suppress even a catalog hit; ``> 0`` is the window.
    #: Like ``profile_name`` it is recorded data, not a derivation: it rides
    #: the wrapper marker (``ctx=``) so reconstruction never re-derives it.
    context_window: int | None = None
    #: Reasoning effort for the OPENAI_TOML shape (issue #100): rendered as
    #: ``model_reasoning_effort`` in the companion TOML profile and recorded
    #: in the marker (``effort=``, after ``ctx=``) like every other explicit
    #: answer. ``None`` means "not managed" — the renderer emits no key and
    #: Codex keeps its own default (the ``ctx=`` sentinel convention). Only
    #: ever non-None for an OPENAI_TOML spec: :func:`build_spec` refuses it
    #: for every other shape.
    effort: str | None = None

    @property
    def name(self) -> str:
        """The on-disk file name. Alias of :attr:`alias`, kept because the
        install/describe layer talks about wrappers by *name*."""
        return self.alias

    @property
    def auth(self) -> str:
        """Auth mode, from the provider — drives the file mode (0o700/0o755)."""
        return self.provider.auth

    @property
    def auth_value(self) -> str:
        """Literal token, from the provider (empty unless ``auth == "literal"``)."""
        return self.provider.auth_value

    @property
    def token_env_var(self) -> str:
        """Env var holding the secret, from the provider.

        Forwarded from the provider rather than made a field of its own: the
        credential belongs to the backend, not to any one wrapper pointing at
        it, so two wrappers on the same provider can never disagree about it.
        """
        return self.provider.token_env_var

    @property
    def window_models(self) -> list[str]:
        """Every model the context-window declaration covers (issue #83).

        The ONE source for that set: the renderer derives the declaration
        from it and the interactive question asks about it, so the two can
        never drift — three review rounds (PR #84) each found a path where
        a hook hand-built a different set than the renderer read. Tier
        slots when :attr:`tier_models` exists (the ANTHROPIC_ENV shape);
        otherwise the single model (OLLAMA_LAUNCH, OPENAI_TOML). The
        subagent model participates when set — it shares the variable.
        """
        if self.tier_models is not None:
            models = [
                self.tier_models.haiku,
                self.tier_models.sonnet,
                self.tier_models.opus,
            ]
        else:
            models = [self.model]
        if self.subagent_model is not None:
            models.append(self.subagent_model)
        return models


def suggest_alias(model: str, agent_name: str, profile_name: str | None = None) -> str:
    """Derive a default alias, e.g. ``glm-5:cloud`` + ``codex`` -> ``glm-5-codex``.

    Strips the registry prefix (``nvidia/…``) and the tag (``:cloud``), then
    replaces anything outside the allowed alias charset. The result still goes
    through :func:`validate_alias` at build time — this is a *suggestion*, not
    a bypass.

    Raises:
        CodeHelperError: the model contributes no usable characters. Without
            this, the alias would collapse to the bare agent name and target
            ``~/.local/bin/claude`` — a real, working symlink on most installs.
    """
    base = model.split("/")[-1].split(":")[0]
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "-" for c in base)
    cleaned = cleaned.strip("-._")
    if not cleaned:
        raise CodeHelperError(
            f"cannot derive a wrapper name from model {model!r} — pass --alias"
        )
    if profile_name:
        profile_suffix = "".join(
            c if (c.isalnum() or c in "._-") else "-" for c in profile_name
        ).strip("-._")
        if profile_suffix:
            return f"{cleaned}-{profile_suffix}-{agent_name}"
    return f"{cleaned}-{agent_name}"


def build_spec(
    *,
    agent: Agent | str,
    provider: Provider | str,
    model: str = "",
    alias: str | None = None,
    shape: ConfigShape | None = None,
    tier_models: TierModels | None = None,
    subagent_model: str | None = None,
    description: str = "",
    profile_name: str | None = None,
    context_window: int | None = None,
    effort: str | None = None,
) -> WrapperSpec:
    """Assemble a :class:`WrapperSpec` from the three axes. Pure, no IO.

    Order matters and is part of the contract: compatibility and the alias are
    validated *here*, which is before any caller resolves a token. A typo must
    never produce an interactive secret prompt for a wrapper that will not be
    written — the old ``get_spec``-first flow had this property and it is
    preserved.

    Raises:
        CodeHelperError: unknown agent/provider, incompatible pairing, missing
            model, or an unusable alias.
    """
    agent_obj = get_agent(agent) if isinstance(agent, str) else agent
    provider_obj = get_provider(provider) if isinstance(provider, str) else provider

    # A REQUIRED-policy provider (its endpoint is the user's own server, e.g.
    # a self-hosted LiteLLM proxy) has an empty registry base_url by
    # construction (see BaseUrlPolicy / _validate_provider). The only way to
    # get a real address into it is model.with_base_url, called by the
    # caller BEFORE build_spec runs. If nobody called it, provider_obj still
    # carries the empty registry default here — refuse rather than silently
    # rendering/patching a malformed address (an OPENAI_TOML profile's
    # base_url derives "/v1/" from an empty root, and ANTHROPIC_BASE_URL
    # would export as an empty string). This is a service-layer invariant,
    # independent of whether any particular CLI/TUI entry point has grown a
    # --base-url flag yet — build_spec must never produce a spec for a
    # REQUIRED provider with no address, no matter how it was reached.
    if (
        provider_obj.base_url_policy is BaseUrlPolicy.REQUIRED
        and not provider_obj.base_url
    ):
        raise CodeHelperError(
            f"provider {provider_obj.name!r} requires a base URL — supply one "
            f"via model.with_base_url(provider, url) before build_spec"
        )

    # An unusable explicit window refuses here, before anything interactive —
    # this is what makes spec_from_installed's garbage marker values
    # (``ctx=abc`` → ValueError, ``ctx=-5`` → lands here) fail closed to
    # None, the same answer as any other unrecognised marker value.
    if context_window is not None and not (0 <= context_window <= 10_000_000):
        raise CodeHelperError(
            f"unusable context window {context_window}: expected 0 (no "
            f"declaration) or a token count in 1..10_000_000"
        )

    # An unknown effort value refuses here, beside the window check (issue
    # #100): Codex hard-rejects unknown ``model_reasoning_effort`` values at
    # config deserialization (the ``wire_api`` precedent), so an unlisted
    # value must fail before anything is rendered — which is also what makes
    # a garbage ``effort=`` marker value fail spec_from_installed closed to
    # None, the same answer as any other unrecognised marker value.
    if effort is not None and effort not in REASONING_EFFORTS:
        raise CodeHelperError(
            f"unusable reasoning effort {effort!r}: expected one of "
            f"{', '.join(REASONING_EFFORTS)}"
        )

    chosen = resolve_shape(agent_obj, provider_obj, preferred=shape)

    # effort is an OPENAI_TOML-only axis (issue #100): it lives in the
    # companion TOML profile, and no other shape has a surface to declare it
    # on. A shape check, never a provider/agent-name check — a future shape
    # with a reasoning-effort surface joins by changing this one condition.
    if effort is not None and chosen is not ConfigShape.OPENAI_TOML:
        raise CodeHelperError("effort applies only to openai-toml wrappers (codex)")

    # There used to be a refusal here: OPENAI_TOML + auth == "secret" was
    # rejected outright, because the wrapper was a one-line
    # ``exec codex --profile <alias> "$@"`` and the profile had no ``env_key``
    # field — a secret provider had no way to carry its token. That is no
    # longer true: ``openai_toml_body`` now writes ``env_key`` (conditionally,
    # only for a secret provider — see ``render.openai_env_key``), and
    # ``_render_openai_toml`` exports that variable before ``exec``, so the
    # token reaches ``codex`` through the process environment instead of
    # through the profile file. The token still lives NOWHERE but the
    # generated script (mode 0o700), so ``_discards_only_secret`` protects it
    # here exactly as it already does for ``ANTHROPIC_ENV``.

    if not model and tier_models is None:
        raise CodeHelperError(
            f"a model is required for {agent_obj.name} + {provider_obj.name}"
        )

    # For the env shape every tier needs a value; a single model fills all three.
    if chosen is ConfigShape.ANTHROPIC_ENV and tier_models is None:
        tier_models = TierModels.uniform(model)

    # The single model that identifies this wrapper: the explicit one, else the
    # sonnet tier (the mid tier is what a user means by "the model" when tiers
    # differ). Resolved once so the alias, description, and spec all agree.
    effective_model = model or (tier_models.sonnet if tier_models else "")

    resolved_alias = (
        alias
        if alias is not None
        else suggest_alias(effective_model, agent_obj.name, profile_name)
    )
    validate_alias(resolved_alias)

    if not description:
        description = f"{agent_obj.name} → {effective_model} via {provider_obj.name}"

    return WrapperSpec(
        alias=resolved_alias,
        agent=agent_obj,
        provider=provider_obj,
        shape=chosen,
        model=effective_model,
        tier_models=tier_models,
        subagent_model=subagent_model,
        description=description,
        profile_name=profile_name,
        context_window=context_window,
        effort=effort,
    )


@dataclass(frozen=True)
class Preset:
    """A named, curated combination — expands into the same axes."""

    alias: str
    agent: str
    provider: str
    shape: ConfigShape | None = None
    model: str = ""
    tier_models: TierModels | None = None
    subagent_model: str | None = None
    description: str = ""
    #: The default endpoint for a provider whose address is REQUIRED (the
    #: user's own server): a preset curated against one concrete instance
    #: ships that instance's address, mirror of how ``deepseek-ollama`` ships
    #: ollama's local default. ``--base-url`` overrides it; a preset for a
    #: FIXED/registry-address provider must leave this empty — ``with_base_url``
    #: refuses a URL for FIXED, and the CLI rejects ``--base-url`` there with
    #: the same teaching message as before this field existed.
    base_url: str = ""


_DEEPSEEK_MODEL = "deepseek-v4-flash:0731-cloud"

# The current Gemini flash, as served by the LiteLLM proxy the
# gemini-litellm preset is curated against (model_name in the proxy's
# config.yaml → litellm_params.model: gemini/gemini-3.7-flash).
_GEMINI_MODEL = "gemini-3.7-flash"

_LITELLM_PROXY_URL = "https://litellm.78.47.183.125.sslip.io"


PRESETS: tuple[Preset, ...] = (
    Preset(
        alias="deepseek-ollama",
        agent="claude",
        provider="ollama-direct",
        shape=ConfigShape.ANTHROPIC_ENV,
        model=_DEEPSEEK_MODEL,
        tier_models=TierModels.uniform(_DEEPSEEK_MODEL),
        subagent_model=_DEEPSEEK_MODEL,
        description="Claude Code → deepseek-v4-flash via the local Ollama daemon",
    ),
    Preset(
        alias="glm",
        agent="claude",
        provider="zai",
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.3",
        tier_models=TierModels.uniform("glm-5.3"),
        subagent_model=None,
        description="Claude Code → Z.ai",
    ),
    Preset(
        alias="glm-ollama",
        agent="claude",
        provider="ollama-direct",
        shape=ConfigShape.OLLAMA_LAUNCH,
        model="glm-5.2:cloud",
        description="Claude Code → glm-5.2:cloud via `ollama launch claude`",
    ),
    Preset(
        alias="gemini-litellm",
        agent="claude",
        provider="litellm",
        shape=ConfigShape.ANTHROPIC_ENV,
        model=_GEMINI_MODEL,
        tier_models=TierModels.uniform(_GEMINI_MODEL),
        # Google serves no Anthropic-compatible endpoint (#74), so Gemini
        # reaches Claude Code only through a translating proxy — this preset
        # is curated against the user's own LiteLLM instance, whose address
        # rides in base_url below (--base-url points it at a different one).
        # The proxy also carries the /v1/responses codex needs, so the codex
        # pairing is `add --agent codex --provider litellm --base-url …`.
        base_url=_LITELLM_PROXY_URL,
        description=f"Claude Code → {_GEMINI_MODEL} via the LiteLLM proxy",
    ),
    Preset(
        alias="bai",
        agent="claude",
        provider="bai",
        shape=ConfigShape.ANTHROPIC_ENV,
        # The current B.AI free-promo workhorse (dashboard "Limited-Time Free";
        # verified through /v1/messages) as the default — the paid claude tier
        # is one --model override away. Promos rotate; re-curate when they do.
        model="qwen3.8-flash",
        tier_models=TierModels.uniform("qwen3.8-flash"),
        # subagent_model stays None (the glm-preset convention). base_url
        # stays empty — mandatory for a FIXED-address provider (the bai
        # registry entry carries the only documented host; --base-url is
        # refused), the mirror image of the gemini-litellm preset, which
        # exists precisely to carry one.
        description="Claude Code → B.AI",
    ),
)


def preset_names() -> list[str]:
    return [p.alias for p in PRESETS]


def get_preset(name: str) -> Preset:
    """Look up a preset by name.

    Raises:
        CodeHelperError: unknown name. The message keeps the substring
            ``"unknown wrapper"`` that the CLI contract and its tests rely on.
    """
    for preset in PRESETS:
        if preset.alias == name:
            return preset
    known = ", ".join(preset_names())
    raise CodeHelperError(f"unknown wrapper name: {name} (known: {known})")


def spec_from_preset(
    preset: Preset,
    *,
    model_override: str | None = None,
    alias_override: str | None = None,
    profile_name: str | None = None,
    base_url_override: str | None = None,
) -> WrapperSpec:
    """Expand a preset into a spec, applying ``--model`` if given.

    ``model_override`` replaces every tier AND the subagent model, matching the
    pre-refactor behaviour of ``--model`` exactly. ``base_url_override`` (the
    CLI's ``--base-url``) replaces the preset's own ``base_url`` — either is
    substituted onto the provider here, the single point every downstream
    reader of the address reads from.
    """
    tier_models = preset.tier_models
    subagent = preset.subagent_model
    model = preset.model

    if model_override:
        model = model_override
        if tier_models is not None:
            tier_models = TierModels.uniform(model_override)
        if subagent is not None:
            subagent = model_override

    provider: Provider | str = preset.provider
    url = base_url_override or preset.base_url
    if url:
        # Raises for a FIXED provider — a preset for one must not carry a
        # base_url at all (see Preset.base_url).
        provider = with_base_url(get_provider(preset.provider), url)

    return build_spec(
        agent=preset.agent,
        provider=provider,
        model=model,
        alias=alias_override or preset.alias,
        shape=preset.shape,
        tier_models=tier_models,
        subagent_model=subagent,
        # The preset's description names the preset's model, so it is wrong
        # the moment --model replaces it. Passing "" lets build_spec generate
        # one for the model actually installed — `list` is the only place a
        # user sees what a wrapper points at, so a stale line misinforms.
        description="" if model_override else preset.description,
        profile_name=profile_name,
    )
