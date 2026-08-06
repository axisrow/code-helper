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
  ``glm-5.2[1m]``); that is not expressible as a single ``--model``, and
  demanding three flags for the common case would be absurd;
- ``code-helper add glm`` must keep working as one word.
"""

from __future__ import annotations

from dataclasses import dataclass

from code_helper.errors import CodeHelperError
from code_helper.services.model import (
    Agent,
    ConfigShape,
    Provider,
    get_agent,
    get_provider,
    resolve_shape,
)
from code_helper.services.naming import validate_alias

__all__ = [
    "WrapperSpec",
    "TierModels",
    "Preset",
    "PRESETS",
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
    repeated (``deepseek``). Collapsing this to a single field would lose the
    former.
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


def suggest_alias(model: str, agent_name: str) -> str:
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

    chosen = resolve_shape(agent_obj, provider_obj, preferred=shape)

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
        alias if alias is not None else suggest_alias(effective_model, agent_obj.name)
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


_DEEPSEEK_MODEL = "deepseek-v4-flash:0731-cloud"


PRESETS: tuple[Preset, ...] = (
    Preset(
        alias="deepseek",
        agent="claude",
        provider="ollama",
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
        model="glm-5-turbo",
        # Genuinely different models per tier — the reason TierModels exists.
        tier_models=TierModels(
            haiku="glm-4.7", sonnet="glm-5-turbo", opus="glm-5.2[1m]"
        ),
        subagent_model=None,
        description="Claude Code → Z.ai",
    ),
    Preset(
        alias="glm-ollama",
        agent="claude",
        provider="ollama",
        shape=ConfigShape.OLLAMA_LAUNCH,
        model="glm-5.2:cloud",
        description="Claude Code → glm-5.2:cloud via `ollama launch claude`",
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
) -> WrapperSpec:
    """Expand a preset into a spec, applying ``--model`` if given.

    ``model_override`` replaces every tier AND the subagent model, matching the
    pre-refactor behaviour of ``--model`` exactly.
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

    return build_spec(
        agent=preset.agent,
        provider=preset.provider,
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
    )
