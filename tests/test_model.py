"""Tests for the agent × provider × model axes and compatibility resolution.

The central property under test: **impossible combinations are rejected by the
data, not by special-case code**. The strongest evidence is
``test_openai_only_provider_is_incompatible_with_claude`` — it uses a provider
that does not exist in the registry, so nothing in ``model.py`` could be
hard-coded to know about it, yet compatibility still comes out right.
"""

from __future__ import annotations

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.model import (
    AGENTS,
    PROVIDERS,
    Agent,
    ConfigShape,
    ModelListAPI,
    Provider,
    compatible_providers,
    get_agent,
    get_provider,
    resolve_shape,
)

# A provider that speaks ONLY the OpenAI protocol — the shape NVIDIA and
# friends have. Deliberately NOT in the registry: it proves compatibility is
# computed, not enumerated.
_OPENAI_ONLY = Provider(
    name="openai-only",
    shapes=frozenset({ConfigShape.OPENAI_TOML}),
    base_url="https://example.invalid/v1",
    auth="secret",
    token_env_var="EXAMPLE_API_KEY",
    model_list_api=ModelListAPI.OPENAI_V1,
    wire_api="chat",
)


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_registry_has_claude_and_codex():
    assert {a.name for a in AGENTS} == {"claude", "codex"}


@pytest.mark.unit
def test_registry_has_ollama_and_zai():
    assert {p.name for p in PROVIDERS} == {"ollama", "zai"}


@pytest.mark.unit
def test_ollama_declares_both_shapes():
    """One provider, two connection mechanisms — deepseek vs glm-ollama."""
    ollama = get_provider("ollama")
    assert ollama.shapes == {ConfigShape.ANTHROPIC_ENV, ConfigShape.OLLAMA_LAUNCH}


@pytest.mark.unit
def test_codex_cannot_be_configured_by_env():
    """Codex has no OPENAI_BASE_URL equivalent — only TOML or a launcher."""
    codex = get_agent("codex")
    assert ConfigShape.ANTHROPIC_ENV not in codex.shapes


@pytest.mark.unit
def test_get_agent_unknown_raises():
    with pytest.raises(CodeHelperError, match="unknown agent"):
        get_agent("nope")


@pytest.mark.unit
def test_get_provider_unknown_raises():
    with pytest.raises(CodeHelperError, match="unknown provider"):
        get_provider("nope")


# --------------------------------------------------------------------------- #
# resolve_shape — the compatibility gate
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_claude_ollama_prefers_direct_http():
    # Both shapes are possible; priority picks the one needing no extra binary.
    shape = resolve_shape(get_agent("claude"), get_provider("ollama"))
    assert shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_codex_ollama_uses_launcher():
    shape = resolve_shape(get_agent("codex"), get_provider("ollama"))
    assert shape is ConfigShape.OLLAMA_LAUNCH


@pytest.mark.unit
def test_claude_zai_uses_env():
    shape = resolve_shape(get_agent("claude"), get_provider("zai"))
    assert shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_openai_only_provider_is_incompatible_with_claude():
    """The headline case: an OpenAI-only endpoint cannot drive Claude Code.

    Verified against reality — such endpoints 404 on `/v1/messages`. No code in
    model.py mentions this provider; the empty shape intersection produces the
    refusal on its own.
    """
    with pytest.raises(CodeHelperError, match="no common configuration"):
        resolve_shape(get_agent("claude"), _OPENAI_ONLY)


@pytest.mark.unit
def test_incompatibility_error_suggests_working_providers():
    with pytest.raises(CodeHelperError, match="works with: ollama, zai"):
        resolve_shape(get_agent("claude"), _OPENAI_ONLY)


@pytest.mark.unit
def test_codex_with_openai_only_provider_resolves_to_toml():
    """Distinct from incompatibility: the pairing is valid, just unimplemented.

    resolve_shape must NOT refuse here — refusing would conflate "impossible"
    with "not built yet". The unsupported-shape error belongs to the renderer.
    """
    shape = resolve_shape(get_agent("codex"), _OPENAI_ONLY)
    assert shape is ConfigShape.OPENAI_TOML


@pytest.mark.unit
def test_preferred_shape_overrides_priority():
    shape = resolve_shape(
        get_agent("claude"),
        get_provider("ollama"),
        preferred=ConfigShape.OLLAMA_LAUNCH,
    )
    assert shape is ConfigShape.OLLAMA_LAUNCH


@pytest.mark.unit
def test_preferred_shape_not_shared_raises():
    with pytest.raises(CodeHelperError, match="cannot use shape"):
        resolve_shape(
            get_agent("claude"),
            get_provider("zai"),
            preferred=ConfigShape.OLLAMA_LAUNCH,
        )


@pytest.mark.unit
def test_resolve_shape_is_pure(tmp_path):
    """Safe to call from a menu: no filesystem effects."""
    before = set(tmp_path.iterdir())
    resolve_shape(get_agent("claude"), get_provider("ollama"))
    assert set(tmp_path.iterdir()) == before


# --------------------------------------------------------------------------- #
# compatible_providers — what a menu is allowed to show
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_compatible_providers_for_claude():
    assert {p.name for p in compatible_providers(get_agent("claude"))} == {
        "ollama",
        "zai",
    }


@pytest.mark.unit
def test_compatible_providers_for_codex_excludes_zai():
    """z.ai is Anthropic-only, and codex cannot speak that protocol."""
    assert {p.name for p in compatible_providers(get_agent("codex"))} == {"ollama"}


@pytest.mark.unit
def test_every_listed_provider_actually_resolves():
    """The menu's filter and the resolver cannot disagree."""
    for agent in AGENTS:
        for provider in compatible_providers(agent):
            assert resolve_shape(agent, provider) in agent.shapes & provider.shapes


# --------------------------------------------------------------------------- #
# Registry validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_agent_binaries_are_safe_to_interpolate():
    """render.py interpolates `binary` unquoted; this is what makes that safe."""
    import re

    for agent in AGENTS:
        assert re.match(r"\A[a-z][a-z0-9_-]*\Z", agent.binary)


@pytest.mark.unit
def test_validate_registries_rejects_bad_binary(monkeypatch):
    import code_helper.services.model as m

    bad = Agent("x", "evil; rm -rf /", frozenset({ConfigShape.OLLAMA_LAUNCH}))
    monkeypatch.setattr(m, "AGENTS", (bad,))
    with pytest.raises(CodeHelperError, match="invalid agent binary"):
        m._validate_registries()


@pytest.mark.unit
def test_validate_registries_rejects_shapeless_agent(monkeypatch):
    import code_helper.services.model as m

    monkeypatch.setattr(m, "AGENTS", (Agent("x", "x", frozenset()),))
    with pytest.raises(CodeHelperError, match="no config shapes"):
        m._validate_registries()
