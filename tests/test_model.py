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
    AuthPolicy,
    BaseUrlPolicy,
    ConfigShape,
    ModelListAPI,
    Provider,
    compatible_providers,
    get_agent,
    get_provider,
    resolve_shape,
    with_auth,
    with_base_url,
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
def test_registry_has_ollama_and_zai_and_litellm():
    assert {p.name for p in PROVIDERS} == {"ollama", "zai", "litellm"}


@pytest.mark.unit
def test_ollama_declares_three_shapes():
    """One provider, three connection mechanisms: direct HTTP (deepseek),
    `ollama launch` (glm-ollama), and a Codex TOML profile (codex × ollama)."""
    ollama = get_provider("ollama")
    assert ollama.shapes == {
        ConfigShape.ANTHROPIC_ENV,
        ConfigShape.OLLAMA_LAUNCH,
        ConfigShape.OPENAI_TOML,
    }
    assert ollama.wire_api == "responses"


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
def test_codex_ollama_uses_toml():
    # Two shapes are possible now; priority picks the TOML profile (no extra
    # binary on PATH). The launcher is still reachable via --shape ollama-launch.
    shape = resolve_shape(get_agent("codex"), get_provider("ollama"))
    assert shape is ConfigShape.OPENAI_TOML


@pytest.mark.unit
def test_codex_ollama_launcher_shape_is_reachable():
    """--shape ollama-launch overrides priority — the launcher path stays open."""
    shape = resolve_shape(
        get_agent("codex"), get_provider("ollama"), preferred=ConfigShape.OLLAMA_LAUNCH
    )
    assert shape is ConfigShape.OLLAMA_LAUNCH


@pytest.mark.unit
def test_claude_zai_uses_env():
    shape = resolve_shape(get_agent("claude"), get_provider("zai"))
    assert shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_litellm_declares_both_shapes():
    """A LiteLLM proxy serves both protocols off one host — the registry must
    say so honestly, the way test_ollama_declares_three_shapes does for
    ollama's three mechanisms."""
    litellm = get_provider("litellm")
    assert litellm.shapes == {ConfigShape.ANTHROPIC_ENV, ConfigShape.OPENAI_TOML}
    assert litellm.wire_api == "chat"
    assert litellm.base_url_policy is BaseUrlPolicy.REQUIRED
    assert litellm.base_url == ""


@pytest.mark.unit
def test_claude_litellm_resolves_to_anthropic_env():
    shape = resolve_shape(get_agent("claude"), get_provider("litellm"))
    assert shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_codex_litellm_resolves_to_openai_toml():
    shape = resolve_shape(get_agent("codex"), get_provider("litellm"))
    assert shape is ConfigShape.OPENAI_TOML


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
        "litellm",
    }


@pytest.mark.unit
def test_compatible_providers_for_codex_excludes_zai():
    """z.ai is Anthropic-only, and codex cannot speak that protocol."""
    assert {p.name for p in compatible_providers(get_agent("codex"))} == {
        "ollama",
        "litellm",
    }


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


# --------------------------------------------------------------------------- #
# BaseUrlPolicy / with_base_url — runtime base_url substitution
# --------------------------------------------------------------------------- #

# Built OUTSIDE the registry, same technique as _OPENAI_ONLY above: proves the
# rules are computed from the policy value, not enumerated per provider name.
_RUNTIME_REQUIRED = Provider(
    name="runtime-required",
    shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
    base_url="",
    base_url_policy=BaseUrlPolicy.REQUIRED,
    auth="secret",
    token_env_var="RUNTIME_API_KEY",
)

_RUNTIME_OVERRIDABLE = Provider(
    name="runtime-overridable",
    shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
    base_url="http://default-host:1/v1",
    base_url_policy=BaseUrlPolicy.OVERRIDABLE,
    auth="secret",
    token_env_var="RUNTIME_API_KEY",
)


@pytest.mark.unit
def test_fixed_provider_passes_through_with_no_url():
    ollama = get_provider("ollama")
    assert with_base_url(ollama, None) is ollama


@pytest.mark.unit
def test_fixed_provider_refuses_an_override():
    ollama = get_provider("ollama")
    with pytest.raises(CodeHelperError, match="fixed base URL"):
        with_base_url(ollama, "http://x/v1")


@pytest.mark.unit
def test_required_provider_refuses_without_a_url():
    with pytest.raises(CodeHelperError, match="needs a base URL"):
        with_base_url(_RUNTIME_REQUIRED, None)


@pytest.mark.unit
def test_required_provider_refuses_an_empty_url():
    with pytest.raises(CodeHelperError, match="needs a base URL"):
        with_base_url(_RUNTIME_REQUIRED, "")


@pytest.mark.unit
def test_required_provider_accepts_a_valid_url():
    got = with_base_url(_RUNTIME_REQUIRED, "http://host:4000/v1")
    assert got.base_url == "http://host:4000/v1"
    # The original registry object is untouched (frozen dataclass + replace).
    assert _RUNTIME_REQUIRED.base_url == ""


@pytest.mark.unit
def test_required_provider_propagates_url_validation():
    with pytest.raises(CodeHelperError, match="http:// or https://"):
        with_base_url(_RUNTIME_REQUIRED, "ftp://x")


@pytest.mark.unit
def test_overridable_provider_keeps_the_default_with_no_override():
    assert with_base_url(_RUNTIME_OVERRIDABLE, None) is _RUNTIME_OVERRIDABLE


@pytest.mark.unit
def test_overridable_provider_accepts_an_override():
    got = with_base_url(_RUNTIME_OVERRIDABLE, "http://custom-host:9/v1")
    assert got.base_url == "http://custom-host:9/v1"


@pytest.mark.unit
def test_fixed_refusal_message_lists_runtime_providers_from_the_registry(
    monkeypatch,
):
    """The error text is DERIVED from PROVIDERS, not a hard-coded name list."""
    import code_helper.services.model as m

    monkeypatch.setattr(m, "PROVIDERS", (get_provider("ollama"), _RUNTIME_REQUIRED))
    with pytest.raises(CodeHelperError, match="runtime-required"):
        with_base_url(get_provider("ollama"), "http://x/v1")


# --------------------------------------------------------------------------- #
# BaseUrlPolicy — registry validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_required_policy_with_a_registry_url_is_rejected():
    import code_helper.services.model as m

    bad = Provider(
        name="bad",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://should-not-be-here",
        base_url_policy=BaseUrlPolicy.REQUIRED,
    )
    with pytest.raises(CodeHelperError, match="base_url_policy=REQUIRED"):
        m._validate_provider(bad)


@pytest.mark.unit
@pytest.mark.parametrize("policy", [BaseUrlPolicy.FIXED, BaseUrlPolicy.OVERRIDABLE])
def test_non_required_policy_with_an_empty_url_is_rejected(policy):
    import code_helper.services.model as m

    bad = Provider(
        name="bad",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="",
        base_url_policy=policy,
    )
    with pytest.raises(CodeHelperError, match="no registry base_url"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_lowercase_token_env_var_is_rejected():
    import code_helper.services.model as m

    bad = Provider(
        name="bad",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
        token_env_var="lower_case_key",
    )
    with pytest.raises(CodeHelperError, match="invalid token_env_var"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_uppercase_token_env_var_is_accepted():
    import code_helper.services.model as m

    ok = Provider(
        name="ok",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
        token_env_var="MY_API_KEY",
    )
    m._validate_provider(ok)  # must not raise


@pytest.mark.unit
def test_secret_provider_with_no_token_env_var_is_rejected():
    """A registry-authoring mistake this import-time check exists to catch
    (cycle-review re-review finding on PR #12): auth='secret' with an empty
    token_env_var would let an OPENAI_TOML wrapper install successfully with
    NO credential wired in at all — openai_env_key returns "" for an empty
    token_env_var, so neither the profile's env_key nor the wrapper's
    `export` line gets written. No shipped provider hits this (zai/litellm
    both carry a real token_env_var); this pins the registry-validation net
    that would catch a future one that doesn't.
    """
    import code_helper.services.model as m

    bad = Provider(
        name="bad-secret",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.bad.invalid/v1",
        auth="secret",
        token_env_var="",  # missing — the whole point of this test
        wire_api="responses",
    )
    with pytest.raises(CodeHelperError, match="declares auth='secret' but has no"):
        m._validate_provider(bad)


# --------------------------------------------------------------------------- #
# AuthPolicy / with_auth — runtime auth override
# --------------------------------------------------------------------------- #

# Built OUTSIDE the registry, same "prove it's computed, not enumerated"
# technique as _OPENAI_ONLY / _RUNTIME_REQUIRED above.
_RUNTIME_AUTH_OVERRIDABLE = Provider(
    name="runtime-auth-overridable",
    shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
    base_url="http://default-host:1/v1",
    auth="literal",
    auth_value="default-token",
    auth_policy=AuthPolicy.OVERRIDABLE,
    token_env_var="RUNTIME_AUTH_API_KEY",
)

# A FIXED provider whose auth is not already "secret" — no shipped provider
# is FIXED+non-secret, so this out-of-registry stand-in is what exercises
# with_auth's refusal branch. Shared by both tests that need one.
_FIXED_LITERAL = Provider(
    name="fixed-literal",
    shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
    base_url="http://x",
    auth="literal",
    auth_value="x",
    auth_policy=AuthPolicy.FIXED,
)


@pytest.mark.unit
def test_fixed_auth_provider_passes_through_with_no_override():
    zai = get_provider("zai")
    assert with_auth(zai, want_secret=False) is zai


@pytest.mark.unit
def test_fixed_auth_provider_already_secret_ignores_override_request():
    """A FIXED provider that is already auth='secret' (zai, litellm) has
    nothing to override to — want_secret=True is a no-op, not a refusal, so a
    caller need not know which providers are already secret."""
    zai = get_provider("zai")
    assert with_auth(zai, want_secret=True) is zai


@pytest.mark.unit
def test_fixed_auth_provider_refuses_an_override_when_not_already_secret():
    with pytest.raises(CodeHelperError, match="fixed auth mode"):
        with_auth(_FIXED_LITERAL, want_secret=True)


@pytest.mark.unit
def test_overridable_auth_provider_keeps_the_default_with_no_override():
    assert (
        with_auth(_RUNTIME_AUTH_OVERRIDABLE, want_secret=False)
        is _RUNTIME_AUTH_OVERRIDABLE
    )


@pytest.mark.unit
def test_overridable_auth_provider_accepts_an_override():
    got = with_auth(_RUNTIME_AUTH_OVERRIDABLE, want_secret=True)
    assert got.auth == "secret"
    assert got.auth_value == ""  # the literal value must not linger, unused
    # The original registry object is untouched (frozen dataclass + replace).
    assert _RUNTIME_AUTH_OVERRIDABLE.auth == "literal"


@pytest.mark.unit
def test_auth_refusal_message_lists_overridable_providers_from_the_registry(
    monkeypatch,
):
    """The error text is DERIVED from PROVIDERS, not a hard-coded name list."""
    import code_helper.services.model as m

    monkeypatch.setattr(m, "PROVIDERS", (_FIXED_LITERAL, _RUNTIME_AUTH_OVERRIDABLE))
    with pytest.raises(CodeHelperError, match="runtime-auth-overridable"):
        with_auth(_FIXED_LITERAL, want_secret=True)


@pytest.mark.unit
def test_ollama_declares_overridable_auth_with_a_token_env_var():
    """ollama's real registry entry: OVERRIDABLE, with a token_env_var already
    present so with_auth has something to substitute in — pins the fix for
    the (incorrect) assumption that a local daemon can never sit behind auth
    (a reverse proxy, or Ollama Cloud, both routinely do)."""
    ollama = get_provider("ollama")
    assert ollama.auth_policy is AuthPolicy.OVERRIDABLE
    assert ollama.auth == "literal"
    assert ollama.token_env_var

    got = with_auth(ollama, want_secret=True)
    assert got.auth == "secret"
    assert got.token_env_var == ollama.token_env_var


# --------------------------------------------------------------------------- #
# AuthPolicy — registry validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_overridable_auth_policy_without_token_env_var_is_rejected():
    import code_helper.services.model as m

    bad = Provider(
        name="bad-overridable",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
        auth="literal",
        auth_value="x",
        auth_policy=AuthPolicy.OVERRIDABLE,
        token_env_var="",  # missing — the whole point of this test
    )
    with pytest.raises(CodeHelperError, match="auth_policy=OVERRIDABLE"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_fixed_auth_policy_without_token_env_var_is_accepted():
    """FIXED doesn't need a token_env_var unless auth='secret' itself (a
    separate, pre-existing check) — a literal-only provider like the
    registry's own ollama-before-this-change shape must stay valid."""
    import code_helper.services.model as m

    ok = Provider(
        name="ok-fixed",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
        auth="literal",
        auth_value="x",
        auth_policy=AuthPolicy.FIXED,
        token_env_var="",
    )
    m._validate_provider(ok)  # must not raise
