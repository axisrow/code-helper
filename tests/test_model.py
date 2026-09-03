"""Tests for the agent × provider × model axes and compatibility resolution.

The central property under test: **impossible combinations are rejected by the
data, not by special-case code**. The strongest evidence is
``test_openai_only_provider_is_incompatible_with_claude`` — it uses a provider
that does not exist in the registry, so nothing in ``model.py`` could be
hard-coded to know about it, yet compatibility still comes out right.
"""

from __future__ import annotations

import pytest

from codehelper.errors import CodeHelperError
from codehelper.services.model import (
    AGENTS,
    PROVIDERS,
    Agent,
    AuthPolicy,
    BaseUrlPolicy,
    ConfigShape,
    ModelListAPI,
    Provider,
    active_providers,
    compatible_providers,
    get_agent,
    get_provider,
    is_provider_disabled,
    provider_storage_names,
    resolve_shape,
    switchable_providers,
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
def test_registry_covers_ollama_launch_integrations():
    """All 15 CLI integrations `ollama launch` supports are registered.

    Superset (not equality) on the two hand-built agents, plus an exact count
    for the whole registry — this is the pin that would catch a launch-only
    agent silently dropped or duplicated, without being so exact it breaks the
    moment a future agent is added deliberately.
    """
    names = {a.name for a in AGENTS}
    assert {"claude", "codex"} <= names
    assert len(AGENTS) == 15


@pytest.mark.unit
def test_launch_only_agents_declare_exactly_ollama_launch():
    """Every agent besides claude/codex is OLLAMA_LAUNCH-only.

    None of them speaks Codex's `--profile <alias>` TOML convention or
    Claude's `ANTHROPIC_*` env vars — each has its own native config format
    that only `ollama launch` itself knows how to write. Declaring a second
    shape for one of them would silently create a wrapper `render.py` cannot
    actually produce correctly.
    """
    for agent in AGENTS:
        if agent.name in ("claude", "codex"):
            continue
        assert agent.shapes == frozenset({ConfigShape.OLLAMA_LAUNCH}), agent.name


@pytest.mark.unit
def test_openai_toml_is_codex_only():
    """OPENAI_TOML — the `--profile <alias>` mechanism — is codex's alone.

    `_render_openai_toml` (render.py) hard-codes `--profile`, which only codex
    understands; this is the registry-side invariant that assumption depends
    on.
    """
    declarers = [a.name for a in AGENTS if ConfigShape.OPENAI_TOML in a.shapes]
    assert declarers == ["codex"]


@pytest.mark.unit
def test_registry_has_builtin_providers():
    assert {p.name for p in PROVIDERS} == {
        "ollama-direct",
        "zai",
        "litellm",
        "gemini",
        "deepseek",
        "deepseek-openai",
        "native",
    }


@pytest.mark.unit
def test_gemini_is_suspended_and_pairs_with_nothing():
    """gemini is registered but deliberately unusable (issue #74): Google's
    OpenAI-compat surface is chat-completions only — `/responses` 404s — while
    Codex hard-rejects `wire_api="chat"`, and no Anthropic-compatible surface
    exists at all. Empty shapes (legal ONLY under `suspended=True`) make every
    pairing resolve to the honest "no common configuration mechanism" instead
    of installing a wrapper that cannot work."""
    gemini = get_provider("gemini")
    assert gemini.shapes == frozenset()
    assert gemini.suspended is True
    # The endpoint/credential fields are kept exactly as the unsuspended
    # entry will need them — unsuspending is one commit (see the registry
    # comment): restore OPENAI_TOML, drop the flag.
    assert gemini.base_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    # The stored base_url IS the complete OpenAI root — the renderer must not
    # append /v1/ to it (that would 404). Pinned here so a regression to a
    # bare-root convention (which openai_base_url would rewrite to
    # .../openai/v1/) is caught at the registry level.
    assert gemini.base_url_is_openai_root is True
    assert gemini.auth == "secret"
    assert gemini.token_env_var == "GEMINI_API_KEY"
    assert gemini.model_list_api is ModelListAPI.OPENAI_V1
    # The value the restored profile will carry — never "chat" again.
    assert gemini.wire_api == "responses"


@pytest.mark.unit
def test_deepseek_is_anthropic_compatible_and_switchable():
    """Portrait of the cloud DeepSeek provider: Anthropic surface for claude,
    switchable, discovery redirected to the OpenAI surface."""
    deepseek = get_provider("deepseek")
    assert deepseek.shapes == {
        ConfigShape.ANTHROPIC_ENV,
        ConfigShape.ANTHROPIC_SETTINGS,
    }
    assert deepseek.base_url == "https://api.deepseek.com/anthropic"
    assert deepseek.auth == "secret"
    assert deepseek.token_env_var == "DEEPSEEK_API_KEY"
    # The /anthropic path serves no OpenAI model list — discovery must be
    # pointed at the OpenAI root explicitly (models_api.list_models uses
    # model_list_url over base_url).
    assert deepseek.model_list_api is ModelListAPI.OPENAI_V1
    assert deepseek.model_list_url == "https://api.deepseek.com"
    assert deepseek.known_models == (
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
    )


@pytest.mark.unit
def test_deepseek_openai_is_openai_only_and_uses_documented_conventions():
    """Portrait of the codex-facing provider: bare OpenAI root (NOT
    is_openai_root — openai_base_url appends /v1/), Responses wire_api."""
    ds = get_provider("deepseek-openai")
    assert ds.shapes == {ConfigShape.OPENAI_TOML}
    assert ds.base_url == "https://api.deepseek.com"
    assert ds.base_url_is_openai_root is False
    assert ds.auth == "secret"
    assert ds.token_env_var == "DEEPSEEK_API_KEY"
    assert ds.model_list_api is ModelListAPI.OPENAI_V1
    assert ds.wire_api == "responses"


@pytest.mark.unit
def test_deepseek_pairing_matrix():
    """claude x deepseek works (ANTHROPIC_ENV); codex x deepseek-openai works
    (OPENAI_TOML); the cross pairings are impossible by shape intersection."""
    assert (
        resolve_shape(get_agent("claude"), get_provider("deepseek"))
        is ConfigShape.ANTHROPIC_ENV
    )
    assert (
        resolve_shape(get_agent("codex"), get_provider("deepseek-openai"))
        is ConfigShape.OPENAI_TOML
    )
    with pytest.raises(CodeHelperError, match="no common configuration"):
        resolve_shape(get_agent("claude"), get_provider("deepseek-openai"))
    with pytest.raises(CodeHelperError, match="no common configuration"):
        resolve_shape(get_agent("codex"), get_provider("deepseek"))


@pytest.mark.unit
def test_claude_gemini_has_no_common_configuration_mechanism():
    with pytest.raises(CodeHelperError, match="no common configuration"):
        resolve_shape(get_agent("claude"), get_provider("gemini"))


@pytest.mark.unit
def test_codex_gemini_is_suspended_not_silently_broken():
    """Issue #74: codex x gemini used to install a profile carrying
    `wire_api="chat"`, which current Codex hard-rejects at config load —
    a wrapper that dies before any request. The pairing must now resolve to
    the honest "no common configuration mechanism" instead."""
    with pytest.raises(CodeHelperError, match="no common configuration"):
        resolve_shape(get_agent("codex"), get_provider("gemini"))


@pytest.mark.unit
def test_ollama_declares_three_shapes():
    """One provider, three connection mechanisms: direct HTTP (deepseek-ollama),
    `ollama launch` (glm-ollama), and a Codex TOML profile (codex × ollama) —
    plus ANTHROPIC_SETTINGS, so a live claude session can `switch` to it too."""
    ollama = get_provider("ollama-direct")
    assert ollama.shapes == {
        ConfigShape.ANTHROPIC_ENV,
        ConfigShape.OLLAMA_LAUNCH,
        ConfigShape.OPENAI_TOML,
        ConfigShape.ANTHROPIC_SETTINGS,
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
    shape = resolve_shape(get_agent("claude"), get_provider("ollama-direct"))
    assert shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_codex_ollama_uses_toml():
    # Two shapes are possible now; priority picks the TOML profile (no extra
    # binary on PATH). The launcher is still reachable via --shape ollama-launch.
    shape = resolve_shape(get_agent("codex"), get_provider("ollama-direct"))
    assert shape is ConfigShape.OPENAI_TOML


@pytest.mark.unit
def test_codex_ollama_launcher_shape_is_reachable():
    """--shape ollama-launch overrides priority — the launcher path stays open."""
    shape = resolve_shape(
        get_agent("codex"),
        get_provider("ollama-direct"),
        preferred=ConfigShape.OLLAMA_LAUNCH,
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
    assert litellm.shapes == {
        ConfigShape.ANTHROPIC_ENV,
        ConfigShape.OPENAI_TOML,
        ConfigShape.ANTHROPIC_SETTINGS,
    }
    assert litellm.wire_api == "responses"
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
    with pytest.raises(CodeHelperError, match="works with: ollama-direct, zai"):
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
        get_provider("ollama-direct"),
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
    resolve_shape(get_agent("claude"), get_provider("ollama-direct"))
    assert set(tmp_path.iterdir()) == before


# --------------------------------------------------------------------------- #
# compatible_providers — what a menu is allowed to show
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_compatible_providers_for_claude():
    assert {p.name for p in compatible_providers(get_agent("claude"))} == {
        "ollama-direct",
        "zai",
        "litellm",
        "deepseek",
    }


@pytest.mark.unit
def test_compatible_providers_for_codex_excludes_zai_and_suspended():
    """z.ai is Anthropic-only, and codex cannot speak that protocol; gemini
    is suspended (#74) and pairs with no agent at all."""
    assert {p.name for p in compatible_providers(get_agent("codex"))} == {
        "ollama-direct",
        "litellm",
        "deepseek-openai",
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
    import codehelper.services.model as m

    bad = Agent("x", "evil; rm -rf /", frozenset({ConfigShape.OLLAMA_LAUNCH}))
    monkeypatch.setattr(m, "AGENTS", (bad,))
    with pytest.raises(CodeHelperError, match="invalid agent binary"):
        m._validate_registries()


@pytest.mark.unit
def test_validate_registries_rejects_shapeless_agent(monkeypatch):
    import codehelper.services.model as m

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
    ollama = get_provider("ollama-direct")
    assert with_base_url(ollama, None) is ollama


@pytest.mark.unit
def test_fixed_provider_refuses_an_override():
    ollama = get_provider("ollama-direct")
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
def test_required_provider_auto_completes_a_bare_host():
    got = with_base_url(_RUNTIME_REQUIRED, "78.47.183.125")
    # No path is guessed — see normalize_base_url's docstring for why.
    assert got.base_url == "https://78.47.183.125:4000"


@pytest.mark.unit
def test_required_provider_keeps_explicit_http_scheme():
    got = with_base_url(_RUNTIME_REQUIRED, "http://127.0.0.1:11434")
    assert got.base_url == "http://127.0.0.1:11434"


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
    import codehelper.services.model as m

    monkeypatch.setattr(
        m, "PROVIDERS", (get_provider("ollama-direct"), _RUNTIME_REQUIRED)
    )
    with pytest.raises(CodeHelperError, match="runtime-required"):
        with_base_url(get_provider("ollama-direct"), "http://x/v1")


# --------------------------------------------------------------------------- #
# BaseUrlPolicy — registry validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_required_policy_with_a_registry_url_is_rejected():
    import codehelper.services.model as m

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
    import codehelper.services.model as m

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
    import codehelper.services.model as m

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
    import codehelper.services.model as m

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
    import codehelper.services.model as m

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


@pytest.mark.unit
def test_chat_wire_api_is_rejected_for_openai_toml_providers():
    """Issue #74: current Codex hard-rejects `wire_api="chat"` at config
    deserialization (its WireApi enum has a single Responses variant,
    openai/codex#7782), so a chat-carrying profile is a wrapper that dies
    before any request. The value must fail at IMPORT time, not at `codex`
    runtime — "responses" is the only wire_api a profile can carry."""
    import codehelper.services.model as m

    bad = Provider(
        name="bad-chat",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.bad.invalid/v1",
        auth="literal",
        auth_value="x",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="chat",
    )
    with pytest.raises(CodeHelperError, match="openai/codex#7782"):
        m._validate_provider(bad)

    # And no SHIPPED provider carries it — the registry itself is clean
    # (gemini, the one carrier, is suspended shapeless and now declares
    # "responses" for its eventual return).
    for provider in PROVIDERS:
        if ConfigShape.OPENAI_TOML in provider.shapes:
            assert provider.wire_api == "responses", provider.name


@pytest.mark.unit
def test_shapeless_provider_requires_the_suspended_flag():
    """Empty shapes pair with nothing, so they are an authoring mistake —
    UNLESS declared suspended (issue #74): a documented endpoint whose real
    surfaces match no shape stays registered and honestly inert. The flag is
    what separates a deliberate suspension from a typo'd entry."""
    import codehelper.services.model as m

    shapeless = Provider(name="shapeless", shapes=frozenset(), base_url="http://x")
    with pytest.raises(CodeHelperError, match="suspended=True is required"):
        m._validate_provider(shapeless)

    suspended = Provider(
        name="suspended",
        shapes=frozenset(),
        suspended=True,
        base_url="https://documented.invalid/surface/",
        auth="secret",
        token_env_var="SUSPENDED_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
    )
    m._validate_provider(suspended)  # must not raise

    # And the registry's one suspended entry really is shapeless + flagged.
    gemini = get_provider("gemini")
    assert gemini.suspended is True
    assert gemini.shapes == frozenset()


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
    import codehelper.services.model as m

    monkeypatch.setattr(m, "PROVIDERS", (_FIXED_LITERAL, _RUNTIME_AUTH_OVERRIDABLE))
    with pytest.raises(CodeHelperError, match="runtime-auth-overridable"):
        with_auth(_FIXED_LITERAL, want_secret=True)


@pytest.mark.unit
def test_ollama_declares_overridable_auth_with_a_token_env_var():
    """ollama's real registry entry: OVERRIDABLE, with a token_env_var already
    present so with_auth has something to substitute in — pins the fix for
    the (incorrect) assumption that a local daemon can never sit behind auth
    (a reverse proxy, or Ollama Cloud, both routinely do)."""
    ollama = get_provider("ollama-direct")
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
    import codehelper.services.model as m

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
    import codehelper.services.model as m

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


# --------------------------------------------------------------------------- #
# ANTHROPIC_SETTINGS / env_reset — the `switch` mechanism
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_anthropic_settings_shape_on_no_agent():
    """No Agent may declare ANTHROPIC_SETTINGS — this is the guard that makes
    wrapper generation provably immune to this shape's existence, rather than
    relying on _SHAPE_PRIORITY ordering to keep it out of the way."""
    from codehelper.services.model import AGENTS

    assert all(ConfigShape.ANTHROPIC_SETTINGS not in a.shapes for a in AGENTS)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("agent_name", "provider_name", "expected"),
    [
        ("claude", "zai", ConfigShape.ANTHROPIC_ENV),
        ("claude", "ollama-direct", ConfigShape.ANTHROPIC_ENV),
        ("claude", "litellm", ConfigShape.ANTHROPIC_ENV),
        ("codex", "ollama-direct", ConfigShape.OPENAI_TOML),
        ("codex", "litellm", ConfigShape.OPENAI_TOML),
        # ("codex", "gemini", ...) removed with #74: the pairing is suspended,
        # resolve_shape raises instead of handing back OPENAI_TOML.
    ],
)
def test_resolve_shape_unchanged_for_existing_pairs(
    agent_name, provider_name, expected
):
    """Pin today's resolve_shape output for every pre-existing pairing —
    adding ANTHROPIC_SETTINGS to ollama/zai/litellm and env_reset to `native`
    must not perturb any of these."""
    shape = resolve_shape(get_agent(agent_name), get_provider(provider_name))
    assert shape is expected


@pytest.mark.unit
def test_claude_native_pairing_is_incompatible():
    """`native` presents only ANTHROPIC_SETTINGS, which no Agent consumes —
    so it can never back a generated wrapper, only a live `switch`."""
    with pytest.raises(CodeHelperError, match="no common configuration mechanism"):
        resolve_shape(get_agent("claude"), get_provider("native"))


@pytest.mark.unit
def test_switchable_providers_includes_native_and_the_settings_providers():

    assert {p.name for p in switchable_providers()} == {
        "ollama-direct",
        "zai",
        "litellm",
        "deepseek",
        "native",
    }


@pytest.mark.unit
def test_env_reset_provider_rejects_a_base_url():
    import codehelper.services.model as m

    bad = Provider(
        name="bad-reset",
        shapes=frozenset({ConfigShape.ANTHROPIC_SETTINGS}),
        base_url="http://x",
        env_reset=True,
    )
    with pytest.raises(CodeHelperError, match="env_reset=True"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_env_reset_provider_rejects_a_credential():
    import codehelper.services.model as m

    bad = Provider(
        name="bad-reset",
        shapes=frozenset({ConfigShape.ANTHROPIC_SETTINGS}),
        auth="secret",
        token_env_var="BAD_API_KEY",
        env_reset=True,
    )
    with pytest.raises(CodeHelperError, match="env_reset=True"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_env_reset_provider_rejects_a_model_list_api():
    import codehelper.services.model as m

    bad = Provider(
        name="bad-reset",
        shapes=frozenset({ConfigShape.ANTHROPIC_SETTINGS}),
        model_list_api=ModelListAPI.OPENAI_V1,
        env_reset=True,
    )
    with pytest.raises(CodeHelperError, match="env_reset=True"):
        m._validate_provider(bad)


@pytest.mark.unit
def test_env_reset_provider_with_no_address_or_credential_is_accepted():
    import codehelper.services.model as m

    ok = Provider(
        name="ok-reset",
        shapes=frozenset({ConfigShape.ANTHROPIC_SETTINGS}),
        env_reset=True,
    )
    m._validate_provider(ok)  # must not raise


# --------------------------------------------------------------------------- #
# runtime disable — active_providers / is_provider_disabled / resolver threading
# (issue #89: disabled-ness is state.json state, independent of `suspended`)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_active_providers_none_means_no_filtering():
    assert active_providers(None) == list(PROVIDERS)


@pytest.mark.unit
def test_active_providers_filters_disabled_keeps_order():
    disabled = frozenset({"ollama-direct"})
    names = [p.name for p in active_providers(disabled)]
    assert "ollama-direct" not in names
    expected = [p.name for p in PROVIDERS if p.name != "ollama-direct"]
    assert names == expected


@pytest.mark.unit
def test_disabled_is_not_suspended():
    """The two flags are orthogonal: disabling at runtime must not touch the
    registry entry, and gemini's static suspension is not a disable."""
    ollama = get_provider("ollama-direct")
    assert not ollama.suspended
    assert is_provider_disabled(ollama, frozenset({"ollama-direct"}))
    assert not is_provider_disabled(ollama, frozenset())
    assert not is_provider_disabled(ollama, None)


@pytest.mark.unit
def test_compatible_providers_excludes_disabled():
    agent = get_agent("claude")
    full = compatible_providers(agent)
    assert any(p.name == "ollama-direct" for p in full)
    filtered = compatible_providers(agent, disabled=frozenset({"ollama-direct"}))
    assert all(p.name != "ollama-direct" for p in filtered)


@pytest.mark.unit
def test_provider_storage_names_is_the_one_spelling_decision_point():
    """``provider_storage_names`` owns "which spellings may identify this
    provider in persisted records": the canonical name plus its retired
    predecessor, nothing else — and an unrenamed provider is just itself."""
    assert provider_storage_names("ollama-direct") == ("ollama-direct", "ollama")
    assert provider_storage_names("zai") == ("zai",)
