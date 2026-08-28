"""Tests for the credential cache and ``resolve_token`` source tracking.

The cache (``~/.config/codehelper/credentials.json``) is the one persistent
copy of a token outside a generated wrapper, and it is a CACHE — never a
session. These tests pin:

- ``load_credentials`` never raises (missing/binary/malformed/oddly-shaped
  files degrade to ``{}``, the same never-fails contract as
  ``models_api.list_models``).
- ``save_credential`` writes ``0o600`` and never clobbers another provider's
  credential.
- ``resolve_token`` priority env > cache > prompt, and reports its source so a
  caller caches only a prompt-resolved value.
- ``token_for_discovery`` never prompts, never raises, returns ``""`` for a
  non-secret provider.
"""

from __future__ import annotations

import json
import stat

import pytest

from codehelper.services.paths import Paths
from codehelper.services.secrets import (
    DEFAULT_PROFILE,
    SOURCE_CACHE,
    SOURCE_ENV,
    SOURCE_PROMPT,
    ResolvedToken,
    credential_for,
    env_cache_conflict,
    invalidate_cached_credential,
    load_credentials,
    profile_names,
    rename_profile,
    resolve_token,
    save_credential,
    seed_default_profile,
    token_for_discovery,
)


def _paths(tmp_path) -> Paths:
    return Paths.from_home(tmp_path)


def _write_credentials_file(paths: Paths, content: str) -> None:
    """Hand-write ``credentials.json``, bypassing ``save_credential`` — for
    tests that need a specific (often malformed) byte layout on disk."""
    paths.credentials_file().parent.mkdir(parents=True, exist_ok=True)
    paths.credentials_file().write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# load_credentials — never raises
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_credentials_missing_file_is_empty(tmp_path):
    assert load_credentials(_paths(tmp_path)) == {}


@pytest.mark.unit
def test_load_credentials_malformed_json_is_empty(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, "{not valid json")
    assert load_credentials(paths) == {}


@pytest.mark.unit
def test_load_credentials_non_dict_payload_is_empty(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, '["litellm", "zai"]')
    assert load_credentials(paths) == {}


@pytest.mark.unit
def test_load_credentials_skips_non_string_values(tmp_path):
    """A stray non-string entry must not cost the user the rest of the cache."""
    paths = _paths(tmp_path)
    _write_credentials_file(
        paths, json.dumps({"litellm": "sk-1", "zai": 12345, "": "empty-key"})
    )
    assert load_credentials(paths) == {"litellm": {DEFAULT_PROFILE: "sk-1"}}


@pytest.mark.unit
def test_load_credentials_skips_empty_values(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, json.dumps({"litellm": "", "zai": "sk-real"}))
    assert load_credentials(paths) == {"zai": {DEFAULT_PROFILE: "sk-real"}}


@pytest.mark.unit
def test_credential_for_missing_provider_is_empty_string(tmp_path):
    assert credential_for(_paths(tmp_path), "litellm") == ""


@pytest.mark.unit
def test_credential_for_finds_a_token_saved_under_a_retired_provider_name(tmp_path):
    """A token cached under "ollama" (before the ollama -> ollama-direct
    rename) must stay reachable when queried under the CURRENT name — the
    old credentials.json entry is not migrated in place, just still found."""
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-legacy")
    assert credential_for(paths, "ollama-direct") == "sk-legacy"


@pytest.mark.unit
def test_credential_for_prefers_the_current_name_over_the_retired_one(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-legacy")
    save_credential(paths, "ollama-direct", "sk-current")
    assert credential_for(paths, "ollama-direct") == "sk-current"


@pytest.mark.unit
def test_profile_names_includes_profiles_saved_under_a_retired_provider_name(
    tmp_path,
):
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-legacy", profile_name="proxy")
    assert "proxy" in profile_names(paths, "ollama-direct")


# --------------------------------------------------------------------------- #
# save_credential
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_save_credential_creates_file_owner_only(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-1")
    mode = stat.S_IMODE(paths.credentials_file().stat().st_mode)
    assert mode == 0o600


@pytest.mark.unit
def test_save_credential_round_trips(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-1")
    assert credential_for(paths, "litellm") == "sk-1"


@pytest.mark.unit
def test_save_credential_does_not_clobber_another_provider(tmp_path):
    """Read-modify-write: one provider's token never overwrites another's."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-1")
    save_credential(paths, "zai", "sk-2")
    creds = load_credentials(paths)
    assert creds == {
        "litellm": {DEFAULT_PROFILE: "sk-1"},
        "zai": {DEFAULT_PROFILE: "sk-2"},
    }


@pytest.mark.unit
def test_save_credential_overwrites_same_provider(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-old")
    save_credential(paths, "litellm", "sk-new")
    assert credential_for(paths, "litellm") == "sk-new"


@pytest.mark.unit
def test_save_credential_keeps_multiple_profiles_for_one_provider(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-one", "work")
    save_credential(paths, "zai", "sk-two", "personal")

    assert profile_names(paths, "zai") == ("personal", "work")
    assert credential_for(paths, "zai", "work") == "sk-one"
    assert credential_for(paths, "zai", "personal") == "sk-two"


@pytest.mark.unit
def test_seed_default_profile_recovers_only_when_provider_has_no_profiles(tmp_path):
    paths = _paths(tmp_path)

    assert seed_default_profile(paths, "zai", "sk-existing") is True
    assert credential_for(paths, "zai") == "sk-existing"

    # A second recovery must not replace the existing default token.
    assert seed_default_profile(paths, "zai", "sk-other") is False
    assert credential_for(paths, "zai") == "sk-existing"


@pytest.mark.unit
def test_seed_default_profile_leaves_malformed_cache_untouched(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, "not json")

    assert seed_default_profile(paths, "zai", "sk-existing") is False
    assert paths.credentials_file().read_text() == "not json"


@pytest.mark.unit
def test_flat_legacy_credential_is_read_as_default_profile(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, json.dumps({"zai": "sk-legacy"}))

    assert profile_names(paths, "zai") == (DEFAULT_PROFILE,)
    assert credential_for(paths, "zai") == "sk-legacy"


@pytest.mark.unit
def test_rename_profile_preserves_the_token(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-one")

    rename_profile(paths, "zai", DEFAULT_PROFILE, "work")

    assert credential_for(paths, "zai", DEFAULT_PROFILE) == ""
    assert credential_for(paths, "zai", "work") == "sk-one"


@pytest.mark.unit
def test_rename_profile_finds_a_profile_stored_under_a_retired_provider_name(
    tmp_path,
):
    """A profile visible via profile_names(paths, "ollama-direct") only
    because it lives under the retired "ollama" key (see
    test_profile_names_includes_profiles_saved_under_a_retired_provider_name)
    must actually be renamable through the current name — not silently
    no-op just because rename_profile only ever looked at the CURRENT
    name's own dict entry."""
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-legacy", profile_name="work")
    assert "work" in profile_names(paths, "ollama-direct")

    rename_profile(paths, "ollama-direct", "work", "personal")

    assert credential_for(paths, "ollama-direct", "personal") == "sk-legacy"
    assert "work" not in profile_names(paths, "ollama-direct")


@pytest.mark.unit
def test_save_credential_empty_token_is_a_noop(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "")
    assert not paths.credentials_file().exists()


@pytest.mark.unit
def test_save_credential_preserves_foreign_file_content(tmp_path):
    """A hand-added unknown key survives a save of a known one."""
    paths = _paths(tmp_path)
    _write_credentials_file(paths, json.dumps({"future-provider": "sk-x"}))
    save_credential(paths, "litellm", "sk-1")
    creds = load_credentials(paths)
    assert creds == {
        "future-provider": {DEFAULT_PROFILE: "sk-x"},
        "litellm": {DEFAULT_PROFILE: "sk-1"},
    }


# --------------------------------------------------------------------------- #
# invalidate_cached_credential — drop a stale entry, never raise
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_invalidate_cached_credential_removes_the_entry(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-old")
    invalidate_cached_credential(paths, "zai")
    assert credential_for(paths, "zai") == ""


@pytest.mark.unit
def test_invalidate_cached_credential_preserves_other_providers(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-old")
    save_credential(paths, "litellm", "sk-other")
    invalidate_cached_credential(paths, "zai")
    assert credential_for(paths, "zai") == ""
    assert credential_for(paths, "litellm") == "sk-other"


@pytest.mark.unit
def test_invalidate_cached_credential_missing_provider_is_a_noop(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-other")
    invalidate_cached_credential(paths, "zai")  # never cached — nothing to do
    assert credential_for(paths, "litellm") == "sk-other"


@pytest.mark.unit
def test_invalidate_cached_credential_missing_file_is_a_noop(tmp_path):
    paths = _paths(tmp_path)
    invalidate_cached_credential(paths, "zai")  # no file at all yet
    assert not paths.credentials_file().exists()


@pytest.mark.unit
def test_invalidate_cached_credential_also_drops_the_retired_name_entry(tmp_path):
    """A token cached under a RETIRED provider name (e.g. "ollama" before the
    ollama -> ollama-direct rename) must not survive invalidating the CURRENT
    name — credential_for falls back to the retired name (see
    test_credential_for_finds_a_token_saved_under_a_retired_provider_name), so
    if invalidation only dropped the current name's entry, a revoked/rotated
    token cached under the old key would resurface on the very next
    credential_for("ollama-direct") call, silently reinstalling it into a
    freshly generated wrapper."""
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-old-revoked")
    assert credential_for(paths, "ollama-direct") == "sk-old-revoked"

    invalidate_cached_credential(paths, "ollama-direct")

    assert credential_for(paths, "ollama-direct") == ""


@pytest.mark.unit
def test_invalidate_cached_credential_retired_name_drop_preserves_other_profiles(
    tmp_path,
):
    """Dropping the retired-name entry during invalidation must only remove
    the targeted profile, not every profile cached under the retired name."""
    paths = _paths(tmp_path)
    save_credential(paths, "ollama", "sk-old-default")
    save_credential(paths, "ollama", "sk-old-proxy", profile_name="proxy")

    invalidate_cached_credential(paths, "ollama-direct")

    assert credential_for(paths, "ollama-direct") == ""
    assert credential_for(paths, "ollama-direct", "proxy") == "sk-old-proxy"


@pytest.mark.unit
def test_invalidate_cached_credential_swallows_oserror(tmp_path, capsys, monkeypatch):
    """Matches cache_freshly_typed_token's own OSError handling (finding K):
    the docstring promises "never raises", but until this fix the body called
    atomic_write with no try/except — a disk-full/permission-denied failure
    there would crash the process after the wrapper install already
    succeeded. Warn and move on instead."""
    import codehelper.services.secrets as secrets

    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-stale")

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("codehelper.backends._atomic.atomic_write", _boom)
    monkeypatch.setattr(secrets, "atomic_write", _boom)
    secrets.invalidate_cached_credential(paths, "zai")

    # Must not raise (proven by reaching this line) and must tell the user.
    assert "warning" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# cache_freshly_typed_token — best-effort: a persistence failure must not
# crash a caller whose wrapper install already succeeded.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cache_freshly_typed_token_swallows_oserror(tmp_path, capsys, monkeypatch):
    """A cache write is best-effort AFTER a successful install (the ordering
    round 1's fix introduced): the install already happened, so a disk-full/
    permission-denied/other OSError writing the cache must not propagate as
    an uncaught exception — that would crash the process for what is, from
    the user's perspective, a fully successful command. Warn and move on."""
    import codehelper.services.secrets as secrets

    paths = _paths(tmp_path)

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(secrets, "save_credential", _boom)
    secrets.cache_freshly_typed_token(
        paths, "litellm", "sk-typed", source=SOURCE_PROMPT, dry_run=False
    )

    # Must not raise (proven by reaching this line) and must tell the user.
    assert "warning" in capsys.readouterr().err.lower()


@pytest.mark.unit
def test_cache_freshly_typed_token_still_raises_nothing_on_success(tmp_path):
    """Sanity check: the try/except added for the OSError case does not
    swallow a normal, successful save — it still lands in the cache."""
    from codehelper.services.secrets import cache_freshly_typed_token

    paths = _paths(tmp_path)
    cache_freshly_typed_token(
        paths, "litellm", "sk-typed", source=SOURCE_PROMPT, dry_run=False
    )
    assert credential_for(paths, "litellm") == "sk-typed"


# --------------------------------------------------------------------------- #
# resolve_token — priority + source
# --------------------------------------------------------------------------- #


def _no_prompt(_prompt):  # pragma: no cover - must not run in env/cache cells
    raise AssertionError("must not prompt when env or cache has the token")


@pytest.mark.unit
def test_resolve_token_env_wins_over_cache(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        environ={"LITELLM_API_KEY": "sk-env"},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-env"
    assert resolved.source == SOURCE_ENV


@pytest.mark.unit
def test_resolve_token_cache_wins_over_prompt(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        environ={},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-cached"
    assert resolved.source == SOURCE_CACHE


@pytest.mark.unit
def test_resolve_token_ignores_cache_for_a_runtime_address_provider(tmp_path):
    """The INSTALL path must not hand a cached token to a different host either.

    Mirrors ``token_for_discovery``'s ``base_url_policy`` gate: a token cached
    for ``litellm`` at one ``--base-url`` must not be silently reused (and
    baked into a wrapper script) for a *different* ``--base-url`` given for
    the same provider name. Falls through to the prompt instead.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached-for-host-a")

    def _fake_prompt(_prompt):
        return "sk-typed-for-host-b"

    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        base_url_policy="required",
        environ={},
        getpass_fn=_fake_prompt,
    )
    assert resolved.value == "sk-typed-for-host-b"
    assert resolved.source == SOURCE_PROMPT


@pytest.mark.unit
def test_resolve_token_trusts_a_named_profiles_cache_for_a_runtime_address_provider(
    tmp_path,
):
    """A NAMED profile's cache is trusted for ANY base_url_policy.

    Unlike the unnamed/default cache (host-isolation guarded elsewhere in
    this file), a profile the user explicitly named via ``--profile`` is a
    deliberate per-invocation choice — the same reasoning that already lets
    an explicit profile win over the environment. This is what makes named
    profiles usable at all for a REQUIRED-policy provider like ``litellm``,
    which the profile feature explicitly documents supporting.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached-for-work", "work")

    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        profile_name="work",
        base_url_policy="required",
        environ={},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-cached-for-work"
    assert resolved.source == SOURCE_CACHE


@pytest.mark.unit
def test_resolve_token_still_uses_a_profiled_cache_for_a_fixed_provider(tmp_path):
    """A named profile's cache is still safe to reuse for a FIXED provider."""
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-cached", "work")
    resolved = resolve_token(
        env_var="ZAI_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="zai",
        profile_name="work",
        base_url_policy="fixed",
        environ={},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-cached"
    assert resolved.source == SOURCE_CACHE


@pytest.mark.unit
def test_resolve_token_falls_back_to_env_for_a_brand_new_uncached_profile(tmp_path):
    """A NEW profile with nothing cached yet must still consult the env var.

    Before this fix, naming an explicit profile skipped the environment
    check outright, so a first-time ``--profile`` use in a headless/CI run
    (no cached token for that profile yet) fell straight to an interactive
    ``getpass`` prompt instead of honouring an already-set env var — a
    scripted install with a brand-new profile name would hang.
    """
    paths = _paths(tmp_path)
    resolved = resolve_token(
        env_var="ZAI_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="zai",
        profile_name="brand-new",
        base_url_policy="fixed",
        environ={"ZAI_API_KEY": "sk-env"},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-env"
    assert resolved.source == SOURCE_ENV


@pytest.mark.unit
def test_resolve_token_still_uses_cache_for_a_fixed_provider(tmp_path):
    """The pre-existing behaviour is preserved for FIXED providers, where the
    address never varies and the cache is safe to reuse."""
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-cached")
    resolved = resolve_token(
        env_var="ZAI_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="zai",
        base_url_policy="fixed",
        environ={},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-cached"
    assert resolved.source == SOURCE_CACHE


@pytest.mark.unit
def test_resolve_token_defaults_to_fixed_for_backward_compatibility(tmp_path):
    """``base_url_policy`` defaults to ``"fixed"`` when omitted — every
    pre-existing caller (and every other test in this file) that doesn't pass
    it keeps the original cache-using behaviour."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        environ={},
        getpass_fn=_no_prompt,
    )
    assert resolved.value == "sk-cached"
    assert resolved.source == SOURCE_CACHE


@pytest.mark.unit
def test_resolve_token_prompt_when_neither_env_nor_cache(tmp_path):
    paths = _paths(tmp_path)

    def _fake_prompt(_prompt):
        return "sk-typed"

    resolved = resolve_token(
        env_var="LITELLM_API_KEY",
        prompt="token: ",
        paths=paths,
        provider_name="litellm",
        environ={},
        getpass_fn=_fake_prompt,
    )
    assert resolved.value == "sk-typed"
    assert resolved.source == SOURCE_PROMPT


@pytest.mark.unit
def test_resolve_token_raises_after_empty_retries(tmp_path):
    from codehelper.errors import CodeHelperError

    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError):
        resolve_token(
            env_var="LITELLM_API_KEY",
            prompt="token: ",
            paths=paths,
            provider_name="litellm",
            environ={},
            getpass_fn=lambda _p: "",
            retries=2,
        )


# --------------------------------------------------------------------------- #
# env_cache_conflict — naming the silently-beaten cache (issue #71)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_env_cache_conflict_names_both_sides_redacted(tmp_path):
    """The exact real-world trap: a stale export beat a good cached key.

    The warning must name the env var, the profile, and BOTH values — each
    redacted (4-char head + 3-char tail) so the full secret never appears
    in terminal output, and it must tell the user how to take the profile
    instead."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached-working-key")
    conflict = env_cache_conflict(
        ResolvedToken("sk-env-stale-token", SOURCE_ENV),
        env_var="LITELLM_API_KEY",
        paths=paths,
        provider_name="litellm",
    )
    assert conflict is not None
    assert "LITELLM_API_KEY" in conflict
    assert "'default'" in conflict
    assert "sk-e...ken" in conflict  # the env side, redacted
    assert "sk-c...key" in conflict  # the cache side, redacted
    assert "sk-env-stale-token" not in conflict
    assert "sk-cached-working-key" not in conflict
    assert "unset LITELLM_API_KEY" in conflict


@pytest.mark.unit
def test_env_cache_conflict_silent_when_tokens_agree(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-same-token")
    assert (
        env_cache_conflict(
            ResolvedToken("sk-same-token", SOURCE_ENV),
            env_var="LITELLM_API_KEY",
            paths=paths,
            provider_name="litellm",
        )
        is None
    )


@pytest.mark.unit
def test_env_cache_conflict_silent_when_no_cache_exists(tmp_path):
    """Env-only: nothing was beaten, so nothing to warn about."""
    assert (
        env_cache_conflict(
            ResolvedToken("sk-env-only-token", SOURCE_ENV),
            env_var="LITELLM_API_KEY",
            paths=_paths(tmp_path),
            provider_name="litellm",
        )
        is None
    )


@pytest.mark.unit
def test_env_cache_conflict_silent_when_cache_won(tmp_path):
    """A cache win is the documented precedence, not a trap — no warning
    even though the two values differ."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    assert (
        env_cache_conflict(
            ResolvedToken("sk-cached", SOURCE_CACHE),
            env_var="LITELLM_API_KEY",
            paths=paths,
            provider_name="litellm",
        )
        is None
    )


@pytest.mark.unit
def test_env_cache_conflict_silent_for_an_explicit_profile(tmp_path):
    """A named profile caches ABOVE the environment, so an env win means the
    profile was empty — nothing disagrees (mirrors resolve_token's rules)."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached-under-default")
    assert (
        env_cache_conflict(
            ResolvedToken("sk-env", SOURCE_ENV),
            env_var="LITELLM_API_KEY",
            paths=paths,
            provider_name="litellm",
            profile_name="work",
        )
        is None
    )


@pytest.mark.unit
def test_env_cache_conflict_silent_for_a_runtime_address_provider(tmp_path):
    """A non-fixed provider never consults the default cache (the
    cross-address rule), so it was never a candidate to be beaten."""
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    assert (
        env_cache_conflict(
            ResolvedToken("sk-env", SOURCE_ENV),
            env_var="LITELLM_API_KEY",
            paths=paths,
            provider_name="litellm",
            base_url_policy="required",
        )
        is None
    )


# --------------------------------------------------------------------------- #
# token_for_discovery — never prompts, never raises
# --------------------------------------------------------------------------- #


class _Provider:
    """Stand-in provider: ``token_for_discovery`` reads these attrs."""

    def __init__(self, *, auth, token_env_var, name, base_url_policy="fixed"):
        self.auth = auth
        self.token_env_var = token_env_var
        self.name = name
        self.base_url_policy = base_url_policy


@pytest.mark.unit
def test_token_for_discovery_env_wins(tmp_path):
    provider = _Provider(auth="secret", token_env_var="LITELLM_API_KEY", name="litellm")
    result = token_for_discovery(
        _paths(tmp_path), provider, environ={"LITELLM_API_KEY": "sk-env"}
    )
    assert result == "sk-env"


@pytest.mark.unit
def test_token_for_discovery_cache_used_without_env(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    provider = _Provider(auth="secret", token_env_var="LITELLM_API_KEY", name="litellm")
    assert token_for_discovery(paths, provider, environ={}) == "sk-cached"


@pytest.mark.unit
def test_token_for_discovery_empty_when_nothing_available(tmp_path):
    provider = _Provider(auth="secret", token_env_var="LITELLM_API_KEY", name="litellm")
    assert token_for_discovery(_paths(tmp_path), provider, environ={}) == ""


@pytest.mark.unit
def test_token_for_discovery_never_prompts(tmp_path):
    """No env, no cache, secret provider — still returns "" without prompting.

    This is the whole point of a separate discovery helper (vs resolve_token):
    discovery must never block a scripted ``--list-models`` on a missing token.
    Unlike ``resolve_token``, this function takes no ``getpass_fn`` at all —
    that absence IS the proof; there is nothing to call to block on.
    """
    provider = _Provider(auth="secret", token_env_var="LITELLM_API_KEY", name="litellm")
    assert token_for_discovery(_paths(tmp_path), provider, environ={}) == ""


@pytest.mark.unit
def test_token_for_discovery_ignores_cache_for_a_runtime_address_provider(tmp_path):
    """A cached token must not follow a provider to an arbitrary runtime URL.

    ``token_for_discovery`` used to key the cache lookup purely by provider
    name — so a token cached for ``litellm`` at one ``--base-url`` was handed
    to a *different* ``--base-url`` for the same provider name, with no check
    that the two addresses have anything to do with each other. Any provider
    whose ``base_url_policy`` is not FIXED (REQUIRED/OVERRIDABLE) can have its
    address changed per-invocation by the caller, so a cached secret must not
    be attached to it automatically — only an explicit env var (which the
    caller set for *this* invocation) is still honoured.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached")
    provider = _Provider(
        auth="secret",
        token_env_var="LITELLM_API_KEY",
        name="litellm",
        base_url_policy="required",
    )
    assert token_for_discovery(paths, provider, environ={}) == ""


@pytest.mark.unit
def test_token_for_discovery_trusts_a_named_profiles_cache_for_a_runtime_address_provider(
    tmp_path,
):
    """A NAMED profile's cache is trusted for ANY base_url_policy.

    Mirrors ``resolve_token``'s same distinction: an explicitly named
    profile is a deliberate per-invocation choice, unlike the unnamed
    cache (still host-isolation guarded below). Without this, model
    discovery could never use an already-cached profile token for a
    REQUIRED-policy provider like ``litellm``.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-cached", "work")
    provider = _Provider(
        auth="secret",
        token_env_var="LITELLM_API_KEY",
        name="litellm",
        base_url_policy="required",
    )
    assert (
        token_for_discovery(paths, provider, profile_name="work", environ={})
        == "sk-cached"
    )


@pytest.mark.unit
def test_token_for_discovery_still_uses_a_profiled_cache_for_a_fixed_provider(
    tmp_path,
):
    """A named profile's cache is still safe to reuse for a FIXED provider."""
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-cached", "work")
    provider = _Provider(
        auth="secret",
        token_env_var="ZAI_API_KEY",
        name="zai",
        base_url_policy="fixed",
    )
    assert (
        token_for_discovery(paths, provider, profile_name="work", environ={})
        == "sk-cached"
    )


@pytest.mark.unit
def test_token_for_discovery_falls_back_to_env_for_a_brand_new_uncached_profile(
    tmp_path,
):
    """A NEW profile with nothing cached yet must still consult the env var.

    Before this fix, naming a profile with no cached entry returned ""
    immediately without ever checking the environment, so ``--list-models
    --profile NAME`` on a first-time profile sent an unauthenticated
    request even when the provider's env var was set — inconsistent with
    ``resolve_token``'s same env fallback for a brand-new profile.
    """
    paths = _paths(tmp_path)
    provider = _Provider(
        auth="secret",
        token_env_var="ZAI_API_KEY",
        name="zai",
        base_url_policy="fixed",
    )
    result = token_for_discovery(
        paths,
        provider,
        profile_name="brand-new",
        environ={"ZAI_API_KEY": "sk-env"},
    )
    assert result == "sk-env"


@pytest.mark.unit
def test_token_for_discovery_still_uses_cache_for_a_fixed_provider(tmp_path):
    """The pre-existing behaviour is preserved for FIXED (registry-address)
    providers — there the cache is safe, since the address never varies."""
    paths = _paths(tmp_path)
    save_credential(paths, "zai", "sk-cached")
    provider = _Provider(
        auth="secret",
        token_env_var="ZAI_API_KEY",
        name="zai",
        base_url_policy="fixed",
    )
    assert token_for_discovery(paths, provider, environ={}) == "sk-cached"


@pytest.mark.unit
def test_token_for_discovery_empty_for_non_secret_provider(tmp_path):
    """A literal/none provider never needs a token — don't even read the cache."""
    paths = _paths(tmp_path)
    save_credential(paths, "dummy", "should-not-be-returned")
    provider = _Provider(auth="literal", token_env_var="DUMMY", name="dummy")
    result = token_for_discovery(paths, provider, environ={"DUMMY": "sk-ignored"})
    assert result == ""


# --------------------------------------------------------------------------- #
# os.environ as the documented default for token_for_discovery
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_token_for_discovery_uses_real_os_environ_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-real-env")
    provider = _Provider(auth="secret", token_env_var="LITELLM_API_KEY", name="litellm")
    assert token_for_discovery(_paths(tmp_path), provider) == "sk-real-env"
