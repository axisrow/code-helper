"""Tests for the credential cache and ``resolve_token`` source tracking.

The cache (``~/.config/code-helper/credentials.json``) is the one persistent
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

from code_helper.services.paths import Paths
from code_helper.services.secrets import (
    SOURCE_CACHE,
    SOURCE_ENV,
    SOURCE_PROMPT,
    credential_for,
    load_credentials,
    resolve_token,
    save_credential,
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
    assert load_credentials(paths) == {"litellm": "sk-1"}


@pytest.mark.unit
def test_load_credentials_skips_empty_values(tmp_path):
    paths = _paths(tmp_path)
    _write_credentials_file(paths, json.dumps({"litellm": "", "zai": "sk-real"}))
    assert load_credentials(paths) == {"zai": "sk-real"}


@pytest.mark.unit
def test_credential_for_missing_provider_is_empty_string(tmp_path):
    assert credential_for(_paths(tmp_path), "litellm") == ""


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
    assert creds == {"litellm": "sk-1", "zai": "sk-2"}


@pytest.mark.unit
def test_save_credential_overwrites_same_provider(tmp_path):
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-old")
    save_credential(paths, "litellm", "sk-new")
    assert credential_for(paths, "litellm") == "sk-new"


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
    assert creds == {"future-provider": "sk-x", "litellm": "sk-1"}


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
    from code_helper.errors import CodeHelperError

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
    save_credential(paths, "ollama", "should-not-be-returned")
    provider = _Provider(auth="literal", token_env_var="DUMMY", name="ollama")
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
