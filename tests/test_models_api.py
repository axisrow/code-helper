"""Tests for provider model listing — all with an injected fetcher, no network.

The load-bearing contract: ``list_models`` NEVER raises. Every failure mode
below asserts a populated ``error`` and an empty list, because a call site is
allowed to ignore the outcome and fall back to manual entry.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from codehelper.services.model import ModelListAPI, Provider, get_provider
from codehelper.services.models_api import list_models

_OLLAMA = Provider(
    name="ollama",
    shapes=frozenset(),
    base_url="http://127.0.0.1:11434",
    auth="literal",
    auth_value="ollama",
    model_list_api=ModelListAPI.OLLAMA_TAGS,
)

_OPENAI = Provider(
    name="example",
    shapes=frozenset(),
    base_url="https://example.invalid/v1",
    auth="secret",
    token_env_var="EXAMPLE_API_KEY",
    model_list_api=ModelListAPI.OPENAI_V1,
)

_NO_LIST = Provider(
    name="zai",
    shapes=frozenset(),
    base_url="https://api.z.ai/api/anthropic",
    model_list_api=ModelListAPI.NONE,
)


def _fetch_returning(payload, *, record=None):
    def _fetch(url, timeout, token):
        if record is not None:
            record.append((url, timeout, token))
        return payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    return _fetch


def _fetch_raising(exc):
    def _fetch(url, timeout, token):
        raise exc

    return _fetch


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ollama_tags_parsed_and_sorted():
    fetch = _fetch_returning({"models": [{"name": "b:1"}, {"name": "a:2"}]})
    result = list_models(_OLLAMA, fetch=fetch)
    assert result.ok
    assert result.models == ("a:2", "b:1")


@pytest.mark.unit
def test_openai_v1_parsed():
    fetch = _fetch_returning(
        {"data": [{"id": "nvidia/nemotron"}, {"id": "meta/llama"}]}
    )
    result = list_models(_OPENAI, fetch=fetch)
    assert result.models == ("meta/llama", "nvidia/nemotron")


@pytest.mark.unit
def test_duplicates_collapsed():
    fetch = _fetch_returning({"models": [{"name": "a"}, {"name": "a"}]})
    assert list_models(_OLLAMA, fetch=fetch).models == ("a",)


@pytest.mark.unit
def test_malformed_entries_are_skipped_not_fatal():
    """One odd record must not cost the user the whole picker."""
    fetch = _fetch_returning(
        {"models": [{"name": "good"}, {}, {"name": ""}, "junk", {"name": 42}]}
    )
    result = list_models(_OLLAMA, fetch=fetch)
    assert result.ok
    assert result.models == ("good",)


@pytest.mark.unit
def test_empty_list_is_success_not_error():
    """Daemon up with no models is a real answer; the UI decides what to do."""
    result = list_models(_OLLAMA, fetch=_fetch_returning({"models": []}))
    assert result.ok
    assert result.models == ()


@pytest.mark.unit
def test_foreign_schema_yields_empty_without_raising():
    result = list_models(_OLLAMA, fetch=_fetch_returning({"unexpected": 1}))
    assert result.models == ()
    assert result.ok  # valid JSON object, just nothing we recognise


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ollama_url():
    calls = []
    list_models(_OLLAMA, fetch=_fetch_returning({"models": []}, record=calls))
    assert calls[0][0] == "http://127.0.0.1:11434/api/tags"


@pytest.mark.unit
def test_openai_url_does_not_duplicate_v1():
    """base_url already ends in /v1 — that's the value Codex's TOML wants."""
    calls = []
    list_models(_OPENAI, fetch=_fetch_returning({"data": []}, record=calls))
    assert calls[0][0] == "https://example.invalid/v1/models"


@pytest.mark.unit
def test_trailing_slash_in_base_url_is_normalised():
    provider = Provider(
        name="p",
        shapes=frozenset(),
        base_url="http://host:1234/",
        model_list_api=ModelListAPI.OLLAMA_TAGS,
    )
    calls = []
    list_models(provider, fetch=_fetch_returning({"models": []}, record=calls))
    assert calls[0][0] == "http://host:1234/api/tags"


@pytest.mark.unit
def test_model_list_url_overrides_base_url():
    provider = Provider(
        name="p",
        shapes=frozenset(),
        base_url="http://ignored",
        model_list_url="http://elsewhere:99",
        model_list_api=ModelListAPI.OLLAMA_TAGS,
    )
    calls = []
    list_models(provider, fetch=_fetch_returning({"models": []}, record=calls))
    assert calls[0][0] == "http://elsewhere:99/api/tags"


# --------------------------------------------------------------------------- #
# Failure modes — none of these may raise
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_daemon_unreachable_reports_error():
    fetch = _fetch_raising(urllib.error.URLError(ConnectionRefusedError(61)))
    result = list_models(_OLLAMA, fetch=fetch)
    assert not result.ok
    assert "could not reach ollama" in result.error
    assert "manually" in result.error  # tells the user the way forward
    assert result.models == ()


@pytest.mark.unit
def test_timeout_reports_error():
    result = list_models(_OLLAMA, fetch=_fetch_raising(TimeoutError("timed out")))
    assert not result.ok
    assert result.models == ()


@pytest.mark.unit
def test_non_json_response_reports_error():
    result = list_models(_OLLAMA, fetch=_fetch_returning(b"<html>nope</html>"))
    assert not result.ok
    assert "non-JSON" in result.error


@pytest.mark.unit
def test_json_but_not_an_object_reports_error():
    result = list_models(_OLLAMA, fetch=_fetch_returning(b"[1,2,3]"))
    assert not result.ok
    assert "unexpected" in result.error


@pytest.mark.unit
def test_api_none_short_circuits_without_fetching():
    def _explode(url, timeout, token):  # pragma: no cover - must not run
        raise AssertionError("fetch must not be called when there is no list API")

    result = list_models(_NO_LIST, fetch=_explode)
    assert not result.ok
    assert "does not publish a model list" in result.error


# --------------------------------------------------------------------------- #
# Auth handling
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_token_sent_only_for_secret_providers():
    calls = []
    list_models(
        _OPENAI, fetch=_fetch_returning({"data": []}, record=calls), token="s3cret"
    )
    assert calls[0][2] == "s3cret"


@pytest.mark.unit
def test_token_not_sent_for_literal_auth_provider():
    calls = []
    list_models(
        _OLLAMA, fetch=_fetch_returning({"models": []}, record=calls), token="s3cret"
    )
    assert calls[0][2] == ""


@pytest.mark.unit
def test_token_never_appears_in_result_or_error():
    """Same invariant as elsewhere in the project: secrets are not echoed."""
    fetch = _fetch_raising(urllib.error.URLError("boom"))
    result = list_models(_OPENAI, fetch=fetch, token="s3cret")
    assert "s3cret" not in (result.error or "")
    assert "s3cret" not in result.source


@pytest.mark.unit
def test_timeout_value_is_forwarded():
    calls = []
    list_models(
        _OLLAMA, fetch=_fetch_returning({"models": []}, record=calls), timeout=0.5
    )
    assert calls[0][1] == 0.5


# --------------------------------------------------------------------------- #
# /v1 normalization (issue #15, point 2) — OPENAI_V1 only
# --------------------------------------------------------------------------- #

_OPENAI_BARE_ROOT = Provider(
    name="litellm",
    shapes=frozenset(),
    base_url="http://host:4000",  # deliberately no /v1 — a bare --base-url
    auth="secret",
    token_env_var="LITELLM_API_KEY",
    model_list_api=ModelListAPI.OPENAI_V1,
)


@pytest.mark.unit
def test_bare_root_openai_url_is_normalized_to_v1():
    calls = []
    list_models(_OPENAI_BARE_ROOT, fetch=_fetch_returning({"data": []}, record=calls))
    assert calls[0][0] == "http://host:4000/v1/models"


@pytest.mark.unit
def test_already_v1_openai_url_is_not_duplicated_via_normalization():
    """Reinforces test_openai_url_does_not_duplicate_v1 through the new
    normalization path explicitly."""
    calls = []
    list_models(_OPENAI, fetch=_fetch_returning({"data": []}, record=calls))
    assert calls[0][0] == "https://example.invalid/v1/models"


@pytest.mark.unit
def test_ollama_tags_are_never_v1_normalized():
    """OLLAMA_TAGS must not gain a /v1 segment — normalization is OPENAI_V1-only."""
    calls = []
    list_models(_OLLAMA, fetch=_fetch_returning({"models": []}, record=calls))
    assert calls[0][0] == "http://127.0.0.1:11434/api/tags"


@pytest.mark.unit
def test_gemini_openai_root_is_not_v1_normalized():
    """Gemini's base_url IS the complete OpenAI root — discovery must hit
    .../openai/models, not a nonexistent .../openai/v1/models."""
    calls = []
    list_models(
        get_provider("gemini"), fetch=_fetch_returning({"data": []}, record=calls)
    )
    assert calls[0][0] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/models"
    )


@pytest.mark.unit
def test_deepseek_anthropic_provider_lists_models_on_the_openai_surface():
    """deepseek's base_url is the /anthropic path, which serves no OpenAI-style
    model list — model_list_url redirects discovery to the OpenAI root."""
    calls = []
    list_models(
        get_provider("deepseek"), fetch=_fetch_returning({"data": []}, record=calls)
    )
    assert calls[0][0] == "https://api.deepseek.com/v1/models"


@pytest.mark.unit
def test_deepseek_openai_bare_root_is_v1_normalized():
    """deepseek-openai's base_url is a bare root — discovery hits /v1/models,
    matching the endpoint the eventual install points Codex at."""
    calls = []
    list_models(
        get_provider("deepseek-openai"),
        fetch=_fetch_returning({"data": []}, record=calls),
    )
    assert calls[0][0] == "https://api.deepseek.com/v1/models"


@pytest.mark.unit
def test_empty_base_url_is_reported_before_any_normalization_or_fetch():
    """The clear "has no base URL configured" message must survive
    normalization — openai_base_url("") would otherwise turn it into a request
    against a bare "/v1/"."""

    def _explode(url, timeout, token):  # pragma: no cover - must not run
        raise AssertionError("fetch must not be called with no base URL")

    provider = Provider(
        name="litellm",
        shapes=frozenset(),
        base_url="",
        auth="secret",
        token_env_var="LITELLM_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
    )
    result = list_models(provider, fetch=_explode)
    assert not result.ok
    assert "has no base URL configured" in result.error


# --------------------------------------------------------------------------- #
# 401/403 — a distinct, actionable message (issue #15, point 1)
# --------------------------------------------------------------------------- #


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/v1/models", code, "x", {}, None
    )


@pytest.mark.unit
def test_401_reports_a_token_specific_message_not_start_it():
    result = list_models(_OPENAI, fetch=_fetch_raising(_http_error(401)))
    assert not result.ok
    assert "start it" not in result.error
    assert "EXAMPLE_API_KEY" in result.error


@pytest.mark.unit
def test_403_reports_a_token_specific_message():
    result = list_models(_OPENAI, fetch=_fetch_raising(_http_error(403)))
    assert not result.ok
    assert "EXAMPLE_API_KEY" in result.error


@pytest.mark.unit
def test_other_http_errors_keep_the_generic_reachability_message():
    """A 500/404 is a reachability-shaped problem, not an auth one — the
    generic wording (and its "manually" fallback hint) still applies."""
    result = list_models(_OPENAI, fetch=_fetch_raising(_http_error(500)))
    assert not result.ok
    assert "could not reach" in result.error
    assert "manually" in result.error


@pytest.mark.unit
def test_401_never_leaks_the_token():
    result = list_models(
        _OPENAI, fetch=_fetch_raising(_http_error(401)), token="s3cret"
    )
    assert "s3cret" not in (result.error or "")
    assert "s3cret" not in result.source
