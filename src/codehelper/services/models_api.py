"""Fetch a provider's model list over HTTP (stdlib ``urllib``, no subprocess).

Two response shapes cover every provider this tool cares about, selected by
:attr:`Provider.model_list_api`:

- :attr:`ModelListAPI.OLLAMA_TAGS` — ``GET {base}/api/tags`` ->
  ``{"models": [{"name": ...}]}``
- :attr:`ModelListAPI.OPENAI_V1` — ``GET {base}/models`` ->
  ``{"data": [{"id": ...}]}`` (the OpenAI standard, served by every
  OpenAI-compatible endpoint)

**This function never raises.** A missing daemon, a timeout, or a garbled
response is not a failure of the operation the user asked for — it just means
the convenience of a picker is unavailable and they should type the model name
instead. Raising would force every call site to wrap a ``try`` around a
"never mind then". Failures come back as :attr:`ModelListResult.error`, which
is a human-readable sentence meant to be printed as-is.

Deliberately no ``subprocess``: parsing ``ollama list``'s table would be
fragile, and shelling out would make this untestable without the binary
installed. The ``fetch`` seam follows the project's existing injection pattern
(``getpass_fn``, ``read_key``, ``print_fn``) so tests never touch the network.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from codehelper.services.model import ModelListAPI, Provider
from codehelper.services.render import openai_base_url

__all__ = ["list_models", "ModelListResult", "Fetcher", "DEFAULT_TIMEOUT"]

#: Short by design: this is an interactive convenience, and a picker that hangs
#: for seconds is worse than one that falls back to manual entry.
DEFAULT_TIMEOUT = 3.0

#: ``(url, timeout, token) -> raw body``. Injected in tests.
Fetcher = Callable[[str, float, str], bytes]


@dataclass(frozen=True)
class ModelListResult:
    """Outcome of a listing attempt.

    ``error`` is ``None`` on success. On failure ``models`` is empty and
    ``error`` holds a printable explanation — callers show it and fall back to
    asking the user to type a model name.
    """

    models: tuple[str, ...]
    source: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every HTTP redirect instead of silently following it.

    ``urllib``'s default redirect handling re-sends the SAME ``Request``
    object — headers included — to the ``Location`` target, even when that
    target is a different host. This function attaches an ``Authorization:
    Bearer <token>`` header for secret-auth providers, so the stock behaviour
    would forward that bearer token to whatever host a (misconfigured,
    load-balancer-fronted, or actively malicious) endpoint redirects to. This
    is a one-shot discovery convenience (``--list-models`` / the TUI's model
    picker), not agent traffic — there is no legitimate reason for it to
    follow a redirect at all, so the correct fix is to refuse one outright
    rather than try to selectively strip the header per-hop. The caller gets
    the resulting :class:`urllib.error.HTTPError` back through the same
    exception path as any other unreachable-endpoint failure (see
    ``list_models``'s ``except`` tuple), so this degrades to the existing
    "could not reach {provider}" message — no new failure mode, just a closed
    one.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802, ARG002 — urllib's handler contract, params arrive positionally
        return None


#: One opener, reused across calls: redirect handling is a fixed policy of
#: this fetcher, not per-request state, and urllib's default OpenerDirector
#: construction is cheap but there is no reason to redo it every call.
_opener = urllib.request.build_opener(_NoRedirectHandler)


def _urlopen_fetch(url: str, timeout: float, token: str) -> bytes:
    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with _opener.open(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


#: Path appended to the provider's base URL, per API shape.
#:
#: Both use a path WITHOUT a version segment. OLLAMA_TAGS hits ``/api/tags``
#: directly off the daemon root. OPENAI_V1 hits ``/models`` off an OpenAI-style
#: root, which :func:`openai_base_url` normalizes to end in ``/v1`` first — so a
#: bare-root ``--base-url`` (``http://host:4000``) and an already-versioned one
#: (``http://host:4000/v1``) both reach ``…/v1/models``, matching the endpoint
#: the eventual install points Codex at (see ``services/render.py``).
_PATHS: dict[ModelListAPI, str] = {
    ModelListAPI.OLLAMA_TAGS: "/api/tags",
    ModelListAPI.OPENAI_V1: "/models",
}


def _parse_ollama_tags(payload: dict) -> tuple[str, ...]:
    entries = payload.get("models")
    if not isinstance(entries, list):
        return ()
    return tuple(
        e["name"]
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"]
    )


def _parse_openai_v1(payload: dict) -> tuple[str, ...]:
    entries = payload.get("data")
    if not isinstance(entries, list):
        return ()
    return tuple(
        e["id"]
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"]
    )


_PARSERS: dict[ModelListAPI, Callable[[dict], tuple[str, ...]]] = {
    ModelListAPI.OLLAMA_TAGS: _parse_ollama_tags,
    ModelListAPI.OPENAI_V1: _parse_openai_v1,
}


def _classify_fetch_error(e: Exception, url: str, provider: Provider) -> str:
    """Turn a fetch exception into a printable ``ModelListResult.error`` string.

    A 401/403 means the endpoint is UP but unauthorized — "start the daemon"
    is the wrong advice there, so it gets its own message naming the token
    env var. Every other failure (connection refused, timeout, 500/404, a
    mid-body ``IncompleteRead``) reads as a reachability fault and falls
    through to the generic "could not reach" wording.
    """
    if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
        return (
            f"{provider.name} rejected the request at {url} (HTTP {e.code}) "
            f"— it requires a token: set {provider.token_env_var} or add one "
            f"to the credentials file"
        )
    return (
        f"could not reach {provider.name} at {url}: {e} "
        f"— start it, or type the model name manually"
    )


def list_models(
    provider: Provider,
    *,
    fetch: Fetcher = _urlopen_fetch,
    timeout: float = DEFAULT_TIMEOUT,
    token: str = "",
) -> ModelListResult:
    """List ``provider``'s models. Never raises — see the module docstring.

    Args:
        provider: Supplies the base URL and which API shape to expect.
        fetch: Injectable transport; the default uses ``urllib``.
        timeout: Seconds to wait.
        token: Sent as ``Authorization: Bearer`` only for ``secret``-auth
            providers. Never echoed into ``source`` or any error text.

    Returns:
        A :class:`ModelListResult`. An empty list with ``error is None`` is a
        legitimate answer (the daemon is up and has no models); deciding what
        to do about it belongs to the UI, not here.
    """
    api = provider.model_list_api
    if api is ModelListAPI.NONE:
        return ModelListResult((), "", f"{provider.name} does not publish a model list")

    base = (provider.model_list_url or provider.base_url).rstrip("/")
    if not base:
        return ModelListResult((), "", f"{provider.name} has no base URL configured")
    # OPENAI_V1 normalizes to an OpenAI ``/v1`` root so a bare ``--base-url``
    # (no ``/v1``) reaches ``…/v1/models`` — matching the endpoint the eventual
    # install points Codex at. Done AFTER the empty-base guard: openai_base_url
    # would turn an empty base into a bare ``/v1/`` and lose the clear message
    # above. OLLAMA_TAGS needs no version segment and is left as-is.
    if api is ModelListAPI.OPENAI_V1:
        # openai_base_url always returns a trailing "/" (e.g. ".../v1/") — this
        # rstrip is NOT redundant with the one above: without it the "url ="
        # concatenation below doubles the slash (".../v1//models").
        base = openai_base_url(base, provider.base_url_is_openai_root).rstrip("/")
    url = base + _PATHS[api]

    auth_token = token if provider.auth == "secret" else ""

    try:
        raw = fetch(url, timeout, auth_token)
    except (
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
    ) as e:
        # ConnectionRefusedError (daemon down) arrives here too — either bare
        # or wrapped in URLError depending on the layer that raised it.
        # ``HTTPException`` is listed separately because it is NOT an
        # ``OSError`` subclass: a server closing mid-body raises
        # ``IncompleteRead``, which would otherwise escape and break the
        # documented never-raises contract every call site relies on.
        #
        # ``HTTPError`` (a URLError subclass) is in the SAME tuple on purpose:
        # a 401/403 means the endpoint is UP but unauthorized, so "start the
        # daemon" is the wrong advice there and gets its own message; every
        # other HTTP status (500, 404, …) reads correctly as a reachability
        # fault and falls through to the generic wording. One ``except`` arm
        # (not a separate clause before this one) because a ``raise`` from a
        # sibling ``except`` escapes the whole ``try`` rather than landing in
        # the next arm — splitting them would leak non-auth HTTP errors out
        # and break the never-raises contract.
        return ModelListResult(
            (),
            url,
            _classify_fetch_error(e, url, provider),
        )

    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return ModelListResult((), url, f"{url} returned a non-JSON response")

    if not isinstance(payload, dict):
        return ModelListResult((), url, f"{url} returned an unexpected response")

    # Malformed individual entries are skipped rather than fatal: a provider
    # adding a field or emitting one odd record should not cost the user the
    # whole picker.
    models = _PARSERS[api](payload)
    return ModelListResult(tuple(sorted(set(models))), url)
