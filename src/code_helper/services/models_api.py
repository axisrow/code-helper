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

from code_helper.services.model import ModelListAPI, Provider

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


def _urlopen_fetch(url: str, timeout: float, token: str) -> bytes:
    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


#: Path appended to the provider's base URL, per API shape. Note OPENAI_V1 uses
#: ``/models``, not ``/v1/models``: an OpenAI-compatible ``base_url`` already
#: ends in ``/v1`` (that is the value Codex itself wants in its TOML), so the
#: version segment must not be duplicated here.
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
    url = base + _PATHS[api]

    auth_token = token if provider.auth == "secret" else ""

    try:
        raw = fetch(url, timeout, auth_token)
    except (
        TimeoutError,
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
        return ModelListResult(
            (),
            url,
            f"could not reach {provider.name} at {url}: {e} "
            f"— start it, or type the model name manually",
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
