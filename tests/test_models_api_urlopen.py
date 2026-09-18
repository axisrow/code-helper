"""Tests for ``urlopen_fetch``'s ACTUAL HTTP behaviour — unlike
``test_models_api.py`` (which injects ``fetch`` and never touches the
network), this file drives a real loopback ``http.server`` to prove the
redirect-refusal fix in ``models_api.urlopen_fetch``: the rest of the
codebase can inject a fake fetcher and stay honest about the JSON-parsing
contract, but nothing short of a real HTTP round-trip proves what
``urllib.request``'s redirect machinery actually does with the
``Authorization`` header.

No third-party HTTP server dependency — this project's ``pyproject.toml``
doesn't declare one, so pulling in ``pytest_httpserver`` here would be a new,
undeclared dependency; the stdlib's own ``http.server`` is enough for the one
scenario this file exists to prove.
"""

from __future__ import annotations

import http.server
import threading
import urllib.error

import pytest

from codehelper.services.models_api import urlopen_fetch


class _RedirectingHandler(http.server.BaseHTTPRequestHandler):
    """Always 302-redirects to a DIFFERENT (fake, unreachable) host, and
    records whether an ``Authorization`` header ever reached it — proving
    a bug would mean the redirect was followed, not merely attempted."""

    seen_auth_header: list[str | None] = []

    def do_GET(self):  # noqa: N802
        type(self).seen_auth_header.append(self.headers.get("Authorization"))
        self.send_response(302)
        self.send_header(
            "Location", "http://198.51.100.1:9/stolen"
        )  # TEST-NET-2, unroutable
        self.end_headers()

    def log_message(self, *_a):  # silence default stderr request logging
        pass


@pytest.fixture
def redirecting_server():
    _RedirectingHandler.seen_auth_header = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.unit
def testurlopen_fetch_refuses_a_redirect_instead_of_following_it(redirecting_server):
    """The core of finding L's fix: a redirecting endpoint must not have its
    ``Location`` followed — proven by the fact fetching raises rather than
    silently returning whatever ``198.51.100.1`` (deliberately unroutable)
    would have sent back."""
    url = f"http://127.0.0.1:{redirecting_server}/models"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urlopen_fetch(url, timeout=2.0, token="sk-should-not-travel")
    assert exc_info.value.code == 302


@pytest.mark.unit
def testurlopen_fetch_sends_the_token_only_to_the_original_host(redirecting_server):
    """Even though the redirect is refused (proven above), confirm the ORIGIN
    request did carry the bearer token — this test would also catch a
    regression where the fix accidentally stopped sending the token at all,
    not just stopped it from leaking to the redirect target."""
    url = f"http://127.0.0.1:{redirecting_server}/models"
    with pytest.raises(urllib.error.HTTPError):
        urlopen_fetch(url, timeout=2.0, token="sk-should-not-travel")
    assert _RedirectingHandler.seen_auth_header == ["Bearer sk-should-not-travel"]
