"""Tests for wrapper-name (alias) validation.

Two independent gates are pinned here:

- :func:`validate_alias` — the semantic rule set, with actionable messages.
- :meth:`Paths.script_for` — the structural guard, tested WITHOUT going
  through ``validate_alias`` (see ``test_script_for_*``). That separation is
  the point: if a future code path forgets to validate, escaping ``bin_dir``
  must still be impossible.
"""

from __future__ import annotations

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.naming import (
    MAX_ALIAS_LENGTH,
    MAX_BASE_URL_LENGTH,
    RESERVED_ALIASES,
    normalize_base_url,
    validate_alias,
    validate_base_url,
)
from code_helper.services.paths import Paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    [
        "glm",
        "glm-codex",
        "deepseek",
        "glm-ollama",
        "my.wrapper_1",
        "a",
        "9lives",
        "x" * MAX_ALIAS_LENGTH,
    ],
)
def test_valid_aliases_pass_through(alias):
    assert validate_alias(alias) == alias


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["", "   ", "\t"])
def test_empty_alias_rejected(alias):
    with pytest.raises(CodeHelperError, match="must not be empty"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", [".", ".."])
def test_dot_aliases_rejected(alias):
    # `bin_dir / "."` is the directory itself — a write there is not a wrapper.
    with pytest.raises(CodeHelperError, match="must not be"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    ["../../etc/passwd", "a/b", "/abs", "sub/dir/name", "a\\b"],
)
def test_path_separators_rejected(alias):
    # The headline vulnerability: these used to reach `bin_dir / name` intact.
    with pytest.raises(CodeHelperError, match="path separator"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["-rf", "--force", "-"])
def test_leading_dash_rejected(alias):
    with pytest.raises(CodeHelperError, match="must not start with"):
        validate_alias(alias)


@pytest.mark.unit
def test_too_long_alias_rejected():
    with pytest.raises(CodeHelperError, match="too long"):
        validate_alias("x" * (MAX_ALIAS_LENGTH + 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    [
        "with space",
        "tab\there",
        "new\nline",
        "nul\x00byte",
        "semi;colon",
        "dollar$sign",
        "back`tick",
        "pipe|char",
        "star*",
        "кириллица",
        ".hidden",
        "_leading",
    ],
)
def test_unsafe_characters_rejected(alias):
    with pytest.raises(CodeHelperError, match="may only contain"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", sorted(RESERVED_ALIASES))
def test_reserved_names_rejected(alias):
    """An alias equal to an agent binary would exec itself forever.

    `~/.local/bin/claude` is also a real symlink on a normal install, so this
    additionally prevents clobbering a working Claude Code entry point.
    """
    with pytest.raises(CodeHelperError, match="reserved name"):
        validate_alias(alias)


# --------------------------------------------------------------------------- #
# Paths.script_for — the structural guard, independent of validate_alias
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["../../etc/passwd", "a/b", "", ".", "..", "sub/dir"],
)
def test_script_for_rejects_names_that_are_not_one_component(tmp_path, name):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="invalid wrapper name"):
        paths.script_for(name)


@pytest.mark.unit
def test_script_for_never_escapes_bin_dir(tmp_path):
    """The guarantee, stated positively: whatever gets through stays inside."""
    paths = Paths.from_home(tmp_path)
    for name in ["glm", "glm-codex", "a.b_c-d"]:
        assert paths.script_for(name).parent == paths.bin_dir


@pytest.mark.unit
def test_script_for_still_does_no_io(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.script_for("glm")
    assert not paths.bin_dir.exists()


# --- normalize_base_url -----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Bare host/IP: scheme + default port + default path are filled in.
        ("78.47.183.125", "https://78.47.183.125:4000/v1"),
        ("host.example.com", "https://host.example.com:4000/v1"),
        # Port present but no path: port kept, path added — no double port.
        ("localhost:4000", "https://localhost:4000/v1"),
        ("192.168.1.10:4000", "https://192.168.1.10:4000/v1"),
        # Port and path both present: unchanged.
        ("192.168.1.10:4000/v1", "https://192.168.1.10:4000/v1"),
        # Already has a scheme: returned untouched (explicit http:// respected).
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("https://api.example.com/v1", "https://api.example.com/v1"),
        # Empty / whitespace-only: stripped to empty.
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


# --- validate_base_url ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:4000/v1",
        "https://litellm.example.com/v1",
        "http://127.0.0.1:4000",
        "https://api.example.com:8443/v1/",
    ],
)
def test_valid_base_urls_pass(url):
    validate_base_url(url)  # must not raise


@pytest.mark.unit
@pytest.mark.parametrize("url", ["", "   "])
def test_empty_base_url_rejected(url):
    with pytest.raises(CodeHelperError, match="empty"):
        validate_base_url(url)


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    ["ftp://host", "localhost:4000", "host.example.com", "ws://host/v1"],
)
def test_base_url_without_http_scheme_rejected(url):
    with pytest.raises(CodeHelperError, match=r"http:// or https://"):
        validate_base_url(url)


@pytest.mark.unit
@pytest.mark.parametrize("url", ["http://", "https://"])
def test_base_url_without_a_host_rejected(url):
    with pytest.raises(CodeHelperError, match="no host"):
        validate_base_url(url)


@pytest.mark.unit
@pytest.mark.parametrize(
    "url", ["http://host/v1?x=1", "http://host/v1#frag", "http://host?x=1#f"]
)
def test_base_url_with_query_or_fragment_rejected(url):
    with pytest.raises(CodeHelperError, match="query string or fragment"):
        validate_base_url(url)


@pytest.mark.unit
def test_base_url_with_embedded_whitespace_rejected():
    with pytest.raises(CodeHelperError, match="whitespace"):
        validate_base_url("http://host with space/v1")


@pytest.mark.unit
def test_too_long_base_url_rejected():
    url = "http://" + "x" * MAX_BASE_URL_LENGTH
    with pytest.raises(CodeHelperError, match="too long"):
        validate_base_url(url)


@pytest.mark.unit
@pytest.mark.parametrize(
    "url", ["http://[bad", "http://[bad]", "http://[::1", "http://["]
)
def test_malformed_ipv6_bracket_url_raises_codehelpererror_not_valueerror(url):
    """urlsplit raises a bare ValueError (not CodeHelperError) for an
    unterminated/invalid IPv6-bracket host (cycle-review round 3, finding
    C2). Every caller of validate_base_url — with_base_url, and
    transitively spec_from_installed's base_url recovery — expects
    CodeHelperError as the only failure mode; an uncaught ValueError here
    escaped spec_from_installed's `except CodeHelperError` and broke its
    documented never-raises contract for a recovered value from a
    hand-edited/truncated wrapper.
    """
    with pytest.raises(CodeHelperError, match="malformed"):
        validate_base_url(url)
