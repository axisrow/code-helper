"""Wrapper-name (alias) validation — the one gate on a user-chosen file name.

Historically the wrapper name was never validated, and it did not need to be:
``get_spec`` only accepted the three names in the registry, so the *set of
reachable names was closed and curated*. Letting the user name a wrapper
anything (``--alias``) removes exactly that precondition, which is why this
module exists.

What it protects against, concretely — every one of these was reachable via
``Paths.script_for`` before validation existed:

- ``"../../etc/passwd"`` — escapes ``~/.local/bin`` entirely
- ``""`` and ``"."`` — resolve to the *directory itself*, not a file in it
- ``"a/b"`` — writes into a subdirectory
- ``"-rf"`` — a name every shell command will read as a flag, not a file

The rule is an **allow-list**, not a deny-list. The name becomes an executable
on ``PATH``; enumerating every dangerous character for that context is not
possible, so anything not explicitly known-safe is rejected.

:func:`validate_alias` is the semantic gate with human-readable errors, called
early (before any token prompt or write). :meth:`Paths.script_for` carries an
independent structural check, so a future code path that forgets to call this
function still cannot escape ``bin_dir``. The duplication is deliberate.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from code_helper.errors import CodeHelperError

__all__ = [
    "validate_alias",
    "MAX_ALIAS_LENGTH",
    "RESERVED_ALIASES",
    "validate_base_url",
    "normalize_base_url",
    "MAX_BASE_URL_LENGTH",
    "DEFAULT_BASE_URL_PORT",
    "DEFAULT_BASE_URL_PATH",
]

#: Cap on alias length. Not a filesystem limit (those are far higher) — a
#: sanity bound so a pasted blob can't become a file name.
MAX_ALIAS_LENGTH = 64

#: Names that must never become a wrapper, independent of shape:
#:
#: - ``code-helper`` — the tool would overwrite its own entry point.
#: - agent binaries (``claude``, ``codex``) — a wrapper named after the binary
#:   it execs is an **infinite recursion** whenever ``~/.local/bin`` precedes
#:   the real binary on ``PATH``: the script re-invokes itself forever. This is
#:   not hypothetical here — ``~/.local/bin/claude`` is a real, working symlink
#:   on a typical install, and clobbering it also breaks Claude Code itself.
#:
#: Kept as a literal rather than derived from the agent registry so that
#: importing this module stays dependency-free (and so the reason above is
#: readable at the point of definition).
RESERVED_ALIASES = frozenset({"code-helper", "claude", "codex"})

#: Must start alphanumeric (no leading ``-``/``.``), then alphanumerics plus
#: ``.``/``_``/``-``. Excludes: path separators, whitespace, control bytes,
#: NUL, shell metacharacters, and non-ASCII look-alikes.
_ALIAS_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def validate_alias(alias: str) -> str:
    """Return ``alias`` unchanged if it is safe as a ``~/.local/bin`` file name.

    Raises:
        CodeHelperError: with a message naming the specific rule broken —
            these are shown directly to the user, so "invalid name" alone
            would not be actionable.
    """
    if not alias or not alias.strip():
        raise CodeHelperError("wrapper name must not be empty")

    # Checked before the regex purely for the error message: `.` and `..` would
    # fail the pattern anyway, but "must not be '.' or '..'" explains why.
    if alias in (".", ".."):
        raise CodeHelperError(f"wrapper name must not be {alias!r}")

    if "/" in alias or "\\" in alias:
        raise CodeHelperError(
            f"wrapper name must not contain a path separator: {alias!r}"
        )

    if alias.startswith("-"):
        raise CodeHelperError(
            f"wrapper name must not start with '-' (it would be read as a "
            f"flag): {alias!r}"
        )

    if len(alias) > MAX_ALIAS_LENGTH:
        raise CodeHelperError(
            f"wrapper name is too long ({len(alias)} > {MAX_ALIAS_LENGTH})"
        )

    if not _ALIAS_RE.match(alias):
        raise CodeHelperError(
            f"wrapper name may only contain letters, digits, '.', '_' and '-', "
            f"and must start with a letter or digit: {alias!r}"
        )

    if alias in RESERVED_ALIASES:
        raise CodeHelperError(
            f"{alias!r} is a reserved name — a wrapper named after the agent "
            f"binary it runs would call itself forever"
        )

    return alias


#: Sanity bound, same class as MAX_ALIAS_LENGTH — not a real protocol limit,
#: just a ceiling so a pasted blob can't become a generated script's URL.
MAX_BASE_URL_LENGTH = 512

#: Default port filled in by :func:`normalize_base_url` when a bare host/IP is
#: supplied without one. Matches a self-hosted LiteLLM proxy's conventional
#: port; a user who wants a different port just types it.
#:
#: Deliberately no matching ``DEFAULT_BASE_URL_PATH``: unlike the port, the
#: right path segment (``/v1`` or none) depends on which :class:`ConfigShape`
#: the provider resolves to — required for OPENAI_TOML, poison for
#: ANTHROPIC_ENV (see :func:`render.anthropic_base_url`) — and the shape is
#: not known yet at this layer. Guessing a path here was the bug: it produced
#: an ``ANTHROPIC_BASE_URL`` that 404s on every request. Leave path completion
#: to each shape's renderer, which knows which protocol it is serving.
DEFAULT_BASE_URL_PORT = "4000"
DEFAULT_BASE_URL_PATH = ""


def normalize_base_url(url: str) -> str:
    """Auto-complete a bare host/IP into a full ``base_url``.

    A user typing ``78.47.183.125`` (no scheme, no port, no path) gets
    ``https://78.47.183.125:4000`` — no path guessed, see
    :data:`DEFAULT_BASE_URL_PATH`. Input that already carries a scheme
    (``http://``/``https://``) is returned untouched — an explicit ``http://``
    is respected, never rewritten to https. Only scheme-less input is touched.

    This is a convenience layer in front of :func:`validate_base_url`, not a
    validator itself: it never raises, and anything it cannot make sense of is
    passed through unchanged so the validator reports it properly. Two shapes
    are deliberately left untouched rather than guessed at:

    - A query string or fragment (``host?x=1``, ``host/path#frag``) —
      :func:`validate_base_url` is documented to reject these; completing
      them into a valid-looking URL would silently discard the part the
      validator exists to catch.
    - A bare IPv6 host (``2001:db8::1``, ``::1``) — its colons are part of
      the address, not a port separator, so the "does netloc contain a
      port?" heuristic below cannot tell the two apart. Left alone, it falls
      through to :func:`validate_base_url`, which reports it as malformed
      (no ``http://``/``https://`` scheme) rather than this function
      guessing a default port into the middle of the address.
    """
    stripped = url.strip()
    if not stripped or "://" in stripped:
        return stripped
    if "?" in stripped or "#" in stripped:
        return stripped
    # More than one colon means this can only be a bare IPv6 address (a
    # "host:port" shape has exactly one). Leave it untouched.
    if stripped.count(":") > 1:
        return stripped
    candidate = f"https://{stripped}"
    try:
        parts = urlsplit(candidate)
    except ValueError:
        # e.g. an unterminated IPv6 bracket — let validate_base_url report it
        # as malformed rather than leaking a bare ValueError here.
        return candidate
    netloc = parts.netloc
    path = parts.path
    if ":" not in netloc:  # no explicit port
        netloc = f"{netloc}:{DEFAULT_BASE_URL_PORT}"
    if not path:
        path = DEFAULT_BASE_URL_PATH
    return f"https://{netloc}{path}"


def validate_base_url(url: str) -> None:
    """Gate on a user-supplied ``base_url`` (``--base-url`` / a TUI prompt).

    An allow-list, the same shape as :func:`validate_alias`: reject anything
    not explicitly known-safe rather than enumerate what's dangerous.

    This is NOT injection defense — it does not need to be. Every channel a
    ``base_url`` reaches (a shell ``export`` via ``_shell_single_quote``, a
    TOML value via ``toml_string``) already quotes it correctly regardless of
    content, and stays the ONLY defense on those channels. This function
    exists purely to catch typos and copy-paste mistakes early, with a
    message that says what's wrong — quoting must never be weakened on the
    assumption that this function already filtered the input.

    Raises:
        CodeHelperError: empty input, a scheme other than http/https (or none
            at all — the single most common typo), no host, a query string or
            fragment, embedded whitespace/control bytes, or over
            :data:`MAX_BASE_URL_LENGTH` characters.
    """
    stripped = url.strip()
    if not stripped:
        raise CodeHelperError("base URL is empty")

    if len(stripped) > MAX_BASE_URL_LENGTH:
        raise CodeHelperError(
            f"base URL is too long ({len(stripped)} > {MAX_BASE_URL_LENGTH})"
        )

    if any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in stripped):
        raise CodeHelperError(f"base URL must not contain whitespace: {url!r}")

    try:
        parts = urlsplit(stripped)
    except ValueError as exc:
        # urlsplit raises ValueError (not CodeHelperError) on some malformed
        # inputs — e.g. an unterminated IPv6 bracket ("http://[bad") raises
        # "Invalid IPv6 URL". Every caller of this function (with_base_url,
        # and transitively spec_from_installed's base_url recovery) expects
        # CodeHelperError as the ONLY failure mode this raises, so a bare
        # ValueError here would escape spec_from_installed's
        # `except CodeHelperError` and break its documented never-raises
        # contract for a recovered value from a hand-edited/truncated file.
        raise CodeHelperError(f"base URL is malformed: {stripped!r} ({exc})") from exc

    if parts.scheme not in ("http", "https"):
        raise CodeHelperError(
            f"base URL must start with http:// or https:// (got: {stripped!r})"
        )

    if not parts.netloc:
        raise CodeHelperError(f"base URL has no host: {stripped!r}")

    if parts.query or parts.fragment:
        raise CodeHelperError(
            f"base URL must not carry a query string or fragment: {stripped!r}"
        )
