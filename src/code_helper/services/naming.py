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

from code_helper.errors import CodeHelperError

__all__ = ["validate_alias", "MAX_ALIAS_LENGTH", "RESERVED_ALIASES"]

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
