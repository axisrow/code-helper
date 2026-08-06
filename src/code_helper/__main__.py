"""Console-script entry point: build parser, dispatch, enforce the error contract.

This is the single place that translates ``CodeHelperError`` (thrown from any
services/backends layer) into the user-facing contract: a one-line
``error: <message>`` on stderr plus exit code 1, with no traceback unless
``--debug`` is passed. Unhandled exceptions (real bugs) are not caught —
Python prints the traceback itself. The error rendering itself lives in
:func:`code_helper.errors.emit_error`, shared with the TUI loop.
"""

import sys

from code_helper.cli.parser import build_parser
from code_helper.errors import CodeHelperError, emit_error

__all__ = ["main", "CodeHelperError"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CodeHelperError as e:
        return emit_error(e, getattr(args, "debug", False))


def cli() -> int:
    """Console-script wrapper: :func:`main` plus the interrupt contract.

    ``MenuCancelled`` must keep propagating out of :func:`main` — that is how
    ``edit-token``'s picker lets a TUI caller tell "go back" from "leave the
    whole TUI", and it is pinned by a test. But at the PROCESS boundary there
    is no such caller, so letting it out prints a raw traceback instead of
    this module's one-line contract. Translating it here, one level above
    ``main``, satisfies both: the exception still escapes ``main`` for the
    TUI, and the plain CLI exits cleanly. 130 is the conventional
    SIGINT-terminated code, matching a real Ctrl-C.
    """
    from code_helper.cli.menu import MenuCancelled

    try:
        return main()
    except (MenuCancelled, KeyboardInterrupt):
        print("cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(cli())
