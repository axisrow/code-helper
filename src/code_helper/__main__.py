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


if __name__ == "__main__":
    sys.exit(main())
