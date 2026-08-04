"""Console-script entry point: build parser, dispatch, enforce the error contract.

This is the single place that translates ``CodeHelperError`` (thrown from any
services/backends layer) into the user-facing contract: a one-line
``error: <message>`` on stderr plus exit code 1, with no traceback unless
``--debug`` is passed. Unhandled exceptions (real bugs) are not caught —
Python prints the traceback itself.
"""

import sys

from code_helper.cli.parser import build_parser
from code_helper.errors import CodeHelperError

__all__ = ["main", "CodeHelperError"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CodeHelperError as e:
        if getattr(args, "debug", False):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
