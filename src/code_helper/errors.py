"""Project-wide exception types.

``CodeHelperError`` is the single expected-error sentinel raised by the
services/backends layers. :func:`emit_error` enforces the user-facing contract
once — a one-line ``error: <message>`` on stderr plus a non-zero exit, with no
traceback unless ``--debug`` is passed — and is called from every dispatch site
(``__main__.main`` and the looping TUI) so the format can never drift between
them.

It lives in its own module (not in ``__main__``) so that the class object is
identical regardless of how the package is invoked — ``python -m code_helper``
runs ``__main__.py`` as the ``__main__`` module, which would otherwise create a
*second* ``CodeHelperError`` distinct from the one imported by
``backends``/``services``. That identity split would defeat the single
``except`` in ``main()`` and leak a traceback in production. One module, one
class.
"""

import sys

__all__ = ["CodeHelperError", "emit_error"]


class CodeHelperError(Exception):
    """Expected helper error → one-line message + non-zero exit, no traceback.

    Raised by services/backends layers when an anticipated failure occurs
    (file not found, foreign script in the way, unknown wrapper name).
    Caught at the dispatch sites (``__main__.main`` and the TUI loop). The
    contract: print ``error: <msg>`` to stderr and exit non-zero; under
    ``--debug`` re-raise so Python emits the full traceback for debugging.
    """


def emit_error(e: CodeHelperError, debug: bool) -> int:
    """Render a ``CodeHelperError`` per the user-facing contract.

    Under ``--debug`` re-raise (so Python prints the full traceback); otherwise
    print ``error: <msg>`` to stderr and return exit code 1. Called from every
    dispatch site so the error format lives in one place.
    """
    if debug:
        raise e
    print(f"error: {e}", file=sys.stderr)
    return 1
