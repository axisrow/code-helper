"""The context-window ceiling shared by every range check and error message.

One home for the number, because it appears in two roles at once: the range
CHECK (``0 <= value <= MAX_CONTEXT_WINDOW`` — ``0`` is the explicit "no
declaration" sentinel, issue #83) and the error TEXT rendered next to it
(interpolated as ``{MAX_CONTEXT_WINDOW:_}`` so the message keeps the
historical ``10_000_000`` spelling byte-for-byte).

The home is a leaf module with no ``codehelper`` imports at all: ``spec``
cannot host it (``render → spec`` is an existing edge, so ``spec → render``
would close a cycle), and the number is needed ABOVE ``state`` too — but
``state`` sits at the bottom of the graph. A leaf is the one place every
consumer — ``cli/parser``, ``services/{spec,state,context_window}`` — can
import from without perturbing the import order that #104 settled.
"""

from __future__ import annotations

__all__ = ["MAX_CONTEXT_WINDOW", "context_window_usable"]

#: Custom-input ceiling: a typo beyond this is a mistake, not a window.
MAX_CONTEXT_WINDOW = 10_000_000


def context_window_usable(value: int) -> bool:
    """``True`` when ``value`` is a storable context-window answer.

    ``0`` counts as usable — it is the explicit "no declaration" sentinel
    (issue #83), never normalized away. Negative values and typo'd
    gigavalues are not: they must not ride an explicit answer into a
    wrapper marker or the live settings (review round 2, PR #84).
    """
    return 0 <= value <= MAX_CONTEXT_WINDOW
