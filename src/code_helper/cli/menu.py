"""Minimal arrow-key terminal menu (stdlib only — no Rich/Typer/curses).

A single-purpose replacement for a "pick one of N wrappers" prompt. The
raw-keypress reading (``_read_key_raw``) is the only part that touches a real
TTY (via ``termios``/``tty``, Unix-only); :func:`select_from_menu` takes it as
an injectable ``read_key`` callable so tests can drive the menu with a fake
key sequence instead of a real terminal — mirroring how
:func:`code_helper.services.secrets.resolve_token` injects ``getpass_fn``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

__all__ = ["select_from_menu", "MenuCancelled"]


class MenuCancelled(Exception):
    """Raised when the user cancels the menu (Ctrl-C or 'q')."""


def _read_key_raw(stream=sys.stdin) -> str:
    """Read one raw keypress from a real TTY.

    Translates arrow escape sequences (``\\x1b[A`` / ``\\x1b[B``) to
    ``"UP"``/``"DOWN"``, Enter to ``"ENTER"``, and Ctrl-C/``q``/``Q`` to
    ``"CANCEL"``. Everything else is ``"OTHER"`` (ignored by the caller).
    """
    import termios
    import tty

    fd = stream.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = stream.read(1)
        if ch == "\x1b":
            ch2 = stream.read(1)
            ch3 = stream.read(1)
            if ch2 == "[":
                if ch3 == "A":
                    return "UP"
                if ch3 == "B":
                    return "DOWN"
            return "OTHER"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch in ("\x03", "q", "Q"):
            return "CANCEL"
        return "OTHER"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_from_menu(
    items: Sequence[str],
    *,
    prompt: str = "select:",
    read_key: Callable[[], str] = _read_key_raw,
    print_fn: Callable[[str], None] = print,
) -> str:
    """Interactively select one of ``items`` with Up/Down + Enter.

    Args:
        items: The choices to present, in order. Must be non-empty.
        prompt: Heading line printed above the list.
        read_key: Source of translated keypresses (``"UP"``/``"DOWN"``/
            ``"ENTER"``/``"CANCEL"``/``"OTHER"``). Defaults to reading a real
            TTY; tests inject a fake sequence instead.
        print_fn: Injectable output sink for the rendered menu.

    Returns:
        The selected item.

    Raises:
        ValueError: ``items`` is empty.
        MenuCancelled: the user cancelled (Ctrl-C or 'q').
    """
    if not items:
        raise ValueError("items must be non-empty")

    index = 0
    while True:
        print_fn(f"\n{prompt}")
        for i, item in enumerate(items):
            marker = ">" if i == index else " "
            print_fn(f" {marker} {item}")

        key = read_key()
        if key == "UP":
            index = (index - 1) % len(items)
        elif key == "DOWN":
            index = (index + 1) % len(items)
        elif key == "ENTER":
            return items[index]
        elif key == "CANCEL":
            raise MenuCancelled()
