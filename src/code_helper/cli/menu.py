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

__all__ = ["select_from_menu", "MenuCancelled", "press_any_key"]


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
    clear: bool = False,
) -> str:
    """Interactively select one of ``items`` with Up/Down + Enter.

    Args:
        items: The choices to present, in order. Must be non-empty.
        prompt: Heading line printed above the list.
        read_key: Source of translated keypresses (``"UP"``/``"DOWN"``/
            ``"ENTER"``/``"CANCEL"``/``"OTHER"``). Defaults to reading a real
            TTY; tests inject a fake sequence instead.
        print_fn: Injectable output sink for the rendered menu.
        clear: On a TTY, clear the *whole* screen (``\\033[2J\\033[H``) before
            the first frame instead of leaving prior content above it. Used by
            the looping TUI so a re-shown menu replaces the previous command's
            output (e.g. ``list``) rather than printing below it. Ignored when
            ``stdout`` is not a TTY.

    On a real terminal (``stdout`` is a TTY) the menu is redrawn *in place* on
    every keypress — a cursor-up + clear-to-end sequence erases the previous
    frame before the next is printed, so arrow-key navigation no longer
    scrolls an endless stack of menus down the screen. The first frame keeps
    a leading blank line for separation (unless ``clear`` clears the screen).
    When ``stdout`` is not a TTY (piped output, or tests with an injected
    ``print_fn``), the menu falls back to reprinting each frame below the last
    — ANSI escapes would only pollute a non-terminal sink.

    Returns:
        The selected item.

    Raises:
        ValueError: ``items`` is empty.
        MenuCancelled: the user cancelled (Ctrl-C or 'q').
    """
    if not items:
        raise ValueError("items must be non-empty")

    redraw = sys.stdout.isatty()
    first = True
    # One rendered frame = prompt line + one line per item (the first frame's
    # leading blank line is spacing, not part of the erasable frame).
    frame_lines = len(items) + 1
    # Erase the previous frame in place: cursor up `frame_lines` lines, then
    # clear to end of screen. Prepended to the prompt in the SAME print_fn
    # call — a separate call would add a stray newline (shifting the cursor)
    # and routing it via sys.stdout would bypass the injectable output sink.
    clear_seq = f"\033[{frame_lines}A\033[J"
    # Clear the whole screen + home cursor, for the `clear` entry mode.
    clear_screen = "\033[2J\033[H"
    index = 0
    while True:
        if redraw and not first:
            print_fn(f"{clear_seq}{prompt}")
        elif redraw and clear:
            print_fn(f"{clear_screen}{prompt}")
        else:
            print_fn(f"\n{prompt}")
        for i, item in enumerate(items):
            marker = ">" if i == index else " "
            print_fn(f" {marker} {item}")
        first = False

        key = read_key()
        if key == "UP":
            index = (index - 1) % len(items)
        elif key == "DOWN":
            index = (index + 1) % len(items)
        elif key == "ENTER":
            return items[index]
        elif key == "CANCEL":
            raise MenuCancelled()


def press_any_key(prompt: str = "") -> None:
    """Block for one keypress on a real TTY; a no-op off a TTY.

    If ``prompt`` is given it is printed without a trailing newline before the
    read, and a newline is printed after, so a caller can show a "press any key
    to continue" hint on the same line as the cursor. When ``stdout`` is not a
    TTY (piped output, tests) it returns immediately so non-interactive runs
    never block waiting on stdin.

    Reads via :func:`_read_key_raw`, which owns the raw ``termios``/``tty``
    setup — keeping all terminal I/O in this module.
    """
    if not sys.stdout.isatty():
        return
    if prompt:
        print(prompt, end="", flush=True)
    _read_key_raw()
    if prompt:
        print()
