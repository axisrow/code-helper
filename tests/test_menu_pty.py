"""Real-PTY verification of ``cli/menu.py``'s raw-mode line editor.

``_read_line_raw`` puts the terminal into ``tty.setraw`` mode, which the
injected-``read_key``/``read_more`` fakes in ``tests/test_menu.py`` cannot
see: they exercise the pure key-translation logic, never the actual bytes a
real terminal driver would receive back (``ONLCR``, cursor-position escapes,
line wrapping at the real terminal width). See CLAUDE.md's testing section —
"Manual PTY verification (``pexpect``) is required for ANSI redraw/hang
behavior" — and ``tests/test_tui.py``'s existing ``pty``/``subprocess``-based
tests, whose harness (``read_until``/spawn/cleanup) this file's helpers
mirror rather than introduce a new dependency (``pexpect``) for.

Each test spawns a tiny driver script (not the full TUI wizard — these tests
are about the line editor's own raw-mode behavior, independent of what
``cli/tui.py`` does with the typed value) that calls
``codehelper.cli.menu.read_line`` directly under a real PTY.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name != "posix", reason="PTY tests require POSIX"),
]

_SRC = str(Path(__file__).resolve().parents[1] / "src")


class _PtySession:
    """One spawned driver process talking over a real pseudo-terminal.

    Mirrors the inline harness in ``tests/test_tui.py``'s
    ``test_provider_first_flow_through_a_real_pty`` — ``read_until`` polls the
    accumulated output for a marker (never blocking on a byte count that may
    arrive split across reads), and ``send`` writes raw bytes to the child's
    stdin. Kept as a class here (rather than closures) so it can be reused
    across several tests without re-deriving the read loop each time.
    """

    def __init__(self, script: str, *, columns: int = 80, lines: int = 24) -> None:
        environment = os.environ.copy()
        # No real HOME needed — the driver script never touches Paths.default().
        source = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (_SRC, source) if part
        )
        self.master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(
            slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", lines, columns, 0, 0)
        )
        self.child = subprocess.Popen(
            [sys.executable, "-c", script],
            env=environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        self._slave_fd = slave_fd
        self.output = bytearray()
        self._search_from = 0

    def read_until(self, marker: str, *, timeout: float = 5) -> None:
        deadline = time.monotonic() + timeout
        marker_bytes = marker.encode()
        while time.monotonic() < deadline:
            found = self.output.find(marker_bytes, self._search_from)
            if found >= 0:
                self._search_from = found + len(marker_bytes)
                return
            ready, _, _ = select.select([self.master_fd], [], [], 0.1)
            if ready:
                try:
                    self.output.extend(os.read(self.master_fd, 4096))
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
        raise AssertionError(
            f"never saw {marker!r} in: {self.output.decode(errors='replace')[-3000:]!r}"
        )

    def send(self, data: str) -> None:
        os.write(self.master_fd, data.encode())

    def close(self) -> None:
        if self.child.poll() is None:
            self.child.kill()
        try:
            self.child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        os.close(self.master_fd)


@pytest.fixture
def pty_session():
    sessions: list[_PtySession] = []

    def _spawn(script: str, **kwargs) -> _PtySession:
        session = _PtySession(script, **kwargs)
        sessions.append(session)
        return session

    yield _spawn
    for session in sessions:
        session.close()


# A driver that reads two lines back to back — this is the exact shape of the
# reported bug: `read_line("First: ")` then `read_line("Second: ")`, with the
# first answer long enough that a bare "\n" (no carriage return) would leave
# the cursor mid-line for the second prompt.
_TWO_PROMPTS = (
    "from codehelper.cli.menu import read_line\n"
    "a = read_line('First: ')\n"
    "b = read_line('Second: ')\n"
    "print('GOT:' + a + '|' + b)\n"
)


def test_enter_returns_cursor_to_column_zero_for_the_next_prompt(pty_session):
    """The bug as reported: a long first answer must not indent the next prompt.

    Before the fix, ``_read_line_raw`` wrote a bare ``\\n`` on Enter. In raw
    mode (``ONLCR`` off) that moves the cursor down WITHOUT returning it to
    column 0, so "Second: " was printed starting at whatever column "First: "
    plus the typed answer ended on — exactly the indentation the user saw.
    """
    session = pty_session(_TWO_PROMPTS)
    session.read_until("First: ")
    session.send("78.47.183.125\r")
    session.read_until("Second: ")
    # The bug, byte for byte: the pre-fix reader wrote a bare "\n" on Enter,
    # so the raw output was "...125\nSecond: " — a line feed with NO carriage
    # return, which every real terminal renders as "Second: " printed
    # starting wherever the cursor happened to be, not column 0. The fixed
    # reader writes "\r\n", so the byte immediately preceding "Second: " must
    # be a "\n" that is itself immediately preceded by "\r".
    # `read_line`'s own first repaint (`redraw()` at the top of
    # `_read_line_raw`) prepends a "\r\033[K" before the prompt text — so the
    # bytes right before "Second: " are that repaint's prefix, not the
    # "\r\n" Enter wrote. Strip exactly one such prefix before checking.
    before_second = bytes(session.output).split(b"Second: ", 1)[0]
    before_second = before_second.removesuffix(b"\r\x1b[K")
    assert before_second.endswith(b"\r\n"), (
        f"Enter did not return the cursor to column 0 before the next "
        f"prompt — got {before_second[-10:]!r} immediately before 'Second: '"
    )
    session.send("done\r")
    session.read_until("GOT:78.47.183.125|done")


def test_esc_and_ctrl_c_leave_the_cursor_at_column_zero(pty_session):
    """Esc/Ctrl-C must not abandon the cursor mid-prompt-line either.

    Both exit the reader via ``MenuCancelled`` rather than returning a value —
    a separate code path from Enter's, and one the original bug report never
    exercised, but it shared the same missing-``\\r`` defect.
    """
    script = (
        "from codehelper.cli.menu import MenuCancelled, read_line\n"
        "try:\n"
        "    read_line('Prompt: ')\n"
        "except MenuCancelled as exc:\n"
        "    print('CANCELLED:' + str(exc.hard))\n"
    )
    session = pty_session(script)
    session.read_until("Prompt: ")
    session.send("partial")
    session.read_until("partial")
    session.send("\x1b")  # lone Esc — soft cancel
    session.read_until("CANCELLED:False")
    # Before the fix, cancelling wrote NOTHING before raising — the raw bytes
    # were "Prompt: partialCANCELLED:False", i.e. the driver's own
    # verification print landed glued to the end of the typed text with no
    # separator at all. The fix must write "\r\n" before raising, so the
    # marker is always preceded by one.
    before_marker = bytes(session.output).split(b"CANCELLED:False", 1)[0]
    assert before_marker.endswith(b"\r\n"), (
        f"Esc did not return the cursor to column 0 before propagating — "
        f"got {before_marker[-10:]!r} immediately before the marker"
    )


def test_backspace_repaints_the_whole_line_instead_of_backing_up_blindly(
    pty_session,
):
    """Backspace must erase by repainting, not by trusting ``\\b`` to work.

    The pre-fix echo wrote a fixed ``"\\b \\b"`` per backspace REGARDLESS of
    terminal width — a value that stays correct (the buffer itself, and thus
    the final returned string, was never wrong: ``chars.pop()`` always
    matched what was typed). The bug was purely what's ON SCREEN: on a narrow
    terminal, once the line has wrapped onto a second physical row, ``\\b``
    at that row's start column 0 is a no-op on real terminal drivers, so a
    character that WAS removed from the buffer stays visibly on screen — a
    silent desync between state and display no return-value check can catch.

    The only observable, terminal-driver-independent proof of a real fix is
    the MECHANISM: every edit must repaint the full line from column 0
    (``\\r`` + ``\\033[K`` + prompt + text) rather than emit an incremental
    ``\\x08`` backspace byte at all. A full repaint cannot desync from the
    buffer regardless of where the terminal thinks the cursor is.
    """
    script = (
        "from codehelper.cli.menu import read_line\n"
        "v = read_line('P: ')\n"
        "print('GOT:' + repr(v))\n"
    )
    session = pty_session(script, columns=20)
    session.read_until("P: ")
    typed = "abcdefghijklmnopqrstuvwxyz"  # long enough to wrap at 20 columns
    session.send(typed)
    session.read_until(typed[-1])
    session.send("\x7f" * len(typed))  # erase everything
    session.send("ok\r")
    session.read_until("GOT:'ok'")
    assert b"\x08" not in bytes(session.output), (
        "backspace used an incremental \\b byte instead of a full-line "
        "repaint — this is exactly the mechanism that silently desyncs "
        "once the line has wrapped onto a second physical row"
    )


def test_long_input_never_echoes_a_row_wider_than_the_terminal(pty_session):
    """A base-URL-length input must never be echoed wider than the terminal.

    ``MAX_BASE_URL_LENGTH`` is 512 (``services/naming.py``) — far past any
    real terminal width. The pre-fix echo wrote every typed character
    unconditionally, so the LAST printed row (the one right before Enter)
    was as wide as the whole input: on an 80-column terminal, a 300+
    character value blew straight past the physical row width the in-place
    redraw's cursor math assumes (see ``_fit``'s docstring on why one
    logical line must stay one physical row). The fix's scrolling window
    guarantees every repaint — the raw text between consecutive ``\\r``s,
    not counting the CSI cursor-back move — fits in `columns`.
    """
    script = (
        "from codehelper.cli.menu import read_line\n"
        "v = read_line('URL: ')\n"
        "print('LEN:' + str(len(v)))\n"
    )
    columns = 80
    session = pty_session(script, columns=columns)
    session.read_until("URL: ")
    long_value = "https://example.com/" + ("a" * 300)
    session.send(long_value)
    session.send("\r")
    session.read_until(f"LEN:{len(long_value)}")
    before_result = bytes(session.output).split(b"LEN:", 1)[0]
    # Each repaint is "\r\033[K<prompt><window>" optionally followed by a
    # cursor-back CSI move; split on \r to get one repaint's printable
    # payload per chunk, then strip the "\033[K" and any trailing CSI move.
    import re

    widest = 0
    for chunk in before_result.split(b"\r"):
        printable = re.sub(rb"\x1b\[K", b"", chunk)
        printable = re.sub(rb"\x1b\[\d+D$", b"", printable)
        widest = max(widest, len(printable))
    assert widest <= columns, (
        f"a single repaint printed {widest} columns of text on an "
        f"{columns}-column terminal — the line wrapped, which desyncs the "
        f"in-place redraw's cursor-up math"
    )


def test_left_right_navigate_and_edit_mid_string(pty_session):
    """Left/Right must move the cursor, not get swallowed as dead escape bytes.

    Types "acb", moves left twice (cursor between 'a' and 'c'), inserts
    nothing extra — instead moves right once and inserts 'X' between 'c' and
    'b', producing "acXb". This is impossible to express with the pre-fix
    append-only editor, which had no way to place the cursor anywhere but the
    end of the line.
    """
    script = (
        "from codehelper.cli.menu import read_line\n"
        "v = read_line('E: ')\n"
        "print('GOT:' + repr(v))\n"
    )
    session = pty_session(script)
    session.read_until("E: ")
    session.send("acb")
    session.read_until("b")
    session.send("\x1b[D\x1b[D")  # Left, Left -> cursor before 'c'
    session.send("\x1b[C")  # Right -> cursor after 'c', before 'b'
    session.send("X")
    session.send("\r")
    session.read_until("GOT:'acXb'")


def test_secret_input_is_masked_and_never_echoes_the_literal_value(pty_session):
    """A token field must never print the typed characters in the clear."""
    script = (
        "from codehelper.cli.menu import read_line\n"
        "v = read_line('Token: ', secret=True)\n"
        "print('GOT:' + repr(v))\n"
    )
    session = pty_session(script)
    session.read_until("Token: ")
    session.send("sk-super-secret")
    session.send("\r")
    # Everything the terminal echoed while the field was live — i.e. before
    # the driver script's own verification `print`, which necessarily
    # contains the plaintext value to prove the round trip worked.
    session.read_until("GOT:'sk-super-secret'")
    echoed_while_typing = bytes(session.output).rsplit(b"GOT:", 1)[0]
    assert b"sk-super-secret" not in echoed_while_typing
    # Each keystroke redraws the whole line, so every frame up to and
    # including the final one is present — the LAST frame (right before the
    # trailing "\r\n" Enter wrote) is what must show exactly one bullet per
    # typed character, proving the full string was masked, not just some.
    last_frame = echoed_while_typing.rstrip(b"\r\n").rsplit(b"\r", 1)[-1]
    assert last_frame == b"\x1b[KToken: " + "•".encode() * len("sk-super-secret")
