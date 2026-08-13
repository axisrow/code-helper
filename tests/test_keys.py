"""Tests for ``cli/menu.py``'s pure key-sequence parser (``_translate``).

Before this module existed, the raw-keypress reader (``_read_key_raw``) was
untested end to end — every ``select_from_menu``/``press_any_key`` test
injects ``read_key`` and bypasses the parser entirely. ``_translate`` is the
seam that makes the parser itself testable without a real TTY: it takes the
first byte plus an injected, non-blocking ``read_more(timeout) -> str | None``
and returns a translated key name.
"""

from __future__ import annotations

import os
import sys

import pytest

from code_helper.cli.menu import _ESC_TIMEOUT, _PAIR_TIMEOUT, _translate


class _StubTermios:
    """Stand-in for ``termios`` — there is no TTY to save/restore under pytest."""

    TCSADRAIN = 0

    def tcgetattr(self, _fd):
        return None

    def tcsetattr(self, _fd, _when, _attrs):
        pass


class _StubTty:
    """Stand-in for ``tty`` — a pipe cannot be put into raw mode."""

    def setraw(self, _fd):
        pass


def _fake_read_more(bytes_available):
    """Return a ``read_more`` that yields ``bytes_available`` in order, then None.

    Also records whether it was ever called with a falsy/zero-ish "block
    forever" expectation is irrelevant here — the fake always returns
    immediately, which is enough to prove ``_translate`` never assumes a
    blocking read.
    """
    it = iter(bytes_available)

    def _read_more(_timeout=None):
        return next(it, None)

    return _read_more


@pytest.mark.unit
def test_lone_esc_is_cancel_without_blocking():
    calls = []

    def read_more(timeout):
        calls.append(timeout)
        return None  # no follow-up byte ever arrives — a lone Esc

    result = _translate("\x1b", read_more)

    assert result == "CANCEL"
    assert calls, "read_more must be consulted (non-blocking) before giving up"


@pytest.mark.unit
def test_csi_up_arrow():
    assert _translate("\x1b", _fake_read_more(["[", "A"])) == "UP"


@pytest.mark.unit
def test_csi_down_arrow():
    assert _translate("\x1b", _fake_read_more(["[", "B"])) == "DOWN"


@pytest.mark.unit
def test_ss3_up_arrow_application_cursor_mode():
    # Real terminals in application cursor mode send \x1bOA instead of
    # \x1b[A for the up arrow — previously always OTHER (arrows silently
    # dead in that mode).
    assert _translate("\x1b", _fake_read_more(["O", "A"])) == "UP"


@pytest.mark.unit
def test_ss3_down_arrow_application_cursor_mode():
    assert _translate("\x1b", _fake_read_more(["O", "B"])) == "DOWN"


@pytest.mark.unit
def test_f5_sequence_is_fully_consumed():
    # F5 is \x1b[15~ — a CSI sequence with TWO parameter bytes before the
    # final '~'. Previously only two bytes after ESC were ever read, leaving
    # '5' and '~' in the buffer to fire as spurious separate keypresses.
    read_more = _fake_read_more(["[", "1", "5", "~"])
    result = _translate("\x1b", read_more)
    assert result == "OTHER"  # F5 has no menu meaning
    # The fake is exhausted — every byte of the sequence was consumed by
    # _translate, none left over to "leak" into a subsequent read.
    assert read_more(0) is None


@pytest.mark.unit
def test_csi_home_tilde_form():
    assert _translate("\x1b", _fake_read_more(["[", "1", "~"])) == "HOME"


@pytest.mark.unit
def test_csi_end_tilde_form():
    assert _translate("\x1b", _fake_read_more(["[", "4", "~"])) == "END"


@pytest.mark.unit
def test_csi_home_letter_form():
    assert _translate("\x1b", _fake_read_more(["[", "H"])) == "HOME"


@pytest.mark.unit
def test_csi_end_letter_form():
    assert _translate("\x1b", _fake_read_more(["[", "F"])) == "END"


@pytest.mark.unit
def test_csi_page_up():
    assert _translate("\x1b", _fake_read_more(["[", "5", "~"])) == "PAGE_UP"


@pytest.mark.unit
def test_csi_page_down():
    assert _translate("\x1b", _fake_read_more(["[", "6", "~"])) == "PAGE_DOWN"


@pytest.mark.unit
def test_csi_cut_off_mid_sequence_is_other_not_a_crash():
    read_more = _fake_read_more(["["])  # nothing follows '['
    assert _translate("\x1b", read_more) == "OTHER"


@pytest.mark.unit
def test_unrecognized_escape_second_byte_is_other():
    assert _translate("\x1b", _fake_read_more(["z"])) == "OTHER"


@pytest.mark.unit
def test_enter_cr_swallows_paired_lf():
    read_more = _fake_read_more(["\n"])
    pushed = []
    assert _translate("\r", read_more, pushed.append) == "ENTER"
    # The LF half of CRLF was consumed — nothing left to fire a second ENTER,
    # and nothing pushed back (the pair byte is genuinely ours to drop).
    assert read_more(0) is None
    assert pushed == []


@pytest.mark.unit
def test_enter_lf_swallows_paired_cr():
    pushed = []
    assert _translate("\n", _fake_read_more(["\r"]), pushed.append) == "ENTER"
    assert pushed == []


@pytest.mark.unit
def test_enter_alone_no_pair_byte():
    pushed = []
    assert _translate("\r", _fake_read_more([]), pushed.append) == "ENTER"
    assert pushed == []


@pytest.mark.unit
def test_enter_pushes_back_a_non_pair_byte():
    """Enter then a fast next keypress: that keypress must NOT be eaten.

    Regression: the parser used to swallow whatever byte showed up after
    Enter. Pressing Enter and immediately an arrow (within the pair window)
    silently dropped the arrow — the navigation move just vanished.
    """
    pushed = []
    assert _translate("\r", _fake_read_more(["\x1b"]), pushed.append) == "ENTER"
    assert pushed == ["\x1b"]


@pytest.mark.unit
def test_enter_pair_peek_uses_the_short_timeout():
    """The CRLF peek must not wait out the 50ms Esc window on every Enter.

    A CRLF pair arrives in one burst from the terminal driver, so the wait
    only needs to cover microseconds; using ``_ESC_TIMEOUT`` here would add
    that latency to literally every Enter keypress.
    """
    seen = []

    def read_more(timeout):
        seen.append(timeout)
        return None

    assert _translate("\r", read_more, lambda _b: None) == "ENTER"
    assert seen == [_PAIR_TIMEOUT]
    assert _PAIR_TIMEOUT < _ESC_TIMEOUT


@pytest.mark.unit
@pytest.mark.parametrize("key", ["q", "Q"])
def test_soft_cancel_keys(key):
    assert _translate(key, _fake_read_more([])) == "CANCEL"


@pytest.mark.unit
def test_ctrl_c_is_a_distinct_hard_cancel():
    # In raw terminal mode ISIG is off, so Ctrl-C never raises
    # KeyboardInterrupt on its own — it must be a distinct translated key
    # from Esc/q's soft CANCEL, or a caller can't tell "go back" from "quit
    # everything" apart (see cli/tui.py's MenuCancelled.hard).
    assert _translate("\x03", _fake_read_more([])) == "HARD_CANCEL"


@pytest.mark.unit
def test_tab_is_a_distinct_key():
    # The Tab key is its own key name (issue #23 — profile switching from the
    # TUI main menu). Like HARD_CANCEL it must stay distinct from the escape
    # sequence bytes; select_from_menu ignores it unless on_tab is given.
    assert _translate("\t", _fake_read_more([])) == "TAB"


@pytest.mark.unit
@pytest.mark.parametrize("key", ["k", "K"])
def test_vim_up(key):
    assert _translate(key, _fake_read_more([])) == "UP"


@pytest.mark.unit
@pytest.mark.parametrize("key", ["j", "J"])
def test_vim_down(key):
    assert _translate(key, _fake_read_more([])) == "DOWN"


@pytest.mark.unit
def test_vim_home():
    assert _translate("g", _fake_read_more([])) == "HOME"


@pytest.mark.unit
def test_vim_end():
    assert _translate("G", _fake_read_more([])) == "END"


@pytest.mark.unit
@pytest.mark.parametrize("digit", list("123456789"))
def test_digit_keys(digit):
    assert _translate(digit, _fake_read_more([])) == f"DIGIT_{digit}"


@pytest.mark.unit
def test_unrelated_key_is_other():
    assert _translate("x", _fake_read_more([])) == "OTHER"


@pytest.mark.unit
def test_key_reader_pushback_survives_into_the_next_call(monkeypatch):
    """The byte pushed back by one keypress is returned by the NEXT one.

    This is the half of the Enter fix that ``_translate`` alone cannot prove:
    a byte peeked past a keypress cannot be un-read from the fd, so it is
    parked in the reader's pending slot and drained by the following call.
    Without that slot, "Enter then Down" typed fast loses the Down.

    Drives a single :class:`KeyReader` (whose pushback slot spans calls)
    over a fake fd (a pipe) with the ``termios``/``tty`` calls stubbed out,
    since there is no TTY under pytest.
    """
    from code_helper.cli.menu import KeyReader

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"\r\x1b[B")  # Enter, immediately followed by Down

    monkeypatch.setitem(sys.modules, "termios", _StubTermios())
    monkeypatch.setitem(sys.modules, "tty", _StubTty())

    class _FakeStream:
        def fileno(self):
            return read_fd

    reader = KeyReader(_FakeStream())
    try:
        assert reader.read() == "ENTER"
        # The ESC that began the Down sequence was pushed back, not eaten.
        assert reader._pending == "\x1b"
        assert reader.read() == "DOWN"
        assert reader._pending is None
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.unit
def test_select_from_menu_default_reader_preserves_pushback_across_calls(monkeypatch):
    """``select_from_menu``'s DEFAULT ``read_key`` must share one reader.

    Regression test for the production TUI shape: the user hits Enter on one
    menu, then IMMEDIATELY types Down intending it for whichever menu opens
    next (``TuiSession._pick`` calls ``select_from_menu`` fresh each time a
    new screen is shown). Enter's CRLF-pair peek parks that Down's leading
    ESC byte in the reader's pushback slot; the byte must be drained by the
    FIRST read of the NEXT ``select_from_menu`` call, not discarded.

    Before the fix, ``read_key=None`` defaulted to ``_read_key_raw``, which
    builds a brand-new, throwaway :class:`KeyReader` on every call — so this
    pushback byte was silently dropped at exactly this boundary, exactly
    where the pre-refactor module-level ``_pending_byte`` global did not
    drop it. Every other test in this suite injects ``read_key`` directly
    and would never catch a regression at this specific boundary.
    """
    import code_helper.cli.menu as menu_module
    from code_helper.cli.menu import select_from_menu

    monkeypatch.setattr(menu_module, "_default_key_reader", None)

    read_fd, write_fd = os.pipe()
    # First menu: Enter selects "only" immediately, but the immediately
    # following Down (intended for the SECOND menu) rides along on the same
    # write and gets peeked-and-parked by Enter's CRLF check.
    # Second menu: that parked Down must be seen as this call's first key.
    os.write(write_fd, b"\r\x1b[B\r")

    monkeypatch.setitem(sys.modules, "termios", _StubTermios())
    monkeypatch.setitem(sys.modules, "tty", _StubTty())

    class _FakeStream:
        def fileno(self):
            return read_fd

    monkeypatch.setattr(sys, "stdin", _FakeStream())
    try:
        first_result = select_from_menu(
            ["only"], read_key=None, print_fn=lambda _: None
        )
        second_result = select_from_menu(
            ["first", "second"], read_key=None, print_fn=lambda _: None
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert first_result == "only"
    # If the parked Down was dropped, the immediate Enter selects "first".
    assert second_result == "second"


@pytest.mark.unit
def test_utf8_layout_key_is_decoded_before_translation():
    """A Cyrillic keypress is one key, not replacement bytes."""
    assert _translate("ф", _fake_read_more([])) == _translate("a", _fake_read_more([]))
    assert _translate("\xd1", _fake_read_more(["\x84"])) == "a"
