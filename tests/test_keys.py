"""Tests for ``cli/menu.py``'s pure key-sequence parser (``_translate``).

Before this module existed, the raw-keypress reader (``_read_key_raw``) was
untested end to end — every ``select_from_menu``/``press_any_key`` test
injects ``read_key`` and bypasses the parser entirely. ``_translate`` is the
seam that makes the parser itself testable without a real TTY: it takes the
first byte plus an injected, non-blocking ``read_more(timeout) -> str | None``
and returns a translated key name.
"""

from __future__ import annotations

import pytest

from code_helper.cli.menu import _translate


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
    assert _translate("\r", read_more) == "ENTER"
    # The LF half of CRLF was consumed — nothing left to fire a second ENTER.
    assert read_more(0) is None


@pytest.mark.unit
def test_enter_lf_swallows_paired_cr():
    read_more = _fake_read_more(["\r"])
    assert _translate("\n", read_more) == "ENTER"


@pytest.mark.unit
def test_enter_alone_no_pair_byte():
    assert _translate("\r", _fake_read_more([])) == "ENTER"


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
