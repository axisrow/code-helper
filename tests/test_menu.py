"""Tests for ``cli/menu.py`` — the stdlib arrow-key selection menu.

``read_key`` is injected as a fake iterator so these tests never touch a real
TTY (mirroring how ``services/secrets.py`` tests inject ``getpass_fn``).
"""

from __future__ import annotations

import sys

import pytest

from code_helper.cli.menu import MenuCancelled, press_any_key, select_from_menu


def _fake_keys(keys):
    it = iter(keys)
    return lambda: next(it)


@pytest.mark.unit
def test_select_from_menu_down_then_enter_picks_second_item():
    result = select_from_menu(
        ["deepseek", "glm"],
        read_key=_fake_keys(["DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "glm"


@pytest.mark.unit
def test_select_from_menu_enter_immediately_picks_first_item():
    result = select_from_menu(
        ["deepseek", "glm"],
        read_key=_fake_keys(["ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "deepseek"


@pytest.mark.unit
def test_select_from_menu_up_wraps_to_last_item():
    result = select_from_menu(
        ["deepseek", "glm"],
        read_key=_fake_keys(["UP", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "glm"


@pytest.mark.unit
def test_select_from_menu_down_wraps_to_first_item():
    result = select_from_menu(
        ["deepseek", "glm"],
        read_key=_fake_keys(["DOWN", "DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "deepseek"


@pytest.mark.unit
def test_select_from_menu_other_key_is_ignored():
    result = select_from_menu(
        ["deepseek", "glm"],
        read_key=_fake_keys(["OTHER", "DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "glm"


@pytest.mark.unit
def test_select_from_menu_cancel_raises():
    with pytest.raises(MenuCancelled):
        select_from_menu(
            ["deepseek", "glm"],
            read_key=_fake_keys(["CANCEL"]),
            print_fn=lambda _: None,
        )


@pytest.mark.unit
def test_select_from_menu_empty_items_raises():
    with pytest.raises(ValueError, match="non-empty"):
        select_from_menu([], read_key=_fake_keys(["ENTER"]))


@pytest.mark.unit
def test_select_from_menu_renders_prompt_and_items():
    lines = []
    select_from_menu(
        ["deepseek", "glm"],
        prompt="pick one:",
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    output = "\n".join(lines)
    assert "pick one:" in output
    assert "deepseek" in output
    assert "glm" in output


class _FakeTTY:
    """Minimal stdout stand-in whose ``isatty()`` is True (for redraw-in-place)."""

    def __init__(self):
        self.writes: list[str] = []

    def write(self, s: str) -> int:
        self.writes.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True


def _run_menu(monkeypatch, keys, *, clear=False, tty=True, hint=None):
    """Drive ``select_from_menu`` with a fake-TTY stdout; return its output.

    The ``sys.stdout`` patch is applied here (in the test body's call stack,
    after pytest installs its capture) so it actually wins — patching from a
    fixture setup would be overwritten by pytest's own capture replacement.
    """
    fake = _FakeTTY()
    if not tty:
        fake.isatty = lambda: False
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu(
        ["deepseek", "glm"], read_key=_fake_keys(keys), clear=clear, hint=hint
    )
    return "".join(fake.writes)


@pytest.mark.unit
def test_select_from_menu_redraws_in_place_on_tty(monkeypatch):
    # The second render erases the first in place: a cursor-up + clear-to-end
    # ANSI sequence is emitted, so the menu stops scrolling down the screen.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"])
    assert "\x1b[4A" in joined  # frame_lines = len(items)+2 = 4 (heading + spacer)
    assert "\x1b[J" in joined  # clear from cursor to end of screen


@pytest.mark.unit
def test_select_from_menu_no_redraw_codes_when_not_a_tty(monkeypatch):
    # Off a TTY: fall back to reprinting, never emit ANSI escapes into the sink.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], tty=False)
    assert "\x1b[" not in joined


@pytest.mark.unit
def test_select_from_menu_clears_screen_on_first_frame_when_clear(monkeypatch):
    # `clear=True` on a TTY: the first frame clears the whole screen + homes
    # the cursor, so a re-shown menu replaces prior output instead of stacking.
    joined = _run_menu(monkeypatch, ["ENTER"], clear=True)
    assert "\x1b[2J" in joined  # clear entire screen
    assert "\x1b[H" in joined  # cursor to home


@pytest.mark.unit
def test_select_from_menu_no_full_clear_when_clear_not_requested(monkeypatch):
    # Without `clear`, the first frame uses a leading blank line — never a
    # full-screen clear (that would wipe the caller's prior output unasked).
    joined = _run_menu(monkeypatch, ["ENTER"])
    assert "\x1b[2J" not in joined


@pytest.mark.unit
def test_select_from_menu_clear_ignored_when_not_a_tty(monkeypatch):
    # `clear` is a no-op off a TTY: no ANSI escapes leak into a non-terminal sink.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], clear=True, tty=False)
    assert "\x1b[" not in joined


@pytest.mark.unit
def test_press_any_key_noop_when_not_a_tty(capsys):
    # Under pytest sys.stdout is captured (not a TTY): never block, never print.
    press_any_key("hint")
    assert capsys.readouterr().out == ""


# --- (value, label) pairs ----------------------------------------------


@pytest.mark.unit
def test_select_from_menu_pair_items_return_value_render_label():
    lines = []
    result = select_from_menu(
        [("deepseek", "deepseek — local Ollama"), ("glm", "glm — Z.ai")],
        read_key=_fake_keys(["DOWN", "ENTER"]),
        print_fn=lines.append,
    )
    output = "\n".join(lines)
    assert result == "glm"
    assert "glm — Z.ai" in output
    assert "glm-ollama" not in output  # label, not a raw value fragment


@pytest.mark.unit
def test_select_from_menu_mixed_str_and_pair_items():
    result = select_from_menu(
        ["deepseek", ("glm", "glm — Z.ai")],
        read_key=_fake_keys(["DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "glm"


# --- hint line -----------------------------------------------------------


@pytest.mark.unit
def test_select_from_menu_hint_is_rendered():
    lines = []
    select_from_menu(
        ["deepseek", "glm"],
        hint="↑/↓ · Enter · Esc back",
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    assert any("Esc back" in line for line in lines)


@pytest.mark.unit
def test_select_from_menu_hint_extends_redraw_frame_by_two_lines(monkeypatch):
    # frame_lines = len(items) + 2 (heading + spacer) + 2 (spacer + hint) = 6
    # for 2 items. If this drifts, the in-place redraw erases the wrong
    # number of lines and the menu visibly creeps down the screen on a real
    # terminal.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], hint="↑/↓ · Enter")
    assert "\x1b[6A" in joined


# --- digit / home / end / page navigation --------------------------------


@pytest.mark.unit
def test_select_from_menu_digit_selects_directly():
    result = select_from_menu(
        ["a", "b", "c"],
        read_key=_fake_keys(["DIGIT_3"]),
        print_fn=lambda _: None,
    )
    assert result == "c"


@pytest.mark.unit
def test_select_from_menu_digit_out_of_range_is_ignored():
    result = select_from_menu(
        ["a", "b"],
        read_key=_fake_keys(["DIGIT_9", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "a"  # DIGIT_9 ignored, ENTER picks the still-first item


@pytest.mark.unit
def test_select_from_menu_home_jumps_to_first():
    result = select_from_menu(
        ["a", "b", "c"],
        read_key=_fake_keys(["DOWN", "DOWN", "HOME", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "a"


@pytest.mark.unit
def test_select_from_menu_end_jumps_to_last():
    result = select_from_menu(
        ["a", "b", "c"],
        read_key=_fake_keys(["END", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "c"


@pytest.mark.unit
def test_select_from_menu_page_up_down_behave_like_home_end():
    assert (
        select_from_menu(
            ["a", "b", "c"],
            read_key=_fake_keys(["PAGE_DOWN", "ENTER"]),
            print_fn=lambda _: None,
        )
        == "c"
    )
    assert (
        select_from_menu(
            ["a", "b", "c"],
            read_key=_fake_keys(["DOWN", "PAGE_UP", "ENTER"]),
            print_fn=lambda _: None,
        )
        == "a"
    )


# --- MenuCancelled.hard ----------------------------------------------------


@pytest.mark.unit
def test_menu_cancelled_soft_by_default_for_esc_or_q():
    with pytest.raises(MenuCancelled) as exc_info:
        select_from_menu(
            ["a", "b"], read_key=_fake_keys(["CANCEL"]), print_fn=lambda _: None
        )
    assert exc_info.value.hard is False


@pytest.mark.unit
def test_menu_cancelled_hard_on_keyboard_interrupt():
    def _raise_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(MenuCancelled) as exc_info:
        select_from_menu(["a", "b"], read_key=_raise_interrupt, print_fn=lambda _: None)
    assert exc_info.value.hard is True


@pytest.mark.unit
def test_menu_cancelled_hard_on_ctrl_c_key():
    # In raw terminal mode ISIG is off, so Ctrl-C is delivered as a
    # translated "HARD_CANCEL" key (see test_keys.py), never as a raised
    # KeyboardInterrupt — this is the path that actually fires on a real TTY.
    with pytest.raises(MenuCancelled) as exc_info:
        select_from_menu(
            ["a", "b"], read_key=_fake_keys(["HARD_CANCEL"]), print_fn=lambda _: None
        )
    assert exc_info.value.hard is True


# --- cursor hide/show -------------------------------------------------------


@pytest.mark.unit
def test_select_from_menu_hides_and_restores_cursor_on_tty(monkeypatch):
    joined = _run_menu(monkeypatch, ["ENTER"])
    assert "\x1b[?25l" in joined  # hidden on entry
    assert "\x1b[?25h" in joined  # restored on exit


@pytest.mark.unit
def test_select_from_menu_restores_cursor_even_on_exception(monkeypatch):
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)

    def _raise_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(MenuCancelled):
        select_from_menu(["a", "b"], read_key=_raise_interrupt)

    joined = "".join(fake.writes)
    assert "\x1b[?25h" in joined


@pytest.mark.unit
def test_select_from_menu_no_cursor_codes_when_not_a_tty(monkeypatch):
    joined = _run_menu(monkeypatch, ["ENTER"], tty=False)
    assert "\x1b[?25l" not in joined
    assert "\x1b[?25h" not in joined


# --- truncation to terminal width ------------------------------------------


@pytest.mark.unit
def test_select_from_menu_truncates_long_row_on_tty(monkeypatch):
    import os
    import shutil as shutil_module

    monkeypatch.setattr(
        shutil_module,
        "get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((40, 24)),
    )
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    long_label = "x" * 80
    select_from_menu([long_label, "b"], read_key=_fake_keys(["ENTER"]))
    joined = "".join(fake.writes)
    # No printed line should exceed the reported terminal width — a longer
    # line would wrap into an extra physical row the redraw's frame_lines
    # counter doesn't know about, and the menu would creep down the screen.
    for line in joined.splitlines():
        assert len(line) <= 40, line
    assert "…" in joined  # the truncation is visibly marked, not silent


@pytest.mark.unit
def test_select_from_menu_no_truncation_when_not_a_tty(monkeypatch):
    long_label = "x" * 200
    lines = []
    fake = _FakeTTY()
    fake.isatty = lambda: False
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu(
        [long_label, "b"], read_key=_fake_keys(["ENTER"]), print_fn=lines.append
    )
    assert any(long_label in line for line in lines)


# --- digit numbering ---------------------------------------------------------


@pytest.mark.unit
def test_select_from_menu_items_are_numbered_from_one():
    lines = []
    select_from_menu(
        ["a", "b", "c"], read_key=_fake_keys(["ENTER"]), print_fn=lines.append
    )
    output = "\n".join(lines)
    assert "1" in output and "2" in output and "3" in output


@pytest.mark.unit
def test_select_from_menu_unnumbered_item_gets_no_digit_and_dot_instead():
    lines = []
    select_from_menu(
        [("a", "a"), ("back", "← назад")],
        unnumbered=frozenset({"back"}),
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    back_line = next(line for line in lines if "← назад" in line)
    assert "1" not in back_line
    assert "·" in back_line


@pytest.mark.unit
def test_select_from_menu_digit_still_selects_correct_item_with_unnumbered_entry():
    # "back" is unnumbered, so "c" (the 3rd NUMBERED item) is still DIGIT_3 —
    # the shown digit and the digit that selects it must never drift apart.
    result = select_from_menu(
        [("a", "a"), ("b", "b"), ("c", "c"), ("back", "← назад")],
        unnumbered=frozenset({"back"}),
        read_key=_fake_keys(["DIGIT_3"]),
        print_fn=lambda _: None,
    )
    assert result == "c"
