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


def _run_menu(monkeypatch, keys, *, clear=False, tty=True):
    """Drive ``select_from_menu`` with a fake-TTY stdout; return its output.

    The ``sys.stdout`` patch is applied here (in the test body's call stack,
    after pytest installs its capture) so it actually wins — patching from a
    fixture setup would be overwritten by pytest's own capture replacement.
    """
    fake = _FakeTTY()
    if not tty:
        fake.isatty = lambda: False
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu(["deepseek", "glm"], read_key=_fake_keys(keys), clear=clear)
    return "".join(fake.writes)


@pytest.mark.unit
def test_select_from_menu_redraws_in_place_on_tty(monkeypatch):
    # The second render erases the first in place: a cursor-up + clear-to-end
    # ANSI sequence is emitted, so the menu stops scrolling down the screen.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"])
    assert "\x1b[3A" in joined  # cursor up by frame_lines = len(items)+1 = 3
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
