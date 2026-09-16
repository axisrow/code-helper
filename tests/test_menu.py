"""Tests for ``cli/menu.py`` — the stdlib arrow-key selection menu.

``read_key`` is injected as a fake iterator so these tests never touch a real
TTY (mirroring how ``services/secrets.py`` tests inject ``getpass_fn``).
"""

from __future__ import annotations

import sys

import pytest

from codehelper.cli.menu import (
    MenuCancelled,
    Section,
    _fit,
    _visible_len,
    press_any_key,
    read_line,
    select_from_menu,
)


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
def test_read_line_uses_the_shared_non_tty_input_contract(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: " typed ")
    assert read_line("Value: ") == "typed"


@pytest.mark.unit
def test_read_line_ctrl_c_is_a_hard_cancel(monkeypatch):
    def interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(MenuCancelled) as exc_info:
        read_line("Value: ")
    assert exc_info.value.hard is True


@pytest.mark.unit
def test_secret_read_rejects_non_ascii_input(monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-ключ")
    with pytest.raises(ValueError, match="ASCII"):
        read_line("Token: ", secret=True)


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


@pytest.mark.unit
def test_select_from_menu_two_line_hint_is_rendered():
    lines = []
    select_from_menu(
        ["deepseek", "glm"],
        hint="↑/↓ · Enter\ns settings · Esc back",
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    assert any("↑/↓ · Enter" in line for line in lines)
    assert any("s settings · Esc back" in line for line in lines)


@pytest.mark.unit
def test_select_from_menu_two_line_hint_extends_redraw_frame_by_three(monkeypatch):
    # frame_lines = len(items) + 2 (heading + spacer) + 3 (spacer + 2 hint
    # rows) = 7 for 2 items. Same creep risk as the one-line pin above — the
    # second hint row must be counted, or the redraw erases one line too few.
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], hint="line one\nline two")
    assert "\x1b[7A" in joined


@pytest.mark.unit
def test_select_from_menu_hint_rejects_more_than_two_lines():
    for bad in ("a\n\nb", "a\nb\nc", "a\rb", "a\nb\n"):
        with pytest.raises(ValueError):
            select_from_menu(
                ["deepseek", "glm"],
                hint=bad,
                read_key=_fake_keys(["ENTER"]),
                print_fn=lambda _: None,
            )


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
def test_select_from_menu_page_up_down_move_by_a_viewport():
    assert (
        select_from_menu(
            ["a", "b", "c"],
            read_key=_fake_keys(["PAGE_DOWN", "ENTER"]),
            print_fn=lambda _: None,
        )
        == "b"
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
        lambda _fallback=(80, 24): os.terminal_size((40, 24)),
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


# --- on_tab / callable prompt & hint (issue #23) ---------------------------


@pytest.mark.unit
def test_select_from_menu_tab_ignored_without_on_tab():
    # Without on_tab, Tab must stay silently ignored (like OTHER) — it must
    # not start doing something in every other menu in the app.
    result = select_from_menu(
        ["a", "b"],
        read_key=_fake_keys(["TAB", "DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "b"


@pytest.mark.unit
def test_select_from_menu_on_tab_invoked_and_frame_redraws(monkeypatch):
    calls = []
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)

    def on_tab():
        calls.append("tab")

    select_from_menu(["a", "b"], read_key=_fake_keys(["TAB", "ENTER"]), on_tab=on_tab)
    joined = "".join(fake.writes)
    assert calls == ["tab"]
    # The Tab handler ran and the menu redrew in place (clear-to-end escape).
    assert "\x1b[J" in joined


# --- on_token / TOKEN key (issue #29) ---------------------------------------


@pytest.mark.unit
def test_translate_t_and_T_return_token():
    # `t`/`T` is the per-row token-rotation key on the main screen (#29). It
    # maps to a caller-side hook name just like Tab, so a menu without on_token
    # ignores it instead of doing something surprising.
    from codehelper.cli.menu import _translate

    assert _translate("t", lambda _t: None) == "TOKEN"
    assert _translate("T", lambda _t: None) == "TOKEN"


@pytest.mark.unit
def test_select_from_menu_on_token_called_with_cursor_value():
    # on_token UNLIKE on_tab receives the value under the cursor: token
    # rotation is per-row, so the handler must know which wrapper to act on.
    # DOWN moves to "b", TOKEN fires on_token("b").
    calls = []
    select_from_menu(
        ["a", "b"],
        read_key=_fake_keys(["DOWN", "TOKEN", "ENTER"]),
        on_token=calls.append,
        print_fn=lambda _: None,
    )
    assert calls == ["b"]


@pytest.mark.unit
def test_select_from_menu_token_ignored_without_on_token():
    # Without on_token, `t` is silently ignored (like OTHER) — a menu that did
    # not opt in must not start reacting to `t`.
    result = select_from_menu(
        ["a", "b"],
        read_key=_fake_keys(["TOKEN", "DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    assert result == "b"


@pytest.mark.unit
def test_select_from_menu_callable_prompt_is_re_evaluated_each_frame():
    renderings = []

    def prompt():
        renderings.append("frame")
        return "header"

    select_from_menu(
        ["a", "b"],
        prompt=prompt,
        read_key=_fake_keys(["DOWN", "ENTER"]),
        print_fn=lambda _: None,
    )
    # Invoked once per frame (initial render + the DOWN redraw), not bound
    # once before the loop — that freshness is what lets a header track an
    # on_tab mutation.
    assert len(renderings) == 2


@pytest.mark.unit
def test_select_from_menu_nullary_callable_label_still_works():
    # Backward compatibility: a plain zero-arg label (the common case) must
    # keep working unchanged — `_call_label`'s TypeError fallback is what
    # lets a label opt into `selected`/`ansi` without every existing label
    # in the project having to grow those params.
    lines = []
    select_from_menu(
        [("a", lambda: "plain label")],
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    assert any("plain label" in line for line in lines)


@pytest.mark.unit
def test_select_from_menu_label_receives_selected_state(monkeypatch):
    # A label declaring `selected`/`ansi` gets told which row the list
    # cursor is actually on — not "always selected", which is the bug that
    # let two chipset rows render as if both carried the cursor at once.
    seen: list[tuple[bool, bool]] = []

    def label(*, selected: bool, ansi: bool) -> str:
        seen.append((selected, ansi))
        return f"row selected={selected}"

    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu(
        [("a", label), ("b", "b")],
        read_key=_fake_keys(["DOWN", "ENTER"]),
    )
    # Two frames rendered (initial + after DOWN); the row is selected first,
    # then not, once the cursor moves to "b".
    assert seen == [(True, True), (False, True)]


@pytest.mark.unit
def test_select_from_menu_label_gets_ansi_false_when_not_a_tty(monkeypatch):
    seen: list[bool] = []

    def label(*, selected: bool = False, ansi: bool = False) -> str:  # noqa: ARG001 — keyword-called by the menu
        seen.append(ansi)
        return "row"

    fake = _FakeTTY()
    fake.isatty = lambda: False
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu([("a", label)], read_key=_fake_keys(["ENTER"]))
    assert seen == [False]


@pytest.mark.unit
def test_select_from_menu_callable_label_reflects_state_change():
    # A callable label that returns different text per call must render the
    # updated text on the redraw frame — the desync symptom from issue #26
    # (a header that updates while the row stays stale).
    state = {"v": "before"}

    def label():
        return f"value={state['v']}"

    lines = []
    select_from_menu(
        [("a", label)],
        prompt=lambda: f"header={state['v']}",
        on_tab=lambda: state.update(v="after"),
        read_key=_fake_keys(["TAB", "ENTER"]),
        print_fn=lines.append,
    )
    joined = "\n".join(lines)
    # Both the header and the row show "after" on the post-Tab redraw —
    # neither is frozen at the pre-Tab "before" value.
    assert "value=after" in joined
    assert "header=after" in joined


@pytest.mark.unit
def test_select_from_menu_callable_hint_is_rendered():
    lines = []
    select_from_menu(
        ["a", "b"],
        hint=lambda: "custom hint",
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    assert any("custom hint" in line for line in lines)


@pytest.mark.unit
def test_select_from_menu_callable_hint_keeps_frame_lines_stable(monkeypatch):
    # A callable hint resolves once, BEFORE frame_lines, so the redraw count
    # matches a plain-string hint (2 items + heading + spacer + hint = 6).
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], hint=lambda: "↑/↓")
    assert "\x1b[6A" in joined


@pytest.mark.unit
def test_select_from_menu_callable_hint_none_adds_no_frame_lines(monkeypatch):
    # A callable hint returning None/empty must not silently add two lines to
    # the erasable frame (a count drift that creeps the menu down the screen).
    joined = _run_menu(monkeypatch, ["DOWN", "ENTER"], hint=lambda: None)
    assert "\x1b[4A" in joined  # 2 items + heading spacer = 4, no hint lines


# --- Section headers (issue #27) --------------------------------------------


@pytest.mark.unit
def test_select_from_menu_section_header_is_rendered_without_digit():
    # A Section renders as a bare label: no digit, no `>` cursor marker. The
    # selectable items on either side keep their digits and are the only rows
    # that can show the cursor.
    lines = []
    select_from_menu(
        [Section("group"), "a", "b"],
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    group_line = next(line for line in lines if "group" in line)
    assert "1" not in group_line  # no digit for a header
    assert ">" not in group_line  # cursor never on a header
    # The actual items still get numbered.
    a_line = next(line for line in lines if line.lstrip().startswith("1"))
    b_line = next(line for line in lines if line.lstrip().startswith("2"))
    assert "a" in a_line
    assert "b" in b_line


@pytest.mark.unit
def test_select_from_menu_numbering_continues_across_sections():
    # Numbering is continuous across section headers: an item in the second
    # section gets the next digit, not a restart at 1. A header is not an item,
    # so it must not reset the digit counter.
    lines = []
    select_from_menu(
        [Section("g1"), "a", "b", Section("g2"), "c"],
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    a_line = next(line for line in lines if line.lstrip().startswith("1"))
    b_line = next(line for line in lines if line.lstrip().startswith("2"))
    c_line = next(line for line in lines if line.lstrip().startswith("3"))
    assert "a" in a_line
    assert "b" in b_line
    assert "c" in c_line


@pytest.mark.unit
def test_select_from_menu_navigation_skips_section_headers():
    # Two sections, two items. The cursor moves over the SELECTABLE items only
    # — ENTER (no nav) picks the first selectable, UP wraps to the last
    # selectable, DOWN cycles back to the first.
    items = [Section("g1"), "a", Section("g2"), "b"]

    # No navigation: first selectable ("a"), not the "g1" header.
    assert (
        select_from_menu(items, read_key=_fake_keys(["ENTER"]), print_fn=lambda _: None)
        == "a"
    )

    # UP from the first wraps to the LAST selectable ("b"), not to "g2".
    assert (
        select_from_menu(
            items, read_key=_fake_keys(["UP", "ENTER"]), print_fn=lambda _: None
        )
        == "b"
    )

    # DOWN, DOWN: a -> b -> a. The cursor jumps both section headers on the
    # wrap (b -> a passes over "g2" and "g1") without ever landing on them.
    assert (
        select_from_menu(
            items,
            read_key=_fake_keys(["DOWN", "DOWN", "ENTER"]),
            print_fn=lambda _: None,
        )
        == "a"
    )


@pytest.mark.unit
def test_select_from_menu_home_end_land_on_selectable_not_header():
    # HOME/END must land on the first/last SELECTABLE item, never on a Section
    # header — even though the header is the very first / very last row.
    items = [Section("g1"), "a", Section("g2"), "b"]

    assert (
        select_from_menu(
            items, read_key=_fake_keys(["HOME", "ENTER"]), print_fn=lambda _: None
        )
        == "a"
    )
    assert (
        select_from_menu(
            items, read_key=_fake_keys(["END", "ENTER"]), print_fn=lambda _: None
        )
        == "b"
    )

    # PAGE_UP/PAGE_DOWN behave the same as HOME/END in this menu.
    assert (
        select_from_menu(
            items, read_key=_fake_keys(["PAGE_DOWN", "ENTER"]), print_fn=lambda _: None
        )
        == "b"
    )
    assert (
        select_from_menu(
            items,
            read_key=_fake_keys(["END", "PAGE_UP", "ENTER"]),
            print_fn=lambda _: None,
        )
        == "a"
    )


@pytest.mark.unit
def test_select_from_menu_section_header_counted_in_frame_lines(monkeypatch):
    # A Section occupies one rendered row, so frame_lines must count it. With
    # Section + 2 items, frame_lines = 3 pairs + heading + spacer = 5; a
    # missing section row in the count would make the in-place redraw erase
    # one row too few and the menu would creep down the screen.
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    select_from_menu(
        [Section("g"), "a", "b"],
        read_key=_fake_keys(["DOWN", "ENTER"]),
        clear=True,
    )
    joined = "".join(fake.writes)
    assert "\x1b[5A" in joined  # 3 pairs (Section + a + b) + 2 = 5


@pytest.mark.unit
def test_select_from_menu_section_only_items_raises():
    # A menu of nothing but section headers has no selectable entry and must
    # fail the same non-empty contract as a truly empty items list — there is
    # nothing for Enter to return.
    with pytest.raises(ValueError, match="non-empty"):
        select_from_menu(
            [Section("only")], read_key=_fake_keys(["ENTER"]), print_fn=lambda _: None
        )


@pytest.mark.unit
def test_section_rejects_multiline_text():
    # A Section must occupy exactly one rendered row (frame_lines counts it
    # as one), so a header containing a newline would desynchronize the
    # in-place redraw. Reject it at construction rather than corrupting the
    # terminal on the first frame.
    with pytest.raises(ValueError, match="single line"):
        Section("group\nsecret")
    with pytest.raises(ValueError, match="single line"):
        Section("group\rsecret")


@pytest.mark.unit
def test_on_key_returns_action_after_menu_has_closed():
    seen = []
    result = select_from_menu(
        ["wrapper"],
        on_key={"a": lambda value: seen.append(value) or "add"},
        read_key=_fake_keys(["a"]),
        print_fn=lambda _: None,
    )
    assert result == "add"
    assert seen == ["wrapper"]


@pytest.mark.unit
def test_numbered_false_hides_digit_column():
    lines = []
    select_from_menu(
        ["wrapper"],
        numbered=False,
        read_key=_fake_keys(["ENTER"]),
        print_fn=lines.append,
    )
    assert not any("·" in line for line in lines)


# --- ANSI-aware `_fit` / `_visible_len` -------------------------------------


@pytest.mark.unit
def test_visible_len_ignores_sgr_wrapper():
    assert _visible_len("\033[7mglm\033[0m") == len("glm")


@pytest.mark.unit
def test_visible_len_matches_plain_len_without_ansi():
    assert _visible_len("plain text") == len("plain text")


@pytest.mark.unit
def test_fit_plain_line_unchanged_when_it_fits():
    assert _fit("short", 10) == "short"


@pytest.mark.unit
def test_fit_plain_line_truncates_like_before():
    # Regression: no ANSI codes involved, behavior matches the pre-SGR `_fit`.
    assert _fit("abcdefgh", 5) == "abcd…"


@pytest.mark.unit
def test_fit_line_with_ansi_that_already_fits_is_untouched():
    line = "\033[7mglm\033[0m"
    assert _fit(line, 10) == line


@pytest.mark.unit
def test_fit_cuts_inside_an_open_span_appends_reset():
    # Visible payload is "applied-glm", width 6 lands mid-span (open \033[7m
    # never closed before the cut) — the reset must be appended so the
    # highlight does not bleed into the rest of the physical row.
    line = "\033[7mapplied-glm\033[0m"
    result = _fit(line, 6)
    assert result.startswith("\033[7m")
    assert result.endswith("\033[0m")
    assert _visible_len(result) <= 6


@pytest.mark.unit
def test_fit_cuts_exactly_at_a_span_boundary_keeps_code_intact():
    # "hi" (2 visible chars) + full reverse span around "x" (1 visible char)
    # = width 3 lands exactly on the boundary right after `\033[7m`.
    line = "hi\033[7mx\033[0m"
    result = _fit(line, 3)
    assert result.count("\033[7m") == line.count("\033[7m")
    # No dangling half-written escape byte sequence.
    assert result.count("\033[") == result.count("m")
