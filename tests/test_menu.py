"""Tests for ``cli/menu.py`` — the stdlib arrow-key selection menu.

``read_key`` is injected as a fake iterator so these tests never touch a real
TTY (mirroring how ``services/secrets.py`` tests inject ``getpass_fn``).
"""

from __future__ import annotations

import pytest

from code_helper.cli.menu import MenuCancelled, select_from_menu


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
