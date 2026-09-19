"""Tests for ``services/limits.py`` — the context-window ceiling's one home."""

from __future__ import annotations

import pytest

from codehelper.services.limits import MAX_CONTEXT_WINDOW, context_window_usable


@pytest.mark.unit
def test_zero_is_usable_the_explicit_no_declaration_sentinel():
    """``0`` is a real answer ("no declaration", issue #83), never normalized
    away — every reader and writer must accept it like any other in-range
    value."""
    assert context_window_usable(0) is True


@pytest.mark.unit
def test_the_range_is_0_through_the_ceiling_inclusive():
    assert context_window_usable(1) is True
    assert context_window_usable(MAX_CONTEXT_WINDOW) is True


@pytest.mark.unit
def test_negatives_and_gigavalues_are_unusable():
    """A hand-edited ``-5`` or a typo'd gigavalue must not ride an explicit
    answer into a wrapper marker or the live settings (PR #84, review
    round 2) — the same range the reader accepts, a writer may record."""
    assert context_window_usable(-1) is False
    assert context_window_usable(MAX_CONTEXT_WINDOW + 1) is False


@pytest.mark.unit
def test_underscore_formatted_spelling_is_byte_identical_with_history():
    """Lockstep pin: every error message interpolates the ceiling as
    ``{MAX_CONTEXT_WINDOW:_}`` so the rendered text stays byte-identical to
    the pre-#106 literals ("10_000_000"). Changing the ceiling changes every
    message WITH it — this test is the conscious-update tripwire."""
    assert f"{MAX_CONTEXT_WINDOW:_}" == "10_000_000"
