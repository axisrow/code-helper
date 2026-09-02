"""Tests for ``services/context_window.py`` — the per-model window resolver.

The resolver answers one question — "does this spec carry an explicit
context window, and if not, should we ask?" — in a strict precedence:
catalog (known models, never prompted) → state (recorded answers) →
interactive menu → ``None``. These tests pin:

- known models are NEVER prompted and NEVER recorded (the catalog owns them);
- recorded answers — including the ``0`` "no declaration" sentinel — are
  returned without prompting;
- unknown + unrecorded models are asked exactly once, the answer is
  persisted, and every menu path (including custom input) records;
- non-interactive resolution degrades to ``None`` without writing;
- mixed recorded tiers stay undeclared (one window per session or none).
"""

from __future__ import annotations

import pytest

from codehelper.cli.menu import MenuCancelled
from codehelper.errors import CodeHelperError
from codehelper.services.context_window import resolve_context_window
from codehelper.services.paths import Paths
from codehelper.services.state import context_window, set_context_window

GLM = "glm-5.3"  # in MODEL_CONTEXT_WINDOWS (1M)

_MENU = {"0": "0", "1": "1000000", "2": "2000000", "c": "custom"}


def _paths(tmp_path) -> Paths:
    return Paths.from_home(tmp_path)


def _select(choice: str):
    """A select_fn that fails the test if the menu is shown at all when
    ``choice`` is the sentinel string ``NEVER``; otherwise always picks."""

    def _pick(items):
        assert choice != "NEVER", "menu was shown"
        for value, _label in items:
            if _MENU.get(choice, choice) in (value, choice):
                return value if choice in _MENU else choice
        raise AssertionError(f"choice {choice!r} not offered: {items}")

    return _pick


@pytest.mark.unit
def test_catalog_hit_returns_none_without_prompting_or_writing(tmp_path):
    """A catalog-known model is the catalog's business — no menu, no state
    write; the caller keeps spec.context_window=None and the renderer derives."""
    paths = _paths(tmp_path)
    assert (
        resolve_context_window(
            paths, [GLM], interactive=True, select_fn=_select("NEVER")
        )
        is None
    )
    assert context_window(paths, GLM) is None


@pytest.mark.unit
def test_recorded_positive_value_is_returned_without_prompting(tmp_path):
    paths = _paths(tmp_path)
    set_context_window(paths, "mystery-3b", 1_000_000)
    assert (
        resolve_context_window(
            paths,
            ["mystery-3b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        == 1_000_000
    )


@pytest.mark.unit
def test_recorded_zero_sentinel_is_returned_without_prompting(tmp_path):
    """``0`` is a real recorded answer — an answered model is never re-asked."""
    paths = _paths(tmp_path)
    set_context_window(paths, "mystery-3b", 0)
    assert (
        resolve_context_window(
            paths,
            ["mystery-3b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        == 0
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "choice,expected", [("0", 0), ("1", 1_000_000), ("2", 2_000_000)]
)
def test_menu_choices_are_recorded(tmp_path, choice, expected):
    paths = _paths(tmp_path)
    value = resolve_context_window(
        paths,
        ["mystery-3b"],
        interactive=True,
        select_fn=_select(choice),
    )
    assert value == expected
    assert context_window(paths, "mystery-3b") == expected


@pytest.mark.unit
def test_custom_input_is_validated_then_recorded(tmp_path):
    """Garbage re-prompts; a valid number (or 'none') is accepted."""
    paths = _paths(tmp_path)
    answers = iter(["abc", "-5", "none"])

    def _read(prompt):
        return next(answers)

    value = resolve_context_window(
        paths,
        ["mystery-3b"],
        interactive=True,
        select_fn=_select("c"),
        read_line_fn=_read,
    )
    assert value == 0
    assert context_window(paths, "mystery-3b") == 0


@pytest.mark.unit
def test_custom_input_gives_up_after_three_bad_answers(tmp_path):
    paths = _paths(tmp_path)

    def _read(prompt):
        return "garbage"

    with pytest.raises(CodeHelperError):
        resolve_context_window(
            paths,
            ["mystery-3b"],
            interactive=True,
            select_fn=_select("c"),
            read_line_fn=_read,
        )
    assert context_window(paths, "mystery-3b") is None


@pytest.mark.unit
def test_unknown_unrecorded_non_interactive_returns_none_and_writes_nothing(tmp_path):
    """Scripts/dry-run never see the menu — honest no-declaration, status quo."""
    paths = _paths(tmp_path)
    assert (
        resolve_context_window(
            paths,
            ["mystery-3b"],
            interactive=False,
            select_fn=_select("NEVER"),
        )
        is None
    )
    assert context_window(paths, "mystery-3b") is None


@pytest.mark.unit
def test_mixed_recorded_tiers_yield_none(tmp_path):
    """One variable declares ONE window for the whole session — tiers that
    recorded different answers stay undeclared unless the user decided."""
    paths = _paths(tmp_path)
    set_context_window(paths, "model-a", 1_000_000)
    set_context_window(paths, "model-b", 0)
    assert (
        resolve_context_window(
            paths,
            ["model-a", "model-b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        is None
    )


@pytest.mark.unit
def test_uniform_recorded_tiers_are_returned(tmp_path):
    paths = _paths(tmp_path)
    set_context_window(paths, "model-a", 2_000_000)
    set_context_window(paths, "model-b", 2_000_000)
    assert (
        resolve_context_window(
            paths,
            ["model-a", "model-b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        == 2_000_000
    )


@pytest.mark.unit
def test_menu_cancel_records_nothing(tmp_path):
    """Esc at the menu aborts: nothing recorded, the question re-asks next
    time — continuing silently would apply a value the user just declined."""
    paths = _paths(tmp_path)

    def _cancel(items):
        raise MenuCancelled(hard=False)

    with pytest.raises(MenuCancelled):
        resolve_context_window(
            paths,
            ["mystery-3b"],
            interactive=True,
            select_fn=_cancel,
        )
    assert context_window(paths, "mystery-3b") is None


@pytest.mark.unit
def test_mixed_catalog_and_recorded_tiers_agreeing_are_explicit(tmp_path):
    """A spec mixing a catalog tier with a state-recorded tier: each model
    resolves (catalog first, state second); agreeing answers become an
    explicit value — the renderer's catalog-only derivation can't see the
    recorded tier, so the resolved value must ride the spec (and marker)."""
    paths = _paths(tmp_path)
    set_context_window(paths, "mystery-3b", 1_000_000)
    assert (
        resolve_context_window(
            paths,
            [GLM, "mystery-3b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        == 1_000_000
    )


@pytest.mark.unit
def test_catalog_tier_disagreeing_with_recorded_tier_yields_none(tmp_path):
    """All models resolve but disagree (catalog 1M vs recorded 'no
    declaration'): honest no-declaration, and no menu — an answered model
    is never re-asked."""
    paths = _paths(tmp_path)
    set_context_window(paths, "mystery-3b", 0)
    assert (
        resolve_context_window(
            paths,
            [GLM, "mystery-3b"],
            interactive=True,
            select_fn=_select("NEVER"),
        )
        is None
    )


@pytest.mark.unit
def test_split_tier_answer_records_under_the_unknown_model(tmp_path):
    """The answer records under the model the menu actually asked about —
    the loop's unknown model, never a caller-supplied fallback name
    (review round 1, PR #84). A catalog tier can't receive the answer:
    it is never recorded, so the question would re-fire forever."""
    paths = _paths(tmp_path)

    def _select(_items, **_kwargs):
        return "2000000"

    resolve_context_window(
        paths,
        ["mystery-haiku", GLM],  # GLM is catalog-known → never recorded
        interactive=True,
        select_fn=_select,
    )
    assert context_window(paths, "mystery-haiku") == 2_000_000
    assert context_window(paths, GLM) is None
