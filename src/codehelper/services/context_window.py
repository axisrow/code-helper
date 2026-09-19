"""Per-model context-window resolution (issue #83).

``MODEL_CONTEXT_WINDOWS`` (``render``) is a hand-maintained catalog: a model
missing from it gets NO ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` declaration, and
Claude Code silently assumes its 200k fallback — for a 1M model that
abandons 80% of the context every session. This module closes the silent
part: an UNKNOWN model is asked once (add/switch), the answer is remembered
per model in ``state.json``, and recorded answers ride the spec — and the
wrapper marker — as explicit data.

Resolution precedence (first hit wins):

1. **catalog** — every model in the spec resolves in ``MODEL_CONTEXT_WINDOWS``
   to one same value: the catalog's business, no prompt, no state write, no
   marker field (the renderer re-derives it; markers for known models stay
   byte-identical — the #82 rule).
2. **state** — every model resolves (catalog first, state second) to one
   same value: that value, returned as EXPLICIT. The renderer's derivation
   is catalog-only, so an answer recorded for an unknown tier can only ride
   the spec (and the marker). A pure-catalog tie already returned in 1.
   Disagreeing answers are an honest ``None`` — one variable declares one
   window for the whole session, and an ANSWERED model is never re-asked.
3. **ask** — some model is unknown AND unrecorded: the menu decides, the
   answer is persisted under that model, and the FULL set is checked for
   agreement before anything may declare (review round 2, PR #84). ``0``
   is a real answer ("no declaration"), never normalized away.
   Non-interactive callers get ``None`` — the status quo, not a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from codehelper.errors import CodeHelperError
from codehelper.services.limits import MAX_CONTEXT_WINDOW
from codehelper.services.paths import Paths
from codehelper.services.render import MODEL_CONTEXT_WINDOWS, uniform_context_window
from codehelper.services.state import context_window, set_context_window

__all__ = ["resolve_context_window"]

#: Consecutive bad custom inputs tolerated before giving up (mirrors
#: ``secrets.resolve_token``'s ``retries=3``).
_RETRIES = 3

#: ``(value, label)`` menu items — value is the string handed back by the
#: menu, ``"custom"`` meaning "ask for a number" and ``"0"`` meaning
#: "no declaration" (Claude Code's native fallback, today's behaviour).
_MENU_ITEMS: Sequence[tuple[str, str]] = (
    ("0", "no declaration — Claude Code's native 200k fallback"),
    ("1000000", "1,000,000 tokens (1M)"),
    ("2000000", "2,000,000 tokens (2M)"),
    ("custom", "custom…"),
)

SelectFn = Callable[[Sequence[tuple[str, str]]], str]
ReadLineFn = Callable[[str], str]


def resolve_context_window(
    paths: Paths,
    models: Iterable[str],
    *,
    interactive: bool = False,
    select_fn: SelectFn | None = None,
    read_line_fn: ReadLineFn | None = None,
) -> int | None:
    """The explicit context window for this spec, ``0`` for "none", or None.

    Args:
        paths: Resolved paths — locates ``state.json``.
        models: EVERY model the declaration would cover — all tiers plus the
            subagent when set. One variable declares one window per session.
        interactive: May a menu be shown? ``--dry-run`` and scripted paths
            pass ``False`` and simply get ``None`` for unknown models.
        select_fn: Menu source (injectable for tests; default: lazy
            ``cli.menu.select_from_menu`` — this module stays free of an
            import-time ``services → cli`` edge).
        read_line_fn: Free-text source for the ``custom`` branch (default:
            lazy ``cli.menu.read_line``).

    Returns:
        ``None`` when the answer is derivable or unavailable (the caller
        keeps ``spec.context_window=None``); ``0`` for an explicit "no
        declaration"; ``> 0`` for an explicit window. ``MenuCancelled``
        propagates — cancelling records nothing and the model is asked
        again next time.
    """
    model_list = list(models)

    # 1. Pure-catalog tie: the renderer re-derives this for free. Return
    # None, NOT the value — an explicit answer would write a marker field
    # for a known model and break the byte-identity rule (issue #82).
    if uniform_context_window(model_list) is not None:
        return None

    # 2. Every model resolves (catalog first, state second, ask third) and
    # the results must AGREE — a fresh answer for one unknown model is
    # recorded for THAT model but still checked against the rest before it
    # may declare a session-wide window (review round 2, PR #84: answering
    # one tier must not over-declare a mix of windows).
    values: set[int] = set()
    for model in model_list:
        value = MODEL_CONTEXT_WINDOWS.get(model)
        if value is None:
            value = context_window(paths, model)
            if value is None:
                # Unknown AND unrecorded: ask once (or degrade honestly).
                # The answer records under THIS model — the one the menu
                # actually names — never under a spec-level fallback, or a
                # split-tier spec would re-ask and mis-key the answer
                # (review round 1, PR #84).
                value = _ask_or_none(
                    paths,
                    model,
                    interactive=interactive,
                    select_fn=select_fn,
                    read_line_fn=read_line_fn,
                )
                if value is None:  # non-interactive: honest no-declaration
                    return None
        values.add(value)
    if len(values) == 1:
        # Reaching here with a single value implies at least one state
        # answer participated (a pure-catalog tie returned at step 1), so
        # it is EXPLICIT: the renderer's catalog-only derivation cannot see
        # it — it must ride the spec.
        (value,) = values
        return value
    return None


def _ask_or_none(
    paths: Paths,
    model: str,
    *,
    interactive: bool,
    select_fn: SelectFn | None,
    read_line_fn: ReadLineFn | None,
) -> int | None:
    """Ask once and record, or — non-interactively — the status-quo ``None``."""
    if not interactive:
        return None
    menu_fn: SelectFn
    if select_fn is None:
        from codehelper.cli.menu import select_from_menu

        def _default_menu(items: Sequence[tuple[str, str]]) -> str:
            return select_from_menu(items, prompt=f"Context window for {model}:")

        menu_fn = _default_menu
    else:
        menu_fn = select_fn

    answer = menu_fn(_MENU_ITEMS)
    if answer == "custom":
        value = _ask_custom(read_line_fn)
    else:
        value = int(answer)  # "0" → 0, "1000000" → 1_000_000
    set_context_window(paths, model, value)
    return value


def _ask_custom(read_line_fn: ReadLineFn | None) -> int:
    """Free-text window: a positive int, or ``none``/``0`` for no declaration.

    Garbage re-prompts; :data:`_RETRIES` consecutive bad answers raise — a
    user fighting the prompt wants out, not an infinite loop.
    """
    if read_line_fn is None:
        from codehelper.cli.menu import read_line

        read_line_fn = read_line
    for _attempt in range(_RETRIES):
        raw = read_line_fn(
            "Context window in tokens (number, or 'none' for no declaration): "
        ).strip()
        if raw.lower() in ("none", "0"):
            return 0
        try:
            value = int(raw)
        except ValueError:
            print(f"not a number: {raw!r}")
            continue
        if 0 < value <= MAX_CONTEXT_WINDOW:
            return value
        print(f"out of range (1..{MAX_CONTEXT_WINDOW}): {value}")
    raise CodeHelperError(
        f"no usable context window after {_RETRIES} attempts — nothing recorded"
    )
