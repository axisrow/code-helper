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
   answer is persisted for ``target_model``. ``0`` is a real answer
   ("no declaration"), never normalized away. Non-interactive callers get
   ``None`` — the status quo, not a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from codehelper.errors import CodeHelperError
from codehelper.services.paths import Paths
from codehelper.services.render import MODEL_CONTEXT_WINDOWS, uniform_context_window
from codehelper.services.state import context_window, set_context_window

__all__ = ["resolve_context_window"]

#: Custom-input ceiling: a typo beyond this is a mistake, not a window.
_MAX_WINDOW = 10_000_000

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
    target_model: str,
    interactive: bool = False,
    select_fn: SelectFn | None = None,
    read_line_fn: ReadLineFn | None = None,
) -> int | None:
    """The explicit context window for this spec, ``0`` for "none", or None.

    Args:
        paths: Resolved paths — locates ``state.json``.
        models: EVERY model the declaration would cover — all tiers plus the
            subagent when set. One variable declares one window per session.
        target_model: The model a new answer is recorded under (``spec.model``).
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

    # 2. Every model resolves (catalog first, state second) to one value.
    values: set[int] = set()
    for model in model_list:
        value = MODEL_CONTEXT_WINDOWS.get(model)
        if value is None:
            value = context_window(paths, model)
            if value is None:
                # Unknown AND unrecorded: ask once (or degrade honestly).
                return _ask_or_none(
                    paths,
                    target_model,
                    interactive=interactive,
                    select_fn=select_fn,
                    read_line_fn=read_line_fn,
                )
        values.add(value)
    if len(values) == 1:
        (value,) = values
        # A tie that involves a recorded answer can't be re-derived by the
        # renderer (it reads the catalog only) — it must ride the spec.
        return (
            None
            if all(MODEL_CONTEXT_WINDOWS.get(m) == value for m in model_list)
            else value
        )
    return None


def _ask_or_none(
    paths: Paths,
    target_model: str,
    *,
    interactive: bool,
    select_fn: SelectFn | None,
    read_line_fn: ReadLineFn | None,
) -> int | None:
    """Ask once and record, or — non-interactively — the status-quo ``None``."""
    if not interactive:
        return None
    menu_fn = select_fn
    if menu_fn is None:
        from codehelper.cli.menu import select_from_menu

        def menu_fn(items: Sequence[tuple[str, str]]) -> str:
            return select_from_menu(items, prompt=f"Context window for {target_model}:")

    answer = menu_fn(_MENU_ITEMS)
    if answer == "custom":
        value = _ask_custom(read_line_fn)
    else:
        value = int(answer)  # "0" → 0, "1000000" → 1_000_000
    set_context_window(paths, target_model, value)
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
        if 0 < value <= _MAX_WINDOW:
            return value
        print(f"out of range (1..{_MAX_WINDOW}): {value}")
    raise CodeHelperError(
        f"no usable context window after {_RETRIES} attempts — nothing recorded"
    )
