"""Arrow-key TUI menu — a thin, looping front end over the CLI handlers.

Contract (see issue #3): CLI is primary, TUI is secondary. The TUI never
reimplements service logic — it only collects arguments (via the same
``select_from_menu`` used by ``edit-token``, plus a plain ``input()`` for a
custom ``--model``) and mutates the shared ``argparse.Namespace`` before
calling the exact same ``_handle_list`` / ``_handle_add`` / ``_handle_edit_token``
from :mod:`code_helper.cli.parser`. It knows nothing about ``render_script`` /
``install_wrapper`` internals — only about command names and their CLI
arguments. The one read-only exception is
:func:`code_helper.services.wrappers.describe_all` (built on ``is_installed``),
used purely to label menu entries (``_handle_edit_token`` in ``parser.py``
builds the same labels the same way) — no state-changing call is ever made
outside a ``_handle_*``.

The menu loops: ``list`` / ``add`` / ``edit-token`` run, then the menu
reappears so the next command can be picked. Navigation is uniform across
every screen:

- **Esc** or ``q`` — go back one level (a sub-menu returns to the main menu;
  the main menu exits).
- **Ctrl-C** — exit the TUI immediately from anywhere, not just "back".
- Every sub-menu also has a visible ``← назад`` entry, since the keyboard
  shortcuts above are not discoverable from the screen alone.

This relies on :class:`code_helper.cli.menu.MenuCancelled`'s ``hard`` flag:
``hard=False`` (Esc/``q``) means "go back one level", ``hard=True`` (Ctrl-C)
means "quit outright" regardless of menu depth. ``run_tui``'s local ``_pick``
folds a soft cancel into the same ``_BACK`` sentinel value returned by
picking the visible ``← назад`` item, so every call site checks the result
ONE way; a hard cancel is left to propagate as ``MenuCancelled`` and is caught
once, at the very top of :func:`run_tui`, which is what makes Ctrl-C unwind
cleanly from any depth instead of being handled ad hoc at each call site.

A ``CodeHelperError`` from ``add`` / ``edit-token`` is printed to stderr and
the menu reappears (under ``--debug`` it is re-raised, mirroring
``__main__.main``). A ``KeyboardInterrupt`` raised while collecting plain
``input()`` (not through ``select_from_menu``) is also caught and treated as
"back to the menu" — Ctrl-C must never crash the TUI, no matter which prompt
it lands in.

There is no ``dry-run`` menu item — ``--dry-run`` is a scripting/debugging
flag, not something a human toggles interactively; ``code-helper --dry-run``
(or ``tui``) still works, the flag is simply read from ``args``, never
mutated here. ``debug`` lives in a ``settings`` sub-menu instead, since it is
one lone toggle that does not deserve main-menu real estate.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["run_tui"]

_LIST = "list"
_ADD = "add"
_EDIT_TOKEN = "edit-token"
_SETTINGS = "settings"
_QUIT = "quit"
_BACK = "__back__"


def _hint(numbered_count: int, *, exit_word: str) -> str:
    """Build a key hint whose digit range matches what's actually numbered.

    A hint that always said "1-9" regardless of how many items were on
    screen (or whether any of them even had a digit) was misleading — see
    CLAUDE.md. ``numbered_count`` must match the number of non-``_BACK``/
    ``_QUIT`` items passed to the same ``select_from_menu`` call, or the
    printed range and the working digits drift apart again.
    """
    digits = "1" if numbered_count == 1 else f"1-{numbered_count}"
    return f"↑/↓ · {digits} · Enter выбрать · Esc/q {exit_word}"


def run_tui(args: argparse.Namespace) -> int:
    """Run the looping arrow-key menu, dispatching into the CLI handlers.

    ``args`` is the parsed root namespace (it already carries ``dry_run``/
    ``debug`` if given on the command line before ``tui``). It is mutated in
    place with whatever the chosen command needs, then handed to the same
    ``_handle_*`` the CLI subcommand would receive.

    Always returns 0: every exit path (``quit``, Esc/``q`` on the main menu,
    Ctrl-C from anywhere) is handled inside this function rather than
    propagated as an exception or a non-zero code.

    ``select_from_menu`` and ``input`` are looked up at call time (the
    ``from ... import`` and the builtin reference resolve on each call, not at
    module-def time) so tests can ``monkeypatch.setattr`` them — a
    ``select = select_from_menu`` default parameter would bind the name at
    def-time and silently ignore the patch.
    """
    from code_helper.cli.menu import MenuCancelled, press_any_key, select_from_menu
    from code_helper.cli.parser import _handle_add, _handle_edit_token, _handle_list
    from code_helper.errors import CodeHelperError, emit_error
    from code_helper.services.paths import Paths
    from code_helper.services.wrappers import WRAPPERS, describe_all, get_spec

    def _pick(
        items: Sequence[tuple[str, str]],
        prompt: str,
        *,
        exit_word: str = "назад",
    ) -> str:
        """``select_from_menu`` with a soft cancel folded into the ``_BACK`` value.

        A soft cancel (Esc/``q``) and picking the visible ``← назад`` item
        already mean the same thing, so this returns ``_BACK`` for either —
        every call site checks the return value ONE way instead of pairing a
        ``try/except`` with a separate ``if choice == _BACK`` underneath it.

        A hard cancel (Ctrl-C, ``MenuCancelled.hard``) is left to propagate as
        ``MenuCancelled`` — caught once, at the very top of :func:`run_tui`,
        so Ctrl-C unwinds cleanly from any menu depth to "exit the TUI".

        ``_BACK``/``_QUIT`` are never given a digit shortcut (``unnumbered``)
        — they're the one item every screen has that isn't a distinct
        "thing to configure", and always sit last. The hint's digit range is
        derived from how many *other* items there are, so it can't drift
        from what actually works — see :func:`_hint`. ``exit_word`` is what
        the hint calls leaving this screen ("назад" everywhere except the
        main menu, where Esc/``q`` means "выход" instead).
        """
        numbered_count = sum(1 for value, _ in items if value not in (_BACK, _QUIT))
        hint = _hint(numbered_count, exit_word=exit_word)
        try:
            return select_from_menu(
                items,
                prompt=prompt,
                hint=hint,
                unnumbered=frozenset({_BACK, _QUIT}),
                clear=True,
            )
        except MenuCancelled as e:
            if e.hard:
                raise
            return _BACK

    def _run(handler, debug: bool) -> None:
        """Call a handler, surfacing ``CodeHelperError`` to the menu loop.

        A hard ``MenuCancelled`` (Ctrl-C) can also propagate out of
        ``handler`` — e.g. ``_handle_edit_token``'s own wrapper picker
        re-raises on ``hard=True`` rather than swallowing it (see
        ``parser.py``). Left uncaught here on purpose: it unwinds straight to
        the top-level ``try/except MenuCancelled`` in :func:`run_tui`, so
        Ctrl-C mid-command still means "leave the TUI now" — no pause, no
        "press any key" prompt for an exit the user already asked for.
        """
        try:
            handler(args)
        except CodeHelperError as e:
            emit_error(e, debug)

    def _pause() -> None:
        """Let the user read a command's output before the menu clears it."""
        press_any_key("\n[нажмите любую клавишу, чтобы вернуться в меню]")

    def _wrapper_items() -> list[tuple[str, str]]:
        """``(name, label)`` pairs for every registered wrapper, with install state.

        Built through :func:`describe_all` — the same wrapper-listing helper
        ``edit-token``'s picker uses (see ``parser.py``) — so the two menus
        can't silently drift apart on either the row layout or the
        ``Paths``/``is_installed`` wiring behind it.
        """
        return describe_all(
            Paths.default(),
            WRAPPERS,
            installed_word="установлена",
            not_installed_word="не установлена",
        )

    def _run_add() -> None:
        """``add`` flow: pick a wrapper, then a model, then dispatch — with a
        visible "← назад" at every step, not just a keyboard shortcut.
        """
        wrapper_items = [*_wrapper_items(), (_BACK, "← назад")]
        name = _pick(wrapper_items, "выберите обёртку для установки:")
        if name == _BACK:
            return

        # `spec.model` is the resolved model whatever the config shape — the
        # old launch_command/sonnet_model branch is gone with the flat spec.
        default_model = get_spec(name).model
        model_items = [
            ("__default__", f"оставить по умолчанию ({default_model})"),
            ("__custom__", "указать свою модель"),
            (_BACK, "← назад"),
        ]
        choice = _pick(model_items, f"модель для {name}:")
        if choice == _BACK:
            return

        if choice == "__custom__":
            try:
                model = input("введите модель: ").strip()
            except KeyboardInterrupt:
                print()
                return
            args.model = model or None
        else:
            args.model = None

        args.name = name
        _run(_handle_add, getattr(args, "debug", False))
        _pause()

    def _run_settings() -> None:
        while True:
            debug = getattr(args, "debug", False)
            items = [
                ("debug", f"debug: {'вкл' if debug else 'выкл'}"),
                (_BACK, "← назад"),
            ]
            choice = _pick(items, "настройки")
            if choice == _BACK:
                return
            if choice == "debug":
                args.debug = not debug
                continue

    def _main_loop() -> int:
        while True:
            items = [
                (_LIST, "list         показать обёртки и их состояние"),
                (_ADD, "add          установить или обновить обёртку"),
                (_EDIT_TOKEN, "edit-token   заменить токен обёртки"),
                (_SETTINGS, "settings     отладочные настройки"),
                (_QUIT, "quit         выход"),
            ]

            # The main menu is the one call site that isn't "go back one
            # level" on cancel — Esc/q here means "leave the TUI" — so it
            # checks _BACK itself rather than sharing _run_add/_run_settings'
            # "return to caller" meaning.
            choice = _pick(
                items,
                "code-helper — управление обёртками Claude Code",
                exit_word="выход",
            )
            if choice in (_QUIT, _BACK):
                return 0

            if choice == _LIST:
                _run(_handle_list, getattr(args, "debug", False))
                _pause()
                continue

            if choice == _ADD:
                _run_add()
                continue

            if choice == _EDIT_TOKEN:
                args.name = None
                _run(_handle_edit_token, getattr(args, "debug", False))
                _pause()
                continue

            if choice == _SETTINGS:
                _run_settings()
                continue

    try:
        return _main_loop()
    except MenuCancelled:
        # Ctrl-C from any depth: exit the TUI cleanly, exit code 0 — matches
        # the historical top-level-cancel behavior.
        return 0
