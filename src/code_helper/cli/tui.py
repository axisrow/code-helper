"""Arrow-key TUI menu — a thin, looping front end over the CLI handlers.

Contract (see issue #3): CLI is primary, TUI is secondary. The TUI never
reimplements service logic — it only collects arguments (via the same
``select_from_menu`` used by ``edit-token``, plus a plain ``input()`` for
``--model``) and mutates the shared ``argparse.Namespace`` before calling the
exact same ``_handle_list`` / ``_handle_add`` / ``_handle_edit_token`` from
:mod:`code_helper.cli.parser`. It knows nothing about ``render_script`` /
``install_wrapper`` internals — only about command names and their CLI
arguments.

The menu loops: ``list`` / ``add`` / ``edit-token`` run, then the menu
reappears so the next command can be picked. Only ``quit`` (and a top-level
Ctrl-C / ``q``) exits. A ``CodeHelperError`` from ``add`` / ``edit-token`` is
printed to stderr and the menu reappears (under ``--debug`` it is re-raised,
mirroring ``__main__.main``); cancelling a sub-menu returns to the main menu.
``dry-run`` / ``debug`` are toggles that redraw the menu instead of running a
command. After a command runs, the screen clears and the menu re-renders at the
top — but only after a keypress (``[press any key to return to the menu]``), so
the command's output stays readable instead of being wiped instantly.
"""

from __future__ import annotations

import argparse

__all__ = ["run_tui"]

_LIST = "list"
_ADD = "add"
_EDIT_TOKEN = "edit-token"
_QUIT = "quit"


def run_tui(args: argparse.Namespace) -> int:
    """Run the looping arrow-key menu, dispatching into the CLI handlers.

    ``args`` is the parsed root namespace (it already carries ``dry_run``/
    ``debug`` if given on the command line before ``tui``). It is mutated in
    place with whatever the chosen command needs, then handed to the same
    ``_handle_*`` the CLI subcommand would receive.

    The menu loops until ``quit`` or a top-level cancel, so this always
    returns 0. A ``CodeHelperError`` from ``add`` / ``edit-token`` is caught
    here, printed to stderr, and the loop continues — unless ``--debug`` is
    set, in which case it is re-raised to ``main`` for a full traceback
    (matching the CLI path's behavior).

    ``select_from_menu`` and ``input`` are looked up at call time (the
    ``from ... import`` and the builtin reference resolve on each call, not at
    module-def time) so tests can ``monkeypatch.setattr`` them — a
    ``select = select_from_menu`` default parameter would bind the name at
    def-time and silently ignore the patch.
    """
    from code_helper.cli.menu import MenuCancelled, press_any_key, select_from_menu
    from code_helper.cli.parser import _handle_add, _handle_edit_token, _handle_list
    from code_helper.errors import CodeHelperError, emit_error
    from code_helper.services.wrappers import WRAPPERS

    def _pick(items: list[str], prompt: str) -> str | None:
        """``select_from_menu`` with cancellation folded to ``None``."""
        try:
            return select_from_menu(items, prompt=prompt, clear=True)
        except MenuCancelled:
            return None

    def _run(handler, debug: bool) -> None:
        """Call a handler, surfacing ``CodeHelperError`` to the menu loop."""
        try:
            handler(args)
        except CodeHelperError as e:
            emit_error(e, debug)

    def _pause() -> None:
        """Let the user read a command's output before the menu clears it.

        On a real TTY, block for one keypress; the menu loop then re-renders
        with a screen clear, so the output is readable yet does not stack into
        the next frame. A no-op off a TTY (tests / pipes) so the loop never
        blocks waiting on a non-interactive stdin.
        """
        press_any_key("\n[press any key to return to the menu]")

    while True:
        dry_run = getattr(args, "dry_run", False)
        debug = getattr(args, "debug", False)
        items = [
            _LIST,
            _ADD,
            _EDIT_TOKEN,
            f"dry-run: {'on' if dry_run else 'off'}",
            f"debug: {'on' if debug else 'off'}",
            _QUIT,
        ]

        choice = _pick(items, "code-helper — pick a command:")
        if choice is None:
            print("cancelled")
            return 0

        if choice == _LIST:
            _run(_handle_list, debug)
            _pause()
            continue

        if choice == _ADD:
            name = _pick([w.name for w in WRAPPERS], "select a wrapper to add:")
            if name is None:
                print("cancelled")
                continue
            args.name = name
            args.model = input("model override (Enter = default): ").strip() or None
            _run(_handle_add, debug)
            _pause()
            continue

        if choice == _EDIT_TOKEN:
            args.name = None
            _run(_handle_edit_token, debug)
            _pause()
            continue

        if choice.startswith("dry-run:"):
            args.dry_run = not dry_run
            continue

        if choice.startswith("debug:"):
            args.debug = not debug
            continue

        if choice == _QUIT:
            return 0
