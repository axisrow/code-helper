"""Arrow-key TUI menu — a thin, single-pass front end over the CLI handlers.

Contract (see issue #3): CLI is primary, TUI is secondary. The TUI never
reimplements service logic — it only collects arguments (via the same
``select_from_menu`` used by ``edit-token``, plus a plain ``input()`` for
``--model``) and mutates the shared ``argparse.Namespace`` before calling the
exact same ``_handle_list`` / ``_handle_add`` / ``_handle_edit_token`` from
:mod:`code_helper.cli.parser`. It knows nothing about ``render_script`` /
``install_wrapper`` internals — only about command names and their CLI
arguments.

One pass: pick an item, run it, return its exit code. ``dry-run``/``debug``
are toggles that redraw the menu instead of exiting; ``quit`` and menu
cancellation (Ctrl-C / ``q``) both return 0 without touching the filesystem.
"""

from __future__ import annotations

import argparse

__all__ = ["run_tui"]

_LIST = "list"
_ADD = "add"
_EDIT_TOKEN = "edit-token"
_QUIT = "quit"


def run_tui(args: argparse.Namespace) -> int:
    """Run the single-pass arrow-key menu and dispatch into a CLI handler.

    ``args`` is the parsed root namespace (it already carries ``dry_run``/
    ``debug`` if given on the command line before ``tui``). It is mutated in
    place with whatever the chosen command needs, then handed to the same
    ``_handle_*`` the CLI subcommand would receive.

    ``select_from_menu`` and ``input`` are looked up at call time (the
    ``from ... import`` and the builtin reference resolve on each call, not at
    module-def time) so tests can ``monkeypatch.setattr`` them — a
    ``select = select_from_menu`` default parameter would bind the name at
    def-time and silently ignore the patch.

    Returns the exit code of the dispatched command, or 0 for ``quit``/cancel.
    """
    from code_helper.cli.menu import MenuCancelled, select_from_menu
    from code_helper.cli.parser import _handle_add, _handle_edit_token, _handle_list
    from code_helper.services.wrappers import WRAPPERS

    def _pick(items: list[str], prompt: str) -> str | None:
        """``select_from_menu`` with cancellation folded to ``None``."""
        try:
            return select_from_menu(items, prompt=prompt)
        except MenuCancelled:
            return None

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
            return _handle_list(args)

        if choice == _ADD:
            name = _pick([w.name for w in WRAPPERS], "select a wrapper to add:")
            if name is None:
                print("cancelled")
                return 0
            args.name = name
            args.model = input("model override (Enter = default): ").strip() or None
            return _handle_add(args)

        if choice == _EDIT_TOKEN:
            args.name = None
            return _handle_edit_token(args)

        if choice.startswith("dry-run:"):
            args.dry_run = not dry_run
            continue

        if choice.startswith("debug:"):
            args.debug = not debug
            continue

        if choice == _QUIT:
            return 0
