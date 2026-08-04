"""Argparse parser builder for the ``code-helper`` CLI.

The dispatch contract: each subcommand registers a handler via
``set_defaults(func=...)``, and :func:`code_helper.__main__.main` calls
``args.func(args)``. Handlers are THIN SHELLS — resolve ``Paths.default()``,
delegate to a service, return its int. They do NOT catch/print/exit: a
:class:`CodeHelperError` propagates to :func:`main`, which formats it as
one-line stderr + exit 1 (full traceback under ``--debug``).

The subcommands: ``list`` (registry + install state), ``add <name>``
(install/update a wrapper, optionally ``--model``), ``edit-token [<name>]``
(rotate a secret-auth wrapper's token; an arrow-key menu picks the wrapper
when ``<name>`` is omitted, via :mod:`code_helper.cli.menu`).

Root flags (``--debug`` / ``--dry-run``) attach via a single shared parent
parser so they parse BOTH before and after the subcommand.
"""

import argparse


def _handle_list(args: argparse.Namespace) -> int:
    """Show the wrapper registry + whether each is installed."""
    from code_helper.services.paths import Paths
    from code_helper.services.wrappers import list_wrappers

    list_wrappers(Paths.default())
    return 0


def _handle_add(args: argparse.Namespace) -> int:
    """Install (or update) the named wrapper."""
    from code_helper.services.paths import Paths
    from code_helper.services.secrets import resolve_token
    from code_helper.services.wrappers import get_spec, install_wrapper

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)
    spec = get_spec(args.name)

    if spec.auth == "secret":
        token = resolve_token(
            env_var=spec.token_env_var,
            prompt=f"{spec.name} token ({spec.token_env_var}): ",
        )
    else:
        token = spec.auth_value

    wrote = install_wrapper(
        paths,
        args.name,
        token=token,
        model_override=args.model,
        dry_run=dry_run,
    )
    if not wrote:
        print("no changes")
    return 0


def _handle_edit_token(args: argparse.Namespace) -> int:
    """Interactively rotate the token of a secret-auth wrapper.

    Unlike ``add``, this always prompts via ``getpass`` directly — it never
    calls :func:`code_helper.services.secrets.resolve_token`, which would
    silently return an existing ``token_env_var`` value instead of the new
    one the user is trying to type in.
    """
    import getpass

    from code_helper.cli.menu import MenuCancelled, select_from_menu
    from code_helper.errors import CodeHelperError
    from code_helper.services.paths import Paths
    from code_helper.services.wrappers import (
        WRAPPERS,
        get_spec,
        install_wrapper,
        is_installed,
    )

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)

    if args.name:
        spec = get_spec(args.name)
    else:
        secret_specs = [w for w in WRAPPERS if w.auth == "secret"]
        if not secret_specs:
            raise CodeHelperError("no wrapper has an editable (secret) token")
        try:
            chosen = select_from_menu(
                [w.name for w in secret_specs],
                prompt="select a wrapper to edit its token:",
            )
        except MenuCancelled:
            print("cancelled")
            return 0
        spec = get_spec(chosen)

    if spec.auth != "secret":
        raise CodeHelperError(f"{spec.name} has no editable token (auth={spec.auth})")

    state = "installed" if is_installed(paths, spec.name) else "not installed"
    print(f"{spec.name}: currently {state}")

    token = getpass.getpass(f"new {spec.name} token ({spec.token_env_var}): ")
    if not token:
        raise CodeHelperError("no token entered — aborting")

    wrote = install_wrapper(paths, spec.name, token=token, dry_run=dry_run)
    if not wrote:
        print("no changes")
    return 0


def _handle_bare(args: argparse.Namespace) -> int:
    """Bare ``code-helper`` (no subcommand) → print help."""
    args._parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the root ``code-helper`` argparse parser.

    The root parser owns the global flags (``--debug`` / ``--dry-run``) and
    the subcommand dispatch table.
    """
    # Global flags: work BOTH before AND after the subcommand. ONE shared
    # parent parser (SUPPRESS defaults) is attached to the root parser AND
    # every subparser via parents=[sub_flags], so `--debug`/`--dry-run` parse
    # in either position. SUPPRESS means a subparser copy does not override a
    # value the root already parsed (argparse subparser quirk).
    sub_flags = argparse.ArgumentParser(add_help=False)
    sub_flags.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show full traceback on error",
    )
    sub_flags.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="preview without writing",
    )

    parser = argparse.ArgumentParser(
        prog="code-helper",
        description="Manage generated Claude Code wrapper scripts in ~/.local/bin.",
        parents=[sub_flags],
    )
    parser.set_defaults(func=_handle_bare, _parser=parser)

    subparsers = parser.add_subparsers(
        dest="cmd",
        required=False,
        metavar="<command>",
    )

    p_list = subparsers.add_parser(
        "list",
        help="show the wrapper registry + install state",
        parents=[sub_flags],
    )
    p_list.set_defaults(func=_handle_list)

    p_add = subparsers.add_parser(
        "add",
        help="install (or update) a wrapper script",
        parents=[sub_flags],
    )
    p_add.add_argument("name", help="wrapper name (see `code-helper list`)")
    p_add.add_argument(
        "--model",
        default=None,
        help="override the wrapper's default model(s)",
    )
    p_add.set_defaults(func=_handle_add)

    p_edit_token = subparsers.add_parser(
        "edit-token",
        help="interactively rotate a wrapper's secret token",
        parents=[sub_flags],
    )
    p_edit_token.add_argument(
        "name",
        nargs="?",
        default=None,
        help="wrapper name (omit to pick interactively with an arrow-key menu)",
    )
    p_edit_token.set_defaults(func=_handle_edit_token)

    return parser
