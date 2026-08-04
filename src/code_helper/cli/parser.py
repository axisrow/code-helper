"""Argparse parser builder for the ``code-helper`` CLI.

The dispatch contract: each subcommand registers a handler via
``set_defaults(func=...)``, and :func:`code_helper.__main__.main` calls
``args.func(args)``. Handlers are THIN SHELLS — resolve ``Paths.default()``,
delegate to a service, return its int. They do NOT catch/print/exit: a
:class:`CodeHelperError` propagates to :func:`main`, which formats it as
one-line stderr + exit 1 (full traceback under ``--debug``).

The subcommands: ``list`` (registry + install state), ``add <name>``
(install/update a wrapper, optionally ``--model``), ``remove <name>``
(uninstall a wrapper if code-helper-managed).

Root flags (``--debug`` / ``--dry-run``) attach via a single shared parent
parser so they parse BOTH before and after the subcommand.
"""

import argparse


def _handle_list(args: argparse.Namespace) -> int:
    """Show the wrapper registry + whether each is installed/foreign."""
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


def _handle_bare(args: argparse.Namespace) -> int:
    """Bare ``code-helper`` (no subcommand) → print help."""
    args._parser.print_help()
    return 0


def _handle_remove(args: argparse.Namespace) -> int:
    """Uninstall the named wrapper, only if code-helper-managed."""
    from code_helper.services.paths import Paths
    from code_helper.services.wrappers import uninstall_wrapper

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)
    removed = uninstall_wrapper(paths, args.name, dry_run=dry_run)
    if not removed:
        print("no changes")
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

    p_remove = subparsers.add_parser(
        "remove",
        help="uninstall a wrapper script (only if code-helper-managed)",
        parents=[sub_flags],
    )
    p_remove.add_argument("name", help="wrapper name to remove")
    p_remove.set_defaults(func=_handle_remove)

    return parser
