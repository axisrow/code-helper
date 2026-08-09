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
when ``<name>`` is omitted, via :mod:`code_helper.cli.menu``), and ``tui``
(single-pass arrow-key menu over the three commands above, via
:mod:`code_helper.cli.tui`). A bare ``code-helper`` (no subcommand) also
opens the TUI — it is the discoverable default for a new user, while every
subcommand remains fully scriptable on its own.

Root flags (``--debug`` / ``--dry-run``) attach via a single shared parent
parser so they parse BOTH before and after the subcommand.
"""

import argparse
import sys


def _handle_list_axes(what: str) -> int:
    """Print the agent/provider registries, or the compatibility matrix.

    ``matrix`` is executable documentation: it is rendered by calling
    ``resolve_shape`` itself, so what it shows and what ``add`` accepts cannot
    disagree. It is where a user sees that some pairings are simply blank.
    """
    from code_helper.errors import CodeHelperError
    from code_helper.services.model import AGENTS, PROVIDERS, resolve_shape

    if what == "agents":
        for agent in AGENTS:
            shapes = ", ".join(sorted(s.value for s in agent.shapes))
            print(f"{agent.name:10} {agent.description:24} [{shapes}]")
        return 0

    if what == "providers":
        for provider in PROVIDERS:
            shapes = ", ".join(sorted(s.value for s in provider.shapes))
            print(f"{provider.name:10} {provider.description:24} [{shapes}]")
        return 0

    # Resolve every cell first: the column has to be as wide as the widest
    # SHAPE it will hold, not the widest provider name, or the values collide.
    rows: list[tuple[str, list[str]]] = []
    for agent in AGENTS:
        cells = []
        for provider in PROVIDERS:
            try:
                cells.append(resolve_shape(agent, provider).value)
            except CodeHelperError:
                cells.append("—")  # genuinely impossible, not merely unbuilt
        rows.append((agent.name, cells))

    label_width = max([len(a.name) for a in AGENTS] + [0]) + 2
    widths = [
        max([len(p.name)] + [len(cells[i]) for _, cells in rows]) + 2
        for i, p in enumerate(PROVIDERS)
    ]

    header = "".join(p.name.ljust(w) for p, w in zip(PROVIDERS, widths, strict=True))
    print(" " * label_width + header)
    for name, cells in rows:
        print(
            name.ljust(label_width)
            + "".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))
        )
    return 0


def _handle_list(args: argparse.Namespace) -> int:
    """Show installed wrappers, or one of the registries behind them."""
    from code_helper.services.paths import Paths
    from code_helper.services.wrappers import list_wrappers

    what = getattr(args, "what", "wrappers")
    if what != "wrappers":
        return _handle_list_axes(what)

    list_wrappers(Paths.default())
    return 0


def _confirm_overwrite(path) -> bool:
    """Ask before clobbering a foreign file — only when there is a TTY to ask.

    Off a TTY this returns False WITHOUT reading stdin, which is what makes a
    scripted run fail fast with the ``--force`` hint instead of blocking
    forever on input that will never arrive. Same ``isatty`` gating as
    ``menu.press_any_key``.
    """
    if not sys.stdin.isatty():
        return False
    answer = input(
        f"{path} exists and was not created by code-helper. Overwrite? [y/N] "
    )
    return answer.strip().lower() in ("y", "yes")


def _confirm_set_default(path, preview: str) -> bool:
    """Ask before ``set-default`` patches/restores a real file — TTY-gated.

    Deliberately NOT :func:`_confirm_overwrite`: that prompt's wording ("was
    not created by code-helper") is misleading here — ``set-default``'s target
    is ALWAYS foreign by definition (Codex's own config), so that phrasing
    would fire on every single successful use rather than flag anything
    unusual. This prompt instead shows the diff/preview so the user can see
    exactly what is about to change before confirming. Same off-a-TTY
    fail-fast contract as ``_confirm_overwrite`` (no stdin read, so a scripted
    run without ``--force`` fails immediately with the hint, never hangs).
    """
    if not sys.stdin.isatty():
        return False
    if preview:
        print(preview)
    answer = input(f"About to write {path}. Continue? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _parse_shape(raw: str | None):
    """``--shape`` string -> ``ConfigShape``, or None when not given.

    Kept out of the argparse layer (no ``choices=``) so the error reads like
    every other domain error this CLI raises, and so the valid set comes from
    the enum rather than a hand-maintained list. Without this, a bogus value
    escaped as a raw ``ValueError`` traceback — and, because ``main`` only
    catches ``CodeHelperError``, still exited 0.
    """
    from code_helper.errors import CodeHelperError
    from code_helper.services.model import ConfigShape

    if not raw:
        return None
    try:
        return ConfigShape(raw)
    except ValueError:
        valid = ", ".join(s.value for s in ConfigShape)
        raise CodeHelperError(f"unknown shape: {raw} (valid: {valid})") from None


def _handle_add(args: argparse.Namespace) -> int:
    """Install (or update) a wrapper — from a preset, or from the three axes.

    Disambiguation rule (deterministic, so it survives future name overlaps):

    1. ``--agent``/``--provider`` given -> constructor; ``name`` must be absent.
    2. otherwise ``name`` is a PRESET, even if an agent happens to share
       the name.
    3. a bare ``name`` that is not a preset but IS an agent gets a message
       showing the constructor form rather than a plain "unknown".
    """
    from code_helper.errors import CodeHelperError
    from code_helper.services.model import (
        get_agent,
        get_provider,
        resolve_shape,
        with_base_url,
    )
    from code_helper.services.models_api import list_models
    from code_helper.services.paths import Paths
    from code_helper.services.secrets import (
        SOURCE_ENV,
        cache_freshly_typed_token,
        credential_for,
        invalidate_cached_credential,
        resolve_token,
        token_for_discovery,
    )
    from code_helper.services.spec import (
        build_spec,
        get_preset,
        spec_from_preset,
        suggest_alias,
    )
    from code_helper.services.wrappers import install_wrapper

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)
    agent_name = getattr(args, "agent", None)
    provider_name = getattr(args, "provider", None)
    # Read through getattr throughout: the TUI builds this Namespace itself and
    # only fills the fields its flow uses, so an absent attribute is normal
    # here, not a bug.
    name = getattr(args, "name", None)
    model = getattr(args, "model", None)
    alias = getattr(args, "alias", None)
    base_url = getattr(args, "base_url", None)
    using_axes = agent_name is not None or provider_name is not None

    if using_axes and name:
        raise CodeHelperError(
            "give either a preset name or --agent/--provider, not both"
        )

    if not using_axes and base_url:
        # Checked before get_preset so the message is about the flag, not
        # about an unrecognised preset name.
        raise CodeHelperError(
            "--base-url applies to the constructor form only "
            "(--agent/--provider) — a preset carries its own provider"
        )

    if using_axes:
        if not agent_name or not provider_name:
            raise CodeHelperError("--agent and --provider must be given together")
        agent = get_agent(agent_name)
        # The ONE substitution point: every downstream reader of base_url
        # (list_models below, resolve_shape/build_spec, the eventual
        # renderer) reads it off this provider object, so subbing it in here
        # — before --list-models, before build_spec — is enough for all of
        # them to see the right value.
        provider = with_base_url(get_provider(provider_name), base_url)

        if getattr(args, "list_models", False):
            result = list_models(provider, token=token_for_discovery(paths, provider))
            if not result.ok:
                raise CodeHelperError(result.error)
            for available in result.models:
                print(available)
            return 0

        if not model:
            raise CodeHelperError(
                f"--model is required (try: code-helper add --agent {agent.name} "
                f"--provider {provider.name} --list-models)"
            )

        shape = _parse_shape(getattr(args, "shape", None))
        # Resolve compatibility BEFORE anything interactive: a bad pairing must
        # never reach a secret prompt for a wrapper that will not be written.
        resolve_shape(agent, provider, preferred=shape)
        spec = build_spec(
            agent=agent,
            provider=provider,
            model=model,
            alias=alias or suggest_alias(model, agent.name),
            shape=shape,
        )
    else:
        if not name:
            raise CodeHelperError("give a preset name, or --agent with --provider")
        try:
            preset = get_preset(name)
        except CodeHelperError as unknown_preset:
            # A bare agent name is a likely mistake worth teaching, not just
            # rejecting.
            try:
                agent = get_agent(name)
            except CodeHelperError:
                # Re-raise get_preset's own message: it names the known
                # presets, and that hint matters most in exactly this case.
                raise unknown_preset from None
            raise CodeHelperError(
                f"unknown wrapper name: {name} — {name} is an agent; "
                f"try: code-helper add --agent {name} --provider ollama "
                f"--model <model>"
            ) from None
        spec = spec_from_preset(preset, model_override=model, alias_override=alias)

    if spec.auth == "secret":
        resolved = resolve_token(
            env_var=spec.token_env_var,
            prompt=f"{spec.name} token ({spec.token_env_var}): ",
            paths=paths,
            provider_name=spec.provider.name,
            base_url_policy=spec.provider.base_url_policy,
        )
        token = resolved.value
    else:
        resolved = None
        token = spec.auth_value

    wrote = install_wrapper(
        paths,
        spec,
        token=token,
        dry_run=dry_run,
        force=getattr(args, "force", False),
        confirm=_confirm_overwrite,
    )
    # Cache only once install_wrapper has returned WITHOUT raising: a refusal
    # (foreign-file guard, discard-only-secret guard) raises CodeHelperError
    # and skips this line entirely, so a token typed for an install that never
    # happened is never persisted. ``wrote`` itself is deliberately NOT part
    # of the gate — ``wrote=False`` means "install_wrapper no-opped because
    # the content was already byte-identical", not a refusal, and the token
    # that produced that byte-identical content is exactly the one worth
    # having cached.
    if resolved is not None:
        cache_freshly_typed_token(
            paths,
            spec.provider.name,
            token,
            source=resolved.source,
            dry_run=dry_run,
        )
        # A non-prompt source (env/cache) is never itself written to the
        # cache — see cache_freshly_typed_token's docstring, an env value
        # already outlives this process. But an env-sourced token that
        # disagrees with what's cached for this provider means the cache is
        # stale relative to what's actually installed: an env-free run later
        # would resolve that stale cache value and silently revert the
        # wrapper to it (a rotated/revoked credential resurrected with no
        # confirmation). Invalidate rather than "helpfully" overwrite it with
        # the env value — env values aren't meant to be cached, and dropping
        # the stale entry is enough to make the next env-free run fall
        # through to a fresh prompt instead of reusing either value.
        #
        # Deliberately NOT gated on ``wrote``: staleness is a fact about
        # whether the cache disagrees with the token just resolved, not about
        # whether THIS call happened to change any bytes. A byte-identical
        # reinstall (``wrote=False`` — the wrapper already has this exact env
        # token) with a stale, DIFFERENT cache entry is just as much a
        # staleness hazard as a real write: the entry is still there, still
        # wrong, and still waiting for an env-free run to resurrect it. Only
        # ``dry_run`` is excluded — a dry run changes nothing on disk, so
        # there is nothing yet to reconcile the cache against.
        if resolved.source == SOURCE_ENV and not dry_run:
            cached = credential_for(paths, spec.provider.name)
            if cached and cached != token:
                invalidate_cached_credential(paths, spec.provider.name)
    if not wrote:
        print("no changes")
    return 0


def _handle_edit_token(args: argparse.Namespace) -> int:
    """Interactively rotate the token of a secret-auth wrapper.

    Unlike ``add``, this always prompts via ``getpass`` directly — it never
    calls :func:`code_helper.services.secrets.resolve_token`, which would
    silently return an existing ``token_env_var`` value OR a cached
    ``credentials.json`` value instead of the NEW one the user is trying to
    type in. The freshly typed token is cached AFTER the install succeeds, so
    the cache tracks the rotation rather than going stale.
    """
    import getpass

    from code_helper.cli.menu import MenuCancelled, select_from_menu
    from code_helper.errors import CodeHelperError
    from code_helper.services.paths import Paths
    from code_helper.services.secrets import SOURCE_PROMPT, cache_freshly_typed_token
    from code_helper.services.wrappers import (
        WRAPPERS,
        describe_all,
        discover_managed,
        get_spec,
        install_wrapper,
        is_installed,
        spec_from_installed,
    )

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)

    def _resolve(name: str):
        """The INSTALLED wrapper's spec, falling back to the preset registry.

        Reading the installed script first is what makes rotation faithful:
        it preserves a model the user chose with ``--model`` (re-expanding the
        preset would silently revert it) and it reaches wrappers built from
        the axes, which have no preset to look up at all.
        """
        return spec_from_installed(paths, name) or get_spec(name)

    if args.name:
        spec = _resolve(args.name)
    else:
        # Presets plus anything the constructor installed — the latter are
        # first-class wrappers and were previously unreachable from here.
        secret_specs = [w for w in WRAPPERS if w.auth == "secret"]
        for found in discover_managed(paths):
            installed = spec_from_installed(paths, found)
            if installed is not None and installed.auth == "secret":
                secret_specs.append(installed)
        if not secret_specs:
            raise CodeHelperError("no wrapper has an editable (secret) token")
        items = describe_all(
            paths,
            secret_specs,
            installed_word="installed",
            not_installed_word="not installed",
        )
        try:
            chosen = select_from_menu(
                items,
                prompt="select a wrapper to edit its token:",
            )
        except MenuCancelled as e:
            if e.hard:
                # Ctrl-C: let it propagate so a TUI caller can treat this as
                # "leave the TUI" rather than "command succeeded, pause and
                # show the menu again" — see cli/tui.py's top-level catch.
                # The plain CLI path has no such catch either, so Ctrl-C at
                # `code-helper edit-token`'s picker behaves like Ctrl-C
                # anywhere else in the CLI: an uncaught KeyboardInterrupt.
                raise
            print("cancelled")
            return 0
        spec = _resolve(chosen)

    if spec.auth != "secret":
        raise CodeHelperError(f"{spec.name} has no editable token (auth={spec.auth})")

    state = "installed" if is_installed(paths, spec.name) else "not installed"
    print(f"{spec.name}: currently {state}")

    token = getpass.getpass(f"new {spec.name} token ({spec.token_env_var}): ")
    if not token:
        raise CodeHelperError("no token entered — aborting")

    # Pass the resolved spec, never the preset NAME: a name re-expands the
    # preset from scratch and discards whatever model this wrapper was
    # actually installed with.
    wrote = install_wrapper(paths, spec, token=token, dry_run=dry_run)
    # Keep the credential cache in step with the rotation: if this was a
    # rotation, the cached value is now stale and the next ``add`` would hand
    # out the old token. This command always prompts (never env/cache — see the
    # docstring above), so the source is unconditionally "prompt". Cached
    # regardless of ``wrote`` — matching ``_handle_add``'s rule (see its
    # comment): ``wrote=False`` means install_wrapper no-opped because the
    # typed token already matches what's installed byte-for-byte, which is
    # exactly the token worth having cached, not a reason to skip caching.
    cache_freshly_typed_token(
        paths, spec.provider.name, token, source=SOURCE_PROMPT, dry_run=dry_run
    )
    if not wrote:
        print("no changes")
    return 0


def _handle_set_default(args: argparse.Namespace) -> int:
    """Patch (or restore) Codex's OWN ``~/.codex/config.toml`` default.

    Thin shell, same contract as every other handler here: resolve
    ``Paths.default()``, delegate to the service, propagate
    ``CodeHelperError``. The one command in this project that touches an
    agent's own configuration file — see ``services/codex_default.py`` for
    why that is safe (patch, never replace; always backed up first).
    """
    from code_helper.errors import CodeHelperError
    from code_helper.services.codex_default import apply_set_default, restore_default
    from code_helper.services.model import get_agent, get_provider, with_base_url
    from code_helper.services.paths import Paths

    paths = Paths.default()
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)
    restore = getattr(args, "restore", False)
    agent_name = getattr(args, "agent", None)
    provider_name = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    base_url = getattr(args, "base_url", None)

    if restore and (agent_name or provider_name or model or base_url):
        raise CodeHelperError(
            "--restore cannot be combined with --agent/--provider/--model/--base-url"
        )
    if getattr(args, "slot", None) is not None and not restore:
        raise CodeHelperError("--slot only applies together with --restore")

    if restore:
        slot = getattr(args, "slot", None) or 1
        wrote = restore_default(
            paths,
            slot=slot,
            catalog_json=getattr(args, "catalog_json", None),
            dry_run=dry_run,
            force=force,
            confirm=_confirm_set_default,
        )
        if not wrote:
            print("no changes")
        return 0

    if not agent_name or not provider_name:
        raise CodeHelperError("--agent and --provider must be given together")
    if not model:
        raise CodeHelperError("--model is required")

    agent = get_agent(agent_name)
    # Same substitution point as _handle_add's — and here it is not merely
    # convenient but load-bearing: without it, a runtime-base_url provider
    # with an empty registry base_url would make openai_base_url("") return
    # "/v1/", and _verify_patch_applied would compare the WRITTEN "/v1/"
    # against the EXPECTED "/v1/" (both derived from the same empty value) —
    # a successful set-default that silently breaks codex, with the
    # verification net unable to catch it because both sides of the check
    # are wrong in the same way.
    provider = with_base_url(get_provider(provider_name), base_url)

    wrote = apply_set_default(
        paths,
        agent=agent,
        provider=provider,
        model=model,
        catalog_json=getattr(args, "catalog_json", None),
        dry_run=dry_run,
        force=force,
        confirm=_confirm_set_default,
    )
    if not wrote:
        print("no changes")
    return 0


def _handle_tui(args: argparse.Namespace) -> int:
    """``tui`` subcommand (and bare ``code-helper``) → the arrow-key menu.

    Thin shell: delegate to :func:`code_helper.cli.tui.run_tui`, which
    dispatches into the same ``_handle_*`` functions as the CLI subcommands.
    """
    from code_helper.cli.tui import run_tui

    return run_tui(args)


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
    parser.set_defaults(func=_handle_tui)

    subparsers = parser.add_subparsers(
        dest="cmd",
        required=False,
        metavar="<command>",
    )

    p_list = subparsers.add_parser(
        "list",
        help="show wrappers, or the available agents/providers",
        parents=[sub_flags],
    )
    p_list.add_argument(
        "what",
        nargs="?",
        default="wrappers",
        choices=["wrappers", "agents", "providers", "matrix"],
        help="what to list (default: wrappers)",
    )
    p_list.set_defaults(func=_handle_list)

    p_add = subparsers.add_parser(
        "add",
        help="install (or update) a wrapper script",
        parents=[sub_flags],
    )
    # `name` is ALWAYS a preset. The constructor is selected by --provider,
    # never by guessing whether `name` looks more like a preset or an agent —
    # a guess would silently change meaning the day a preset and an agent
    # share a name. See _handle_add for the full rule.
    p_add.add_argument(
        "name",
        nargs="?",
        default=None,
        help="preset name (see `code-helper list`); omit when using --agent",
    )
    p_add.add_argument(
        "--agent",
        default=None,
        help="agent to run, e.g. claude or codex (see `code-helper list agents`)",
    )
    p_add.add_argument(
        "--provider",
        default=None,
        help="model backend (see `code-helper list providers`)",
    )
    p_add.add_argument(
        "--model",
        default=None,
        help="model name; required with --agent, an override for a preset",
    )
    p_add.add_argument(
        "--alias",
        default=None,
        help="wrapper file name (default: <model>-<agent>)",
    )
    p_add.add_argument(
        "--shape",
        default=None,
        help="force a config mechanism when several are possible",
    )
    p_add.add_argument(
        "--base-url",
        default=None,
        help="backend URL for a provider with no address in the registry "
        "(e.g. litellm: http://localhost:4000/v1); constructor form only",
    )
    p_add.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="overwrite a file code-helper did not create",
    )
    p_add.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="print the provider's models and exit (writes nothing)",
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

    p_set_default = subparsers.add_parser(
        "set-default",
        help="patch Codex's own ~/.codex/config.toml default (the ONE command "
        "that touches an agent's own config)",
        parents=[sub_flags],
    )
    # No argparse mutually-exclusive group here: that mechanism expresses
    # "flag A vs flag B", not "flag A vs THIS TRIPLET together" — --restore is
    # validated against --agent/--provider/--model explicitly in the handler
    # instead, which also gives a clearer error message than argparse's own.
    p_set_default.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="restore config.toml from a backup slot instead of patching",
    )
    p_set_default.add_argument(
        "--agent",
        default=None,
        help="agent to default onto, e.g. codex (see `code-helper list agents`)",
    )
    p_set_default.add_argument(
        "--provider",
        default=None,
        help="model backend (see `code-helper list providers`)",
    )
    p_set_default.add_argument(
        "--model",
        default=None,
        help="model name to make the default",
    )
    p_set_default.add_argument(
        "--base-url",
        default=None,
        help="backend URL for a provider with no address in the registry "
        "(e.g. litellm: http://localhost:4000/v1); REQUIRED for such a "
        "provider, or config.toml would be patched with a malformed URL",
    )
    p_set_default.add_argument(
        "--catalog-json",
        default=None,
        help="path to the model catalog (default: ~/.codex/model.json)",
    )
    p_set_default.add_argument(
        "--slot",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="backup slot for --restore (1=newest, default: 1)",
    )
    p_set_default.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="skip the confirmation prompt",
    )
    p_set_default.set_defaults(func=_handle_set_default)

    p_tui = subparsers.add_parser(
        "tui",
        help="open the arrow-key menu (also the bare `code-helper` default)",
        parents=[sub_flags],
    )
    p_tui.set_defaults(func=_handle_tui)

    return parser
