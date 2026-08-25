"""Argparse parser builder for the ``codehelper`` CLI.

The dispatch contract: each subcommand registers a handler via
``set_defaults(func=...)``, and :func:`codehelper.__main__.main` calls
``args.func(args)``. Handlers are THIN SHELLS — resolve ``Paths.default()``,
delegate to a service, return its int. They do NOT catch/print/exit: a
:class:`CodeHelperError` propagates to :func:`main`, which formats it as
one-line stderr + exit 1 (full traceback under ``--debug``).

The subcommands: ``list`` (registry + install state), ``add <name>``
(install/update a wrapper, optionally ``--model``), ``edit-token [<name>]``
(rotate a secret-auth wrapper's token; an arrow-key menu picks the wrapper
when ``<name>`` is omitted, via :mod:`codehelper.cli.menu``), and ``tui``
(single-pass arrow-key menu over the three commands above, via
:mod:`codehelper.cli.tui`). A bare ``codehelper`` (no subcommand) also
opens the TUI — it is the discoverable default for a new user, while every
subcommand remains fully scriptable on its own.

Root flags (``--debug`` / ``--dry-run``) attach via a single shared parent
parser so they parse BOTH before and after the subcommand.

Imports: every ``services.*`` dependency is a top-level import — the layering
is a DAG (``cli`` → ``services`` → ``backends``), so nothing here needs to be
deferred to break a cycle. Two deliberate exceptions remain, both about
LATE BINDING rather than cycles:

* :mod:`codehelper.cli.menu` is imported inside the handlers that use it,
  because the test suite patches ``menu.select_from_menu`` / ``menu.read_line``
  by name;
* ``models_api``, ``codex_default`` and ``secrets`` are imported as MODULES and
  called through the module, for the same reason.

A ``from … import name`` in either case would bind the original function at
import time and no patch would ever be seen. :mod:`codehelper.cli.tui` stays
local as well — it imports this module back, so that one IS a cycle.
"""

import argparse
import os
import sys
from dataclasses import replace

from codehelper.cli.requests import (
    AddRequest,
    EditTokenRequest,
    ProxyRequest,
    RemoveRequest,
    SetDefaultRequest,
    SwitchRequest,
)
from codehelper.errors import CodeHelperError

# Modules, not names — see this module's docstring on late binding.
from codehelper.services import claude_settings, codex_default, models_api, secrets
from codehelper.services.agents import all_agents, load_user_agents_strict
from codehelper.services.agents import get_agent as get_any_agent
from codehelper.services.claude_settings import current_switch
from codehelper.services.codex_default import restore_default
from codehelper.services.model import (
    PROVIDERS,
    ConfigShape,
    get_agent,
    get_provider,
    resolve_shape,
    with_auth,
    with_base_url,
)
from codehelper.services.paths import Paths
from codehelper.services.profiles import (
    NewProfileOutcome,
    classify_new_profile,
    validate_new_profile_name,
)
from codehelper.services.secrets import (
    DEFAULT_PROFILE,
    SOURCE_ENV,
    SOURCE_PROMPT,
    ResolvedToken,
    cache_freshly_typed_token,
    credential_for,
    invalidate_cached_credential,
    profile_names,
    rename_profile,
    token_for_discovery,
    valid_active_profile,
)
from codehelper.services.spec import (
    build_spec,
    get_preset,
    spec_from_preset,
    suggest_alias,
)
from codehelper.services.state import active_selection
from codehelper.services.wrappers import (
    WRAPPERS,
    describe_all,
    discover_managed,
    get_spec,
    install_wrapper,
    is_installed,
    list_wrappers,
    remove_wrapper,
    spec_from_installed,
    token_from_installed,
)


def _handle_list_axes(what: str) -> int:
    """Print the agent/provider registries, or the compatibility matrix.

    ``matrix`` is executable documentation: it is rendered by calling
    ``resolve_shape`` itself, so what it shows and what ``add`` accepts cannot
    disagree. It is where a user sees that some pairings are simply blank.
    """

    if what == "agents":
        for agent in all_agents(Paths.default()):
            shapes = ", ".join(sorted(s.value for s in agent.shapes))
            print(f"{agent.name:10} {agent.description:24} [{shapes}]")
        return 0

    if what == "providers":
        for provider in PROVIDERS:
            shapes = ", ".join(sorted(s.value for s in provider.shapes))
            # A provider whose ONLY shape is ANTHROPIC_SETTINGS can never
            # back a generated wrapper (no Agent declares that shape — see
            # ConfigShape.ANTHROPIC_SETTINGS) — every `add`/`matrix` cell for
            # it is a blank "—" by design, which reads as a bug without this
            # label. Data-driven off provider.shapes, not provider.name.
            tag = (
                " (switch-only)"
                if provider.shapes == {ConfigShape.ANTHROPIC_SETTINGS}
                else ""
            )
            print(f"{provider.name:10} {provider.description:24} [{shapes}]{tag}")
        return 0

    # Resolve every cell first: the column has to be as wide as the widest
    # SHAPE it will hold, not the widest provider name, or the values collide.
    agents = all_agents(Paths.default())
    rows: list[tuple[str, list[str]]] = []
    for agent in agents:
        cells = []
        for provider in PROVIDERS:
            try:
                cells.append(resolve_shape(agent, provider).value)
            except CodeHelperError:
                cells.append("—")  # genuinely impossible, not merely unbuilt
        rows.append((agent.name, cells))

    label_width = max([len(a.name) for a in agents] + [0]) + 2
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

    what = getattr(args, "what", "wrappers")
    if what != "wrappers":
        return _handle_list_axes(what)

    # CLI visibility for the TUI's active-profile pre-selection (issue #23): a
    # one-line header naming the active provider/profile, when one is set and
    # still valid. Kept in the parser (not wrappers.describe_all, which three
    # call sites share) because this is CLI presentation, not a wrapper row.

    paths = Paths.default()
    selection = active_selection(paths)
    if selection is not None:
        provider, _ = selection
        profile = valid_active_profile(paths, provider)
        if profile:
            print(f"Active profile: {provider}/{profile}")

    list_wrappers(paths)
    return 0


def _ask_yes_no(prompt: str) -> bool:
    """Read one y/N answer with the same cancellation contract as every
    other TUI text field (Esc/Ctrl-C via :func:`menu.read_line`), instead of
    a bare ``input()`` that leaves Esc's raw bytes in the answer and lets
    Ctrl-C escape as an uncaught ``KeyboardInterrupt``. A soft cancel (Esc) —
    like a blank or non-"y" answer — means "no": declining a destructive
    confirmation is not a cancel-worthy event. A hard cancel (Ctrl-C)
    propagates, because that gesture means "leave the whole program", not
    "decline this prompt" — the same hard/soft split every other TUI reader
    makes (see ``TuiSession._read_text``).
    """
    from codehelper.cli.menu import MenuCancelled, read_line

    try:
        answer = read_line(prompt)
    except MenuCancelled as exc:
        if exc.hard:
            raise
        return False
    return answer.strip().lower() in ("y", "yes")


def _confirm_overwrite(path) -> bool:
    """Ask before clobbering a foreign file — only when there is a TTY to ask.

    Off a TTY this returns False WITHOUT reading stdin, which is what makes a
    scripted run fail fast with the ``--force`` hint instead of blocking
    forever on input that will never arrive. Same ``isatty`` gating as
    ``menu.press_any_key``.
    """
    if not sys.stdin.isatty():
        return False
    return _ask_yes_no(
        f"{path} exists and was not created by codehelper. Overwrite? [y/N] "
    )


def _confirm_set_default(path, preview: str) -> bool:
    """Ask before patching/restoring a real foreign config file — TTY-gated.

    Shared by both ``set-default`` (Codex's ``config.toml``) and ``switch``
    (Claude's ``settings.json``) — same target-is-foreign-by-definition
    situation, same diff-first prompt. Deliberately NOT :func:`_confirm_overwrite`:
    that prompt's wording ("was not created by codehelper") is misleading
    here — the target is ALWAYS foreign by definition, so that phrasing would
    fire on every single successful use rather than flag anything unusual.
    This prompt instead shows the diff/preview so the user can see exactly
    what is about to change before confirming. Same off-a-TTY fail-fast
    contract as ``_confirm_overwrite`` (no stdin read, so a scripted run
    without ``--force`` fails immediately with the hint, never hangs). For
    ``switch``, ``preview`` has already had any token value redacted by
    ``claude_settings._redacted_preview`` before it reaches this function.
    """
    if not sys.stdin.isatty():
        return False
    if preview:
        print(preview)
    return _ask_yes_no(f"About to write {path}. Continue? [y/N] ")


def _read_profile_name(prompt: str) -> str | None:
    """Read one profile name, returning ``None`` on cancel (Esc or Ctrl-C).

    Used by ``_handle_edit_token``'s interactive create/rename branches.
    ``edit-token``'s picker already distinguishes hard cancel (Ctrl-C at the
    menu, propagated) from soft (Esc, exit 0); a cancel at this follow-up
    text prompt always matches the soft path instead, so ``MenuCancelled``
    is caught regardless of its ``hard`` flag.
    """
    from codehelper.cli.menu import MenuCancelled, read_line

    try:
        return read_line(prompt)
    except MenuCancelled:
        return None


def _parse_shape(raw: str | None):
    """``--shape`` string -> ``ConfigShape``, or None when not given.

    Kept out of the argparse layer (no ``choices=``) so the error reads like
    every other domain error this CLI raises, and so the valid set comes from
    the enum rather than a hand-maintained list. Without this, a bogus value
    escaped as a raw ``ValueError`` traceback — and, because ``main`` only
    catches ``CodeHelperError``, still exited 0.
    """

    if not raw:
        return None
    try:
        return ConfigShape(raw)
    except ValueError:
        valid = ", ".join(s.value for s in ConfigShape)
        raise CodeHelperError(f"unknown shape: {raw} (valid: {valid})") from None


def _add_resolve_provider(req, paths):
    """Resolve the constructor-form provider (axes branch only).

    The ONE substitution point: every downstream reader of ``base_url``
    (``list_models``, ``resolve_shape``/``build_spec``, the eventual renderer)
    reads it off the returned provider object, so subbing it in here — before
    ``--list-models``, before ``build_spec`` — is enough for all of them to
    see the right value. ``with_auth`` is the matching substitution for auth
    (``--auth secret``), same reasoning, same call order.

    Raises:
        CodeHelperError: ``--agent``/``--provider`` not given together.
    """

    if not req.agent or not req.provider:
        raise CodeHelperError("--agent and --provider must be given together")
    agent = get_any_agent(paths, req.agent)
    provider = with_base_url(get_provider(req.provider), req.base_url)
    provider = with_auth(provider, want_secret=req.auth == "secret")
    return agent, provider


def _add_list_models_or_none(req, paths, agent, provider, profile_name):
    """Run ``--list-models`` if requested, printing models and returning 0.

    Returns ``None`` when ``--list-models`` was not requested, so the caller
    falls through to spec construction. The token for discovery follows the
    env → cache (FIXED provider only) → unauthenticated fallback — never a
    prompt, so an optional listing never blocks a script on stdin.
    """

    if not req.list_models:
        return None
    result = models_api.list_models(
        provider,
        token=token_for_discovery(paths, provider, profile_name=profile_name),
    )
    if not result.ok:
        raise CodeHelperError(result.error)
    for available in result.models:
        print(available)
    return 0


def _add_resolve_spec(req, paths, agent, provider, profile_name):
    """Build the ``WrapperSpec`` — constructor path.

    Resolves compatibility BEFORE anything interactive: a bad pairing must
    never reach a secret prompt for a wrapper that will not be written.
    """

    if not req.model:
        raise CodeHelperError(
            f"--model is required (try: codehelper add --agent {agent.name} "
            f"--provider {provider.name} --list-models)"
        )
    shape = _parse_shape(req.shape)
    resolve_shape(agent, provider, preferred=shape)
    return build_spec(
        agent=agent,
        provider=provider,
        model=req.model,
        alias=req.alias or suggest_alias(req.model, agent.name, profile_name),
        shape=shape,
        profile_name=profile_name,
    )


def _add_spec_from_preset(req, paths, profile_name):
    """Build the ``WrapperSpec`` — preset path.

    Disambiguation: ``name`` that is not a preset but IS an agent gets a
    teaching message showing the constructor form rather than a plain
    "unknown". Issue #23 preset-branch fallback: pick up the stored active
    profile for the preset's own provider when ``--profile`` is absent.

    Raises:
        CodeHelperError: unknown preset/agent name.
    """

    if not req.name:
        raise CodeHelperError("give a preset name, or --agent with --provider")
    try:
        preset = get_preset(req.name)
    except CodeHelperError as unknown_preset:
        # A bare agent name is a likely mistake worth teaching, not just
        # rejecting.
        try:
            get_any_agent(paths, req.name)
        except CodeHelperError:
            # Re-raise get_preset's own message: it names the known presets,
            # and that hint matters most in exactly this case.
            raise unknown_preset from None
        raise CodeHelperError(
            f"unknown wrapper name: {req.name} — {req.name} is an agent; "
            f"try: codehelper add --agent {req.name} --provider ollama "
            f"--model <model>"
        ) from None
    if profile_name is None:
        profile_name = valid_active_profile(paths, preset.provider)
    return spec_from_preset(
        preset,
        model_override=req.model,
        alias_override=req.alias,
        profile_name=profile_name,
    )


def _warn_env_cache_conflict(resolved, **axes) -> None:
    """Print the env-vs-cache token disagreement (issue #71) to stderr.

    Emitted at RESOLUTION time — before any write/dry-run branch — so a
    ``--dry-run`` shows it too (it is exactly the diagnostic a dry run
    needs), and the TUI's silent chip hot-apply, which captures stdout
    only, still surfaces it on the terminal.
    """
    conflict = secrets.env_cache_conflict(resolved, **axes)
    if conflict:
        print(conflict, file=sys.stderr)


def _add_resolve_token(spec, req, paths, profile_name):
    """Resolve the token for ``spec``: pre-typed → env/cache/prompt → literal.

    Returns ``(token, resolved)`` where ``resolved`` is the
    :class:`ResolvedToken` for a secret spec (or ``None`` for literal/none
    auth), so the caller can decide whether to cache it.
    """

    if spec.auth != "secret":
        return spec.auth_value, None
    if req.profile_token is not None:
        if not req.profile_token:
            raise CodeHelperError("no token entered — aborting")
        resolved = ResolvedToken(req.profile_token, SOURCE_PROMPT)
    else:
        resolved = secrets.resolve_token(
            env_var=spec.token_env_var,
            prompt=f"{spec.name} token ({spec.token_env_var}): ",
            paths=paths,
            provider_name=spec.provider.name,
            profile_name=profile_name,
            base_url_policy=spec.provider.base_url_policy,
        )
    _warn_env_cache_conflict(
        resolved,
        env_var=spec.token_env_var,
        paths=paths,
        provider_name=spec.provider.name,
        profile_name=profile_name,
        base_url_policy=spec.provider.base_url_policy,
    )
    return resolved.value, resolved


def _add_install_and_cache(spec, req, paths, token, resolved, profile_name):
    """Install the wrapper, then keep the credential cache in step.

    Caches only once ``install_wrapper`` has returned WITHOUT raising: a
    refusal (foreign-file guard, discard-only-secret guard) raises
    :class:`CodeHelperError` earlier and skips this entirely. See the inline
    comments for the staleness-invalidation rule (env-sourced token that
    disagrees with the cache → invalidate, not overwrite).

    Returns ``True`` iff anything was written.
    """

    dry_run = req.dry_run
    wrote = install_wrapper(
        paths,
        spec,
        token=token,
        dry_run=dry_run,
        force=req.force,
        confirm=_confirm_overwrite,
    )
    if resolved is not None:
        cache_profile = profile_name or DEFAULT_PROFILE
        cache_freshly_typed_token(
            paths,
            spec.provider.name,
            token,
            profile_name=cache_profile,
            source=resolved.source,
            dry_run=dry_run,
        )
        # A non-prompt source (env/cache) is never itself written to the cache
        # — see cache_freshly_typed_token's docstring. But an env-sourced
        # token that disagrees with what's cached means the cache is stale
        # relative to what's installed: invalidate rather than overwrite.
        # Deliberately NOT gated on ``wrote`` — staleness is about whether the
        # cache disagrees with the resolved token, not whether this call
        # changed bytes. Only ``dry_run`` is excluded.
        if resolved.source == SOURCE_ENV and not dry_run:
            cached = credential_for(paths, spec.provider.name, cache_profile)
            if cached and cached != token:
                invalidate_cached_credential(paths, spec.provider.name, cache_profile)
    if (
        resolved is not None
        and req.profile_rename_from
        and req.profile_rename_to
        and not dry_run
    ):
        rename_profile(
            paths,
            spec.provider.name,
            req.profile_rename_from,
            req.profile_rename_to,
        )
    return wrote


def _handle_add(args: argparse.Namespace | AddRequest) -> int:
    """Install (or update) a wrapper — from a preset, or from the three axes.

    Disambiguation rule (deterministic, so it survives future name overlaps):

    1. ``--agent``/``--provider`` given -> constructor; ``name`` must be absent.
    2. otherwise ``name`` is a PRESET, even if an agent happens to share
       the name.
    3. a bare ``name`` that is not a preset but IS an agent gets a message
       showing the constructor form rather than a plain "unknown".

    Thin orchestrator over the ``_add_*`` pipeline: validate flags → resolve
    provider (axes) → early ``--list-models`` return → build spec → resolve
    token → install + cache. Each stage lives in its own function so the
    branching surface stays readable; see the individual docstrings for the
    invariants each carries.
    """

    paths = Paths.default()
    req = args if isinstance(args, AddRequest) else AddRequest.from_namespace(args)
    using_axes = req.agent is not None or req.provider is not None

    # Issue #23: an explicit --profile always wins; otherwise the CLI picks up
    # the TUI's stored active profile for this provider as a default, if it
    # still exists (valid_active_profile cross-checks against profile_names).
    # The implicit injection applies ONLY to the provider's own default
    # endpoint: a caller-supplied --base-url names a host the stored active
    # profile was never authorized for, so the caller must opt in explicitly
    # with --profile (or an env token) there — never a silent persisted
    # pointer. The constructor branch knows the provider immediately; the
    # preset branch (no custom base-url possible) falls back inside
    # _add_spec_from_preset.
    profile_name = req.profile
    if profile_name is None and req.provider and req.base_url is None:
        profile_name = valid_active_profile(paths, req.provider)

    if using_axes and req.name:
        raise CodeHelperError(
            "give either a preset name or --agent/--provider, not both"
        )
    if not using_axes and req.base_url:
        # Checked before get_preset so the message is about the flag, not
        # about an unrecognised preset name.
        raise CodeHelperError(
            "--base-url applies to the constructor form only "
            "(--agent/--provider) — a preset carries its own provider"
        )
    if not using_axes and req.auth:
        raise CodeHelperError(
            "--auth applies to the constructor form only "
            "(--agent/--provider) — a preset carries its own provider"
        )

    if using_axes:
        agent, provider = _add_resolve_provider(req, paths)
        early = _add_list_models_or_none(req, paths, agent, provider, profile_name)
        if early is not None:
            return early
        spec = _add_resolve_spec(req, paths, agent, provider, profile_name)
    else:
        spec = _add_spec_from_preset(req, paths, profile_name)
        # _add_spec_from_preset may have filled profile_name via its own
        # active-profile fallback; re-read it off the built spec.
        profile_name = spec.profile_name

    # The alias reservation (naming.validate_alias) covers only the static
    # built-in registry. A user-defined agent's binary is a real agent binary
    # too, and a wrapper named after it would shadow the real executable on
    # PATH exactly as a built-in one would — so reject it here, at the single
    # boundary every wrapper creation flows through, before any token prompt.
    # The STRICT read: a corrupt registry must fail closed, not read as "no
    # user agents" and let a wrapper silently shadow a real binary.
    user_binaries = {a.binary for a in load_user_agents_strict(paths)}
    if spec.alias in user_binaries:
        raise CodeHelperError(
            f"{spec.alias!r} is a reserved name — a wrapper named after a "
            f"user-defined agent's binary would shadow the real one on PATH"
        )

    token, resolved = _add_resolve_token(spec, req, paths, profile_name)
    wrote = _add_install_and_cache(spec, req, paths, token, resolved, profile_name)
    if not wrote:
        print("no changes")
    elif not req.dry_run and str(paths.bin_dir) not in os.environ.get("PATH", "").split(
        os.pathsep
    ):
        # `wrote` is True under --dry-run too ("would write" — install_wrapper's
        # documented contract), so without this guard a dry run would warn
        # about a PATH problem for a file it never actually created.
        print(
            f"warning: {paths.bin_dir} is not on PATH — add it to your shell "
            f"profile to run '{spec.alias}'",
            file=sys.stderr,
        )
    return 0


def _edit_token_resolve_profile(
    paths, provider_name: str
) -> tuple[str, str | None, str | None] | None:
    """Pick (or create) a token profile for ``provider_name`` interactively.

    Returns ``(profile_name, rename_from, rename_to)`` — where ``rename_*``
    carry the prior profile's rename when a new profile is created alongside
    a single existing one, or ``None`` otherwise. Returns ``None`` on a soft
    cancel (Esc at the menu, Ctrl-C at a name prompt), signalling the caller
    to print "cancelled" and exit 0.

    The new-profile naming decision is owned by :mod:`codehelper.services.profiles`
    (issue #37, P1.2): this function maps the classifier's outcome to the
    CLI's raises (lowercase messages, SAME and COLLISION_NEW merged). See
    ``_new_profile`` in ``cli/tui.py`` for the TUI's own mapping.
    """
    from codehelper.cli.menu import MenuCancelled, select_from_menu

    names = list(profile_names(paths, provider_name))
    if not names:
        return None  # no menu to show — caller falls through to DEFAULT_PROFILE

    profile_items = [
        (name, "default" if name == DEFAULT_PROFILE else name) for name in names
    ]
    profile_items.append(("__new_profile__", "create new profile"))
    try:
        selected = select_from_menu(
            profile_items,
            prompt="select token profile to rotate:",
        )
    except MenuCancelled as e:
        if e.hard:
            # Ctrl-C: propagate so a TUI caller treats this as "leave", not
            # "succeed + re-show menu" — see cli/tui.py's top-level catch.
            raise
        return None

    if selected != "__new_profile__":
        return selected, None, None

    if len(names) == 1:
        current_name = _read_profile_name(f"name for current profile ({names[0]}): ")
        if current_name is None:
            return None
        new_name = _read_profile_name("name for new profile: ")
        if new_name is None:
            return None
        outcome = classify_new_profile(names, current_name, new_name)
        if outcome is NewProfileOutcome.EMPTY:
            raise CodeHelperError("profile names cannot be empty")
        # CLI merges SAME and COLLISION_NEW into one message, matching the
        # pre-refactor `current_name == new_name or new_name in names` check.
        # COLLISION_RENAMED is NOT mapped here: the original CLI never checked
        # it in this branch, and downstream rename_profile surfaces it.
        if outcome in (NewProfileOutcome.SAME, NewProfileOutcome.COLLISION_NEW):
            raise CodeHelperError("profile names must be unique")
        return new_name, names[0], current_name

    # Multi-profile branch: only a new name is collected.
    new_name = _read_profile_name("name for new profile: ")
    if new_name is None:
        return None
    outcome = validate_new_profile_name(new_name, names)
    if outcome in (NewProfileOutcome.EMPTY, NewProfileOutcome.COLLISION_NEW):
        raise CodeHelperError("new profile name must be non-empty and unique")
    return new_name, None, None


def _edit_token_spec(paths, name: str | None):
    """The spec whose token ``edit-token`` will rotate.

    Reading the INSTALLED script first is what makes rotation faithful: it
    preserves a model the user chose with ``--model`` (re-expanding the preset
    would silently revert it) and it reaches wrappers built from the axes,
    which have no preset to look up at all.

    With no ``name``, an arrow-key menu picks among every secret-auth wrapper
    (presets plus anything the constructor installed). Returns ``None`` on a
    soft cancel at that menu, signalling the caller to print "cancelled" and
    exit 0; a hard cancel (Ctrl-C) propagates.
    """
    from codehelper.cli.menu import MenuCancelled, select_from_menu

    def _resolve(alias: str):
        return spec_from_installed(paths, alias) or get_spec(alias)

    if name:
        return _resolve(name)

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
            # `codehelper edit-token`'s picker behaves like Ctrl-C
            # anywhere else in the CLI: an uncaught KeyboardInterrupt.
            raise
        return None
    return _resolve(chosen)


def _edit_token_prompt_token(spec, profile_name: str, given: str | None) -> str:
    """The token to install — the request's own, else a fresh ``getpass``.

    Validation is deliberately applied to BOTH paths: a caller-supplied token
    (the TUI's) goes through the same empty/ASCII gate as a typed one.
    """
    import getpass

    token = given
    if token is None:
        token = getpass.getpass(
            f"new {spec.name} token ({spec.token_env_var}) for {profile_name}: "
        )
    if not token:
        raise CodeHelperError("no token entered — aborting")
    if not token.isascii():
        raise CodeHelperError("token must contain ASCII characters")
    return token


def _handle_edit_token(args: argparse.Namespace | EditTokenRequest) -> int:
    """Interactively rotate the token of a secret-auth wrapper.

    Unlike ``add``, this always prompts via ``getpass`` directly — it never
    calls :func:`codehelper.services.secrets.resolve_token`, which would
    silently return an existing ``token_env_var`` value OR a cached
    ``credentials.json`` value instead of the NEW one the user is trying to
    type in. The freshly typed token is cached AFTER the install succeeds, so
    the cache tracks the rotation rather than going stale.
    """

    req = (
        args
        if isinstance(args, EditTokenRequest)
        else EditTokenRequest.from_namespace(args)
    )

    paths = Paths.default()
    dry_run = req.dry_run
    profile_rename_from = req.profile_rename_from
    profile_rename_to = req.profile_rename_to

    spec = _edit_token_spec(paths, req.name)
    if spec is None:
        print("cancelled")
        return 0

    if spec.auth != "secret":
        raise CodeHelperError(f"{spec.name} has no editable token (auth={spec.auth})")

    profile_name = req.profile
    # Only offer the profile menu when profiles exist; with none cached the
    # original code skipped straight to DEFAULT_PROFILE (no menu to pick
    # from). _edit_token_resolve_profile returns None on a soft cancel
    # (Esc/Ctrl-C at the menu or a name prompt) — distinct from the "no
    # profiles" skip, which never calls it.
    if profile_name is None and profile_names(paths, spec.provider.name):
        resolved = _edit_token_resolve_profile(paths, spec.provider.name)
        if resolved is None:
            # Soft cancel → "cancelled" + exit 0, matching the picker's own
            # soft-cancel behavior.
            print("cancelled")
            return 0
        profile_name, profile_rename_from, profile_rename_to = resolved
    if profile_name is None:
        profile_name = DEFAULT_PROFILE
    spec = replace(spec, profile_name=profile_name)

    state = "installed" if is_installed(paths, spec.name) else "not installed"
    print(f"{spec.name}: currently {state}")

    token = _edit_token_prompt_token(spec, profile_name, req.profile_token)

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
        paths,
        spec.provider.name,
        token,
        profile_name=profile_name,
        source=SOURCE_PROMPT,
        dry_run=dry_run,
    )
    if profile_rename_from and profile_rename_to and not dry_run:
        rename_profile(
            paths,
            spec.provider.name,
            profile_rename_from,
            profile_rename_to,
        )
    if not wrote:
        print("no changes")
    return 0


def _confirm_remove(paths: list) -> bool:
    """Ask before ``remove`` deletes real files — TTY-gated, same contract as
    :func:`_confirm_set_default` (no stdin read off a TTY, so a scripted run
    without ``--force`` fails fast instead of hanging on input that will
    never arrive).

    Lists every file about to be unlinked (the wrapper plus any owned
    Codex config/catalog siblings) so the user sees the full blast radius —
    a single ``y`` answer, unlike ``set-default``'s config patch, can never
    be undone.
    """
    if not sys.stdin.isatty():
        return False
    print("About to remove:")
    for path in paths:
        print(f"  {path}")
    return _ask_yes_no("Continue? [y/N] ")


def _handle_remove(args: argparse.Namespace | RemoveRequest) -> int:
    """Remove one managed wrapper and its owned companion files."""

    req = (
        args if isinstance(args, RemoveRequest) else RemoveRequest.from_namespace(args)
    )
    remove_wrapper(
        Paths.default(),
        req.name,
        dry_run=req.dry_run,
        force=req.force,
        confirm=_confirm_remove,
    )
    return 0


def _handle_set_default(args: argparse.Namespace | SetDefaultRequest) -> int:
    """Patch (or restore) Codex's OWN ``~/.codex/config.toml`` default.

    Thin shell, same contract as every other handler here: resolve
    ``Paths.default()``, delegate to the service, propagate
    ``CodeHelperError``. The one command in this project that touches an
    agent's own configuration file — see ``services/codex_default.py`` for
    why that is safe (patch, never replace; always backed up first).
    """

    req = (
        args
        if isinstance(args, SetDefaultRequest)
        else SetDefaultRequest.from_namespace(args)
    )

    paths = Paths.default()

    if req.restore and (req.agent or req.provider or req.model or req.base_url):
        raise CodeHelperError(
            "--restore cannot be combined with --agent/--provider/--model/--base-url"
        )
    if req.slot is not None and not req.restore:
        raise CodeHelperError("--slot only applies together with --restore")

    if req.restore:
        slot = req.slot or 1
        wrote = restore_default(
            paths,
            slot=slot,
            dry_run=req.dry_run,
            force=req.force,
            confirm=_confirm_set_default,
        )
        if not wrote:
            print("no changes")
        return 0

    if not req.agent or not req.provider:
        raise CodeHelperError("--agent and --provider must be given together")
    if not req.model:
        raise CodeHelperError("--model is required")

    agent = get_agent(req.agent)
    # Same substitution point as _handle_add's — and here it is not merely
    # convenient but load-bearing: without it, a runtime-base_url provider
    # with an empty registry base_url would make openai_base_url("") return
    # "/v1/", and _verify_patch_applied would compare the WRITTEN "/v1/"
    # against the EXPECTED "/v1/" (both derived from the same empty value) —
    # a successful set-default that silently breaks codex, with the
    # verification net unable to catch it because both sides of the check
    # are wrong in the same way.
    provider = with_base_url(get_provider(req.provider), req.base_url)

    wrote = codex_default.apply_set_default(
        paths,
        agent=agent,
        provider=provider,
        model=req.model,
        dry_run=req.dry_run,
        force=req.force,
        confirm=_confirm_set_default,
    )
    if not wrote:
        print("no changes")
    return 0


def _switch_resolve_token(
    provider, req: SwitchRequest, paths, *, non_interactive: bool = False
) -> str:
    """The token for a `switch --provider ...` (explicit-axes) invocation.

    Mirrors ``_add_resolve_token`` exactly, but works off a bare
    :class:`~codehelper.services.model.Provider` rather than a
    :class:`~codehelper.services.spec.WrapperSpec` — `switch` never builds
    one (see ``_handle_switch``'s docstring on why). An `env_reset` provider
    never reaches this: :func:`_handle_switch` returns before calling it.

    ``non_interactive=True`` is the chip hot-apply path (``from_preset``): a
    token is taken from env/cache or the switch FAILS CLEANLY — it never
    prompts, because a prompt inside the running menu would swallow the
    user's keystrokes and block the very no-prompt apply the chip promises.
    """
    if provider.auth != "secret":
        return provider.auth_value
    kwargs = {}
    if non_interactive:

        def _no_prompt(_prompt: str) -> str:
            raise CodeHelperError(
                f"no {provider.name} token available for a non-interactive "
                f"switch — set {provider.token_env_var} or cache one first"
            )

        kwargs["getpass_fn"] = _no_prompt
    resolved = secrets.resolve_token(
        env_var=provider.token_env_var,
        prompt=f"{provider.name} token ({provider.token_env_var}): ",
        paths=paths,
        provider_name=provider.name,
        profile_name=req.profile,
        base_url_policy=provider.base_url_policy,
        **kwargs,
    )
    _warn_env_cache_conflict(
        resolved,
        env_var=provider.token_env_var,
        paths=paths,
        provider_name=provider.name,
        profile_name=req.profile,
        base_url_policy=provider.base_url_policy,
    )
    return resolved.value


def _switch_axes_from_wrapper(req: SwitchRequest, paths):
    """Resolve ``(provider, tier_models, token, subagent_model)`` from an
    already-installed wrapper — the ``--from-wrapper`` fast path.

    Guaranteed to reach the SAME backend that wrapper's own script would:
    ``tier_models``/``base_url`` are recovered from the rendered body
    (``wrappers.spec_from_installed``) and the token from the same file
    (``wrappers.token_from_installed``) — zero prompts, zero re-derivation.

    An ``ollama-launch`` claude wrapper is converted to Ollama's
    Anthropic-compatible live-settings target, just like its preset chip.
    ``switch`` drives a LIVE CLAUDE session, so a wrapper belonging to any
    other agent (codex shares Ollama) fails closed rather than retargeting
    claude to a foreign selection.
    """
    if not req.from_wrapper:
        raise CodeHelperError(
            "internal error: _switch_axes_from_wrapper called with no "
            "--from-wrapper name"
        )  # pragma: no cover — _handle_switch only calls this when truthy
    name: str = req.from_wrapper
    spec = spec_from_installed(paths, name)
    if spec is None:
        raise CodeHelperError(
            f"no installed wrapper named {name!r} — see `codehelper list`"
        )
    if spec.agent.name != "claude":
        raise CodeHelperError(
            f"wrapper {name!r} is a {spec.agent.name} wrapper — "
            f"`switch --from-wrapper` can only retarget a claude session"
        )
    provider, tier_models, subagent_model = claude_settings.live_axes_for_spec(spec)
    token = ""
    if spec.auth == "secret":
        token = token_from_installed(paths, name, spec.provider.name) or ""
        if not token:
            raise CodeHelperError(
                f"could not recover a token from wrapper {name!r} "
                f"— re-install it or use --provider/--model instead"
            )
    elif spec.auth == "literal":
        token = spec.auth_value
    return provider, tier_models, token, subagent_model


def _switch_axes_from_preset(req: SwitchRequest, paths):
    """Resolve a curated Claude chip without consulting ``~/.local/bin``.

    Presets are backend choices in the chipset.  Their optional wrapper is a
    separate convenience for launching a *new* process, so a foreign script
    named ``deepseek`` must never make the DeepSeek chip disappear or fail.
    ``ollama-launch`` presets are converted to Ollama's Anthropic-compatible
    live-settings form; that is the only way to retarget an existing process.

    A preset, unlike an installed wrapper, freezes no token — so the token is
    resolved NON-interactively (env/cache or a clean error): a chip press is a
    no-prompt hot-apply, and a hidden prompt inside the running menu would
    swallow keystrokes.
    """
    if not req.from_preset:
        raise CodeHelperError("internal error: missing preset for chipset switch")
    spec = spec_from_preset(get_preset(req.from_preset))
    if spec.agent.name != "claude":
        raise CodeHelperError(f"preset {spec.name!r} is not a Claude backend")
    provider, tier_models, subagent_model = claude_settings.live_axes_for_spec(spec)
    return (
        provider,
        tier_models,
        _switch_resolve_token(provider, req, paths, non_interactive=True),
        subagent_model,
    )


def _switch_axes_from_flags(req: SwitchRequest, paths):
    """Resolve ``(provider, tier_models, token, subagent_model)`` from the
    explicit ``--provider``/``--model``/``--haiku``/... flags."""
    from codehelper.services.spec import TierModels

    provider_name = req.provider
    if provider_name is None:
        raise CodeHelperError(
            "give a provider — `switch <provider>`, `switch --provider P "
            "--model M`, or `switch --from-wrapper NAME`"
        )
    provider = with_auth(
        with_base_url(get_provider(provider_name), req.base_url), req.auth == "secret"
    )

    if provider.env_reset:
        return provider, None, "", None

    if req.haiku or req.sonnet or req.opus:
        if not (req.haiku and req.sonnet and req.opus):
            raise CodeHelperError(
                "--haiku/--sonnet/--opus must be given together (or use "
                "--model for all three)"
            )
        tier_models = TierModels(haiku=req.haiku, sonnet=req.sonnet, opus=req.opus)
    elif req.model:
        tier_models = TierModels.uniform(req.model)
    else:
        raise CodeHelperError(
            "a model is required — pass --model, or --haiku/--sonnet/--opus, "
            "or --from-wrapper"
        )

    token = _switch_resolve_token(provider, req, paths)
    return provider, tier_models, token, req.subagent_model


def _handle_switch(args: argparse.Namespace | SwitchRequest) -> int:
    """Live-patch Claude Code's OWN ``~/.claude/settings.json`` ``env`` block.

    Unlike ``add``/``set-default``, this never builds a ``WrapperSpec`` —
    ``build_spec`` requires a valid alias and carries install semantics
    (file name, mode, ownership marker) meaningless for a settings.json
    patch. It reuses ``TierModels`` (the value type both mechanisms share)
    directly, via ``claude_settings.apply_switch``.

    Distinct from ``set-default``: that command persists a PREFERENCE
    (patches Codex's OWN ``~/.codex/config.toml``, applies at Codex's next
    launch); this command changes what an ALREADY RUNNING ``claude`` does on
    its NEXT PROMPT — no restart, because Claude Code re-reads
    ``settings.json`` between prompts. See ``services/claude_settings.py``'s
    module docstring for the mechanism and its caveats (process-env vs
    settings.json precedence inside a wrapper session; the token now lives
    in a typically-``0o644``, often-synced file).
    """
    req = (
        args if isinstance(args, SwitchRequest) else SwitchRequest.from_namespace(args)
    )
    paths = Paths.default()

    if req.status:
        live = current_switch(paths)
        print(live if live is not None else "native (no override in settings.json)")
        return 0

    if req.restore and (
        req.provider
        or req.from_wrapper
        or req.model
        or req.haiku
        or req.sonnet
        or req.opus
        or req.subagent_model
        or req.base_url
        or req.auth
        or req.profile
    ):
        raise CodeHelperError(
            "--restore cannot be combined with a provider or model flags"
        )
    if req.slot is not None and not req.restore:
        raise CodeHelperError("--slot only applies together with --restore")
    switch_sources = sum(
        bool(value) for value in (req.provider, req.from_wrapper, req.from_preset)
    )
    if switch_sources > 1:
        raise CodeHelperError("give exactly one switch source")

    if req.restore:
        slot = req.slot or 1
        wrote = claude_settings.restore_settings(
            paths,
            slot=slot,
            dry_run=req.dry_run,
            force=req.force,
            confirm=_confirm_set_default,
        )
        if not wrote:
            print("no changes")
        return 0

    if req.from_preset:
        provider, tier_models, token, subagent_model = _switch_axes_from_preset(
            req, paths
        )
    elif req.from_wrapper:
        provider, tier_models, token, subagent_model = _switch_axes_from_wrapper(
            req, paths
        )
    else:
        provider, tier_models, token, subagent_model = _switch_axes_from_flags(
            req, paths
        )

    wrote = claude_settings.apply_switch(
        paths,
        provider=provider,
        tier_models=tier_models,
        token=token,
        subagent_model=subagent_model,
        dry_run=req.dry_run,
        # A provider switch is an explicitly requested hot-apply operation.
        # A chip press (the TUI) opts in via force=True and skips the prompt;
        # the interactive CLI command keeps its confirmation gate unless
        # --force is passed. The service still validates the complete target,
        # protects unrelated keys, snapshots a backup and rejects stale writes.
        force=req.force,
        confirm=_confirm_set_default,
    )
    if not wrote:
        print("no changes")
    return 0


def _handle_proxy(args: argparse.Namespace | ProxyRequest) -> int:
    """Toggle / configure the proxy keys in ``~/.claude/settings.json``.

    Patches the SAME file as ``switch`` but a disjoint set of keys: this
    command owns ``PROXY_ENV_KEYS`` + ``NO_PROXY_ENV_KEYS``, ``switch`` owns
    the ``ANTHROPIC_*`` block. Neither can disturb the other — see
    ``services/proxy.py`` for why that is enforced mechanically rather than
    by convention.

    Thin by design: the verb-to-address resolution and the
    save-before-blank ordering both live in ``proxy.set_proxy_state``, so
    every entry point (this command and the TUI's three) gets them from one
    place rather than by routing discipline. What stays here is what is
    genuinely CLI-shaped — the flag contradiction check and status printing.
    """
    from codehelper.services import proxy as proxy_service

    req = args if isinstance(args, ProxyRequest) else ProxyRequest.from_namespace(args)
    paths = Paths.default()

    if req.status or (req.action is None and req.url is None and req.no_proxy is None):
        status = proxy_service.proxy_status(paths)
        if status.enabled:
            print(f"on — {status.display_url}")
        elif status.saved_url:
            print(f"off (saved: {status.display_saved_url})")
        else:
            print("off (no address configured)")
        if status.no_proxy:
            print(f"NO_PROXY: {status.no_proxy}")
        return 0

    # `--url` is "set the address", which only means something switched on;
    # an explicit `off` alongside it would be contradictory rather than
    # merely redundant, so it is rejected instead of silently picking one.
    if req.url is not None and req.action == "off":
        raise CodeHelperError("--url sets an address to use — it cannot go with `off`")

    wrote = proxy_service.set_proxy_state(
        paths,
        action=req.action,
        url=req.url,
        no_proxy=req.no_proxy,
        dry_run=req.dry_run,
        force=req.force,
        confirm=_confirm_set_default,
    )
    if not wrote:
        print("no changes")
    return 0


def _handle_tui(args: argparse.Namespace) -> int:
    """``tui`` subcommand (and bare ``codehelper``) → the arrow-key menu.

    Thin shell: delegate to :func:`codehelper.cli.tui.run_tui`, which
    dispatches into the same ``_handle_*`` functions as the CLI subcommands.
    """
    from codehelper.cli.tui import run_tui

    return run_tui(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the root ``codehelper`` argparse parser.

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
        prog="codehelper",
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
        help="preset name (see `codehelper list`); omit when using --agent",
    )
    p_add.add_argument(
        "--agent",
        default=None,
        help="agent to run, e.g. claude or codex (see `codehelper list agents`)",
    )
    p_add.add_argument(
        "--provider",
        default=None,
        help="model backend (see `codehelper list providers`)",
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
        "(e.g. litellm: http://localhost:4000); a bare host/IP like "
        "78.47.183.125 is auto-completed to https://78.47.183.125:4000; "
        "constructor form only",
    )
    p_add.add_argument(
        "--auth",
        default=None,
        choices=["secret"],
        help="override a provider's default auth mode to a secret token "
        "(e.g. ollama behind a reverse proxy or Ollama Cloud); only for "
        "providers that declare it overridable; constructor form only",
    )
    p_add.add_argument(
        "--profile",
        default=None,
        help="token profile to use (e.g. work or personal)",
    )
    p_add.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="overwrite a file codehelper did not create",
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
    p_edit_token.add_argument(
        "--profile",
        default=None,
        help="token profile to update (default: default)",
    )
    p_edit_token.set_defaults(func=_handle_edit_token)

    p_remove = subparsers.add_parser(
        "remove",
        help="remove a wrapper and its managed companion files",
        parents=[sub_flags],
    )
    p_remove.add_argument("name", help="wrapper name to remove")
    p_remove.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="remove an unmanaged wrapper file too",
    )
    p_remove.set_defaults(func=_handle_remove)

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
        help="agent to default onto, e.g. codex (see `codehelper list agents`)",
    )
    p_set_default.add_argument(
        "--provider",
        default=None,
        help="model backend (see `codehelper list providers`)",
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

    p_switch = subparsers.add_parser(
        "switch",
        help="live-patch Claude Code's own ~/.claude/settings.json — changes "
        "an ALREADY RUNNING claude's backend on its next prompt, no restart "
        "(distinct from set-default, which persists a preference for the "
        "NEXT launch)",
        parents=[sub_flags],
    )
    p_switch.add_argument(
        "provider_positional",
        metavar="provider",
        nargs="?",
        default=None,
        help="provider to switch to, e.g. zai, or native to clear the "
        "override (see `codehelper list providers`); sugar for --provider",
    )
    # No argparse mutually-exclusive group — same reasoning as set-default's:
    # the handler validates the combinations explicitly for a clearer error.
    p_switch.add_argument(
        "--from-wrapper",
        default=None,
        help="lift the model/token straight off an already-installed "
        "anthropic-env wrapper (e.g. glm) — zero prompts, same backend that "
        "wrapper's own script reaches",
    )
    p_switch.add_argument(
        "--provider",
        default=None,
        help="model backend (see `codehelper list providers`); alternative "
        "to the positional form",
    )
    p_switch.add_argument(
        "--model",
        default=None,
        help="model name for all three tiers (haiku/sonnet/opus)",
    )
    p_switch.add_argument(
        "--haiku",
        default=None,
        help="haiku-tier model — must be given together with --sonnet/--opus",
    )
    p_switch.add_argument(
        "--sonnet",
        default=None,
        help="sonnet-tier model — must be given together with --haiku/--opus",
    )
    p_switch.add_argument(
        "--opus",
        default=None,
        help="opus-tier model — must be given together with --haiku/--sonnet",
    )
    p_switch.add_argument(
        "--subagent-model",
        default=None,
        help="CLAUDE_CODE_SUBAGENT_MODEL override (omit to leave it unset)",
    )
    p_switch.add_argument(
        "--base-url",
        default=None,
        help="backend URL for a provider with no address in the registry "
        "(e.g. litellm: http://localhost:4000); a bare host/IP like "
        "78.47.183.125 is auto-completed",
    )
    p_switch.add_argument(
        "--auth",
        default=None,
        choices=["secret"],
        help="override a provider's default auth mode to a secret token; "
        "only for providers that declare it overridable",
    )
    p_switch.add_argument(
        "--profile",
        default=None,
        help="token profile to use (e.g. work or personal)",
    )
    p_switch.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="restore settings.json's env block from a backup slot instead of patching",
    )
    p_switch.add_argument(
        "--slot",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="backup slot for --restore (1=newest, default: 1)",
    )
    p_switch.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="print the currently-active switch and exit (writes nothing)",
    )
    p_switch.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="skip the confirmation prompt (the TUI chip hot-applies either way)",
    )
    p_switch.set_defaults(func=_handle_switch)

    p_proxy = subparsers.add_parser(
        "proxy",
        help="turn the proxy in ~/.claude/settings.json on or off, set its "
        "address, or edit NO_PROXY (independent of `switch`, which owns the "
        "ANTHROPIC_* keys in the same file)",
        parents=[sub_flags],
    )
    p_proxy.add_argument(
        "action",
        metavar="on|off|toggle",
        nargs="?",
        choices=("on", "off", "toggle"),
        default=None,
        help="turn the proxy on, off, or flip it; omit to report status",
    )
    p_proxy.add_argument(
        "--url",
        default=None,
        help="set the proxy address, e.g. http://127.0.0.1:8118, and turn it "
        "on (validated first — Claude Code refuses to start on a URL it "
        "cannot parse, and does not support SOCKS)",
    )
    p_proxy.add_argument(
        "--no-proxy",
        dest="no_proxy",
        default=None,
        help="set the bypass list, e.g. 'localhost,127.0.0.1,.example.com' "
        "(space- or comma-separated, '*' bypasses everything); written to "
        "both NO_PROXY and no_proxy, and never touched by on/off",
    )
    p_proxy.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="print the current proxy state and exit (writes nothing)",
    )
    p_proxy.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="skip the confirmation prompt",
    )
    p_proxy.set_defaults(func=_handle_proxy)

    p_tui = subparsers.add_parser(
        "tui",
        help="open the arrow-key menu (also the bare `codehelper` default)",
        parents=[sub_flags],
    )
    p_tui.set_defaults(func=_handle_tui)

    return parser
