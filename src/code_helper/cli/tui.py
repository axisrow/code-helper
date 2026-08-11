"""Interactive menu for creating and maintaining code-helper wrappers.

The TUI is deliberately a thin English-only front end over the CLI handlers.
It owns navigation and collects interactive values; ``parser.py`` remains the
single place that installs wrappers and updates the credential cache.

There are no pause screens.  Every selection enters another menu and every
submenu has an explicit ``Back`` entry (Esc/q has the same meaning).  Ctrl-C
leaves the TUI from any depth.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

__all__ = ["run_tui"]

_ADD = "add"
_SETTINGS = "settings"
_PROFILE = "profile"
_QUIT = "quit"
_BACK = "__back__"
_NEW_PROFILE = "__new_profile__"
_USE_PROFILE = "__use_profile__"
_REPLACE_TOKEN = "__replace_token__"

# profile name, token typed in this flow, old profile name, new old-profile name
ProfileChoice = tuple[str, str | None, str | None, str | None]


def _hint(
    numbered_count: int,
    *,
    exit_word: str,
    tab_provider: str | None = None,
    has_token_key: bool = False,
) -> str:
    """Return the uniform navigation hint for a menu.

    ``tab_provider``, when given, names the provider Tab would cycle — e.g.
    ``"zai"`` renders ``· Tab: zai profile`` — so the hint says what Tab does
    rather than just that it does something. It cycles *profiles within one
    provider*, never providers themselves (that is what the Profile screen is
    for), and this makes the distinction visible instead of implied.

    ``has_token_key=True`` adds ``· t: token`` — the main screen (issue #29)
    binds ``t`` to per-row token rotation, and the hint must say so. It is
    always shown on the main screen even when the cursor is on a non-secret
    wrapper, because ``t`` there is a silent no-op (no dead-end screen), and
    hiding the hint only on non-secret rows would make it flicker as the user
    moves the cursor.
    """
    from code_helper.cli.menu import MAX_DIGIT_ITEMS

    usable = min(numbered_count, MAX_DIGIT_ITEMS)
    digits = "1" if usable == 1 else f"1-{usable}"
    hint = f"Up/Down · {digits} · Enter select · Esc {exit_word} · Ctrl-C quit"
    if tab_provider:
        hint += f" · Tab: {tab_provider} profile"
    if has_token_key:
        hint += " · t: token"
    return hint


def run_tui(args: argparse.Namespace) -> int:
    """Run the hierarchical interactive UI and always return a shell status."""
    from code_helper.cli.menu import MenuCancelled, Section, read_line, select_from_menu
    from code_helper.cli.parser import _handle_add, _handle_edit_token
    from code_helper.errors import CodeHelperError, emit_error
    from code_helper.services.model import (
        AGENTS,
        PROVIDERS,
        AuthPolicy,
        BaseUrlPolicy,
        Provider,
        resolve_shape,
        with_auth,
        with_base_url,
    )
    from code_helper.services.models_api import list_models
    from code_helper.services.paths import Paths
    from code_helper.services.secrets import (
        DEFAULT_PROFILE,
        profile_names,
        seed_default_profile,
        token_for_discovery,
        valid_active_profile,
    )
    from code_helper.services.spec import suggest_alias
    from code_helper.services.state import (
        active_selection,
        set_active_selection,
        set_default_wrapper,
    )
    from code_helper.services.wrappers import (
        WRAPPERS,
        describe_all,
        discover_managed,
        get_spec,
        spec_from_installed,
        token_from_installed,
        valid_default_wrapper,
    )

    def _pick(
        items: Sequence[tuple[str, str]],
        prompt: str | Callable[[], str],
        *,
        exit_word: str = "back",
        on_tab: Callable[[], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        tab_provider: str | None = None,
    ) -> str:
        # `items` may now contain `Section` headers (issue #29 main screen) —
        # they are not `(value, label)` pairs, so skip them when counting
        # numbered rows. `select_from_menu` does the same via `isinstance`.
        numbered = sum(
            1
            for entry in items
            if not isinstance(entry, Section) and entry[0] not in (_BACK, _QUIT)
        )
        try:
            return select_from_menu(
                items,
                prompt=prompt,
                hint=_hint(
                    numbered,
                    exit_word=exit_word,
                    tab_provider=tab_provider,
                    has_token_key=on_token is not None,
                ),
                on_tab=on_tab,
                on_token=on_token,
                unnumbered=frozenset({_BACK, _QUIT}),
                clear=True,
            )
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return _BACK

    def _run(handler) -> None:
        try:
            handler(args)
        except CodeHelperError as exc:
            emit_error(exc, getattr(args, "debug", False))

    def _read_text(prompt: str) -> str | None:
        try:
            return read_line(prompt)
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return None

    def _read_token(prompt: str) -> str | None:
        while True:
            try:
                token = read_line(prompt, secret=True)
            except MenuCancelled as exc:
                if exc.hard:
                    raise
                return None
            except ValueError as exc:
                print(f"Invalid token: {exc}")
                continue
            if not token:
                print("No token entered.")
                return None
            return token

    def _recover_default(provider_name: str) -> None:
        """Best-effort profile-cache recovery from an installed wrapper."""
        paths = Paths.default()
        if profile_names(paths, provider_name):
            return
        for name in discover_managed(paths):
            token = token_from_installed(paths, name, provider_name)
            if token:
                seed_default_profile(paths, provider_name, token)
                return

    def _new_profile(
        names: list[str], provider_name: str
    ) -> ProfileChoice | str | None:
        """Collect a new profile and token before model discovery."""
        if not names:
            token = _read_token(f"Token for {provider_name}: ")
            return (DEFAULT_PROFILE, token, None, None) if token else None

        if len(names) == 1:
            old_name = names[0]
            renamed_old = _read_text(f"Name for current profile ({old_name}): ")
            if renamed_old is None:
                return _BACK
            new_name = _read_text("Name for new profile: ")
            if new_name is None:
                return _BACK
            if not renamed_old or not new_name:
                print("Profile names cannot be empty.")
                return None
            if renamed_old == new_name:
                print("Profile names must be different.")
                return None
            if renamed_old in names and renamed_old != old_name:
                print(f"Profile {renamed_old!r} already exists.")
                return None
            if new_name in names:
                print(f"Profile {new_name!r} already exists.")
                return None
            token = _read_token(f"Token for {provider_name} ({new_name}): ")
            return (new_name, token, old_name, renamed_old) if token else _BACK

        new_name = _read_text("Name for new profile: ")
        if new_name is None:
            return _BACK
        if not new_name:
            print("Profile name cannot be empty.")
            return None
        if new_name in names:
            print(f"Profile {new_name!r} already exists.")
            return None
        token = _read_token(f"Token for {provider_name} ({new_name}): ")
        return (new_name, token, None, None) if token else _BACK

    def _select_profile(provider_name: str, *, editing: bool) -> ProfileChoice | None:
        """Choose a profile, optionally replacing its token, or create one.

        A new/replaced token is intentionally kept only on the Namespace until
        the shared handler has installed the wrapper successfully.
        """
        while True:
            _recover_default(provider_name)
            names = list(profile_names(Paths.default(), provider_name))
            if not names:
                return _new_profile(names, provider_name)

            # The stored active profile is the pre-selection: put it FIRST with
            # a marker so the cursor (index 0) lands on it and Enter accepts it,
            # while arrows/digits can still pick another — a default, not a trap.
            # A stale/missing profile yields None and the normal order stands.
            active = valid_active_profile(Paths.default(), provider_name)
            items: list[tuple[str, str]] = []
            for name in names:
                label = "default" if name == DEFAULT_PROFILE else name
                if name == active:
                    label = f"{label} (active)"
                    items.insert(0, (name, label))
                else:
                    items.append((name, label))
            items.extend(((_NEW_PROFILE, "Add profile"), (_BACK, "Back")))
            selected = _pick(items, f"Token profile for {provider_name}:")
            if selected == _BACK:
                return None
            if selected == _NEW_PROFILE:
                created = _new_profile(names, provider_name)
                if created == _BACK:
                    continue
                return created

            if editing:
                token = _read_token(f"New token for {provider_name} ({selected}): ")
                return (selected, token, None, None) if token else None

            action = _pick(
                [
                    (_USE_PROFILE, "Use profile"),
                    (_REPLACE_TOKEN, "Replace token"),
                    (_BACK, "Back"),
                ],
                f"Profile: {selected}",
            )
            if action == _BACK:
                continue
            if action == _USE_PROFILE:
                return selected, None, None, None
            token = _read_token(f"New token for {provider_name} ({selected}): ")
            if token:
                return selected, token, None, None

    def _all_wrapper_specs():
        """Presets plus managed constructor wrappers, without duplicate names."""
        paths = Paths.default()
        specs = list(WRAPPERS)
        known = {spec.name for spec in specs}
        for name in discover_managed(paths):
            if name not in known and (spec := spec_from_installed(paths, name)):
                specs.append(spec)
                known.add(name)
        return specs

    def _wrapper_rows(paths: Paths) -> list:
        """Menu items for the main screen's wrapper list, grouped by agent.

        Each agent that has ANY wrapper gets a non-selectable ``Section``
        header (issue #27) followed by its wrappers' rows; the rows come from
        :func:`describe_all` with a live ``defaults`` map so the one wrapper
        that is ``default_wrapper`` for its agent is prefixed with ``●``
        (issue #29). The map is resolved ONCE here (two ``valid_default_wrapper``
        calls — one per agent — not one per row), because the main screen
        rebuilds it every loop iteration and re-reading ``state.json`` per
        row would multiply I/O needlessly. Agents with no wrappers are skipped
        (no empty ``Section``), so a fresh install shows only the agents that
        actually have presets or installed wrappers.
        """
        specs = _all_wrapper_specs()
        rows: list = []
        for agent in AGENTS:
            agent_specs = [s for s in specs if s.agent.name == agent.name]
            if not agent_specs:
                continue
            rows.append(Section(agent.name))
            rows.extend(
                describe_all(
                    paths,
                    agent_specs,
                    installed_word="installed",
                    not_installed_word="not installed",
                    # Resolve the default only for agents that actually have
                    # wrappers — a per-agent ``valid_default_wrapper`` call
                    # reads ``state.json``, so skipping empty agents avoids
                    # wasted I/O on the main screen's hot render path.
                    defaults={agent.name: valid_default_wrapper(paths, agent.name)},
                )
            )
        return rows

    def _resolve_spec(alias: str) -> object | None:
        """Resolve ``alias`` to a spec — installed wrapper first, else preset.

        Returns ``None`` (after emitting the error) when resolution fails, so
        callers bail out without repeating the try/except. Shared by ``_on_token``
        and the main screen's Enter-on-wrapper branch, which both need the same
        "installed wrapper takes precedence over a same-named preset" lookup.
        """
        try:
            return spec_from_installed(Paths.default(), alias) or get_spec(alias)
        except CodeHelperError as exc:
            emit_error(exc, getattr(args, "debug", False))
            return None

    def _on_token(alias: str) -> None:
        """Rotate the token of wrapper ``alias`` (the ``t`` key on the main screen).

        Reuses the exact wiring the old ``_run_list`` had for a selected
        wrapper — profile pick then ``_handle_edit_token`` — moved here
        because #29 inlines the wrapper list into the main screen and binds
        token rotation to ``t`` instead of Enter. Enter is now reserved for
        ``set_default_wrapper``. A non-secret wrapper is a SILENT no-op: the
        old dead-end ``"{alias} has no editable token."`` screen is gone, so
        ``t`` on a non-secret row simply does nothing rather than trapping the
        user in a one-item ``Back`` menu.
        """
        spec = _resolve_spec(alias)
        if spec is None:
            return
        if spec.auth != "secret":
            return  # silent no-op — no dead-end screen (issue #29)
        profile = _select_profile(spec.provider.name, editing=True)
        if profile is None:
            return
        args.name = alias
        (
            args.profile,
            args.profile_token,
            args.profile_rename_from,
            args.profile_rename_to,
        ) = profile
        _run(_handle_edit_token)

    def _run_add() -> None:
        """Create a wrapper through provider → profile → model → agent → name."""
        # An OVERRIDABLE provider (auth_policy) gets a SECOND row rather than
        # an extra interstitial screen: `ollama` (registry default, no token)
        # and `ollama (with token)` (--auth secret equivalent) both pick the
        # same provider — so the common "just want ollama" path stays a
        # single Enter, exactly as before this axis existed. Each row's menu
        # value is its own unique key into `provider_choices`, a direct
        # lookup rather than string-encoding the auth choice into the value
        # itself (`select_from_menu` already separates `value` from the
        # rendered `label` for exactly this reason).
        provider_items: list[tuple[str, str]] = []
        provider_choices: dict[str, tuple[Provider, bool]] = {}
        for provider in PROVIDERS:
            provider_items.append(
                (provider.name, f"{provider.name} — {provider.description}")
            )
            provider_choices[provider.name] = (provider, False)
            if (
                provider.auth_policy is AuthPolicy.OVERRIDABLE
                and provider.auth != "secret"
            ):
                secret_value = f"{provider.name}:secret"
                provider_items.append(
                    (
                        secret_value,
                        f"{provider.name} (with token) — reverse proxy / cloud auth",
                    )
                )
                provider_choices[secret_value] = (provider, True)
        while True:  # provider level
            selection = _pick([*provider_items, (_BACK, "Back")], "Select a provider:")
            if selection == _BACK:
                return
            provider, want_secret_auth = provider_choices[selection]
            provider = with_auth(provider, want_secret=want_secret_auth)

            typed_url: str | None = None
            if provider.base_url_policy is not BaseUrlPolicy.FIXED:
                default = f" [{provider.base_url}]" if provider.base_url else ""
                typed_url = _read_text(f"Base URL for {provider.name}{default}: ")
                if typed_url is None:
                    continue
                if provider.base_url_policy is BaseUrlPolicy.REQUIRED and not typed_url:
                    continue
                try:
                    provider = with_base_url(provider, typed_url or None)
                except CodeHelperError as exc:
                    emit_error(exc, getattr(args, "debug", False))
                    continue

            # Profile is the previous level for model selection. If it is
            # cancelled, restart at provider rather than jumping to main.
            while True:
                profile: ProfileChoice | None
                if provider.auth == "secret":
                    profile = _select_profile(provider.name, editing=False)
                    if profile is None:
                        break
                else:
                    profile = ("", None, None, None)

                profile_name, profile_token, rename_from, rename_to = profile
                discovery_token = profile_token or token_for_discovery(
                    Paths.default(), provider, profile_name=profile_name or None
                )
                result = list_models(provider, token=discovery_token)
                model_items = [(model, model) for model in result.models]
                model_items.append(("__custom__", "Enter model manually"))
                model_items.append((_BACK, "Back"))
                prompt = f"Select a model for {provider.name}:"
                if not result.ok:
                    prompt = (
                        f"Model discovery unavailable for {provider.name}; "
                        "enter a model:"
                    )

                # Model is the previous level for agent selection.
                while True:
                    model = _pick(model_items, prompt)
                    if model == _BACK:
                        break
                    if model == "__custom__":
                        typed_model = _read_text("Model: ")
                        if typed_model is None:
                            continue
                        if not typed_model:
                            continue
                        model = typed_model

                    agent_items: list[tuple[str, str]] = []
                    for agent in AGENTS:
                        try:
                            resolve_shape(agent, provider)
                        except CodeHelperError:
                            continue
                        agent_items.append(
                            (agent.name, f"{agent.name} — {agent.description}")
                        )

                    # Agent is the previous level for alias input.
                    while True:
                        agent_name = _pick(
                            [*agent_items, (_BACK, "Back")], "Select an agent:"
                        )
                        if agent_name == _BACK:
                            break
                        try:
                            default_alias = suggest_alias(
                                model, agent_name, profile_name or None
                            )
                        except CodeHelperError as exc:
                            emit_error(exc, getattr(args, "debug", False))
                            break
                        alias = _read_text(f"Command name [{default_alias}]: ")
                        if alias is None:
                            continue

                        args.name = None
                        args.agent = agent_name
                        args.provider = provider.name
                        args.model = model
                        args.alias = alias or default_alias
                        args.shape = None
                        args.base_url = typed_url
                        args.auth = "secret" if want_secret_auth else None
                        args.profile = profile_name or None
                        args.profile_token = profile_token
                        args.profile_rename_from = rename_from
                        args.profile_rename_to = rename_to
                        _run(_handle_add)
                        return
                    # Back from alias/agent level returns to model selection.
                # Back from model returns to profile selection.
            # Back from profile returns to provider selection.

    def _run_settings() -> None:
        while True:
            debug = getattr(args, "debug", False)
            choice = _pick(
                [
                    ("debug", f"Debug: {'on' if debug else 'off'}"),
                    (_BACK, "Back"),
                ],
                "Settings:",
            )
            if choice == _BACK:
                return
            args.debug = not debug

    def _run_profile_screen() -> None:
        """Choose the active provider and its active profile (the Tab cycle)."""
        paths = Paths.default()
        provider_items = [
            (p.name, f"{p.name} — {p.description}") for p in _secret_providers()
        ]
        provider = _pick([*provider_items, (_BACK, "Back")], "Active profile provider:")
        if provider == _BACK:
            return
        names = list(profile_names(paths, provider))
        if not names:
            print(f"No profiles for {provider}.")
            return
        current = valid_active_profile(paths, provider)
        items = [
            (
                name,
                f"{'default' if name == DEFAULT_PROFILE else name}"
                f"{' (active)' if name == current else ''}",
            )
            for name in names
        ]
        items.extend([(_BACK, "Back")])
        choice = _pick(items, f"Active profile for {provider}:")
        if choice == _BACK:
            return
        set_active_selection(paths, provider, choice)

    def _secret_providers() -> list:
        """Providers that can have named token profiles at all.

        NOT just ``p.auth == "secret"``: an OVERRIDABLE provider's registry
        entry always keeps its default ``auth`` (``with_auth`` returns a
        RUNTIME copy, never mutates ``PROVIDERS`` — see ``services/model.py``)
        — so ``ollama`` stays ``auth="literal"`` in this list even after an
        `add --auth secret` install has cached real profiles for it under
        that same provider name. Filtering on ``auth_policy`` instead of the
        registry's snapshot ``auth`` is what keeps the Profile screen and
        Tab's fallback scan able to find those profiles.
        """
        return [
            p
            for p in PROVIDERS
            if p.auth == "secret" or p.auth_policy is AuthPolicy.OVERRIDABLE
        ]

    def _resolve_tab_provider() -> str | None:
        """Resolve which provider Tab/the header should act on.

        The stored active selection is preferred, but it is only ever
        *written* from the Profile screen (`_run_profile_screen`) or a prior
        Tab press — on a fresh install (or after credentials were cleared)
        nothing has ever written it, which used to make Tab a permanent
        no-op with no indication why. Falling back to whichever secret
        provider already has cached profiles makes Tab work immediately,
        matching issue #23's intent of skipping the manual walk rather than
        requiring one first.
        """
        paths = Paths.default()
        selection = active_selection(paths)
        if selection is not None:
            stored, _ = selection
            if profile_names(paths, stored):
                return stored
        for provider in _secret_providers():
            if profile_names(paths, provider.name):
                return provider.name
        return None

    def _tab_profile(provider: str) -> str | None:
        """The profile Tab currently shows/would land on for ``provider``.

        Mirrors ``_on_tab``'s own fallback: a stored-but-stale or never-set
        active profile is treated as "before the first profile", so this
        agrees with what one Tab press would select — without writing
        anything (``valid_active_profile`` only reads; nothing here has the
        side effect ``_on_tab`` has via ``set_active_selection``).
        """
        paths = Paths.default()
        stored = valid_active_profile(paths, provider)
        if stored:
            return stored
        names = profile_names(paths, provider)
        return names[0] if names else None

    def _active_label(tab_provider: str | None) -> str:
        """``"provider/profile"``, ``"provider"``, or ``""`` for the header
        and the Profile row — the one place both derive their text from.

        Takes the already-resolved provider rather than re-resolving it
        (see the cache in the main loop below) — ``_resolve_tab_provider``
        reads ``state.json`` and, on its fallback path, loops every secret
        provider's cached profiles, so re-deriving it per caller here would
        turn one main-menu redraw into several rounds of that I/O.
        """
        if not tab_provider:
            return ""
        profile = _tab_profile(tab_provider)
        return f"{tab_provider}/{profile}" if profile else tab_provider

    def _on_tab() -> None:
        """Cycle the active profile of the current provider on Tab."""
        paths = Paths.default()
        provider = _tab_provider_cache["value"]
        if not provider:
            return
        names = list(profile_names(paths, provider))
        if not names:
            return
        # A valid stored profile advances to the next; a stale or never-set
        # one is "before the first profile", so Tab lands on names[0] (the
        # same frame the header already claims) rather than skipping it.
        stored = valid_active_profile(paths, provider)
        idx = names.index(stored) if stored in names else -1
        set_active_selection(paths, provider, names[(idx + 1) % len(names)])
        # The selection just changed — refresh so the header/row/hint
        # reflect it on the next read instead of the pre-Tab provider.
        _refresh_active_label()

    # `_resolve_tab_provider()` does real I/O (a `state.json` read and,
    # on its fallback path, a `profile_names` scan of every secret
    # provider), and `_tab_profile` beneath `_active_label` reads the
    # profile cache too — too expensive to re-run on every menu redraw
    # frame. `_main_prompt` and the Profile row label are BOTH callables
    # re-evaluated every frame (see CLAUDE.md's callable-prompt contract),
    # so they read this cache instead of re-deriving the label themselves;
    # the cache is refreshed once per main-loop iteration and again by
    # `_on_tab` the moment Tab actually changes the selection. Storing the
    # fully-rendered label (not just the provider) lets header and row share
    # ONE I/O pass per refresh instead of each doing its own
    # `valid_active_profile`/`profile_names` read on every frame.
    _tab_provider_cache: dict[str, str | None] = {"value": None, "label": ""}

    def _refresh_active_label() -> None:
        """Resolve the active provider and its rendered label, once.

        Called once per main-loop iteration and by `_on_tab` after it
        changes the selection; both the header and the Profile row read the
        cached ``label`` from then on instead of re-deriving it per frame.
        """
        value = _resolve_tab_provider()
        _tab_provider_cache["value"] = value
        _tab_provider_cache["label"] = _active_label(value)

    def _main_prompt() -> str:
        """Live main-menu header, showing the active provider/profile."""
        label = _tab_provider_cache["label"]
        return f"code-helper — {label}" if label else "code-helper"

    def _profile_row_label() -> str:
        label = _tab_provider_cache["label"]
        return f"Profile: {label}" if label else "Profile"

    try:
        while True:
            # Resolved ONCE per loop iteration and reused for the row label,
            # the hint gate, and (via the cached label) the live header and
            # Profile row — see `_refresh_active_label`/`_main_prompt` above
            # for why re-deriving it per reader would multiply the I/O.
            _refresh_active_label()
            tab_provider = _tab_provider_cache["value"]
            paths = Paths.default()
            # The wrapper list IS the main screen now (issue #29): grouped by
            # agent via `Section`, with `●` on the default wrapper and `t` for
            # per-row token rotation. Enter on a wrapper row makes it the
            # default for its agent (`set_default_wrapper`); the service rows
            # (`Add`/`Profile`/`Settings`/`Quit`) sit below the wrappers.
            choice = _pick(
                [
                    *_wrapper_rows(paths),
                    (_ADD, "Add"),
                    (_PROFILE, _profile_row_label),
                    (_SETTINGS, "Settings"),
                    (_QUIT, "Quit"),
                ],
                _main_prompt,
                exit_word="quit",
                on_tab=_on_tab,
                tab_provider=tab_provider,
                on_token=_on_token,
            )
            if choice in (_BACK, _QUIT):
                return 0
            if choice == _ADD:
                _run_add()
            elif choice == _PROFILE:
                _run_profile_screen()
            elif choice == _SETTINGS:
                _run_settings()
            else:
                # A wrapper alias — Enter makes it the default for its agent.
                # `set_default_wrapper` is the raw store (#28); it does not
                # validate, but `choice` came straight from `_wrapper_rows`,
                # which only yields aliases that exist as presets or installed
                # managed wrappers, so the write is always of a real alias.
                spec = _resolve_spec(choice)
                if spec is None:
                    continue
                set_default_wrapper(paths, spec.agent.name, choice)
    except MenuCancelled as exc:
        if exc.hard:
            raise
        return 0
