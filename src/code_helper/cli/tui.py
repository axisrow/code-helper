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

_LIST = "list"
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


def _hint(numbered_count: int, *, exit_word: str, tab: bool = False) -> str:
    """Return the uniform navigation hint for a menu."""
    from code_helper.cli.menu import MAX_DIGIT_ITEMS

    usable = min(numbered_count, MAX_DIGIT_ITEMS)
    digits = "1" if usable == 1 else f"1-{usable}"
    hint = f"Up/Down · {digits} · Enter select · Esc {exit_word} · Ctrl-C quit"
    if tab:
        hint += " · Tab profile"
    return hint


def run_tui(args: argparse.Namespace) -> int:
    """Run the hierarchical interactive UI and always return a shell status."""
    from code_helper.cli.menu import MenuCancelled, read_line, select_from_menu
    from code_helper.cli.parser import _handle_add, _handle_edit_token
    from code_helper.errors import CodeHelperError, emit_error
    from code_helper.services.model import (
        AGENTS,
        PROVIDERS,
        BaseUrlPolicy,
        resolve_shape,
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
        active_profile,
        active_provider,
        set_active_profile,
        set_active_provider,
    )
    from code_helper.services.wrappers import (
        WRAPPERS,
        describe_all,
        discover_managed,
        get_spec,
        spec_from_installed,
        token_from_installed,
    )

    def _pick(
        items: Sequence[tuple[str, str]],
        prompt: str | Callable[[], str],
        *,
        exit_word: str = "back",
        on_tab: Callable[[], None] | None = None,
    ) -> str:
        numbered = sum(1 for value, _ in items if value not in (_BACK, _QUIT))
        try:
            return select_from_menu(
                items,
                prompt=prompt,
                hint=_hint(numbered, exit_word=exit_word, tab=on_tab is not None),
                on_tab=on_tab,
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

    def _run_list() -> None:
        """Browse wrappers; selecting a secret wrapper rotates its profile token."""
        while True:
            specs = _all_wrapper_specs()
            rows = describe_all(
                Paths.default(),
                specs,
                installed_word="installed",
                not_installed_word="not installed",
            )
            choice = _pick([*rows, (_BACK, "Back")], "Wrappers:")
            if choice == _BACK:
                return
            try:
                spec = spec_from_installed(Paths.default(), choice) or get_spec(choice)
            except CodeHelperError as exc:
                emit_error(exc, getattr(args, "debug", False))
                continue
            if spec.auth != "secret":
                _pick([(_BACK, "Back")], f"{choice} has no editable token.")
                continue
            profile = _select_profile(spec.provider.name, editing=True)
            if profile is None:
                continue
            args.name = choice
            (
                args.profile,
                args.profile_token,
                args.profile_rename_from,
                args.profile_rename_to,
            ) = profile
            _run(_handle_edit_token)

    def _run_add() -> None:
        """Create a wrapper through provider → profile → model → agent → name."""
        provider_items = [
            (provider.name, f"{provider.name} — {provider.description}")
            for provider in PROVIDERS
        ]
        while True:  # provider level
            provider_name = _pick(
                [*provider_items, (_BACK, "Back")], "Select a provider:"
            )
            if provider_name == _BACK:
                return
            provider = next(item for item in PROVIDERS if item.name == provider_name)

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
            (p.name, f"{p.name} — {p.description}")
            for p in PROVIDERS
            if p.auth == "secret"
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
        set_active_provider(paths, provider)
        set_active_profile(paths, provider, choice)

    def _main_prompt() -> str:
        """Live main-menu header, showing the active provider/profile."""
        paths = Paths.default()
        provider = active_provider(paths)
        if not provider:
            return "code-helper"
        profile = valid_active_profile(paths, provider)
        if profile:
            return f"code-helper — {provider}/{profile}"
        return f"code-helper — {provider}"

    def _on_tab() -> None:
        """Cycle the active profile of the current provider on Tab."""
        paths = Paths.default()
        provider = active_provider(paths)
        if not provider:
            return
        names = list(profile_names(paths, provider))
        if not names:
            return
        current = active_profile(paths, provider)
        idx = names.index(current) if current in names else -1
        set_active_profile(paths, provider, names[(idx + 1) % len(names)])

    try:
        while True:
            choice = _pick(
                [
                    (_LIST, "List"),
                    (_ADD, "Add"),
                    (_PROFILE, "Profile"),
                    (_SETTINGS, "Settings"),
                    (_QUIT, "Quit"),
                ],
                _main_prompt,
                exit_word="quit",
                on_tab=_on_tab,
            )
            if choice in (_BACK, _QUIT):
                return 0
            if choice == _LIST:
                _run_list()
            elif choice == _ADD:
                _run_add()
            elif choice == _PROFILE:
                _run_profile_screen()
            else:
                _run_settings()
    except MenuCancelled as exc:
        if exc.hard:
            raise
        return 0
