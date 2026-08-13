"""Interactive menu for creating and maintaining code-helper wrappers.

The TUI is deliberately a thin English-only front end over the CLI handlers.
It owns navigation and collects interactive values; ``parser.py`` remains the
single place that installs wrappers and updates the credential cache.

Navigation has no pause screens: every selection enters another menu and every
submenu has an explicit ``Back`` entry (Esc/q has the same meaning). Results and
validation errors are acknowledged so they remain visible. Ctrl-C leaves the TUI
from any depth.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

__all__ = ["run_tui"]

_ADD = "add"
_SETTINGS = "settings"
_PROFILE = "profile"
_SET_DEFAULT = "set-default"
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
    return TuiSession(args).run()


class TuiSession:
    """One interactive UI session — replaces the former ``run_tui`` closure.

    The 24 nested functions inside ``run_tui`` shared state through three
    implicit channels: the ``args`` Namespace (mutated 21 times before
    dispatch), the ``_tab_provider_cache`` dict, and Python closures. This
    class makes those channels EXPLICIT as ``self.args`` and
    ``self._tab_provider``/``self._tab_label``, so the coupling is visible and
    the methods are individually testable. ``run`` is the main loop; every
    ``_run_*`` is a screen; the ``_``-prefixed helpers are UI primitives
    shared across screens.

    The dispatch contract is unchanged: ``_run(handler)`` still calls
    ``handler(self.args)``, so the CLI handlers (``_handle_add`` etc.) see
    the same Namespace they always did. The TUI's mutation of ``self.args``
    before each dispatch is the load-bearing bridge — see P2.4's
    ``AddRequest``/``EditTokenRequest``/``SetDefaultRequest`` for the typed
    view the handlers themselves consume.
    """

    __slots__ = ("args", "_tab_provider", "_tab_label")

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # `_resolve_tab_provider` does real I/O (`state.json` + a profile
        # scan), and `_main_prompt`/`_profile_row_label` are callables
        # re-evaluated every redraw frame — so they read this cache instead
        # of re-deriving per frame. Refreshed once per main-loop iteration
        # and again by `_on_tab` the moment Tab changes the selection.
        self._tab_provider: str | None = None
        self._tab_label: str = ""

    # --- UI primitives ---------------------------------------------------

    def _pick(
        self,
        items: Sequence[tuple[str, str]],
        prompt: str | Callable[[], str],
        *,
        exit_word: str = "back",
        on_tab: Callable[[], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        tab_provider: str | None = None,
    ) -> str:
        from code_helper.cli.menu import MenuCancelled, Section, select_from_menu

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

    def _notify(self, text: str) -> None:
        """Show a result until it is acknowledged instead of clearing it."""
        from code_helper.cli.menu import press_any_key

        if text:
            print(text)
        press_any_key("Press any key to continue...")

    def _run(self, handler) -> None:
        """Dispatch a CLI handler and preserve its user-facing output."""
        import contextlib
        import io

        from code_helper.errors import CodeHelperError

        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                handler(self.args)
        except CodeHelperError as exc:
            output.write(f"error: {exc}\n")
        self._notify(output.getvalue().rstrip())

    def _read_text(self, prompt: str) -> str | None:
        from code_helper.cli.menu import MenuCancelled, read_line

        try:
            return read_line(prompt)
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return None

    def _read_token(self, prompt: str) -> str | None:
        from code_helper.cli.menu import MenuCancelled, read_line

        while True:
            try:
                token = read_line(prompt, secret=True)
            except MenuCancelled as exc:
                if exc.hard:
                    raise
                return None
            except ValueError as exc:
                self._notify(f"Invalid token: {exc}")
                continue
            if not token:
                self._notify("No token entered.")
                return None
            return token

    # --- profiles --------------------------------------------------------

    def _recover_default(self, provider_name: str) -> None:
        """Best-effort profile-cache recovery from an installed wrapper."""
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import (
            profile_names,
            seed_default_profile,
        )
        from code_helper.services.wrappers import (
            discover_managed,
            token_from_installed,
        )

        paths = Paths.default()
        if profile_names(paths, provider_name):
            return
        for name in discover_managed(paths):
            token = token_from_installed(paths, name, provider_name)
            if token:
                seed_default_profile(paths, provider_name, token)
                return

    def _new_profile(
        self, names: list[str], provider_name: str
    ) -> ProfileChoice | str | None:
        """Collect a new profile and token before model discovery."""
        from code_helper.services.profiles import (
            NewProfileOutcome,
            classify_new_profile,
            validate_new_profile_name,
        )
        from code_helper.services.secrets import DEFAULT_PROFILE

        if not names:
            token = self._read_token(f"Token for {provider_name}: ")
            return (DEFAULT_PROFILE, token, None, None) if token else None

        if len(names) == 1:
            old_name = names[0]
            renamed_old = self._read_text(f"Name for current profile ({old_name}): ")
            if renamed_old is None:
                return _BACK
            new_name = self._read_text("Name for new profile: ")
            if new_name is None:
                return _BACK
            # The naming decision is owned by services.profiles (issue #37,
            # P1.2): the same rules the CLI applies, but this TUI reacts with
            # print + return None and splits the outcomes into separate
            # messages — word-for-word unchanged.
            outcome = classify_new_profile(names, renamed_old, new_name)
            if outcome is NewProfileOutcome.EMPTY:
                self._notify("Profile names cannot be empty.")
                return None
            if outcome is NewProfileOutcome.SAME:
                self._notify("Profile names must be different.")
                return None
            if outcome is NewProfileOutcome.COLLISION_RENAMED:
                self._notify(f"Profile {renamed_old!r} already exists.")
                return None
            if outcome is NewProfileOutcome.COLLISION_NEW:
                self._notify(f"Profile {new_name!r} already exists.")
                return None
            token = self._read_token(f"Token for {provider_name} ({new_name}): ")
            return (new_name, token, old_name, renamed_old) if token else _BACK

        new_name = self._read_text("Name for new profile: ")
        if new_name is None:
            return _BACK
        outcome = validate_new_profile_name(new_name, names)
        if outcome is NewProfileOutcome.EMPTY:
            self._notify("Profile name cannot be empty.")
            return None
        if outcome is NewProfileOutcome.COLLISION_NEW:
            self._notify(f"Profile {new_name!r} already exists.")
            return None
        token = self._read_token(f"Token for {provider_name} ({new_name}): ")
        return (new_name, token, None, None) if token else _BACK

    def _select_profile(
        self, provider_name: str, *, editing: bool
    ) -> ProfileChoice | None:
        """Choose a profile, optionally replacing its token, or create one.

        A new/replaced token is intentionally kept only on the Namespace until
        the shared handler has installed the wrapper successfully.
        """
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import (
            DEFAULT_PROFILE,
            profile_names,
            valid_active_profile,
        )

        while True:
            self._recover_default(provider_name)
            names = list(profile_names(Paths.default(), provider_name))
            if not names:
                return self._new_profile(names, provider_name)

            # The stored active profile is the pre-selection: put it FIRST
            # with a marker so the cursor (index 0) lands on it.
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
            selected = self._pick(items, f"Token profile for {provider_name}:")
            if selected == _BACK:
                return None
            if selected == _NEW_PROFILE:
                created = self._new_profile(names, provider_name)
                if created == _BACK:
                    continue
                return created

            if editing:
                token = self._read_token(
                    f"New token for {provider_name} ({selected}): "
                )
                return (selected, token, None, None) if token else None

            action = self._pick(
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
            token = self._read_token(f"New token for {provider_name} ({selected}): ")
            if token:
                return selected, token, None, None

    # --- wrappers --------------------------------------------------------

    def _all_wrapper_specs(self):
        """Presets plus managed constructor wrappers, without duplicate names."""
        from code_helper.services.paths import Paths
        from code_helper.services.wrappers import (
            WRAPPERS,
            discover_managed,
            spec_from_installed,
        )

        paths = Paths.default()
        specs = []
        known = set()
        for spec in WRAPPERS:
            resolved = spec_from_installed(paths, spec.name) or spec
            specs.append(resolved)
            known.add(resolved.name)
        for name in discover_managed(paths):
            if name not in known and (spec := spec_from_installed(paths, name)):
                specs.append(spec)
                known.add(name)
        return specs

    def _wrapper_rows(self, paths) -> list:
        """Menu items for the main screen's wrapper list, grouped by agent."""
        from code_helper.cli.menu import Section
        from code_helper.services.model import AGENTS
        from code_helper.services.wrappers import (
            describe_all,
            valid_default_wrapper,
        )

        specs = self._all_wrapper_specs()
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
                    defaults={agent.name: valid_default_wrapper(paths, agent.name)},
                )
            )
        return rows

    def _resolve_spec(self, alias: str) -> object | None:
        """Resolve ``alias`` to a spec — installed wrapper first, else preset."""
        from code_helper.errors import CodeHelperError
        from code_helper.services.paths import Paths
        from code_helper.services.wrappers import get_spec, spec_from_installed

        try:
            return spec_from_installed(Paths.default(), alias) or get_spec(alias)
        except CodeHelperError as exc:
            self._notify(f"error: {exc}")
            return None

    # --- actions ---------------------------------------------------------

    def _on_token(self, alias: str) -> None:
        """Rotate the token of wrapper ``alias`` (``t`` on the main screen)."""
        from code_helper.cli.parser import _handle_edit_token

        spec = self._resolve_spec(alias)
        if spec is None:
            return
        if spec.auth != "secret":
            self._notify(f"{alias} has no editable token.")
            return
        profile = self._select_profile(spec.provider.name, editing=True)
        if profile is None:
            return
        self.args.name = alias
        (
            self.args.profile,
            self.args.profile_token,
            self.args.profile_rename_from,
            self.args.profile_rename_to,
        ) = profile
        self._run(_handle_edit_token)

    def _run_set_default(self, paths) -> None:
        """Patch ``~/.codex/config.toml`` with codex's default wrapper."""
        from code_helper.cli.parser import _handle_set_default
        from code_helper.services.model import BaseUrlPolicy
        from code_helper.services.wrappers import valid_default_wrapper

        alias = valid_default_wrapper(paths, "codex")
        if alias is None:
            self._notify(
                "error: no default wrapper set for codex — pick an installed "
                "codex wrapper (Enter on its row) first"
            )
            return
        spec = self._resolve_spec(alias)
        if spec is None:
            return
        self.args.agent = spec.agent.name
        self.args.provider = spec.provider.name
        self.args.model = spec.model
        # FIXED providers reject a --base-url even when it equals their own
        # registry default; forward the resolved address only for
        # REQUIRED/OVERRIDABLE (issue #30 follow-up).
        self.args.base_url = (
            spec.provider.base_url
            if spec.provider.base_url_policy is not BaseUrlPolicy.FIXED
            else None
        )
        self.args.restore = False
        self.args.slot = None
        self.args.catalog_json = None
        self.args.force = False
        self._run(_handle_set_default)

    def _run_add(self) -> None:
        """Create a wrapper through provider → profile → model → agent → name."""
        from code_helper.cli.parser import _handle_add
        from code_helper.errors import CodeHelperError
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
        from code_helper.services.secrets import token_for_discovery
        from code_helper.services.spec import suggest_alias

        # An OVERRIDABLE provider gets a SECOND row rather than an extra
        # screen: `ollama` (default) and `ollama (with token)` both pick the
        # same provider.
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
            selection = self._pick(
                [*provider_items, (_BACK, "Back")], "Select a provider:"
            )
            if selection == _BACK:
                return
            provider, want_secret_auth = provider_choices[selection]
            provider = with_auth(provider, want_secret=want_secret_auth)

            typed_url: str | None = None
            if provider.base_url_policy is not BaseUrlPolicy.FIXED:
                default = f" [{provider.base_url}]" if provider.base_url else ""
                typed_url = self._read_text(f"Base URL for {provider.name}{default}: ")
                if typed_url is None:
                    continue
                if provider.base_url_policy is BaseUrlPolicy.REQUIRED and not typed_url:
                    continue
                try:
                    provider = with_base_url(provider, typed_url or None)
                except CodeHelperError as exc:
                    self._notify(f"error: {exc}")
                    continue

            # Profile is the previous level for model selection.
            while True:
                profile: ProfileChoice | None
                if provider.auth == "secret":
                    profile = self._select_profile(provider.name, editing=False)
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
                    model = self._pick(model_items, prompt)
                    if model == _BACK:
                        break
                    if model == "__custom__":
                        typed_model = self._read_text("Model: ")
                        if typed_model is None or not typed_model:
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
                        agent_name = self._pick(
                            [*agent_items, (_BACK, "Back")], "Select an agent:"
                        )
                        if agent_name == _BACK:
                            break
                        try:
                            default_alias = suggest_alias(
                                model, agent_name, profile_name or None
                            )
                        except CodeHelperError as exc:
                            self._notify(f"error: {exc}")
                            break
                        alias = self._read_text(f"Command name [{default_alias}]: ")
                        if alias is None:
                            continue

                        self.args.name = None
                        self.args.force = False
                        self.args.agent = agent_name
                        self.args.provider = provider.name
                        self.args.model = model
                        self.args.alias = alias or default_alias
                        self.args.shape = None
                        self.args.base_url = typed_url
                        self.args.auth = "secret" if want_secret_auth else None
                        self.args.profile = profile_name or None
                        self.args.profile_token = profile_token
                        self.args.profile_rename_from = rename_from
                        self.args.profile_rename_to = rename_to
                        self._run(_handle_add)
                        return

    def _run_settings(self) -> None:
        while True:
            debug = getattr(self.args, "debug", False)
            choice = self._pick(
                [
                    ("debug", f"Debug: {'on' if debug else 'off'}"),
                    (
                        "dry-run",
                        f"Dry-run: {'on' if getattr(self.args, 'dry_run', False) else 'off'}",
                    ),
                    (_BACK, "Back"),
                ],
                "Settings:",
            )
            if choice == _BACK:
                return
            if choice == "debug":
                self.args.debug = not debug
            else:
                self.args.dry_run = not getattr(self.args, "dry_run", False)

    def _run_profile_screen(self) -> None:
        """Choose the active provider and its active profile (the Tab cycle)."""
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import (
            DEFAULT_PROFILE,
            profile_names,
            valid_active_profile,
        )
        from code_helper.services.state import set_active_selection

        paths = Paths.default()
        provider_items = [
            (p.name, f"{p.name} — {p.description}") for p in self._secret_providers()
        ]
        provider = self._pick(
            [*provider_items, (_BACK, "Back")], "Active profile provider:"
        )
        if provider == _BACK:
            return
        names = list(profile_names(paths, provider))
        if not names:
            self._notify(f"No profiles for {provider}.")
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
        choice = self._pick(items, f"Active profile for {provider}:")
        if choice == _BACK:
            return
        set_active_selection(paths, provider, choice)

    # --- active-label subsystem -----------------------------------------

    @staticmethod
    def _secret_providers() -> list:
        """Providers that can have named token profiles at all.

        NOT just ``p.auth == "secret"``: an OVERRIDABLE provider's registry
        entry keeps its default ``auth`` (``with_auth`` returns a RUNTIME
        copy, never mutates ``PROVIDERS``). Filtering on ``auth_policy``
        keeps the Profile screen and Tab's fallback scan able to find
        profiles cached under an OVERRIDABLE provider.
        """
        from code_helper.services.model import PROVIDERS, AuthPolicy

        return [
            p
            for p in PROVIDERS
            if p.auth == "secret" or p.auth_policy is AuthPolicy.OVERRIDABLE
        ]

    def _resolve_tab_provider(self) -> str | None:
        """Resolve which provider Tab/the header should act on."""
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import profile_names
        from code_helper.services.state import active_selection

        paths = Paths.default()
        selection = active_selection(paths)
        if selection is not None:
            stored, _ = selection
            if profile_names(paths, stored):
                return stored
        for provider in self._secret_providers():
            if profile_names(paths, provider.name):
                return provider.name
        return None

    def _tab_profile(self, provider: str) -> str | None:
        """The profile Tab currently shows/would land on for ``provider``."""
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import profile_names, valid_active_profile

        paths = Paths.default()
        stored = valid_active_profile(paths, provider)
        if stored:
            return stored
        names = profile_names(paths, provider)
        return names[0] if names else None

    def _active_label(self, tab_provider: str | None) -> str:
        """``"provider/profile"``, ``"provider"``, or ``""`` for header/row."""
        if not tab_provider:
            return ""
        profile = self._tab_profile(tab_provider)
        return f"{tab_provider}/{profile}" if profile else tab_provider

    def _on_tab(self) -> None:
        """Cycle the active profile of the current provider on Tab."""
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import profile_names, valid_active_profile
        from code_helper.services.state import set_active_selection

        paths = Paths.default()
        provider = self._tab_provider
        if not provider:
            return
        names = list(profile_names(paths, provider))
        if not names:
            return
        stored = valid_active_profile(paths, provider)
        idx = names.index(stored) if stored in names else -1
        set_active_selection(paths, provider, names[(idx + 1) % len(names)])
        self._refresh_active_label()

    def _refresh_active_label(self) -> None:
        """Resolve the active provider and its rendered label, once."""
        self._tab_provider = self._resolve_tab_provider()
        self._tab_label = self._active_label(self._tab_provider)

    def _main_prompt(self) -> str:
        """Live main-menu header, showing the active provider/profile."""
        return f"code-helper — {self._tab_label}" if self._tab_label else "code-helper"

    def _profile_row_label(self) -> str:
        return f"Profile: {self._tab_label}" if self._tab_label else "Profile"

    # --- main loop -------------------------------------------------------

    def run(self) -> int:
        """The main menu loop — show wrappers + service rows, dispatch."""
        from code_helper.cli.menu import MenuCancelled
        from code_helper.services.paths import Paths
        from code_helper.services.state import set_default_wrapper

        try:
            while True:
                self._refresh_active_label()
                tab_provider = self._tab_provider
                paths = Paths.default()
                # The wrapper list IS the main screen (issue #29): grouped by
                # agent via `Section`, with `●` on the default wrapper and `t`
                # for per-row token rotation. Enter makes a wrapper the default
                # for its agent; the service rows sit below the wrappers.
                choice = self._pick(
                    [
                        *self._wrapper_rows(paths),
                        (_ADD, "Add"),
                        (_PROFILE, self._profile_row_label),
                        (_SET_DEFAULT, "Apply codex default"),
                        (_SETTINGS, "Settings"),
                        (_QUIT, "Quit"),
                    ],
                    self._main_prompt,
                    exit_word="quit",
                    on_tab=self._on_tab,
                    tab_provider=tab_provider,
                    on_token=self._on_token,
                )
                if choice in (_BACK, _QUIT):
                    return 0
                if choice == _ADD:
                    self._run_add()
                elif choice == _PROFILE:
                    self._run_profile_screen()
                elif choice == _SET_DEFAULT:
                    self._run_set_default(paths)
                elif choice == _SETTINGS:
                    self._run_settings()
                else:
                    # A wrapper alias — Enter makes it the default for its
                    # agent. `set_default_wrapper` is the raw store (#28);
                    # `choice` came from `_wrapper_rows`, which only yields
                    # real aliases, so the write is always of a real alias.
                    from code_helper.services.wrappers import is_installed

                    spec = self._resolve_spec(choice)
                    if spec is None:
                        continue
                    if not is_installed(paths, choice):
                        self._notify("Wrapper not installed — use Add first.")
                        continue
                    set_default_wrapper(paths, spec.agent.name, choice)
                    self._notify(f"Default wrapper set to {choice}.")
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return 0
