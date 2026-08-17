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
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from code_helper.services.paths import Paths

__all__ = ["run_tui"]

_ADD = "add"
_SETTINGS = "settings"
_PROFILE = "profile"
_HELP = "help"
_QUIT = "quit"
_BACK = "__back__"
_NEW_PROFILE = "__new_profile__"
_USE_PROFILE = "__use_profile__"
_REPLACE_TOKEN = "__replace_token__"

#: Prefix of a main-screen chipset row's value. The row namespace is what
#: gives Enter its meaning (``agent:<name>`` applies the highlighted chip,
#: anything else is a wrapper alias) — the same dispatch-on-prefix idiom
#: ``token:``/``remove:`` already use, rather than a second mode flag.
_AGENT_ROW = "agent:"

#: The chip that means "no override" — displayed on every agent row, applied
#: through that agent's own reset mechanism (see :class:`_AgentBackend`).
_NATIVE_CHIP = "native"

#: The trailing ACTION chip on every agent row — not a backend, never
#: "applied", Enter opens Add pre-scoped to that row's agent instead of
#: switching to anything. Exists so an agent with no wrappers yet (e.g. a
#: fresh install's `codex` row, which otherwise renders as a single bare
#: `native` chip) advertises how to get its first one, instead of reading as
#: broken. See ``CLAUDE.md``'s amended chip-strip rule: a row carries backend
#: chips plus exactly one of these, visually distinct, never applied.
_ADD_CHIP = "+ add"

#: SGR codes for the chip strip: reverse video marks the chip under the
#: cursor (fzf/less-style), bold marks the chip currently applied when the
#: cursor is elsewhere, dim marks the action chip so it reads as "not a
#: backend" even before you notice it never carries a `✓`. `menu._fit` is
#: ANSI-aware and strips these safely when a row is truncated to the
#: terminal width.
_REVERSE = "\033[7m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# profile name, token typed in this flow, old profile name, new old-profile name
ProfileChoice = tuple[str, str | None, str | None, str | None]


@dataclass(frozen=True)
class _AgentBackend:
    """How ONE agent's backend is read and applied, as data.

    The main screen shows one chipset row per agent, and the two agents differ
    along three axes at once: how to read which backend is currently applied,
    how to apply a wrapper, and how to clear the override. Expressing that as
    ``if agent.name == "claude"`` is exactly what this project's conventions
    forbid, so the differences live here as a per-agent record and every
    consumer is a dict lookup.

    An agent with no entry in :data:`_AGENT_BACKENDS` simply gets no chipset
    row — the honest degradation when a new agent is added to ``AGENTS``
    before its backend mechanism exists, and the single place a future agent
    (Copilot, say) has to be described to gain a fully working row.

    The two apply hooks are :class:`TuiSession` METHOD NAMES, resolved on the
    session with ``getattr`` at the moment a chip is applied — not function
    objects captured when this table is built. Capturing the functions would
    freeze whatever they were at import time, so a test that patches
    ``TuiSession._apply_switch_wrapper`` would silently keep calling the
    original; a name stays honest about the fact that the session is what
    ultimately does the work.

    Attributes:
        read_applied: ``(paths) -> provider name | None``. MUST never raise —
            it runs once per main-loop iteration on the UI path, where an
            unreadable config means "nothing applied", not a crash.
        apply_wrapper: Method name taking ``(spec)`` — point the agent at an
            installed wrapper's backend.
        apply_native: Method name taking no arguments — clear the override.
        lifecycle: When the change takes effect, rendered at the end of the
            row. The claude/codex difference (a running session vs. the next
            launch) is real and must be visible, not implied.
    """

    read_applied: Callable[[Paths], str | None]
    apply_wrapper: str
    apply_native: str
    lifecycle: str


def _hint(
    numbered_count: int,
    *,
    exit_word: str,
    chips: bool = False,
    has_token_key: bool = False,
) -> str:
    """Return the uniform navigation hint for a menu.

    ``chips=True`` is the main screen, whose rows are not a flat list: the
    agent rows carry a horizontal chip strip that Left/Right moves through
    and Enter applies. The hint must name that, because a chipset is not
    discoverable from an Up/Down hint alone.

    ``has_token_key=True`` adds ``· t: token`` — the main screen (issue #29)
    binds ``t`` to per-row token rotation, and the hint must say so. It is
    always shown on the main screen even when the cursor is on a non-secret
    wrapper, because ``t`` there is a silent no-op (no dead-end screen), and
    hiding the hint only on non-secret rows would make it flicker as the user
    moves the cursor.

    Every key named here MUST actually be reachable — a hint is a promise.
    See ``menu._PASSTHROUGH`` for the bug this class of drift already caused
    once, and ``test_tui`` for the pin that now prevents it.
    """
    from code_helper.cli.menu import MAX_DIGIT_ITEMS

    usable = min(numbered_count, MAX_DIGIT_ITEMS)
    if usable:
        digits = "1" if usable == 1 else f"1-{usable}"
        hint = f"Up/Down · {digits} · Enter select · Esc {exit_word} · Ctrl-C quit"
    elif chips:
        # The editing keys are named here rather than left behind `?`: on the
        # main screen they are the only way to add or change a wrapper, and a
        # key nobody can see is a key nobody presses. `a` is split OUT of the
        # `e/d` cluster rather than folded into "edit" — `e` only rotates a
        # token (it dispatches to the same handler as `t`), so grouping `a`
        # under "edit" mislabels what pressing it does. Kept short (`?` alone,
        # not `? keys`) to stay well inside 80 columns — `_fit` would
        # otherwise truncate the tail and silently eat the exit hint, which is
        # exactly the bug a PTY run caught here.
        hint = (
            f"↑↓ row · ←→ chip · Enter apply · a add · e/d edit · ? · Esc {exit_word}"
        )
    else:
        hint = f"Up/Down · Enter select · Esc {exit_word} · Ctrl-C quit"
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

    The typed handlers receive frozen ``AddRequest``/``EditTokenRequest``/
    ``RemoveRequest``/``SetDefaultRequest`` instances directly. ``self.args``
    now carries only the session configuration supplied by argparse (``debug``
    and ``dry_run``, which the Settings screen toggles); no screen mutates it
    to pass an argument into a handler any more.
    """

    __slots__ = (
        "args",
        "_tab_provider",
        "_tab_label",
        "_slot_label",
        "_applied",
        "_chips",
        "_chip_index",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Every cache below is filled once per main-loop iteration by
        # `_refresh_active_label`, because each is read from a label callable
        # that the menu re-evaluates on EVERY redraw frame — including pure
        # cursor movement. Deriving any of them per frame would put a file
        # read or a directory scan on the keystroke path.
        self._tab_provider: str | None = None
        self._tab_label: str = ""
        #: The profile screen's rendered slot strip.
        self._slot_label: str = ""
        #: agent name -> the provider its config currently names, or None.
        self._applied: dict[str, str | None] = {}
        #: agent name -> its chip strip (`native` plus installed wrappers).
        self._chips: dict[str, list] = {}
        #: agent name -> chip cursor. Ephemeral on purpose: persisting it
        #: would create a second source of truth about what is selected,
        #: competing with the config files that actually decide.
        self._chip_index: dict[str, int] = {}

    # --- UI primitives ---------------------------------------------------

    def _pick(
        self,
        items: Sequence[object],
        prompt: str | Callable[[], str],
        *,
        exit_word: str = "back",
        on_tab: Callable[[], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_key: dict[str, Callable[[str], object]] | None = None,
        chips: bool = False,
        numbered: bool = True,
    ) -> str:
        from code_helper.cli.menu import MenuCancelled, Section, select_from_menu

        numbered_count = sum(
            1
            for entry in items
            if not isinstance(entry, Section) and entry[0] not in (_BACK, _QUIT)  # type: ignore[index]
        )
        try:
            return select_from_menu(
                items,  # type: ignore[arg-type]
                prompt=prompt,
                hint=_hint(
                    numbered_count if numbered else 0,
                    exit_word=exit_word,
                    chips=chips,
                    has_token_key=on_token is not None,
                ),
                on_tab=on_tab,
                on_token=on_token,
                on_key=on_key,
                unnumbered=frozenset({_BACK, _QUIT}),
                numbered=numbered,
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

    def _run(self, handler, request=None) -> None:
        """Dispatch a CLI handler/request and preserve its user-facing output.

        Tees stdout instead of fully redirecting it: some handlers (e.g.
        ``set-default``'s ``_confirm_set_default``) print a preview and then
        block on ``input()`` for a yes/no confirmation. A full
        ``redirect_stdout`` buffers that preview into the capture and never
        shows it before ``input()`` blocks, so the user would confirm a
        destructive config write blind. Writing to both the real stdout and
        the buffer keeps that output live while still letting ``_notify``
        replay the full transcript afterwards so it survives the next
        redraw.
        """
        import sys

        from code_helper.errors import CodeHelperError

        class _Tee:
            """Fan writes out to the real stdout AND a capture buffer.

            Must behave enough like the stream it replaces that code reached
            through the handler cannot tell the difference. ``isatty`` in
            particular is not optional: ``menu.read_line`` — which every
            confirmation prompt goes through — calls it to decide between
            raw-mode line editing and a plain ``input()``. Without it, any
            handler that asked for confirmation died with an
            ``AttributeError`` instead of prompting, which is exactly what
            happened to ``set-default``'s prompt from inside the TUI. It
            answers for the REAL terminal (the first stream), because that is
            where the prompt is actually rendered and read.
            """

            def __init__(self, *streams) -> None:
                self._streams = streams

            def write(self, text: str) -> int:
                for stream in self._streams:
                    stream.write(text)
                return len(text)

            def flush(self) -> None:
                for stream in self._streams:
                    stream.flush()

            def isatty(self) -> bool:
                return self._streams[0].isatty()

            def fileno(self) -> int:
                return self._streams[0].fileno()

            @property
            def encoding(self) -> str:
                return getattr(self._streams[0], "encoding", "utf-8")

        import io

        buffer = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, buffer)
        try:
            handler(self.args if request is None else request)
        except CodeHelperError as exc:
            if getattr(self.args, "debug", False):
                sys.stdout = real_stdout
                raise
            print(f"error: {exc}")
        finally:
            sys.stdout = real_stdout
        self._notify(buffer.getvalue().rstrip())

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
        from code_helper.services.agents import all_agents
        from code_helper.services.wrappers import (
            column_header,
            describe_all_columns,
            valid_default_wrapper,
        )

        agents = all_agents(paths)
        specs = self._all_wrapper_specs()
        # Columns are aligned across ALL agents in one describe_all_columns
        # call, not one call per agent — a per-agent call would compute its
        # own name/provider widths from only that agent's wrappers, and the
        # single header above the first group would then misalign against
        # every later group whose widths differ.
        described = describe_all_columns(
            paths,
            specs,
            defaults={
                agent.name: valid_default_wrapper(paths, agent.name) for agent in agents
            },
        )
        by_name = dict(described)
        rows: list = []
        for agent in agents:
            agent_specs = [s for s in specs if s.agent.name == agent.name]
            if not agent_specs:
                continue
            if not rows:
                # Once, above the first group: three unlabelled columns read
                # as noise ("ollama" alone says nothing about being a
                # provider). Repeating it per agent would be louder than the
                # data it describes.
                rows.append(Section(column_header(described)))
            # The bare agent name would repeat the chipset row verbatim; the
            # count says what this section actually is — the wrappers behind
            # those chips.
            rows.append(Section(f"{agent.name} — {len(agent_specs)} wrappers"))
            rows.extend(
                (spec.name, "  ".join(by_name[spec.name])) for spec in agent_specs
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
            return
        profile = self._select_profile(spec.provider.name, editing=True)
        if profile is None:
            return
        from code_helper.cli.requests import EditTokenRequest

        profile_name, profile_token, rename_from, rename_to = profile
        self._run(
            _handle_edit_token,
            EditTokenRequest(
                name=alias,
                profile=profile_name,
                profile_token=profile_token,
                profile_rename_from=rename_from,
                profile_rename_to=rename_to,
                dry_run=getattr(self.args, "dry_run", False),
                debug=getattr(self.args, "debug", False),
            ),
        )

    # --- applying a chip -------------------------------------------------

    def _apply_switch_wrapper(self, spec) -> None:
        """claude: retarget the RUNNING session at ``spec``'s backend."""
        from code_helper.cli.parser import _handle_switch

        self._run(_handle_switch, self._switch_request(from_wrapper=spec.name))

    def _apply_switch_native(self) -> None:
        """claude: clear the managed ``env`` block from settings.json."""
        from code_helper.cli.parser import _handle_switch

        self._run(_handle_switch, self._switch_request(provider=_NATIVE_CHIP))

    def _switch_request(self, *, provider=None, from_wrapper=None):
        """A :class:`SwitchRequest` carrying only the axis a chip varies.

        Every other field is meaningless from the main screen (no model or
        token prompts, no restore), so the shared fields are filled once here
        instead of at each call site — which also keeps the two apply paths
        from drifting apart if ``SwitchRequest`` grows another field.
        """
        from code_helper.cli.requests import SwitchRequest

        return SwitchRequest(
            provider=provider,
            from_wrapper=from_wrapper,
            model=None,
            haiku=None,
            sonnet=None,
            opus=None,
            subagent_model=None,
            base_url=None,
            auth=None,
            profile=None,
            restore=False,
            slot=None,
            status=False,
            dry_run=getattr(self.args, "dry_run", False),
            force=False,
            debug=getattr(self.args, "debug", False),
        )

    def _apply_set_default_wrapper(self, spec) -> None:
        """codex: patch config.toml so the NEXT launch uses ``spec``."""
        from code_helper.cli.parser import _handle_set_default
        from code_helper.cli.requests import SetDefaultRequest
        from code_helper.services.model import BaseUrlPolicy

        # FIXED providers reject a --base-url even when it equals their own
        # registry default; forward the resolved address only for
        # REQUIRED/OVERRIDABLE (issue #30 follow-up).
        self._run(
            _handle_set_default,
            SetDefaultRequest(
                agent=spec.agent.name,
                provider=spec.provider.name,
                model=spec.model,
                base_url=(
                    spec.provider.base_url
                    if spec.provider.base_url_policy is not BaseUrlPolicy.FIXED
                    else None
                ),
                restore=False,
                slot=None,
                catalog_json=None,
                dry_run=getattr(self.args, "dry_run", False),
                force=False,
                debug=getattr(self.args, "debug", False),
            ),
        )

    def _apply_set_default_native(self) -> None:
        """codex: remove the managed region from config.toml.

        Deliberately NOT ``set-default --restore``: restore rolls the file
        back to a backup snapshot, undoing unrelated hand-edits made since.
        This removes only the region this tool owns — "stop overriding",
        not "undo my last change".
        """
        from code_helper.cli.parser import _confirm_set_default
        from code_helper.services.codex_default import clear_default
        from code_helper.services.paths import Paths

        # Reuses the CLI's own confirm — same diff-first prompt, and (the part
        # a local re-implementation would have silently dropped) the same
        # off-a-TTY refusal, so a scripted run can never be talked into
        # patching config.toml through the TUI.
        self._run(
            lambda _req: clear_default(
                Paths.default(),
                dry_run=getattr(self.args, "dry_run", False),
                confirm=_confirm_set_default,
            ),
            None,
        )

    def _add_provider_choices(self, agent):
        """Return provider menu rows and their runtime-auth choices for ``agent``.

        Filtered to ``compatible_providers(agent)`` — NOT raw ``PROVIDERS`` —
        so a switch-only entry (``native``: presents only
        ``ANTHROPIC_SETTINGS``, which no ``Agent`` consumes) never appears as
        an ``add`` choice, and so a provider that only a DIFFERENT agent can
        reach never appears either. This is equivalent to what
        ``resolve_shape`` would accept for each provider, computed without
        actually calling it per provider.
        """
        from code_helper.services.model import AuthPolicy, compatible_providers

        items: list[tuple[str, str]] = []
        choices = {}
        for provider in compatible_providers(agent):
            items.append((provider.name, f"{provider.name} — {provider.description}"))
            choices[provider.name] = (provider, False)
            if (
                provider.auth_policy is AuthPolicy.OVERRIDABLE
                and provider.auth != "secret"
            ):
                value = f"{provider.name}:secret"
                items.append(
                    (
                        value,
                        f"{provider.name} (with token) — reverse proxy / cloud auth",
                    )
                )
                choices[value] = (provider, True)
        return items, choices

    @staticmethod
    def _breadcrumb(*parts: str) -> str:
        """``"codex › ollama "`` — a prefix naming choices already made.

        Prepended to a wizard step's prompt so a user several screens into
        Add still sees what they picked earlier (agent, provider, …) without
        a dedicated summary screen. Empty when there is nothing to show yet
        (the very first step), so it never dangles a bare ``"› "``.
        """
        return f"{' › '.join(parts)} › " if parts else ""

    def _choose_add_provider(self, agent):
        """Choose and configure a provider for ``agent``, or ``_BACK``/``None``."""
        from code_helper.errors import CodeHelperError
        from code_helper.services.model import BaseUrlPolicy, with_auth, with_base_url

        items, choices = self._add_provider_choices(agent)
        selection = self._pick(
            [*items, (_BACK, "Back")], f"Select a provider for {agent.name}:"
        )
        if selection == _BACK:
            return _BACK
        provider, want_secret_auth = choices[selection]
        provider = with_auth(provider, want_secret=want_secret_auth)
        typed_url: str | None = None
        if provider.base_url_policy is not BaseUrlPolicy.FIXED:
            required = provider.base_url_policy is BaseUrlPolicy.REQUIRED
            default = (
                " (required)"
                if required
                else (f" [{provider.base_url}]" if provider.base_url else "")
            )
            # Re-ask on empty for a REQUIRED field — the same "validate, then
            # loop" shape `_read_token` uses — so a stray Enter re-prompts the
            # URL instead of dumping the user back to provider selection.
            # `None` (Esc) still means cancel; only an empty submit loops.
            while True:
                typed_url = self._read_text(f"Base URL for {provider.name}{default}: ")
                if typed_url is None:
                    return None
                if required and not typed_url:
                    self._notify(f"A base URL is required for {provider.name}.")
                    continue
                break
            try:
                provider = with_base_url(provider, typed_url or None)
            except CodeHelperError as exc:
                self._notify(f"error: {exc}")
                return None
        return provider, want_secret_auth, typed_url

    def _choose_add_model(self, provider, profile_name, profile_token, agent_name: str):
        """Discover and choose a model; ``_BACK`` returns to profile choice."""
        from code_helper.services.models_api import list_models
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import token_for_discovery

        discovery_token = profile_token or token_for_discovery(
            Paths.default(), provider, profile_name=profile_name or None
        )
        result = list_models(provider, token=discovery_token)
        # Discovery is always tried first and is the source of truth; the
        # registry's `known_models` is only consulted when discovery comes
        # back with nothing (structurally unavailable, e.g. zai, or a
        # transient failure) — never used to override a real result. Which
        # source fed the menu is shown in the prompt so a stale built-in
        # entry is never mistaken for something the endpoint just confirmed.
        models = result.models
        using_known = not models and provider.known_models
        if using_known:
            models = provider.known_models
        items = [(model, model) for model in models]
        items.extend((("__custom__", "Enter model manually"), (_BACK, "Back")))
        # The breadcrumb names the agent already scoped by the row that opened
        # Add, so a model step reached several screens in still shows what it
        # is for.
        breadcrumb = self._breadcrumb(agent_name)
        if using_known:
            base = (
                f"Select a model for {provider.name} "
                "(known models — discovery unavailable):"
            )
        elif not result.ok:
            base = f"Model discovery unavailable for {provider.name}; enter a model:"
        else:
            base = f"Select a model for {provider.name}:"
        model = self._pick(items, f"{breadcrumb}{base}")
        if model != "__custom__":
            return model
        typed = self._read_text("Model: ")
        return typed or None

    def _choose_add_agent(
        self,
        provider,
        model,
        profile_name,
        profile,
        typed_url,
        want_secret_auth,
        agent_name: str,
    ):
        """Choose an alias and dispatch Add for the pre-scoped ``agent_name``.

        ``agent_name`` is always resolved up front by :meth:`_run_add` (from a
        chipset row's ``a``/``+ add``, or the unscoped wrapper picker), so this
        step never asks "which agent" — it goes straight to the alias.
        """
        from code_helper.cli.parser import _handle_add
        from code_helper.cli.requests import AddRequest
        from code_helper.errors import CodeHelperError
        from code_helper.services.spec import suggest_alias

        try:
            default_alias = suggest_alias(model, agent_name, profile_name or None)
        except CodeHelperError as exc:
            self._notify(f"error: {exc}")
            return _BACK
        # The alias prompt is the last screen before a real filesystem write
        # (`_handle_add` below), and the furthest from the choices that led
        # here — the breadcrumb is what lets the user confirm "yes, this is
        # the pairing I meant" without a dedicated summary screen.
        breadcrumb = self._breadcrumb(agent_name, provider.name, model)
        alias = self._read_text(f"{breadcrumb}Command name [{default_alias}]: ")
        if alias is None:
            return None
        _, profile_token, rename_from, rename_to = profile
        self._run(
            _handle_add,
            AddRequest(
                name=None,
                agent=agent_name,
                provider=provider.name,
                model=model,
                alias=alias or default_alias,
                shape=None,
                base_url=typed_url,
                auth="secret" if want_secret_auth else None,
                profile=profile_name or None,
                profile_token=profile_token,
                profile_rename_from=rename_from,
                profile_rename_to=rename_to,
                list_models=False,
                dry_run=getattr(self.args, "dry_run", False),
                force=False,
                debug=getattr(self.args, "debug", False),
            ),
        )
        return True

    def _run_add_provider(
        self, provider, want_secret_auth, typed_url, agent_name: str
    ) -> bool:
        """Run profile → model → alias for one pre-scoped agent.

        ``agent_name`` is always resolved by :meth:`_run_add` before this is
        called — the agent is never chosen here.
        """
        while True:
            if provider.auth == "secret":
                profile = self._select_profile(provider.name, editing=False)
                if profile is None:
                    return False
            else:
                profile = ("", None, None, None)
            profile_name, profile_token, _, _ = profile
            while True:
                model = self._choose_add_model(
                    provider, profile_name, profile_token, agent_name=agent_name
                )
                if model == _BACK:
                    break
                if model is None:
                    continue
                while True:
                    outcome = self._choose_add_agent(
                        provider,
                        model,
                        profile_name,
                        profile,
                        typed_url,
                        want_secret_auth,
                        agent_name=agent_name,
                    )
                    if outcome is True:
                        return True
                    if outcome == _BACK:
                        break
                    if outcome is None:
                        # Esc at the alias prompt: there is no agent menu to
                        # re-open (the agent was resolved up front), so looping
                        # here would spin silently on a screen that never
                        # renders. Fall back to the model menu instead, the
                        # nearest enclosing screen that still exists.
                        break

    def _choose_add_kind(self):
        """The `a` entry point: agent or wrapper? Returns ``_BACK`` on Esc.

        A wrapper is an agent × provider × model pairing; an agent is a new
        CLI integration entry. They are different objects and the user must
        say which one they mean before anything else is asked — see
        ``CLAUDE.md``'s "add is agent-first" note.
        """
        return self._pick(
            [
                ("wrapper", "Wrapper — an agent × provider × model pairing"),
                ("agent", "Agent — a new CLI integration"),
                (_BACK, "Back"),
            ],
            "What do you want to add?",
        )

    def _run_add(self, agent_name: str | None = None) -> None:
        """Create a wrapper (or, unscoped, an agent) via the Add flow.

        ``agent_name`` pre-scopes the agent — set when Add was entered from a
        specific agent's row (``a`` on that row, or its ``+ add`` chip). That
        already answers "wrapper for which agent", so the kind/agent screens
        below are skipped and the flow goes straight to provider selection.
        """
        from code_helper.services.agents import get_agent
        from code_helper.services.paths import Paths

        if agent_name is None:
            kind = self._choose_add_kind()
            if kind == _BACK:
                return
            if kind == "agent":
                self._run_add_agent()
                return
            agent_name = self._choose_add_agent_for_wrapper()
            if agent_name is None:
                return
        agent = get_agent(Paths.default(), agent_name)
        while True:
            selected = self._choose_add_provider(agent)
            if selected == _BACK:
                return
            if selected is None:
                continue
            provider, want_secret_auth, typed_url = selected
            if self._run_add_provider(
                provider, want_secret_auth, typed_url, agent_name=agent_name
            ):
                return

    def _choose_add_agent_for_wrapper(self) -> str | None:
        """Which agent the new wrapper is for — the unscoped Wrapper branch.

        Returns ``None`` on Esc/Back (caller re-shows the kind screen by
        simply returning, same as every other Back in this flow).
        """
        from code_helper.services.agents import all_agents
        from code_helper.services.paths import Paths

        items = [
            (agent.name, f"{agent.name} — {agent.description}")
            for agent in all_agents(Paths.default())
        ]
        choice = self._pick([*items, (_BACK, "Back")], "Add a wrapper for which agent?")
        return None if choice == _BACK else choice

    def _run_add_agent(self) -> None:
        """Register a new user-defined agent (the Agent branch of `a`).

        Collects a name (and, only if it differs, a binary), persists it via
        ``services/agents.py`` merged with the built-in registry, and reports
        the honest degradation: the new agent gets `add`/`list`/`remove`
        support immediately, but no chipset row (no live-patchable config
        this project knows how to read for it — see ``_AgentBackend``'s
        docstring).
        """
        from code_helper.errors import CodeHelperError
        from code_helper.services.agents import add_user_agent
        from code_helper.services.paths import Paths

        name = self._read_text("Agent name (its executable name on PATH): ")
        if not name:
            return
        name = name.strip()
        binary = self._read_text(f"Binary name [{name}]: ")
        binary = (binary or "").strip() or name
        description = self._read_text("Description (optional): ")
        description = (description or "").strip()

        try:
            add_user_agent(Paths.default(), name, binary, description)
        except CodeHelperError as exc:
            self._notify(f"error: {exc}")
            return
        self._notify(
            f"Added agent {name!r}. It has no chipset row (no live-patchable "
            f"config to read), but `add`/`list`/`remove` work for it now."
        )

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
        """Pick the active token profile — the ``p`` screen.

        This is where profile switching lives now. It used to be spread
        across the main screen as Tab (cycle within the current provider)
        plus digits 1-0 (jump to a numbered slot), while this screen itself
        was unreachable — nothing dispatched to it. The chipset needed
        Left/Right and Shift+Tab for backends, so profile switching moved
        here WHOLE rather than being split further: Tab and the digit slots
        work exactly as they did, on the screen that is about profiles.
        """
        from code_helper.services.paths import Paths
        from code_helper.services.secrets import (
            DEFAULT_PROFILE,
            profile_names,
            valid_active_profile,
        )
        from code_helper.services.state import set_active_selection

        paths = Paths.default()
        self._refresh_profile_label()
        provider_items = [
            (p.name, f"{p.name} — {p.description}") for p in self._secret_providers()
        ]
        keys: dict[str, Callable[[str], object]] = {}
        for i in range(10):
            keys[f"DIGIT_{i}"] = lambda _value, i=i: self._on_slot(
                9 if i == 0 else i - 1
            )
        provider = self._pick(
            [self._slot_section(), *provider_items, (_BACK, "Back")],
            "Active profile provider:",
            on_tab=self._on_tab,
            on_key=keys,
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
        self._refresh_profile_label()

    def _on_slot(self, slot: int) -> None:
        from code_helper.services.paths import Paths
        from code_helper.services.profiles import profile_slots
        from code_helper.services.state import set_active_selection

        slots = profile_slots(Paths.default())
        if slot < len(slots):
            provider, profile = slots[slot]
            set_active_selection(Paths.default(), provider, profile)
            self._refresh_profile_label()

    def _slot_section(self):
        """The profile screen's slot strip — `1 zai/default  2 zai/work`.

        Reads the cache rather than rescanning: this is a callable label, so
        it is re-evaluated on every redraw frame, and `profile_slots` walks
        every provider's profiles. `_refresh_profile_label` refills it when a
        slot actually changes.
        """
        from code_helper.cli.menu import Section

        return Section(lambda: self._slot_label)

    def _refresh_profile_label(self) -> None:
        """Refresh only the profile-related caches.

        Split from :meth:`_refresh_active_label` because Tab and the digit
        slots change a PROFILE, not a backend — re-reading each agent's
        config file there would be work no keypress on that screen can
        invalidate.
        """
        from code_helper.services.paths import Paths
        from code_helper.services.profiles import profile_slots

        self._tab_provider = self._resolve_tab_provider()
        self._tab_label = self._active_label(self._tab_provider)
        self._slot_label = "  ".join(
            f"{i + 1} {provider}/{profile}"
            for i, (provider, profile) in enumerate(profile_slots(Paths.default()))
        )

    def _refresh_active_label(self) -> None:
        """Refresh every per-iteration cache the main screen reads.

        Called ONCE per main-loop iteration. Everything filled here does real
        I/O (a profile scan, plus one config read per agent), and every
        consumer is a label callable that the menu re-evaluates on EVERY
        redraw frame — including pure cursor movement. Reading any of it from
        those callables would turn one screen's worth of I/O into one
        keystroke's worth.
        """
        from code_helper.services.paths import Paths

        paths = Paths.default()
        self._refresh_profile_label()
        self._applied = {
            name: backend.read_applied(paths)
            for name, backend in _AGENT_BACKENDS.items()
        }
        self._chips = {name: self._chips_for(name, paths) for name in _AGENT_BACKENDS}
        # A wrapper can be removed (`d`) or added (`a`) between iterations, so
        # a chip cursor parked past the end of a now-shorter strip is normal,
        # not a bug — clamp rather than reset, so an unaffected row keeps its
        # position.
        for name, chips in self._chips.items():
            if self._chip_index.get(name, 0) >= len(chips):
                self._chip_index[name] = max(0, len(chips) - 1)

    def _chips_for(self, agent_name: str, paths) -> list:
        """The chip strip for ``agent_name``: native, its wrappers, then Add.

        A BACKEND chip is an ALREADY-INSTALLED wrapper, never a bare
        provider: the wrapper is where a backend's model, token and base URL
        were resolved and frozen when it was created. That is what lets Enter
        apply a backend chip with no prompts and no second guess about which
        model to use — and it is why there is no "provider with no wrapper"
        chip to explain away.

        The trailing :data:`_ADD_CHIP` is the one deliberate exception: not a
        backend, never "applied", present on every row (including one with no
        wrappers yet) so that row always has a visible way to get its first
        one instead of reading as empty/broken.
        """
        from code_helper.services.wrappers import is_installed

        return [
            _NATIVE_CHIP,
            *(
                spec
                for spec in self._all_wrapper_specs()
                if spec.agent.name == agent_name and is_installed(paths, spec.name)
            ),
            _ADD_CHIP,
        ]

    @staticmethod
    def _chip_name(chip) -> str:
        return chip if isinstance(chip, str) else chip.name

    def _chip_is_applied(self, agent_name: str, chip) -> bool:
        """Whether ``chip`` is the backend currently in the agent's config.

        The applied state is read back as a PROVIDER name (that is what the
        config file records), while a chip is a wrapper — so a wrapper chip
        matches when its provider matches. Two wrappers on the same provider
        are therefore indistinguishable here and the first one is marked; the
        same honest limitation ``current_switch`` documents for two providers
        sharing one base URL. The action chip is never applied — it isn't a
        backend at all.
        """
        if chip == _ADD_CHIP:
            return False
        applied = self._applied.get(agent_name)
        if chip == _NATIVE_CHIP:
            return applied is None
        return applied is not None and chip.provider.name == applied

    def _chip_row(self, agent_name: str) -> Callable[..., str]:
        """A label callable rendering one agent's chip strip.

        Re-evaluated on every redraw frame, so it reads ONLY the caches
        ``_refresh_active_label`` fills — never the filesystem. Returns one
        logical line: the menu truncates to the terminal width and counts one
        row per entry, and a strip that wrapped would desynchronise the
        in-place frame erase.

        Accepts ``selected``/``ansi`` from ``menu._call_label``: the chip
        cursor (Left/Right, applied by Enter) is drawn ONLY when this row is
        the one the list cursor `>` sits on. Without that, every agent row
        would show its own remembered ``_chip_index`` at once — two (or N)
        highlighted blocks on screen for a list with a single cursor.
        """

        def label(*, selected: bool = True, ansi: bool = True) -> str:
            chips = self._chips.get(agent_name, [])
            cursor = self._chip_index.get(agent_name, 0)
            rendered = []
            for index, chip in enumerate(chips):
                is_add = chip == _ADD_CHIP
                name = self._chip_name(chip)
                applied = self._chip_is_applied(agent_name, chip)
                text = f"✓ {name}" if applied else name
                if selected and index == cursor:
                    rendered.append(
                        f"{_REVERSE}{text}{_RESET}" if ansi else f"[{text}]"
                    )
                elif applied:
                    rendered.append(f"{_BOLD}{text}{_RESET}" if ansi else text)
                elif is_add:
                    # Dim, never bold/`✓`: this is the one non-backend chip on
                    # the strip, so it must never be mistaken for one that's
                    # applied (see `_chip_is_applied`'s `_ADD_CHIP` case).
                    rendered.append(f"{_DIM}{text}{_RESET}" if ansi else text)
                else:
                    rendered.append(text)
            return f"{agent_name:<8}{'  '.join(rendered)}"

        return label

    def _chip_move(self, value: str, delta: int) -> None:
        """Move the focused agent row's chip cursor. Pure in-memory.

        Returns ``None`` so the menu redraws instead of exiting, which is what
        makes left/right free of I/O: nothing is read or written until Enter
        applies a chip. A no-op when the row cursor is on a wrapper row.
        """
        if not value.startswith(_AGENT_ROW):
            return None
        agent_name = value.removeprefix(_AGENT_ROW)
        chips = self._chips.get(agent_name, [])
        if chips:
            current = self._chip_index.get(agent_name, 0)
            self._chip_index[agent_name] = (current + delta) % len(chips)
        return None

    def _apply_chip(self, agent_name: str) -> None:
        """Apply the highlighted chip of ``agent_name`` (Enter on its row).

        The already-applied short-circuit is trusted ONLY for the native
        chip: ``applied is None`` is an exact, unambiguous read (see
        ``_chip_is_applied``). A wrapper chip's "applied" is a heuristic —
        matched by PROVIDER NAME only, because that is all a config file
        records — so two installed wrappers sharing a provider (different
        model/profile/token/base-url) are indistinguishable there and the
        first one is reported applied even when the SECOND is the one
        actually live. Short-circuiting Enter on that heuristic would make
        the second wrapper permanently unreachable through the chipset — the
        exact distinction it exists to expose — so a wrapper chip always
        re-applies; the backend's own apply path already no-ops safely when
        the resolved config truly hasn't changed.
        """
        chips = self._chips.get(agent_name, [])
        if not chips:
            return
        chip = chips[min(self._chip_index.get(agent_name, 0), len(chips) - 1)]
        if chip == _ADD_CHIP:
            # Not a backend — opens Add pre-scoped to this row's agent,
            # exactly what `a` on this row does (see `run()`'s "a" binding).
            self._run_add(agent_name)
            return
        backend = _AGENT_BACKENDS[agent_name]
        if chip == _NATIVE_CHIP:
            if self._chip_is_applied(agent_name, chip):
                self._notify(f"{agent_name} is already on {self._chip_name(chip)}.")
                return
            getattr(self, backend.apply_native)()
        else:
            getattr(self, backend.apply_wrapper)(chip)

    def _main_prompt(self) -> str:
        """Live main-menu header.

        Deliberately does NOT repeat which backend is applied: the chipset
        rows say that in place, and a header that restates it is the kind of
        duplication this screen was redesigned to remove.
        """
        if self._tab_label:
            # Labelled: a bare "zai/axisrow" up here reads as a model or an
            # endpoint, which is exactly what the rest of the screen is about.
            return f"code-helper{' ' * 20}profile: {self._tab_label}"
        return "code-helper"

    def _show_help(self) -> None:
        from code_helper.cli.menu import press_any_key

        print("a add (agent row: scoped to it) · t/e token · d delete")
        print("p profiles · s settings · ←→ + Enter on the + add chip: same as a")
        print("Up/Down row · Left/Right chip · Enter apply · Esc quit · Ctrl-C quit")
        press_any_key("Press any key to continue...")

    def _profile_row_label(self) -> str:
        return f"Profile: {self._tab_label}" if self._tab_label else "Profile"

    # --- main loop -------------------------------------------------------

    def run(self) -> int:
        """The main menu loop — show wrappers + service rows, dispatch."""
        from code_helper.cli.menu import MenuCancelled, Section
        from code_helper.services.paths import Paths
        from code_helper.services.state import set_default_wrapper

        try:
            while True:
                self._refresh_active_label()
                paths = Paths.default()
                keys = {
                    # `a` on an agent chipset row already answers "wrapper for
                    # which agent" — skip straight to a scoped Add instead of
                    # asking again. Elsewhere on the main screen (a wrapper
                    # row, or no row at all) `a` opens the unscoped kind
                    # picker (agent vs wrapper).
                    "a": lambda value: (
                        f"add:{value.removeprefix(_AGENT_ROW)}"
                        if value.startswith(_AGENT_ROW)
                        else _ADD
                    ),
                    "p": lambda _alias: _PROFILE,
                    "s": lambda _alias: _SETTINGS,
                    "?": lambda _alias: _HELP,
                    "TOKEN": lambda alias: f"token:{alias}",
                    "e": lambda alias: f"token:{alias}",
                    "d": lambda alias: f"remove:{alias}",
                    # Left/Right (and Shift+Tab, the same move backwards) only
                    # ever reposition the chip cursor and return None, so the
                    # menu redraws without exiting — the whole point of
                    # applying on Enter is that moving costs no I/O.
                    "LEFT": lambda value: self._chip_move(value, -1),
                    "RIGHT": lambda value: self._chip_move(value, +1),
                    "BACK_TAB": lambda value: self._chip_move(value, -1),
                }
                choice = self._pick(
                    [
                        *(
                            (f"{_AGENT_ROW}{name}", self._chip_row(name))
                            for name in self._chips
                        ),
                        # Separates the chipset from the wrapper list below.
                        Section(""),
                        *self._wrapper_rows(paths),
                    ],
                    self._main_prompt,
                    exit_word="quit",
                    on_key=keys,
                    chips=True,
                    numbered=False,
                )
                if choice in (_BACK, _QUIT):
                    return 0
                if choice == _ADD:
                    self._run_add()
                elif choice.startswith("add:"):
                    self._run_add(choice.removeprefix("add:"))
                elif choice == _PROFILE:
                    self._run_profile_screen()
                elif choice.startswith(_AGENT_ROW):
                    self._apply_chip(choice.removeprefix(_AGENT_ROW))
                elif choice == _SETTINGS:
                    self._run_settings()
                elif choice == _HELP:
                    self._show_help()
                elif choice.startswith("token:"):
                    self._on_token(choice.removeprefix("token:"))
                elif choice.startswith("remove:"):
                    from code_helper.cli.parser import _handle_remove
                    from code_helper.cli.requests import RemoveRequest

                    # `force` stays False: removing an unmanaged file is an
                    # explicit-command-line decision, never a keypress.
                    self._run(
                        _handle_remove,
                        RemoveRequest(
                            name=choice.removeprefix("remove:"),
                            dry_run=getattr(self.args, "dry_run", False),
                            force=False,
                            debug=getattr(self.args, "debug", False),
                        ),
                    )
                else:
                    # A wrapper alias — Enter makes it the default for its
                    # agent. `set_default_wrapper` is the raw store (#28);
                    # `choice` came from `_wrapper_rows`, which only yields
                    # real aliases, so the write is always of a real alias.
                    from code_helper.services.wrappers import is_installed, is_managed

                    spec = self._resolve_spec(choice)
                    if spec is None:
                        continue
                    if not is_installed(paths, choice):
                        self._notify("Wrapper not installed — use Add first.")
                        continue
                    if not is_managed(paths, choice):
                        # valid_default_wrapper (the only reader that
                        # matters — the `●` marker and set-default both go
                        # through it) requires is_installed AND is_managed.
                        # Writing a default for an unmanaged file would
                        # "succeed" here and silently vanish on the very
                        # next read.
                        self._notify(
                            f"{choice} is not a code-helper-managed wrapper "
                            "— cannot set it as default."
                        )
                        continue
                    set_default_wrapper(paths, spec.agent.name, choice)
                    self._notify(f"Default wrapper set to {choice}.")
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return 0


#: How each agent's backend is read and applied — the one place the two
#: agents' differences are written down. Defined after `TuiSession` because
#: its callables are that class's own methods; keeping it at module level
#: (rather than as a class attribute) keeps it importable by tests that pin
#: the table's shape without constructing a session.
#:
#: The two mechanisms are genuinely different, not two spellings of one:
#: claude's `switch` rewrites `~/.claude/settings.json`, which a RUNNING
#: session re-reads between prompts; codex's `set-default` patches
#: `~/.codex/config.toml`, which is read at launch. Hence the differing
#: `lifecycle` strings — the distinction is shown, not hidden.
_AGENT_BACKENDS: dict[str, _AgentBackend] = {}


def _register_agent_backends() -> None:
    """Populate :data:`_AGENT_BACKENDS`, importing services lazily.

    A function rather than a module-level literal so the service imports stay
    inside a call — `cli/tui.py` is imported by `__main__` on every run,
    including `--help`, and the rest of this module already defers its
    service imports for that reason.
    """
    from code_helper.services.claude_settings import current_switch
    from code_helper.services.codex_default import current_default

    _AGENT_BACKENDS.update(
        {
            "claude": _AgentBackend(
                read_applied=current_switch,
                apply_wrapper="_apply_switch_wrapper",
                apply_native="_apply_switch_native",
                lifecycle="live",
            ),
            "codex": _AgentBackend(
                read_applied=current_default,
                apply_wrapper="_apply_set_default_wrapper",
                apply_native="_apply_set_default_native",
                lifecycle="next launch",
            ),
        }
    )
    # A method name is only as safe as its spelling: resolve every hook now,
    # at import, so a typo is an immediate ImportError rather than an
    # AttributeError the first time someone presses Enter on that row.
    for backend in _AGENT_BACKENDS.values():
        for hook in (backend.apply_wrapper, backend.apply_native):
            if not callable(getattr(TuiSession, hook, None)):
                raise AttributeError(f"TuiSession has no apply hook {hook!r}")


_register_agent_backends()
