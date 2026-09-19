"""Interactive menu for creating and maintaining codehelper wrappers.

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
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from codehelper.services.paths import Paths
    from codehelper.services.spec import WrapperSpec

__all__ = ["run_tui"]

_ADD = "add"
#: Main-screen action rows: add a new CLI integration (agent) and add a
#: wrapper for any agent. Distinct from the per-row `+ add` chip (`_ADD_CHIP`),
#: which is scoped to that row's agent; these are unscoped entry points.
#:
#: The values are namespaced with a ``:``-containing prefix so they can never
#: collide with a wrapper alias: ``validate_alias`` rejects ``:``, so a wrapper
#: literally named ``add-agent``/``add-wrapper`` (both valid aliases) stays
#: selectable as a wrapper instead of being shadowed by these action rows.
_ADD_AGENT = "__action:add-agent"
_ADD_WRAPPER = "__action:add-wrapper"
_SETTINGS = "settings"
_PROFILE = "profile"
_HELP = "help"
_QUIT = "quit"
_BACK: Final = "__back__"
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

#: The proxy row's value, and its two mutually-exclusive chips.
#:
#: Rendered like an agent row and driven by the SAME chip cursor, but it is
#: NOT in ``_AGENT_BACKENDS``: the proxy is not an agent backend, has no
#: wrapper, and applies to whatever backend is selected. What it borrows is
#: the chipset's disambiguation — showing every option with `✓` on the live
#: one. A single "Proxy: on" row would leave Enter ambiguous (is `on` the
#: state, or the button?); `✓ on   off` cannot be misread, because Enter
#: applies the chip under the cursor, exactly as it does one row above.
_PROXY_ROW = "proxy:"
_PROXY_ON = "on"
_PROXY_OFF = "off"
_PROXY_CHIPS = (_PROXY_ON, _PROXY_OFF)

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
        read_applied_model: ``(paths) -> model | None``, or ``None`` when this
            agent's config does not record a model this project can read back.
            Same never-raise contract as ``read_applied``, and it runs on the
            same once-per-iteration path. Optional because the two mechanisms
            genuinely differ: codex's ``config.toml`` names the model right
            next to the provider, while claude's richer readback is already
            covered by an exact-env match.
        apply_wrapper: Method name taking ``(spec)`` — point the agent at an
            installed wrapper's backend.
        apply_native: Method name taking no arguments — clear the override.
        lifecycle: When the change takes effect, rendered at the end of the
            row. The claude/codex difference (a running session vs. the next
            launch) is real and must be visible, not implied.
        chip_is_applied: Method name taking ``(agent_name, chip)`` — whether a
            backend chip is the one currently applied. Defaults to matching by
            provider name; claude overrides it with an exact-env match plus
            the chip's own token and codex with a provider+model match, each
            because its readback is richer than a provider name alone (see
            :meth:`_chip_is_applied_switch`, :meth:`_chip_is_applied_codex`).
        chip_switch_token: Method name taking ``(chip)`` — the token Enter on
            ``chip`` would apply, or ``None`` when it cannot resolve one.
            Feeds :meth:`_chip_is_applied_switch`: two accounts on one
            backend differ in nothing but the token, so a readback that
            ignores it marks every chip on that backend applied. Optional —
            only agents whose readback is token-exact carry it.
        exact_readback: Whether this agent's ``chip_is_applied`` hook is an
            EXACT comparison of the applied target — the property that makes
            Enter on a chip already reading applied a safe silent no-op.
            claude compares the full managed env plus the chip's own token;
            codex's provider+model pair cannot tell two same-axes wrappers
            apart, so codex keeps ``False`` and Enter always re-applies.
            The flag exists so ``_apply_chip`` stays free of an
            ``agent_name`` branch — the one that shipped there read
            ``agent_name == "claude"``.
    """

    read_applied: Callable[[Paths], str | None]
    apply_wrapper: str
    apply_native: str
    lifecycle: str
    chip_is_applied: str
    read_applied_model: Callable[[Paths], str | None] | None = None
    chip_switch_token: str | None = None
    exact_readback: bool = False


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
    from codehelper.cli.menu import MAX_DIGIT_ITEMS

    usable = min(numbered_count, MAX_DIGIT_ITEMS)
    if usable:
        digits = "1" if usable == 1 else f"1-{usable}"
        hint = f"Up/Down · {digits} · Enter select · Esc {exit_word} · Ctrl-C quit"
    elif chips:
        # TWO lines — `menu` accepts exactly one "\n" in a hint and counts
        # both rows into frame_lines, so the in-place redraw stays honest.
        # The editing keys are named here rather than left behind `?`: on the
        # main screen they are the only way to add, rename or change a
        # wrapper, and a key nobody can see is a key nobody presses. `a` is
        # split OUT of the `e/d` cluster rather than folded into "edit" —
        # `e` renames the focused wrapper, `t` rotates its token, so grouping
        # `a` under "edit" mislabels what pressing it does. Two lines are
        # what lets every name stay whole: cramming `s settings` into one
        # line forced `↵`/`d del`-style truncations and 80 columns was
        # exactly the budget `_fit` truncates past (the bug a PTY run caught
        # here). `p` is named too — bound in run()'s keys, pinned against
        # `_translate_char`, never before hinted.
        hint = (
            f"↑↓ row · ←→ chip · Enter apply · a add · e edit · t token · d delete\n"
            f"s settings · p profiles · ? · Esc {exit_word}"
        )
    else:
        hint = f"Up/Down · Enter select · Esc {exit_word} · Ctrl-C quit"
    if has_token_key:
        hint += " · t: token"
    return hint


def run_tui(args: argparse.Namespace) -> int:
    """Run the hierarchical interactive UI and always return a shell status."""
    return TuiSession(args).run()


class _Tee:
    """Fan writes out to the real stdout AND a capture buffer.

    Module-scoped (not nested in ``_run``) so the class is built once at import
    rather than recreated on every ``_run`` call — including silent chip
    applies, which never instantiate it. It closes over nothing from ``_run``.

    Must behave enough like the stream it replaces that code reached through
    the handler cannot tell the difference. ``isatty`` in particular is not
    optional: ``menu.read_line`` — which every confirmation prompt goes
    through — calls it to decide between raw-mode line editing and a plain
    ``input()``. Without it, any handler that asked for confirmation died with
    an ``AttributeError`` instead of prompting, which is exactly what happened
    to ``set-default``'s prompt from inside the TUI. It answers for the REAL
    terminal (the first stream), because that is where the prompt is actually
    rendered and read.
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
        "_applied_model",
        "_claude_active_env",
        "_proxy",
        "_chips",
        "_chip_index",
        "_chip_switch_tokens",
        "_tokens_reveal",
        "_state_snapshot",
        "_creds_snapshot",
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
        #: agent name -> the MODEL its config currently names, or None. Only
        #: an agent whose `_AgentBackend` carries `read_applied_model` appears
        #: here; the rest read as None and their predicate ignores it.
        self._applied_model: dict[str, str | None] = {}
        #: Managed Claude env snapshot used for exact chip readback.
        self._claude_active_env: dict[str, str] | None = None
        #: The proxy snapshot the chipset row renders (state + address).
        self._proxy = None
        #: agent name -> its chip strip (`native` plus installed wrappers).
        self._chips: dict[str, list] = {}
        #: agent name -> chip cursor. Ephemeral on purpose: persisting it
        #: would create a second source of truth about what is selected,
        #: competing with the config files that actually decide.
        self._chip_index: dict[str, int] = {}
        #: agent name -> {chip name -> the token Enter on that chip would
        #: apply, or None}. Only an agent whose `_AgentBackend` carries
        #: `chip_switch_token` appears here. Filled once per iteration; the
        #: chip_is_applied predicate reads ONLY this on a redraw frame.
        self._chip_switch_tokens: dict[str, dict[str, str | None]] = {}
        #: The Tokens screen's show/hide state. PER-VISIT: `_run_tokens_screen`
        #: resets it to False on every entry, so a re-visit can never open
        #: showing full credentials with no new action by the user.
        self._tokens_reveal: bool = False
        #: The per-iteration ``state.json`` snapshot (issue #110): loaded ONCE
        #: by `_refresh_active_label` and fed to every reader that would
        #: otherwise re-read the file — the profile resolvers, the disabled
        #: gates, the proxy row, the per-agent default-wrapper lookups.
        self._state_snapshot: dict[str, object] = {}
        #: The per-iteration ``credentials.json`` snapshot (issue #110), fed
        #: to the profile resolvers and the chip-token cache leg the same way.
        self._creds_snapshot: dict[str, dict[str, str]] = {}

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
        from codehelper.cli.menu import MenuCancelled, Section, select_from_menu

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
        from codehelper.cli.menu import press_any_key

        if text:
            print(text)
        press_any_key("Press any key to continue...")

    def _run(
        self, handler, request=None, *, silent: bool = False, live: bool = False
    ) -> bool:
        """Dispatch a CLI handler/request and show its output exactly once.

        Three output modes:

        - **default** — stdout is captured and, on success, replayed once by
          ``_notify`` (transcript + pause). Correct for handlers that only
          print: nothing appears until the finished transcript can be shown,
          and it survives the next redraw.
        - **live=True** — stdout is teed to the real terminal, for handlers
          that print a prompt and then block on ``input()`` (set-default's
          config confirm, remove's file list, add's foreign-overwrite
          confirm): the prompt must be visible before the blocking read.
          The transcript is therefore already on screen — success replays
          nothing, ``_notify("")`` keeps just the pause.
        - **silent=True** — capture without any display: the chip hot-apply
          path, where the redraw itself reflects the new state. An error is
          still surfaced and paused on; a failed apply must never be silent.

        On an error the ``error:`` line is printed first; in default mode the
        captured transcript is replayed after it (diagnostic), while in live
        mode the transcript is already on screen from the tee and only the
        pause is shown.

        Returns True iff the handler completed without an error — callers
        that synthesize their own success message must gate it on this, not
        on inspecting the filesystem afterwards (the handler may have failed
        on a collision with an existing target that still exists).
        """
        import io
        import sys

        from codehelper.errors import CodeHelperError

        buffer = io.StringIO()
        real_stdout = sys.stdout
        # live tees (prompts must be visible mid-flow); default and silent
        # capture only — the transcript is shown once, after the handler.
        sys.stdout = _Tee(real_stdout, buffer) if live and not silent else buffer
        try:
            handler(self.args if request is None else request)
        except CodeHelperError as exc:
            if getattr(self.args, "debug", False):
                sys.stdout = real_stdout
                raise
            sys.stdout = real_stdout
            print(f"error: {exc}")
            if live and not silent:
                # live: the transcript is already on screen from the tee —
                # replaying it here would show it twice (the #97 symptom on
                # the error path). The pause alone keeps it readable.
                self._notify("")
            else:
                self._notify(buffer.getvalue().rstrip())
            return False
        finally:
            sys.stdout = real_stdout
        if live and not silent:
            self._notify("")
        elif not silent:
            self._notify(buffer.getvalue().rstrip())
        return True

    def _read_text(self, prompt: str) -> str | None:
        from codehelper.cli.menu import MenuCancelled, read_line

        try:
            return read_line(prompt)
        except MenuCancelled as exc:
            if exc.hard:
                raise
            return None

    def _read_token(self, prompt: str) -> str | None:
        from codehelper.cli.menu import MenuCancelled, read_line

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
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import (
            profile_names,
            seed_default_profile,
        )
        from codehelper.services.wrappers import (
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
    ) -> ProfileChoice | Literal["__back__"] | None:
        """Collect a new profile and token before model discovery."""
        from codehelper.services.profiles import (
            NewProfileOutcome,
            classify_new_profile,
            validate_new_profile_name,
        )
        from codehelper.services.secrets import DEFAULT_PROFILE

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
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import (
            DEFAULT_PROFILE,
            profile_names,
            valid_active_profile,
        )

        while True:
            self._recover_default(provider_name)
            names = list(profile_names(Paths.default(), provider_name))
            if not names:
                created = self._new_profile(names, provider_name)
                return None if created == _BACK else created

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

    def _all_wrapper_specs(self, *, state: dict[str, object] | None = None):
        """Presets plus managed constructor wrappers, without duplicate names.

        Wrappers of a runtime-disabled provider are excluded (issue #89) —
        the provider vanishes from the chipset and this list together.

        ``state`` is the per-iteration snapshot (issue #110) for the
        disabled-provider filter; ``None`` falls back to the session's
        :attr:`_state_snapshot`, so the per-iteration row builder never
        re-reads ``state.json``.
        """
        from codehelper.services.model import is_provider_disabled
        from codehelper.services.paths import Paths
        from codehelper.services.state import disabled_providers
        from codehelper.services.wrappers import (
            WRAPPERS,
            discover_managed,
            spec_from_installed,
        )

        paths = Paths.default()
        disabled = disabled_providers(paths, state=state or self._state_snapshot)
        specs = []
        known = set()
        for spec in WRAPPERS:
            resolved = spec_from_installed(paths, spec.name) or spec
            if is_provider_disabled(resolved.provider, disabled):
                continue
            specs.append(resolved)
            known.add(resolved.name)
        for name in discover_managed(paths):
            if name in known:
                continue
            spec = spec_from_installed(paths, name)
            if spec and not is_provider_disabled(spec.provider, disabled):
                specs.append(spec)
                known.add(name)
        return specs

    def _wrapper_rows(self, paths) -> list:
        """Menu items for the main screen's wrapper list, grouped by agent."""
        from codehelper.cli.menu import Section
        from codehelper.services.agents import all_agents
        from codehelper.services.wrappers import (
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
                agent.name: valid_default_wrapper(
                    paths, agent.name, state=self._state_snapshot
                )
                for agent in agents
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

    def _resolve_spec(self, alias: str) -> WrapperSpec | None:
        """Resolve ``alias`` to a spec — installed wrapper first, else preset."""
        from codehelper.errors import CodeHelperError
        from codehelper.services.paths import Paths
        from codehelper.services.wrappers import get_spec, spec_from_installed

        try:
            return spec_from_installed(Paths.default(), alias) or get_spec(alias)
        except CodeHelperError as exc:
            self._notify(f"error: {exc}")
            return None

    # --- actions ---------------------------------------------------------

    def _on_token(self, alias: str) -> None:
        """Rotate the token of wrapper ``alias`` (``t`` on the main screen)."""
        from codehelper.cli.parser import _handle_edit_token

        spec = self._resolve_spec(alias)
        if spec is None:
            return
        if spec.auth != "secret":
            return
        profile = self._select_profile(spec.provider.name, editing=True)
        if profile is None:
            return
        from codehelper.cli.requests import EditTokenRequest

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

    def _chip_silent(self) -> bool:
        """A chip press is silent only for a real write.

        The silence is justified by the chipset redraw already showing the new
        [applied] state. A dry run writes nothing, so the redraw does not
        change — silence would swallow the only feedback (the preview). In
        dry-run the chip must surface its preview and pause like a normal run.
        """
        return not getattr(self.args, "dry_run", False)

    def _apply_switch_wrapper(self, spec) -> None:
        """claude: retarget the RUNNING session at ``spec``'s backend."""
        from codehelper.cli.parser import _handle_switch
        from codehelper.services.spec import preset_names

        source = (
            self._switch_request(from_preset=spec.name)
            if spec.name in preset_names()
            else self._switch_request(from_wrapper=spec.name)
        )
        # silent: a force=True switch chip never prompts — no echo, no pause.
        self._run(_handle_switch, source, silent=self._chip_silent())

    def _apply_switch_native(self) -> None:
        """claude: clear the managed ``env`` block from settings.json."""
        from codehelper.cli.parser import _handle_switch

        self._run(
            _handle_switch,
            self._switch_request(provider=_NATIVE_CHIP),
            silent=self._chip_silent(),
        )

    def _switch_request(self, *, provider=None, from_wrapper=None, from_preset=None):
        """A :class:`SwitchRequest` carrying only the axis a chip varies.

        Every other field is meaningless from the main screen (no model or
        token prompts, no restore), so the shared fields are filled once here
        instead of at each call site — which also keeps the two apply paths
        from drifting apart if ``SwitchRequest`` grows another field.
        """
        from codehelper.cli.requests import SwitchRequest

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
            # Chips are hot-apply controls.  The target has already been
            # resolved and validated locally; a second interactive prompt
            # turns a one-key backend switch into a blocking CLI flow.
            force=True,
            debug=getattr(self.args, "debug", False),
            from_preset=from_preset,
        )

    def _apply_set_default_wrapper(self, spec) -> None:
        """codex: patch config.toml so the NEXT launch uses ``spec``."""
        from codehelper.cli.parser import _handle_set_default
        from codehelper.cli.requests import SetDefaultRequest
        from codehelper.services.model import BaseUrlPolicy

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
                dry_run=getattr(self.args, "dry_run", False),
                force=False,
                debug=getattr(self.args, "debug", False),
            ),
            live=True,
        )

    def _apply_set_default_native(self) -> None:
        """codex: remove the managed region from config.toml.

        Deliberately NOT ``set-default --restore``: restore rolls the file
        back to a backup snapshot, undoing unrelated hand-edits made since.
        This removes only the region this tool owns — "stop overriding",
        not "undo my last change". Routes through ``_handle_set_default``'s
        ``native`` operation (issue #47) — the same handler path the CLI
        ``set-default --native`` takes, never a second implementation.
        """
        from codehelper.cli.parser import _handle_set_default
        from codehelper.cli.requests import SetDefaultRequest

        # force stays False: unlike the hot-apply chips, this keep-the-prompt
        # routing reuses the handler's confirm gate — same diff-first prompt,
        # and (the part a local re-implementation would have silently
        # dropped) the same off-a-TTY refusal, so a scripted run can never be
        # talked into patching config.toml through the TUI.
        self._run(
            _handle_set_default,
            SetDefaultRequest(
                agent=None,
                provider=None,
                model=None,
                base_url=None,
                restore=False,
                slot=None,
                native=True,
                dry_run=getattr(self.args, "dry_run", False),
                force=False,
                debug=getattr(self.args, "debug", False),
            ),
            live=True,
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
        from codehelper.services.model import AuthPolicy, compatible_providers
        from codehelper.services.paths import Paths
        from codehelper.services.state import disabled_providers

        items: list[tuple[str, str]] = []
        choices = {}
        # Disabled providers are not offered (issue #89) — compatible_providers
        # carries the filter so an OVERRIDABLE provider's `:secret` twin
        # (the same provider re-keyed) cannot leak back in below it.
        disabled = disabled_providers(Paths.default())
        for provider in compatible_providers(agent, disabled=disabled):
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
        from codehelper.errors import CodeHelperError
        from codehelper.services.model import BaseUrlPolicy, with_auth, with_base_url

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
        from codehelper.services.models_api import list_models
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import token_for_discovery

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
        # Fall back to the registry's known_models only when discovery
        # actually FAILED (not result.ok) — a successful discovery that
        # legitimately returned zero models is a real answer, not a reason to
        # substitute the built-in list and mislabel it "discovery unavailable".
        using_known = not result.ok and provider.known_models
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
        from codehelper.cli.parser import _handle_add
        from codehelper.cli.requests import AddRequest
        from codehelper.errors import CodeHelperError
        from codehelper.services.spec import suggest_alias

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
            # live: the foreign-overwrite confirm must be visible before the
            # blocking read.
            live=True,
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
        from codehelper.services.agents import get_agent
        from codehelper.services.paths import Paths

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
        from codehelper.services.agents import all_agents
        from codehelper.services.paths import Paths

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
        from codehelper.errors import CodeHelperError
        from codehelper.services.agents import add_user_agent
        from codehelper.services.paths import Paths

        name = (
            self._read_text("Agent name (its executable name on PATH): ") or ""
        ).strip()
        if not name:
            return
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
        """The ``s`` screen: the proxy EDITORS plus the two session flags.

        On/off deliberately does NOT live here — the main screen's proxy
        chipset row owns it, and a second toggle would be two controls for
        one setting. What remains is what the chipset row cannot express:
        the address itself and the NO_PROXY bypass list. Both dispatch into
        ``parser._handle_proxy`` with a ``ProxyRequest``, the same handler
        the CLI uses.
        """
        from codehelper.cli.parser import _handle_proxy
        from codehelper.services.paths import Paths
        from codehelper.services.proxy import proxy_status

        while True:
            debug = getattr(self.args, "debug", False)
            status = proxy_status(Paths.default())
            choice = self._pick(
                [
                    (
                        "proxy-url",
                        f"Proxy address: {status.display_restorable_url or '(none)'}",
                    ),
                    ("proxy-no-proxy", f"NO_PROXY: {status.no_proxy or '(none)'}"),
                    ("providers", "Providers"),
                    ("tokens", "Tokens"),
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
            elif choice == "dry-run":
                self.args.dry_run = not getattr(self.args, "dry_run", False)
            elif choice == "tokens":
                self._run_tokens_screen()
            elif choice == "providers":
                self._run_providers_screen()
            elif choice == "proxy-url":
                entered = self._read_text("Proxy URL (e.g. http://127.0.0.1:8118): ")
                if entered is None or not entered.strip():
                    continue
                self._run(_handle_proxy, self._proxy_request(url=entered.strip()))
            elif choice == "proxy-no-proxy":
                entered = self._read_text(
                    "NO_PROXY (comma- or space-separated, '*' bypasses all): "
                )
                if entered is None:
                    continue
                self._run(_handle_proxy, self._proxy_request(no_proxy=entered.strip()))

    def _run_providers_screen(self) -> None:
        """Settings → Providers: runtime disable/enable per provider (issue #89).

        One row per non-``env_reset`` registry provider, labelled with its
        current state; Enter toggles by dispatching the REAL
        ``parser._handle_disable``/``_handle_enable`` — the pick IS the
        confirmation (``yes=True``), the same opt-in a chip press makes. The
        submenu loops, so the label of the just-toggled row doubles as the
        operation's result. ``env_reset`` providers (``native``) are absent:
        they are the agent's own backend-clearing entry and ``disable``
        refuses them — showing an action that can only fail would be noise.
        """
        from codehelper.cli.parser import _handle_disable, _handle_enable
        from codehelper.cli.requests import DisableRequest, EnableRequest
        from codehelper.services.model import PROVIDERS, is_provider_disabled
        from codehelper.services.paths import Paths
        from codehelper.services.state import disabled_providers

        while True:
            paths = Paths.default()
            disabled = disabled_providers(paths)
            dry_run = getattr(self.args, "dry_run", False)
            debug = getattr(self.args, "debug", False)
            rows = []
            states = {}
            for provider in PROVIDERS:
                if provider.env_reset:
                    continue
                off = is_provider_disabled(provider, disabled)
                states[provider.name] = off
                rows.append(
                    (
                        provider.name,
                        f"{provider.name}: {'disabled' if off else 'enabled'}"
                        f"{'  (suspended)' if provider.suspended else ''}",
                    )
                )
            choice = self._pick([*rows, (_BACK, "Back")], "Providers:")
            if choice == _BACK:
                return
            if states.get(choice):
                self._run(
                    _handle_enable,
                    EnableRequest(name=choice, dry_run=dry_run, debug=debug),
                )
            else:
                self._run(
                    _handle_disable,
                    DisableRequest(
                        name=choice,
                        # The submenu pick IS the confirmation.
                        yes=True,
                        force=False,
                        dry_run=dry_run,
                        debug=debug,
                    ),
                )

    def _proxy_request(self, **overrides) -> object:
        """A ``ProxyRequest`` carrying this session's flags plus ``overrides``.

        ``force=True`` for the same reason a chip press hot-applies: the user
        already chose the action on a screen that showed the current state,
        so a second yes/no prompt would be asking them to confirm the
        keystroke they just made. ``--dry-run`` still short-circuits the write
        inside the service.
        """
        from codehelper.cli.requests import ProxyRequest

        fields = {
            "action": None,
            "url": None,
            "no_proxy": None,
            "status": False,
            "dry_run": getattr(self.args, "dry_run", False),
            "force": True,
            "debug": getattr(self.args, "debug", False),
        }
        fields.update(overrides)
        return ProxyRequest(**fields)

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
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import (
            DEFAULT_PROFILE,
            profile_names,
            valid_active_profile,
        )
        from codehelper.services.state import set_active_selection

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
        rename_keys: dict[str, Callable[[str], object]] = {
            "e": lambda value: None if value == _BACK else f"rename:{value}"
        }
        while True:
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
            choice = self._pick(
                items,
                f"Active profile for {provider} (e: rename):",
                on_key=rename_keys,
            )
            if choice == _BACK:
                return
            if isinstance(choice, str) and choice.startswith("rename:"):
                from codehelper.cli.parser import _handle_rename
                from codehelper.cli.requests import RenameRequest

                old_name = choice.removeprefix("rename:")
                new_name = self._read_text(f"New name for {old_name}: ")
                if not new_name or new_name == old_name:
                    continue
                # silent: same no-double-echo contract as the wrapper rename.
                renamed = self._run(
                    _handle_rename,
                    RenameRequest(
                        kind="profile",
                        provider=provider,
                        name=old_name,
                        new_name=new_name,
                        dry_run=getattr(self.args, "dry_run", False),
                        debug=getattr(self.args, "debug", False),
                    ),
                    silent=True,
                )
                names = list(profile_names(paths, provider))
                if renamed:
                    self._notify(f"renamed profile {old_name} -> {new_name}")
                continue
            set_active_selection(paths, provider, choice)
            return

    def _run_tokens_screen(self) -> None:
        """The read-only Tokens screen — the Settings screen's viewer twin of
        the CLI's ``codehelper tokens``.

        Lists every cached credential (``credentials.json``, provider ×
        profile, active selection marked) plus one row per registry token
        env var, masked by default via ``secrets.mask_token``. The ``s`` key
        toggles reveal (``s`` is already in ``menu._PASSTHROUGH`` — see the
        shipped-once ``w`` bug this project pins in test_tui.py); Enter on a
        row does nothing on purpose: this screen edits nothing, rotation
        lives on ``edit-token``/the ``t`` action, removal on ``remove``.

        Rows are read ONCE per entry (a credential store the user is only
        looking at does not change mid-screen), so the redraw loop does no
        I/O — the toggle mutates ``self._tokens_reveal`` and lets the menu
        redraw with re-formatted labels. The on_key handler MUST return
        None: ``menu._dispatch_key`` turns a non-None return into a menu
        selection, and a toggle is not a selection.

        Reveal is PER-VISIT state, reset on every entry: a second visit must
        never open showing full credentials with no new action by the user —
        that is the accident the mask exists to prevent. The rows come from
        ``parser._token_view_rows``, the same source the CLI command renders,
        so the two views cannot drift (including the ``(nothing cached)``
        row when only env vars exist).
        """
        from codehelper.cli.parser import _token_view_rows
        from codehelper.services.paths import Paths

        # Per-visit reset FIRST — before anything can render a frame.
        self._tokens_reveal = False

        paths = Paths.default()
        rows, env_rows = _token_view_rows(paths)

        while True:
            reveal = self._tokens_reveal

            items: list[object] = [
                (
                    f"{provider_name}/{profile_name}",
                    f"{provider_name}/{profile_name}: {self._shown(token, reveal)}"
                    + ("  ← active" if is_active else ""),
                )
                for provider_name, profile_name, token, is_active in rows
            ]
            if not rows:
                # CLI parity: name the empty cache explicitly, so an
                # env-vars-only screen does not read as "the store is empty".
                items.append(("no-cache", "(nothing cached)"))
            items += [
                (
                    f"env/{env_var}",
                    f"{env_var}: "
                    + (self._shown(value, reveal) if value else "not set"),
                )
                for env_var, value in env_rows
            ]
            items += [
                ("toggle-reveal", f"Reveal values: {'on' if reveal else 'off'}"),
                (_BACK, "Back"),
            ]
            keys: dict[str, Callable[[str], object]] = {
                "s": lambda _value: self._toggle_tokens_reveal()
            }
            choice = self._pick(
                items,
                "Stored tokens (s: show/hide, Enter: back):",
                on_key=keys,
                numbered=False,
            )
            if choice == _BACK:
                return
            if choice == "toggle-reveal":
                # Two ways to the same toggle, like the Settings screen's
                # Debug/Dry-run rows: the `s` key or Enter on the row itself.
                self._toggle_tokens_reveal()
                continue
            # Enter on a token row is a no-op — this screen is a viewer.
            # The loop redraws unchanged.

    @staticmethod
    def _shown(value: str, reveal: bool) -> str:
        """The screen's current rendering of one token value — the shared
        ``secrets.render_token``, so this view and the CLI command cannot
        drift on how a value is displayed."""
        from codehelper.services.secrets import render_token

        return render_token(value, reveal)

    def _toggle_tokens_reveal(self) -> None:
        """Flip the Tokens screen's show/hide state; returns None so
        ``menu._dispatch_key`` treats ``s`` as state mutation, not selection.
        """
        self._tokens_reveal = not self._tokens_reveal

    # --- active-label subsystem -----------------------------------------

    @staticmethod
    def _secret_providers(*, state: dict[str, object] | None = None) -> list:
        """Providers that can have named token profiles at all.

        NOT just ``p.auth == "secret"``: an OVERRIDABLE provider's registry
        entry keeps its default ``auth`` (``with_auth`` returns a RUNTIME
        copy, never mutates ``PROVIDERS``). Filtering on ``auth_policy``
        keeps the Profile screen and Tab's fallback scan able to find
        profiles cached under an OVERRIDABLE provider.

        ``state`` is the per-iteration snapshot (issue #110); ``None`` reads
        ``state.json`` — the event-path callers keep their fresh read.
        """
        from codehelper.services.model import AuthPolicy, active_providers
        from codehelper.services.paths import Paths
        from codehelper.services.state import disabled_providers

        disabled = disabled_providers(Paths.default(), state=state)
        # A disabled provider's profiles are not offered as choices (issue
        # #89) — the tokens view hides its rows the same way.
        return [
            p
            for p in active_providers(disabled)
            if p.auth == "secret" or p.auth_policy is AuthPolicy.OVERRIDABLE
        ]

    def _resolve_tab_provider(
        self,
        *,
        state: dict[str, object] | None = None,
        creds: dict[str, dict[str, str]] | None = None,
    ) -> str | None:
        """Resolve which provider Tab/the header should act on.

        ``state`` / ``creds`` are the per-iteration snapshots (issue #110);
        ``None`` reads the files — the event-path callers keep their fresh
        read.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import profile_names
        from codehelper.services.state import active_selection

        paths = Paths.default()
        selection = active_selection(paths, state=state)
        if selection is not None:
            stored, _ = selection
            if profile_names(paths, stored, creds=creds):
                return stored
        for provider in self._secret_providers(state=state):
            if profile_names(paths, provider.name, creds=creds):
                return provider.name
        return None

    def _tab_profile(
        self,
        provider: str,
        *,
        state: dict[str, object] | None = None,
        creds: dict[str, dict[str, str]] | None = None,
    ) -> str | None:
        """The profile Tab currently shows/would land on for ``provider``.

        ``state`` / ``creds`` are the per-iteration snapshots (issue #110);
        ``None`` reads the files — the event-path callers keep their fresh
        read.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import profile_names, valid_active_profile

        paths = Paths.default()
        stored = valid_active_profile(paths, provider, state=state, creds=creds)
        if stored:
            return stored
        names = profile_names(paths, provider, creds=creds)
        return names[0] if names else None

    def _active_label(
        self,
        tab_provider: str | None,
        *,
        state: dict[str, object] | None = None,
        creds: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """``"provider/profile"``, ``"provider"``, or ``""`` for header/row."""
        if not tab_provider:
            return ""
        profile = self._tab_profile(tab_provider, state=state, creds=creds)
        return f"{tab_provider}/{profile}" if profile else tab_provider

    def _on_tab(self) -> None:
        """Cycle the active profile of the current provider on Tab."""
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import profile_names, valid_active_profile
        from codehelper.services.state import set_active_selection

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
        from codehelper.services.paths import Paths
        from codehelper.services.profiles import profile_slots
        from codehelper.services.state import set_active_selection

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
        from codehelper.cli.menu import Section

        return Section(lambda: self._slot_label)

    def _refresh_profile_label(
        self,
        *,
        state: dict[str, object] | None = None,
        creds: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Refresh only the profile-related caches.

        Split from :meth:`_refresh_active_label` because Tab and the digit
        slots change a PROFILE, not a backend — re-reading each agent's
        config file there would be work no keypress on that screen can
        invalidate.

        ``state`` / ``creds`` are the per-iteration snapshots (issue #110);
        ``None`` reads the files — the event-path callers (Tab, digit slots,
        the Profile screen) keep their fresh read.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.profiles import profile_slots

        self._tab_provider = self._resolve_tab_provider(state=state, creds=creds)
        self._tab_label = self._active_label(
            self._tab_provider, state=state, creds=creds
        )
        self._slot_label = "  ".join(
            f"{i + 1} {provider}/{profile}"
            for i, (provider, profile) in enumerate(
                profile_slots(Paths.default(), creds=creds)
            )
        )

    def _refresh_active_label(self) -> None:
        """Refresh every per-iteration cache the main screen reads.

        Called ONCE per main-loop iteration. Everything filled here does real
        I/O (a profile scan, plus one config read per agent), and every
        consumer is a label callable that the menu re-evaluates on EVERY
        redraw frame — including pure cursor movement. Reading any of it from
        those callables would turn one screen's worth of I/O into one
        keystroke's worth.

        The per-file stores are also loaded ONCE here and fed to every
        reader below as an optional preloaded kwarg (issue #110): one
        ``load_state`` instead of one per profile resolver / disabled gate /
        proxy row / default-wrapper lookup. The readers default to reading
        the file themselves, so nothing off this loop changes.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import load_credentials
        from codehelper.services.state import load_state

        paths = Paths.default()
        self._state_snapshot = load_state(paths)
        self._creds_snapshot = load_credentials(paths)
        self._refresh_profile_label(
            state=self._state_snapshot, creds=self._creds_snapshot
        )
        self._applied = {
            name: backend.read_applied(paths)
            for name, backend in _AGENT_BACKENDS.items()
        }
        self._applied_model = {
            name: backend.read_applied_model(paths)
            for name, backend in _AGENT_BACKENDS.items()
            if backend.read_applied_model is not None
        }
        from codehelper.services.claude_settings import active_switch_env
        from codehelper.services.proxy import proxy_status

        self._claude_active_env = active_switch_env(paths)
        # Read here, never from the proxy row's label callable: the menu
        # re-evaluates that on every redraw frame, cursor movement included.
        self._proxy = proxy_status(paths, state=self._state_snapshot)
        self._chips = {
            name: self._chips_for(name, paths, state=self._state_snapshot)
            for name in _AGENT_BACKENDS
        }
        # Token readback for the agents whose chip_is_applied is token-exact.
        # Same I/O class as the reads above — once per iteration, never per
        # redraw frame — and resolved through the apply-path resolvers so the
        # ✓ cannot drift from what a chip press actually writes.
        self._chip_switch_tokens = {
            name: {
                self._chip_name(chip): getattr(self, backend.chip_switch_token)(
                    chip, paths, state=self._state_snapshot, creds=self._creds_snapshot
                )
                for chip in chips
            }
            for name, backend in _AGENT_BACKENDS.items()
            if backend.chip_switch_token is not None
            for chips in (self._chips[name],)
        }
        # A wrapper can be removed (`d`) or added (`a`) between iterations, so
        # a chip cursor parked past the end of a now-shorter strip is normal,
        # not a bug — clamp rather than reset, so an unaffected row keeps its
        # position.
        for name, chips in self._chips.items():
            if self._chip_index.get(name, 0) >= len(chips):
                self._chip_index[name] = max(0, len(chips) - 1)

    def _chips_for(
        self,
        agent_name: str,
        paths,
        *,
        state: dict[str, object] | None = None,
    ) -> list:
        """The chip strip for ``agent_name``: native, targets, then Add.

        Curated Claude presets are live backend targets in their own right;
        they do not depend on an optional launch-wrapper file.  Managed ad-hoc
        wrappers add their recovered target to the same strip.

        The trailing :data:`_ADD_CHIP` is the one deliberate exception: not a
        backend, never "applied", present on every row (including one with no
        wrappers yet) so that row always has a visible way to get its first
        one instead of reading as empty/broken.

        ``state`` is the per-iteration snapshot (issue #110) for the
        disabled-provider filter; ``None`` reads ``state.json``.
        """
        from codehelper.services.model import is_provider_disabled
        from codehelper.services.spec import PRESETS, spec_from_preset
        from codehelper.services.state import disabled_providers
        from codehelper.services.wrappers import discover_managed, spec_from_installed

        disabled = disabled_providers(paths, state=state)

        def _live(spec) -> bool:
            # A disabled provider produces NO chip at all (issue #89) — not a
            # greyed-out one: its wrappers were deleted on disable, so a chip
            # here would promise an Enter that cannot resolve.
            return not is_provider_disabled(spec.provider, disabled)

        presets = []
        for preset in PRESETS:
            if preset.agent != agent_name:
                continue
            spec = spec_from_preset(preset)
            if _live(spec):
                presets.append(spec)
        # A managed constructor wrapper has no registry entry and therefore
        # is an additional chip.  A preset name is deliberately not recovered
        # from disk: its chip means the canonical preset, never whatever file
        # happens to have claimed the same alias.  ``discover_managed`` already
        # excludes preset names, so every name here is an ad-hoc wrapper.
        ad_hoc = [
            spec
            for name in discover_managed(paths)
            if (spec := spec_from_installed(paths, name)) is not None
            and spec.agent.name == agent_name
            and _live(spec)
        ]
        return [_NATIVE_CHIP, *presets, *ad_hoc, _ADD_CHIP]

    @staticmethod
    def _chip_name(chip) -> str:
        return chip if isinstance(chip, str) else chip.name

    def _chip_is_applied(self, agent_name: str, chip) -> bool:
        """Whether ``chip`` is the backend currently in the agent's config.

        A chip is a wrapper while the config records a backend, so how
        precisely the two can be matched is per-agent and lives in
        ``_AgentBackend.chip_is_applied``, resolved here by method name —
        never an ``if agent.name == ...`` branch. Each agent matches as
        exactly as its config allows: claude by exact env plus the chip's own
        token (:meth:`_chip_is_applied_switch`), codex by provider AND model
        (:meth:`_chip_is_applied_codex`). ``_chip_is_applied_provider`` — the
        provider-name-only default — is the weakest of the three and marks
        EVERY chip sharing that provider, not merely the first: the predicate
        runs per chip with nothing tracking order. Any agent whose config
        records more than a provider must therefore carry its own predicate
        rather than fall back to it. The action chip is never applied — it
        isn't a backend at all.
        """
        if chip == _ADD_CHIP:
            return False
        applied = self._applied.get(agent_name)
        if chip == _NATIVE_CHIP:
            return applied is None
        backend = _AGENT_BACKENDS[agent_name]
        return getattr(self, backend.chip_is_applied)(agent_name, chip)

    def _chip_is_applied_provider(self, agent_name: str, chip) -> bool:
        """Default readback: a backend chip matches when its provider is applied."""
        applied = self._applied.get(agent_name)
        return applied is not None and chip.provider.name == applied

    def _chip_is_applied_codex(self, agent_name: str, chip) -> bool:
        """codex readback: the applied chip is the provider AND model pair.

        ``config.toml`` records ``model`` next to ``model_provider``, and
        ``_patch_value_for`` writes the wrapper's own ``spec.model`` there
        verbatim — so equality against ``chip.model`` is exact, not a
        heuristic. Matching on the provider alone marked EVERY chip sharing
        that provider (the predicate is evaluated per chip, with nothing
        tracking which came first), which is how two codex wrappers on one
        Ollama endpoint both rendered as applied.

        A config that names a provider but no readable ``model`` falls back to
        the provider-only answer rather than reporting nothing applied: a
        hand-written ``config.toml`` may legitimately omit the key, and
        silently dropping the ``✓`` would read as "native", which is a
        different and equally wrong claim.
        """
        if not self._chip_is_applied_provider(agent_name, chip):
            return False
        model = self._applied_model.get(agent_name)
        return model is None or chip.model == model

    def _chip_switch_token(
        self,
        chip,
        paths,
        *,
        state: dict[str, object] | None = None,
        creds: dict[str, dict[str, str]] | None = None,
    ) -> str | None:
        """The token Enter on ``chip`` would apply, or ``None`` when it has none.

        Asked through the SAME resolvers the apply path uses —
        ``_switch_axes_from_preset`` for a preset chip,
        ``_switch_axes_from_wrapper`` for an installed one — so the readback
        and the write can never disagree about which account a chip names. A
        resolver raising is the apply path failing cleanly (no non-interactive
        token available, unreadable wrapper), which is NOT a no-op: ``None``,
        never a guess.

        ``state`` / ``creds`` are the per-iteration snapshots (issue #110)
        for the resolvers' disabled gate and cache leg; ``None`` reads the
        files.
        """
        from codehelper.cli.parser import (
            _switch_axes_from_preset,
            _switch_axes_from_wrapper,
        )
        from codehelper.errors import CodeHelperError
        from codehelper.services.spec import preset_names

        if isinstance(chip, str):  # native / + add — not backends
            return None
        try:
            if chip.name in preset_names():
                _, _, token, _, _ = _switch_axes_from_preset(
                    self._switch_request(from_preset=chip.name),
                    paths,
                    state=state,
                    creds=creds,
                )
            else:
                _, _, token, _, _ = _switch_axes_from_wrapper(
                    self._switch_request(from_wrapper=chip.name),
                    paths,
                    state=state,
                )
        except CodeHelperError:
            return None
        return token or None

    def _chip_is_applied_switch(self, agent_name: str, chip) -> bool:
        """claude readback: the managed env equals exactly what Enter writes.

        ``matches_switch_spec`` checks the axes (provider, tier models,
        subagent) against the managed snapshot — but it builds the expected
        env with the LIVE token substituted in, which used to leave every
        chip on one backend indistinguishable: two accounts on the same
        provider with the same tier models differ in NOTHING but the token,
        and all of them rendered ``✓`` at once (three zai chips shipped
        doing exactly that). So the token comes from the CHIP instead, via
        the same resolver a chip press uses (:meth:`_chip_switch_token`):
        applied now means Enter would be a no-op, which is also exactly when
        the ``_apply_chip`` guard is allowed to swallow the press.

        The comparison is uniform across auth modes. A literal chip also
        resolves a token at apply time — its ``auth_value``, through the very
        same resolvers — so an early ``True`` for non-secret chips claimed a
        no-op Enter never was: on an OVERRIDABLE provider (ollama-direct) a
        secret wrapper and the literal preset share provider AND tier models,
        and only the token says whose credential Enter would write. This
        shipped as exactly that shortcut and marked every such chip applied
        at once.

        Both inputs are refreshed once per main-loop iteration by
        ``_refresh_active_label`` — this reads only caches, never the
        filesystem on a redraw frame.
        """
        from codehelper.services.claude_settings import matches_switch_spec

        env = self._claude_active_env
        if not matches_switch_spec(env, chip):
            return False
        chip_token = self._chip_switch_tokens.get(agent_name, {}).get(
            self._chip_name(chip)
        )
        return chip_token is not None and chip_token == (env or {}).get(
            "ANTHROPIC_AUTH_TOKEN", ""
        )

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

    def _proxy_row(self) -> Callable[..., str]:
        """A label callable rendering the proxy row's ``on``/``off`` chips.

        Deliberately the same shape as :meth:`_chip_row`: `✓` on the live
        option, reverse video under the cursor, bold when applied but not
        focused. Reads only the caches ``_refresh_active_label`` fills — this
        is re-evaluated on every redraw frame, cursor movement included.

        The address rides along as a dim tail when one is known, because
        "off" alone leaves the obvious next question (off from what?)
        unanswered, and it is what makes `on` mean something specific.
        """

        def label(*, selected: bool = True, ansi: bool = True) -> str:
            cursor = self._chip_index.get(_PROXY_ROW, 0)
            rendered = []
            for index, chip in enumerate(_PROXY_CHIPS):
                applied = (chip == _PROXY_ON) == bool(
                    self._proxy and self._proxy.enabled
                )
                text = f"✓ {chip}" if applied else chip
                if selected and index == cursor:
                    rendered.append(
                        f"{_REVERSE}{text}{_RESET}" if ansi else f"[{text}]"
                    )
                elif applied:
                    rendered.append(f"{_BOLD}{text}{_RESET}" if ansi else text)
                else:
                    rendered.append(text)
            row = f"{'proxy':<8}{'  '.join(rendered)}"
            address = self._proxy.display_restorable_url if self._proxy else ""
            if address:
                tail = f"  {address}"
                row += f"{_DIM}{tail}{_RESET}" if ansi else tail
            return row

        return label

    def _apply_proxy_chip(self) -> None:
        """Apply the proxy chip under the cursor (Enter on the proxy row).

        Applying the state that is already live is a silent no-op, matching
        how a selected claude chip behaves — Enter on `✓ on` should not
        rewrite the file or flash a message.

        Turning on with no address anywhere is the one case that cannot just
        act: the handler would raise, so ask for the address instead of
        showing an error the user can do nothing about from here.
        """
        from codehelper.cli.parser import _handle_proxy

        chip = _PROXY_CHIPS[self._chip_index.get(_PROXY_ROW, 0) % len(_PROXY_CHIPS)]
        want_on = chip == _PROXY_ON
        live_on = bool(self._proxy and self._proxy.enabled)
        if want_on == live_on:
            return
        if want_on and not (self._proxy and self._proxy.restorable_url):
            entered = self._read_text("Proxy URL (e.g. http://127.0.0.1:8118): ")
            if entered is None or not entered.strip():
                return
            self._run(_handle_proxy, self._proxy_request(url=entered.strip()))
            return
        self._run(
            _handle_proxy,
            self._proxy_request(action=_PROXY_ON if want_on else _PROXY_OFF),
            silent=self._chip_silent(),
        )

    def _chip_move(self, value: str, delta: int) -> None:
        """Move the focused row's chip cursor. Pure in-memory.

        Returns ``None`` so the menu redraws instead of exiting, which is what
        makes left/right free of I/O: nothing is read or written until Enter
        applies a chip. A no-op when the row cursor is on a wrapper row.
        """
        if value == _PROXY_ROW:
            current = self._chip_index.get(_PROXY_ROW, 0)
            self._chip_index[_PROXY_ROW] = (current + delta) % len(_PROXY_CHIPS)
            return None
        if not value.startswith(_AGENT_ROW):
            return None
        agent_name = value.removeprefix(_AGENT_ROW)
        chips = self._chips.get(agent_name, [])
        if chips:
            current = self._chip_index.get(agent_name, 0)
            self._chip_index[agent_name] = (current + delta) % len(chips)
        return None

    def _focused_chip(self, value: str) -> object | None:
        """Return the chip focused by the chip cursor, or None.

        For non-agent rows, returns the value itself (alias string).
        For agent rows with a chip cursor, resolves the cursor position to
        the actual chip object. Returns None for agent rows with no chips,
        when focused on _NATIVE_CHIP/_ADD_CHIP, or on the proxy row — whose
        chips are settings, not wrappers, and carry nothing token-shaped.
        """
        if value == _PROXY_ROW:
            return None
        if not value.startswith(_AGENT_ROW):
            return value
        agent_name = value.removeprefix(_AGENT_ROW)
        chips = self._chips.get(agent_name, [])
        if not chips:
            return None
        chip_idx = self._chip_index.get(agent_name, 0)
        chip = chips[chip_idx % len(chips)]
        if chip in (_NATIVE_CHIP, _ADD_CHIP):
            return None
        return chip

    def _token_action(self, value: str) -> str | None:
        """Return the token action for the focused row/chip.

        Agent rows are namespaced as ``agent:<name>`` and carry a horizontal
        chip cursor.  Passing that row key directly to ``_on_token`` makes it
        look for a wrapper literally named ``agent:claude``.  Resolve the
        highlighted backend to its real preset/wrapper alias first; native
        and the trailing add chip have no token to edit.
        """
        if value in (_ADD_AGENT, _ADD_WRAPPER):
            # Same guard `d` carries: the action rows name no wrapper, so a
            # focused-value leak here would look up a wrapper literally
            # named "+ add wrapper".
            return None
        chip = self._focused_chip(value)
        if chip is None:
            return None
        return f"token:{self._chip_name(chip)}"

    def _edit_action(self, value: str) -> str | None:
        """Return the edit action for the focused row/chip (issue #100).

        Same focused-chip resolution as :meth:`_token_action` — chipset rows
        and action rows own nothing editable, so they stay inert there. The
        `e` verb is now the FULL editor (model/tiers/provider/effort, with
        rename as a row inside); the Profiles screen keeps its own local `e`
        for profile renames.
        """
        if value in (_ADD_AGENT, _ADD_WRAPPER):
            return None
        chip = self._focused_chip(value)
        if chip is None:
            return None
        return f"edit:{self._chip_name(chip)}"

    def _on_rename(self, alias: str) -> bool:
        """`e` on a wrapper row: move the wrapper to a new alias (issue #95)."""
        from codehelper.cli.parser import _handle_rename
        from codehelper.cli.requests import RenameRequest
        from codehelper.services.paths import Paths
        from codehelper.services.wrappers import is_installed, is_managed

        paths = Paths.default()
        if not is_installed(paths, alias):
            self._notify("Wrapper not installed — use Add first.")
            return False
        if not is_managed(paths, alias):
            self._notify("Refusing to rename an unmanaged wrapper file.")
            return False
        new_name = self._read_text(f"New name for {alias}: ")
        if not new_name or new_name == alias:
            return False
        # silent: the tee would otherwise show the service transcript live AND
        # replay it in the notify pause — the doubled output a live run hit.
        # The confirmation is gated on the handler's RESULT, not on the target
        # existing: a colliding alias leaves the source untouched and must not
        # print a success line after its own error.
        renamed = self._run(
            _handle_rename,
            RenameRequest(
                kind="wrapper",
                provider=None,
                name=alias,
                new_name=new_name,
                dry_run=getattr(self.args, "dry_run", False),
                debug=getattr(self.args, "debug", False),
            ),
            silent=True,
        )
        if renamed:
            self._notify(f"renamed wrapper {alias} -> {new_name}")
        return renamed

    # --- the edit screen (issue #100) ------------------------------------

    def _run_edit(self, alias: str) -> None:
        """`e` on a wrapper row or highlighted chip: the edit screen."""
        from codehelper.services.paths import Paths
        from codehelper.services.wrappers import is_installed, is_managed

        paths = Paths.default()
        if not is_installed(paths, alias):
            self._notify("Wrapper not installed — use Add first.")
            return
        if not is_managed(paths, alias):
            self._notify("Refusing to edit an unmanaged wrapper file.")
            return
        self._run_edit_screen(alias)

    def _run_edit_screen(self, alias: str) -> None:
        """One row per editable axis; Enter opens that axis's picker.

        Which rows exist is shape-driven data (:data:`_EDIT_AXES`). The draft
        is screen-local and the labels read ONLY the draft — the label rule:
        menu re-evaluates labels on every redraw frame, so a label doing I/O
        would re-read the file per frame. One ``spec_from_installed`` per
        screen ENTRY is the whole I/O budget.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.wrappers import UNSET, Unset, spec_from_installed

        spec = spec_from_installed(Paths.default(), alias)
        if spec is None:
            self._notify(f"Cannot edit {alias}: its marker is not recognisable.")
            return
        axes = [axis for axis in _EDIT_AXES.get(spec.shape, ()) if axis.applies(spec)]
        if not axes:
            self._notify(f"No editable axes for a {spec.shape.value} wrapper.")
            return
        draft: dict[str, object] = {
            # Every axis starts UNSET ("keep the recorded value") — None would
            # mean "clear"/"switch to nothing", which is never a starting point.
            "provider": UNSET,
            "auth": UNSET,
            "base_url": UNSET,
            "model": UNSET,
            "tiers": UNSET,
            "subagent": UNSET,
            "effort": UNSET,
            "ctx": UNSET,
        }
        while True:
            drafted = [
                axis
                for axis in axes
                if axis.key != "rename"
                and not isinstance(draft[self._edit_draft_key(axis.key)], Unset)
            ]
            items = [
                (axis.key, self._edit_axis_label(axis, spec, draft)) for axis in axes
            ]
            plural = "es" if len(drafted) != 1 else ""
            apply_label = (
                f"( apply — {len(drafted)} axis{plural} changed )"
                if drafted
                else "( apply )"
            )
            items.extend([("__apply__", apply_label), (_BACK, "Back")])
            choice = self._pick(items, f"Edit wrapper {alias}:")
            if choice == _BACK:
                return
            if choice == "__apply__":
                if drafted and self._apply_edit(alias, draft):
                    return
                continue
            axis = next(a for a in axes if a.key == choice)
            getattr(self, axis.pick)(spec, draft)
            if draft.get("closed"):
                return

    @staticmethod
    def _edit_draft_key(key: str) -> str:
        """Tier-family rows share the ONE ``tiers`` draft value (``None`` =
        back to uniform, a dict = per-field overrides); the rest map 1:1."""
        return "tiers" if key in ("haiku", "sonnet", "opus", "uniform") else key

    @staticmethod
    def _edit_axis_label(axis, spec, draft) -> str:
        """``"model: glm-5.3 -> glm-5.2:cloud"`` — reads ONLY the draft."""
        from codehelper.services.wrappers import Unset

        if axis.key == "rename":
            return "rename — move the wrapper to a new alias"
        key = TuiSession._edit_draft_key(axis.key)
        current = TuiSession._edit_current_value(axis.key, spec)
        value = draft[key]
        if isinstance(value, Unset):
            return f"{axis.label}: {current}"
        if key == "tiers":
            if value is None:
                return f"{axis.label}: {current} -> uniform"
            shown = (
                value.get(axis.key, current)
                if axis.key != "uniform"
                else "mixed (per-tier)"
            )
            return f"{axis.label}: {current} -> {shown}"
        shown = "none (suppressed)" if axis.key == "ctx" and value == 0 else str(value)
        return f"{axis.label}: {current} -> {shown}"

    @staticmethod
    def _edit_current_value(key: str, spec) -> str:
        if key == "provider":
            return spec.provider.name
        if key == "model":
            return spec.model
        if key in ("haiku", "sonnet", "opus"):
            return getattr(spec.tier_models, key) if spec.tier_models else "—"
        if key == "uniform":
            return "per-tier" if spec.tier_models is not None else "uniform"
        if key == "subagent":
            return spec.subagent_model or "—"
        if key == "effort":
            return spec.effort or "—"
        if key == "ctx":
            if spec.context_window is None:
                return "catalog / unset"
            return (
                "none (suppressed)"
                if spec.context_window == 0
                else str(spec.context_window)
            )
        return "—"

    def _edit_provider_view(self, spec, draft):
        """The provider object + profile the pickers should see: the drafted
        provider (auth/URL applied) when one was picked, else the recorded
        one — so a model/tier pick after a provider change discovers against
        the NEW endpoint."""
        from codehelper.services.model import get_provider, with_auth, with_base_url
        from codehelper.services.wrappers import Unset

        if isinstance(draft["provider"], Unset):
            return spec.provider, spec.profile_name
        obj = with_auth(
            get_provider(draft["provider"]), want_secret=draft["auth"] == "secret"
        )
        if draft["base_url"]:
            obj = with_base_url(obj, draft["base_url"])
        return obj, None

    def _apply_edit(self, alias: str, draft) -> bool:
        """Dispatch the drafted edit; True when the screen should close.

        ``silent=self._chip_silent()`` — the ``_run`` rule for chip-grade
        applies: a dry run must surface its preview, a real write is already
        reflected by the redraw. The applied-hint is gated on the POST-edit
        readback, not on ``wrote``: an effort-only codex edit stays applied
        and the hint must not claim otherwise.
        """
        from codehelper.cli.parser import _handle_edit_wrapper
        from codehelper.cli.requests import EditWrapperRequest

        was_applied = self._edit_was_applied(alias)
        from codehelper.services.wrappers import Unset

        req = EditWrapperRequest(
            alias=alias,
            provider=None
            if isinstance(draft["provider"], Unset)
            else draft["provider"],
            auth=None if isinstance(draft["auth"], Unset) else draft["auth"],
            model=draft["model"],
            tier_overrides=draft["tiers"],
            subagent_model=draft["subagent"],
            effort=draft["effort"],
            context_window=draft["ctx"],
            base_url=None
            if isinstance(draft["base_url"], Unset)
            else draft["base_url"],
            dry_run=getattr(self.args, "dry_run", False),
            debug=getattr(self.args, "debug", False),
        )
        wrote = self._run(_handle_edit_wrapper, req, silent=self._chip_silent())
        if not wrote:
            return False
        now_applied = self._edit_was_applied(alias)
        if (
            was_applied is not None
            and was_applied[1]
            and now_applied is not None
            and not now_applied[1]
        ):
            self._notify(
                f"edited {alias} — it is no longer applied; Enter the chip to re-apply."
            )
        return True

    def _edit_was_applied(self, alias: str) -> tuple[str, bool] | None:
        """``(agent, applied?)`` for ``alias`` — None without a backend row.

        The chip predicate consumes a FRESH spec (re-read from the file) and
        :meth:`_refresh_active_label` re-caches the live-config readback, so
        calling this before AND after the apply measures the real delta.
        """
        from codehelper.services.paths import Paths
        from codehelper.services.wrappers import spec_from_installed

        spec = spec_from_installed(Paths.default(), alias)
        if spec is None or spec.agent.name not in _AGENT_BACKENDS:
            return None
        self._refresh_active_label()
        return spec.agent.name, self._chip_is_applied(spec.agent.name, spec)

    def _pick_edit_provider(self, spec, draft) -> None:
        chosen = self._choose_add_provider(spec.agent)
        if chosen in (_BACK, None):
            return
        provider, want_secret, typed_url = chosen
        draft["provider"] = provider.name
        draft["auth"] = "secret" if want_secret else "literal"
        draft["base_url"] = typed_url or None

    def _pick_edit_model(self, spec, draft) -> None:
        provider_obj, profile = self._edit_provider_view(spec, draft)
        model = self._choose_add_model(provider_obj, profile, None, spec.agent.name)
        if model not in (_BACK, None):
            draft["model"] = model

    def _pick_edit_tier(self, spec, draft, tier: str) -> None:
        provider_obj, profile = self._edit_provider_view(spec, draft)
        model = self._choose_add_model(
            provider_obj, profile, None, f"{spec.agent.name} {tier}"
        )
        if model in (_BACK, None):
            return
        tiers = draft["tiers"]
        if isinstance(tiers, dict):
            tiers[tier] = model
        else:
            draft["tiers"] = {tier: model}

    def _pick_edit_haiku(self, spec, draft) -> None:
        self._pick_edit_tier(spec, draft, "haiku")

    def _pick_edit_sonnet(self, spec, draft) -> None:
        self._pick_edit_tier(spec, draft, "sonnet")

    def _pick_edit_opus(self, spec, draft) -> None:
        self._pick_edit_tier(spec, draft, "opus")

    def _pick_edit_uniform(self, spec, draft) -> None:
        """Back to uniform tiers: the model axis carries the value."""
        draft["tiers"] = None

    def _pick_edit_subagent(self, spec, draft) -> None:
        from codehelper.services.models_api import list_models
        from codehelper.services.paths import Paths
        from codehelper.services.secrets import token_for_discovery

        provider_obj, profile = self._edit_provider_view(spec, draft)
        token = token_for_discovery(Paths.default(), provider_obj, profile_name=profile)
        result = list_models(provider_obj, token=token)
        models = result.models or (provider_obj.known_models if not result.ok else [])
        items: list[tuple[str, str]] = [
            ("__unset__", "( unset — no separate subagent model )")
        ]
        items.extend((m, m) for m in models)
        items.extend([("__custom__", "Enter model manually"), (_BACK, "Back")])
        choice = self._pick(items, f"Subagent model for {provider_obj.name}:")
        if choice == "__unset__":
            draft["subagent"] = None
        elif choice == "__custom__":
            typed = self._read_text("Subagent model: ")
            if typed:
                draft["subagent"] = typed
        elif choice not in (_BACK, None):
            draft["subagent"] = choice

    def _pick_edit_effort(self, spec, draft) -> None:
        from codehelper.services.spec import REASONING_EFFORTS

        items: list[tuple[str, str]] = [
            ("__clear__", "( clear — stop managing effort )")
        ]
        items.extend((e, e) for e in REASONING_EFFORTS)
        items.append((_BACK, "Back"))
        choice = self._pick(items, "Reasoning effort (codex wrappers):")
        if choice == "__clear__":
            draft["effort"] = None
        elif choice not in (_BACK, None):
            draft["effort"] = choice

    def _pick_edit_ctx(self, spec, draft) -> None:
        from codehelper.cli.parser import _parse_context_window
        from codehelper.errors import CodeHelperError

        items = [
            ("0", "no declaration — the agent's native fallback"),
            ("1000000", "1,000,000 tokens (1M)"),
            ("2000000", "2,000,000 tokens (2M)"),
            ("__custom__", "custom…"),
            (_BACK, "Back"),
        ]
        choice = self._pick(items, "Context window:")
        if choice in (_BACK, None):
            return
        if choice == "__custom__":
            typed = self._read_text("Context window in tokens (or 'none'): ")
            if not typed:
                return
            try:
                draft["ctx"] = _parse_context_window(typed)
            except CodeHelperError as exc:
                self._notify(f"error: {exc}")
            return
        draft["ctx"] = _parse_context_window(choice)

    def _pick_edit_rename(self, spec, draft) -> None:
        if self._on_rename(spec.alias):
            # The alias moved — this screen's spec is stale; close it.
            draft["closed"] = True

    def _apply_chip(self, agent_name: str) -> None:
        """Apply the highlighted chip of ``agent_name`` (Enter on its row).

        A chip whose backend readback is exact (``_AgentBackend.exact_readback``
        — claude compares its full managed target: endpoint, models, subagent
        and credential) is a silent no-op when it already reads applied.  The
        same holds for native, whose no-override state is exact.  Other agents
        retain their existing provider-level readback, which is not precise
        enough to skip a potentially different wrapper.
        """
        chips = self._chips.get(agent_name, [])
        if not chips:
            return

        chip_idx = self._chip_index.get(agent_name, 0)
        chip = chips[chip_idx % len(chips)]

        if chip == _ADD_CHIP:
            # The + add action — opens Add pre-scoped to this row's agent,
            # exactly what `a` on this row does (see `run()`'s "a" binding).
            self._run_add(agent_name)
            return
        backend = _AGENT_BACKENDS[agent_name]
        if (chip == _NATIVE_CHIP or backend.exact_readback) and self._chip_is_applied(
            agent_name, chip
        ):
            return
        if chip == _NATIVE_CHIP:
            getattr(self, backend.apply_native)()
        else:
            getattr(self, backend.apply_wrapper)(chip)

    def _main_prompt(self) -> str:
        """Live main-menu header.

        Deliberately does NOT repeat which backend is applied: the chipset
        rows say that in place, and a header that restates it is the kind of
        duplication this screen was redesigned to remove.

        The proxy is NOT named here either: it has its own permanent chipset
        row that states both the live setting and what Enter would do, so a
        header tag would be the same duplication.
        """
        if self._tab_label:
            # Labelled: a bare "zai/axisrow" up here reads as a model or an
            # endpoint, which is exactly what the rest of the screen is about.
            return f"codehelper{' ' * 20}profile: {self._tab_label}"
        return "codehelper"

    def _show_help(self) -> None:
        from codehelper.cli.menu import press_any_key

        print("a add (agent row: scoped to it) · t token · e edit · d delete")
        print("+ add agent: new CLI integration · + add wrapper: any agent")
        print("p profiles · s settings (proxy, stored tokens)")
        print("←→ + Enter on the + add chip: same as a")
        print("Up/Down row · Left/Right chip · Enter apply · Esc quit · Ctrl-C quit")
        press_any_key("Press any key to continue...")

    def _profile_row_label(self) -> str:
        return f"Profile: {self._tab_label}" if self._tab_label else "Profile"

    # --- main loop -------------------------------------------------------

    def run(self) -> int:
        """The main menu loop — show wrappers + service rows, dispatch."""
        from codehelper.cli.menu import MenuCancelled, Section
        from codehelper.services.paths import Paths
        from codehelper.services.state import set_default_wrapper

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
                    "TOKEN": self._token_action,
                    "t": self._token_action,
                    "e": self._edit_action,
                    # Chipset rows (agents, proxy) and the two action rows own
                    # no wrapper file, so `d` there must be inert rather than
                    # looking for a wrapper literally named `proxy:` or
                    # `agent:claude` — same guard `_focused_chip` applies to
                    # `t`. Only a row from `_wrapper_rows` is a real alias.
                    "d": lambda alias: (
                        None
                        if alias == _PROXY_ROW
                        or alias.startswith(_AGENT_ROW)
                        or alias in (_ADD_AGENT, _ADD_WRAPPER)
                        else f"remove:{alias}"
                    ),
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
                        # The proxy, rendered as a chipset row of its own —
                        # not an agent, but the same on-screen grammar, so
                        # "which one is live" and "what will Enter do" read
                        # identically to the rows above it.
                        (_PROXY_ROW, self._proxy_row()),
                        # Add a new CLI integration, right below the chipset
                        # rows — the one add action that has no agent row of
                        # its own to hang a `+ add` chip on.
                        (_ADD_AGENT, "+ add agent — a new CLI integration"),
                        # Separates the chipset from the wrapper list below.
                        Section(""),
                        *self._wrapper_rows(paths),
                        # Add a wrapper for any agent (unscoped — asks which
                        # agent first), at the bottom of the screen.
                        (_ADD_WRAPPER, "+ add wrapper"),
                    ],
                    self._main_prompt,
                    exit_word="quit",
                    on_key=keys,
                    chips=True,
                    numbered=False,
                )
                if choice in (_BACK, _QUIT):
                    return 0
                if choice in (_ADD, _ADD_WRAPPER):
                    self._run_add()
                elif choice == _ADD_AGENT:
                    self._run_add_agent()
                elif choice.startswith("add:"):
                    self._run_add(choice.removeprefix("add:"))
                elif choice == _PROFILE:
                    self._run_profile_screen()
                elif choice == _PROXY_ROW:
                    self._apply_proxy_chip()
                elif choice.startswith(_AGENT_ROW):
                    self._apply_chip(choice.removeprefix(_AGENT_ROW))
                elif choice == _SETTINGS:
                    self._run_settings()
                elif choice == _HELP:
                    self._show_help()
                elif choice.startswith("token:"):
                    self._on_token(choice.removeprefix("token:"))
                elif choice.startswith("edit:"):
                    self._run_edit(choice.removeprefix("edit:"))
                elif choice.startswith("remove:"):
                    from codehelper.cli.parser import _handle_remove
                    from codehelper.cli.requests import RemoveRequest

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
                        # live: the file-list confirm must be visible before
                        # the blocking y/N read.
                        live=True,
                    )
                else:
                    # A wrapper alias — Enter makes it the default for its
                    # agent. `set_default_wrapper` is the raw store (#28);
                    # `choice` came from `_wrapper_rows`, which only yields
                    # real aliases, so the write is always of a real alias.
                    from codehelper.services.wrappers import is_installed, is_managed

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
                            f"{choice} is not a codehelper-managed wrapper "
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
    from codehelper.services.claude_settings import current_switch
    from codehelper.services.codex_default import (
        current_default,
        current_default_model,
    )

    _AGENT_BACKENDS.update(
        {
            "claude": _AgentBackend(
                read_applied=current_switch,
                apply_wrapper="_apply_switch_wrapper",
                apply_native="_apply_switch_native",
                lifecycle="live",
                chip_is_applied="_chip_is_applied_switch",
                chip_switch_token="_chip_switch_token",
                exact_readback=True,
            ),
            "codex": _AgentBackend(
                read_applied=current_default,
                apply_wrapper="_apply_set_default_wrapper",
                apply_native="_apply_set_default_native",
                lifecycle="next launch",
                chip_is_applied="_chip_is_applied_codex",
                read_applied_model=current_default_model,
            ),
        }
    )
    # A method name is only as safe as its spelling: resolve every hook now,
    # at import, so a typo is an immediate ImportError rather than an
    # AttributeError the first time someone presses Enter on that row.
    for backend in _AGENT_BACKENDS.values():
        for hook in (
            backend.apply_wrapper,
            backend.apply_native,
            backend.chip_is_applied,
            backend.chip_switch_token,
        ):
            if hook is None:
                continue
            if not callable(getattr(TuiSession, hook, None)):
                raise AttributeError(f"TuiSession has no hook {hook!r}")


_register_agent_backends()


@dataclass(frozen=True)
class _EditAxis:
    """One editable axis row of the `e` edit screen (issue #100).

    Which rows exist is SHAPE-driven data — never an ``if agent.name == ...``
    branch — and ``pick`` is a METHOD NAME resolved via ``getattr`` at press
    time (the ``_AGENT_BACKENDS`` convention, so patching the method takes
    effect). An ollama-launch wrapper therefore gets no tier/subagent/effort
    rows at all: its shape cannot render them — the same honest degradation
    the chipset row implements for launch-only agents.
    """

    key: str
    label: str
    applies: Callable[[object], bool]
    pick: str


_EDIT_AXES: dict[object, tuple[_EditAxis, ...]] = {}


def _register_edit_axes() -> None:
    """Populate :data:`_EDIT_AXES` — the same lazy-import shape as
    :func:`_register_agent_backends` (this module is imported on every run,
    including ``--help``)."""
    from codehelper.services.model import ConfigShape

    def _always(_spec: object) -> bool:
        return True

    def _has_recorded_tiers(spec: object) -> bool:
        # "back to uniform" is meaningful only when tiers differ today.
        return getattr(spec, "tier_models", None) is not None

    def _launch_ctx(spec: object) -> bool:
        # Delegates to the ONE predicate (WrapperSpec.can_declare_context_window):
        # the ctx declaration rides the env (and TOML) sinks; a launch wrapper
        # declares it only when the AGENT itself has an env-capable provider
        # surface — data off shape + agent.shapes, never a name check. The add
        # path and edit_wrapper's re-resolve gate on the same property.
        return getattr(spec, "can_declare_context_window", False)

    _EDIT_AXES.update(
        {
            ConfigShape.ANTHROPIC_ENV: (
                _EditAxis("provider", "provider", _always, "_pick_edit_provider"),
                _EditAxis("model", "model", _always, "_pick_edit_model"),
                _EditAxis("haiku", "haiku", _always, "_pick_edit_haiku"),
                _EditAxis("sonnet", "sonnet", _always, "_pick_edit_sonnet"),
                _EditAxis("opus", "opus", _always, "_pick_edit_opus"),
                _EditAxis(
                    "uniform", "tiers", _has_recorded_tiers, "_pick_edit_uniform"
                ),
                _EditAxis("subagent", "subagent", _always, "_pick_edit_subagent"),
                _EditAxis("ctx", "ctx", _always, "_pick_edit_ctx"),
                _EditAxis("rename", "rename", _always, "_pick_edit_rename"),
            ),
            ConfigShape.OPENAI_TOML: (
                _EditAxis("provider", "provider", _always, "_pick_edit_provider"),
                _EditAxis("model", "model", _always, "_pick_edit_model"),
                _EditAxis("effort", "effort", _always, "_pick_edit_effort"),
                _EditAxis("ctx", "ctx", _always, "_pick_edit_ctx"),
                _EditAxis("rename", "rename", _always, "_pick_edit_rename"),
            ),
            ConfigShape.OLLAMA_LAUNCH: (
                _EditAxis("provider", "provider", _always, "_pick_edit_provider"),
                _EditAxis("model", "model", _always, "_pick_edit_model"),
                _EditAxis("ctx", "ctx", _launch_ctx, "_pick_edit_ctx"),
                _EditAxis("rename", "rename", _always, "_pick_edit_rename"),
            ),
        }
    )
    # A method name is only as safe as its spelling: resolve every pick now,
    # at import, so a typo is an immediate ImportError rather than an
    # AttributeError the first time someone presses Enter on that row.
    missing = sorted(
        {
            axis.pick
            for axes in _EDIT_AXES.values()
            for axis in axes
            if not callable(getattr(TuiSession, axis.pick, None))
        }
    )
    if missing:
        raise AttributeError(f"TuiSession has no edit-axis pick methods: {missing}")


_register_edit_axes()
