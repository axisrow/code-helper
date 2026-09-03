"""Typed request objects for the CLI handlers (issue #37, P2.4).

The three handlers ``_handle_add`` / ``_handle_edit_token`` / ``_handle_set_default``
previously read their inputs off the raw ``argparse.Namespace`` via ``getattr``
(32 reads across the three). The TUI built the SAME shape by mutating a shared
``Namespace`` in place (21 assignments) before dispatching into the same
handlers — a contract described only in comments, invisible to the type checker,
and requiring every new flag to be added symmetrically in three places
(``build_parser``, the TUI assignment, the handler ``getattr``).

This module replaces that with one frozen dataclass per command. Each CLI
handler builds its request from the parsed ``Namespace`` via ``from_namespace``
(the single bridge), then works off typed fields. The TUI constructs these
requests directly, so its dispatch does not mutate the parser Namespace.

The dataclasses are ``frozen=True`` for the same reason ``Paths`` is: a request
handed to a handler is a snapshot of the user's intent, not something the
handler should mutate mid-flight.

Why not change ``build_parser`` / ``main`` / ``args.func(args)``: that contract
is pinned by ~60 ``test_cli_add.py`` tests that drive the CLI through
``main([...])`` and never construct a ``Namespace`` themselves. Keeping
``args.func(args)`` intact means those tests stay green untouched; the
``from_namespace`` bridge is the only new seam.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from codehelper.errors import CodeHelperError

__all__ = [
    "AddRequest",
    "DisableRequest",
    "EditTokenRequest",
    "EnableRequest",
    "ProxyRequest",
    "RemoveRequest",
    "SetDefaultRequest",
    "SwitchRequest",
]


def _g(args: argparse.Namespace, name: str, default: object = None) -> object:
    """``getattr`` with a stable default, for the ``from_namespace`` bridges.

    The TUI builds a ``Namespace`` itself and only fills the fields its flow
    uses, so an absent attribute is normal (not a bug) — exactly the contract
    the handlers' ``getattr(args, ..., default)`` reads already encode. One
    helper so the defaulting rule lives in one place.
    """
    return getattr(args, name, default)


@dataclass(frozen=True)
class AddRequest:
    """Inputs to ``_handle_add`` (the ``add`` subcommand and the TUI's Add flow).

    ``name`` is a preset name; the constructor axes are ``agent``/``provider``/
    ``model``. Exactly one of those two shapes is set per request — the handler
    disambiguates, same as before.
    """

    name: str | None
    agent: str | None
    provider: str | None
    model: str | None
    alias: str | None
    shape: str | None
    base_url: str | None
    auth: str | None
    profile: str | None
    profile_token: str | None
    profile_rename_from: str | None
    profile_rename_to: str | None
    list_models: bool
    dry_run: bool
    force: bool
    debug: bool
    # Explicit context window (issue #83): an int, or 0 for "no declaration".
    # None asks (interactively) or derives — the resolver decides.
    context_window: int | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> AddRequest:
        """Build an ``AddRequest`` from a parsed ``argparse.Namespace``.

        The single bridge between the argparse layer and the handler: every
        ``getattr(args, ...)`` the handler used to do lives here now, so the
        field set the handler consumes and the field set ``build_parser``
        declares cannot drift without a type error here first.
        """
        return cls(
            name=_g(args, "name"),
            agent=_g(args, "agent"),
            provider=_g(args, "provider"),
            model=_g(args, "model"),
            alias=_g(args, "alias"),
            shape=_g(args, "shape"),
            base_url=_g(args, "base_url"),
            auth=_g(args, "auth"),
            profile=_g(args, "profile"),
            profile_token=_g(args, "profile_token"),
            profile_rename_from=_g(args, "profile_rename_from"),
            profile_rename_to=_g(args, "profile_rename_to"),
            list_models=bool(_g(args, "list_models", False)),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
            context_window=_g(args, "context_window"),
        )


@dataclass(frozen=True)
class EditTokenRequest:
    """Inputs to ``_handle_edit_token`` (the ``edit-token`` subcommand and TUI).

    ``name`` may be ``None`` — the handler then offers an interactive picker.
    The four ``profile_*`` fields carry the TUI's collected profile decision
    (selected name, typed token, and an optional rename of the prior profile);
    they are ``None`` when the CLI path picks a profile through its own menu.
    """

    name: str | None
    profile: str | None
    profile_token: str | None
    profile_rename_from: str | None
    profile_rename_to: str | None
    dry_run: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> EditTokenRequest:
        return cls(
            name=_g(args, "name"),
            profile=_g(args, "profile"),
            profile_token=_g(args, "profile_token"),
            profile_rename_from=_g(args, "profile_rename_from"),
            profile_rename_to=_g(args, "profile_rename_to"),
            dry_run=bool(_g(args, "dry_run", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class RemoveRequest:
    """Inputs to ``_handle_remove`` (the ``remove`` subcommand and TUI ``d``).

    The last handler still reading a raw ``Namespace``: the TUI dispatched
    remove by assigning ``self.args.name`` in place, the one surviving instance
    of the mutation pattern the rest of this module replaced. With this request
    the TUI constructs its intent directly, so no screen mutates the parser
    Namespace any more.

    ``force`` carries the ``--force`` flag that lets ``remove_wrapper`` delete
    an unmanaged wrapper file; the TUI never sets it, so a foreign file can
    only be removed from an explicit command line.
    """

    name: str
    dry_run: bool
    force: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> RemoveRequest:
        return cls(
            name=_g(args, "name"),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class DisableRequest:
    """Inputs to ``_handle_disable`` (the ``disable`` subcommand and the TUI's
    Settings → Providers submenu, issue #89).

    ``yes`` skips the file-list confirmation — the TUI's submenu pick IS the
    confirmation, mirroring how the chipset's apply hooks opt in via force.
    ``force`` threads into each ``remove_wrapper`` call for unmanaged files.
    """

    name: str
    yes: bool
    force: bool
    dry_run: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> DisableRequest:
        return cls(
            name=_g(args, "name"),
            yes=bool(_g(args, "yes", False)),
            force=bool(_g(args, "force", False)),
            dry_run=bool(_g(args, "dry_run", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class EnableRequest:
    """Inputs to ``_handle_enable`` (the ``enable`` subcommand and the TUI's
    Settings → Providers submenu, issue #89)."""

    name: str
    dry_run: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> EnableRequest:
        return cls(
            name=_g(args, "name"),
            dry_run=bool(_g(args, "dry_run", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class ProxyRequest:
    """Inputs to ``_handle_proxy`` (the ``proxy`` subcommand and the TUI's
    Settings screen).

    ``action`` is the positional verb — ``on`` / ``off`` / ``toggle`` — or
    ``None`` for a bare ``codehelper proxy``, which reports status. ``url``
    sets a new address (and implies turning the proxy on); ``no_proxy``
    edits the bypass list on its own, without touching the proxy chain.

    All three axes are independent rather than one enum: setting an address
    while the proxy is off is a real case (configure now, enable later), and
    so is editing the bypass list without changing anything else.
    """

    action: str | None
    url: str | None
    no_proxy: str | None
    status: bool
    dry_run: bool
    force: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ProxyRequest:
        return cls(
            action=_g(args, "action"),
            url=_g(args, "url"),
            no_proxy=_g(args, "no_proxy"),
            status=bool(_g(args, "status", False)),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class SetDefaultRequest:
    """Inputs to ``_handle_set_default`` (the ``set-default`` subcommand and TUI).

    ``restore`` with an optional ``slot`` reads a backup back; otherwise
    ``agent``/``provider``/``model`` (and ``base_url`` for a runtime-address
    provider) define the patch to apply.
    """

    agent: str | None
    provider: str | None
    model: str | None
    base_url: str | None
    restore: bool
    slot: int | None
    dry_run: bool
    force: bool
    debug: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SetDefaultRequest:
        return cls(
            agent=_g(args, "agent"),
            provider=_g(args, "provider"),
            model=_g(args, "model"),
            base_url=_g(args, "base_url"),
            restore=bool(_g(args, "restore", False)),
            slot=_g(args, "slot"),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
        )


@dataclass(frozen=True)
class SwitchRequest:
    """Inputs to ``_handle_switch`` (the ``switch`` subcommand and TUI ``w``).

    Live-patches ``~/.claude/settings.json`` — a DIFFERENT file from
    ``set-default``'s ``~/.codex/config.toml`` — so a currently RUNNING
    ``claude`` picks up the new backend on its next prompt, no restart.

    ``provider`` is the positional-or-flag axis (``switch zai`` and
    ``switch --provider zai`` mean the same thing — the handler rejects
    giving both). ``from_wrapper`` is the no-prompts fast path: lift the
    model/token straight off an already-installed ``anthropic-env`` wrapper,
    guaranteeing the same backend that wrapper's own script would reach.
    ``model``/``haiku``/``sonnet``/``opus``/``subagent_model`` are the
    explicit-axes path, mutually exclusive with ``from_wrapper``.
    """

    provider: str | None
    from_wrapper: str | None
    model: str | None
    haiku: str | None
    sonnet: str | None
    opus: str | None
    subagent_model: str | None
    base_url: str | None
    auth: str | None
    profile: str | None
    restore: bool
    slot: int | None
    status: bool
    dry_run: bool
    force: bool
    debug: bool
    # Internal TUI fast path.  Unlike ``from_wrapper``, a preset is registry
    # data, not a pathname: selecting it must work even when no wrapper has
    # been installed (or an unrelated executable owns that alias on PATH).
    from_preset: str | None = None
    # Explicit context window (issue #83): an int, or 0 for "no declaration".
    # Only meaningful on the explicit-axes path; --from-wrapper/--from-preset
    # carry their own recorded answer, and mixing them is rejected.
    context_window: int | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SwitchRequest:
        # `provider_positional` (the `switch <name>` form) and `--provider`
        # are two argparse destinations for the same axis. Checked HERE,
        # while both raw values are still visible — merging them with a bare
        # `or` first (as build_spec-style requests usually do) would make
        # `switch zai --provider zai` look like a single value and silently
        # swallow the conflict.
        positional = _g(args, "provider_positional")
        flag = _g(args, "provider")
        if positional and flag:
            raise CodeHelperError("give either a provider or --provider, not both")
        provider = positional or flag
        return cls(
            provider=provider,
            from_wrapper=_g(args, "from_wrapper"),
            model=_g(args, "model"),
            haiku=_g(args, "haiku"),
            sonnet=_g(args, "sonnet"),
            opus=_g(args, "opus"),
            subagent_model=_g(args, "subagent_model"),
            base_url=_g(args, "base_url"),
            auth=_g(args, "auth"),
            profile=_g(args, "profile"),
            restore=bool(_g(args, "restore", False)),
            slot=_g(args, "slot"),
            status=bool(_g(args, "status", False)),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
            from_preset=None,
            context_window=_g(args, "context_window"),
        )
