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

__all__ = ["AddRequest", "EditTokenRequest", "RemoveRequest", "SetDefaultRequest"]


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
class SetDefaultRequest:
    """Inputs to ``_handle_set_default`` (the ``set-default`` subcommand and TUI).

    ``restore`` with an optional ``slot`` reads a backup back; otherwise
    ``agent``/``provider``/``model`` (and ``base_url`` for a runtime-address
    provider) define the patch to apply. ``catalog_json`` is an optional
    override for the model-catalog path.
    """

    agent: str | None
    provider: str | None
    model: str | None
    base_url: str | None
    restore: bool
    slot: int | None
    catalog_json: str | None
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
            catalog_json=_g(args, "catalog_json"),
            dry_run=bool(_g(args, "dry_run", False)),
            force=bool(_g(args, "force", False)),
            debug=bool(_g(args, "debug", False)),
        )
