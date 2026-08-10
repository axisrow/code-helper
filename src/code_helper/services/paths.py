"""Pure-domain ``Paths`` object: the root configuration object of the project.

Every filesystem path the tool touches is resolved from a single injected
``home`` through ``Paths.from_home`` — ``~/.local/bin`` (the XDG user-bin dir
where every generated wrapper script lives), ``~/.codex`` (Codex's config
dir, written only by the OPENAI_TOML shape), and ``~/.config/code-helper``
(this tool's own config dir, where ``credentials.json`` caches provider
tokens). This is the single source of truth for resolved paths: no other
module may hard-code those literals.

This module lives in the ``services/`` layer (pure domain services, no side
effects). ``from_home`` is PURE path arithmetic — it performs no IO at all and
no existence checks; directory creation is deferred to the write boundary
(``backends/_atomic.py``'s ``parent.mkdir``).

Usage contract:
- **Tests** inject a tmp home: ``Paths.from_home(tmp_path)``. This is the
  primary isolation mechanism — a test never resolves the developer's real
  ``$HOME``.
- **Production code** calls ``Paths.default()``, a one-line wrapper over
  ``from_home(Path.home())``. The naming split is what makes the isolation
  provable: tests always inject, never call ``default()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from code_helper.errors import CodeHelperError


@dataclass(frozen=True)
class Paths:
    """Frozen bundle of every resolved filesystem path the tool touches.

    Frozen so a ``Paths`` instance handed to a service/backend cannot be
    mutated to silently redirect writes — a tampering guard. All fields are
    ``pathlib.Path``; they are set exclusively by :meth:`from_home`, so an
    instance can never be half-resolved.
    """

    bin_dir: Path
    #: ``~/.codex`` — Codex's per-profile config dir. The OPENAI_TOML shape
    #: writes ``<alias>.config.toml`` / ``<alias>.model.json`` here. Codex's own
    #: ``config.toml`` is left untouched by every command EXCEPT the explicit
    #: ``set-default`` (``services/codex_default.py``), which patches only its
    #: own managed top-level keys and ``[model_providers.X]`` table there —
    #: never a fragment any other command writes.
    codex_dir: Path
    #: ``~/.config/code-helper`` — this tool's own config dir. Only
    #: ``credentials.json`` (the cached provider-token store — see
    #: ``services/secrets.py``) lives here today. NOT a wrapper/source of
    #: truth: a token is baked into each generated script (``0o700``), and
    #: deleting this dir does not break any installed wrapper.
    config_dir: Path

    @classmethod
    def from_home(cls, home: str | Path) -> Paths:
        """Resolve all paths off ``home`` via pure arithmetic (no IO).

        Accepts ``str | Path`` and coerces to ``pathlib.Path``. Does NOT
        validate existence, create directories, or read anything — it
        succeeds on a non-existent home. Symlinks are left as-is (no
        ``.resolve()``).

        - ``bin_dir`` = ``home / ".local" / "bin"`` — the XDG user-bin dir
          where every generated wrapper script (``deepseek``, ``glm``, ...)
          lives. It is typically already on ``PATH``.
        - ``codex_dir`` = ``home / ".codex"`` — Codex's config dir; the
          OPENAI_TOML shape writes profile files here.
        - ``config_dir`` = ``home / ".config" / "code-helper"`` — this tool's
          own config dir (``credentials.json``).

        ``XDG_CONFIG_HOME`` is intentionally NOT consulted: this method is
        documented as PURE path arithmetic off ``home`` (no IO, no env), and
        reading it would also defeat ``conftest.py``'s ``_isolate_home``
        fixture, which swaps only ``HOME``.
        """
        h = Path(home)
        return cls(
            bin_dir=h / ".local" / "bin",
            codex_dir=h / ".codex",
            config_dir=h / ".config" / "code-helper",
        )

    @classmethod
    def default(cls) -> Paths:
        """Production entry point: ``from_home(Path.home())``.

        A one-line thin wrapper with no alternate resolution path. Tests
        never call this — they inject ``tmp_path`` via :meth:`from_home`
        directly; that naming split is what makes the isolation provable.
        """
        return cls.from_home(Path.home())

    def _require_single_component(self, name: str, *, noun: str) -> None:
        """Raise unless ``name`` is a single path component.

        ``Path(name).name != name`` is the whole check: it rejects
        ``"../../etc/passwd"``, ``"a/b"``, ``""``, ``"."`` and ``".."`` alike,
        without ``resolve()`` — so this stays pure arithmetic with no IO. The
        single shared predicate behind :meth:`script_for` and the
        ``~/.codex`` accessors: each public accessor still enforces it
        independently (the structural-safety argument is preserved), but the
        condition itself lives in one place.

        Args:
            noun: The human-facing label in the error (``"wrapper name"`` for
                :meth:`script_for`, ``"alias"`` for the ``~/.codex`` accessors).

        Raises:
            CodeHelperError: ``name`` is not a single path component.
        """
        if name in ("", ".", "..") or Path(name).name != name:
            raise CodeHelperError(
                f"invalid {noun} (must be a single path component): {name!r}"
            )

    def script_for(self, name: str) -> Path:
        """Return the resolved path of the wrapper script named ``name``.

        Pure arithmetic (``bin_dir / name``) — no existence check, no IO.

        ``name`` must be a single path component. This is a STRUCTURAL guard,
        independent of :func:`code_helper.services.naming.validate_alias`:
        that function owns the human-facing rules and runs early, this one
        guarantees that no code path — including a future one that forgets to
        validate — can address a file outside ``bin_dir``. The cross-module
        duplication with :func:`validate_alias` is deliberate; the failure it
        prevents is writing an executable to an arbitrary filesystem location.
        """
        self._require_single_component(name, noun="wrapper name")
        return self.bin_dir / name

    def _single_component(self, alias: str, *, suffix: str) -> Path:
        """``codex_dir / <alias>.<suffix>`` with the same single-component guard
        as :meth:`script_for`.

        The alias flows into a filename written under ``~/.codex``, so the same
        structural guard applies (independent of :func:`validate_alias`, which
        owns the human-facing rules and runs earlier). The suffix is appended
        after the guard so it cannot smuggle a path separator.
        """
        self._require_single_component(alias, noun="alias")
        return self.codex_dir / f"{alias}.{suffix}"

    def codex_config_for(self, alias: str) -> Path:
        """``~/.codex/<alias>.config.toml`` — the OPENAI_TOML profile.

        Pure arithmetic, no IO. Codex resolves the profile name from the file
        basename, so ``<alias>.config.toml`` is addressed as ``--profile
        <alias>``.
        """
        return self._single_component(alias, suffix="config.toml")

    def codex_catalog_for(self, alias: str) -> Path:
        """``~/.codex/<alias>.model.json`` — the model catalog for the profile.

        Gives Codex a ``context_window`` for models it does not ship knowledge
        of (e.g. ``glm-5.2:cloud``). Pure arithmetic, no IO.
        """
        return self._single_component(alias, suffix="model.json")

    def codex_main_config(self) -> Path:
        """``~/.codex/config.toml`` — Codex's OWN default config.

        Distinct from :meth:`codex_config_for`: that is a per-alias PROFILE
        file this tool owns outright; this is Codex's single top-level config
        file, which this tool never owns and only ever patches (never
        replaces) via ``services/codex_default.py``. Pure arithmetic, no IO,
        no existence check — same contract as every other accessor here.
        """
        return self.codex_dir / "config.toml"

    def codex_main_config_backup(self, slot: int) -> Path:
        """``~/.codex/config.toml.bak<slot>`` — one of three rotating backups.

        ``slot`` must be 1, 2, or 3 (1 = most recent). ``set-default`` rotates
        these FIFO-style before every real patch: 2→3 (oldest lost), 1→2,
        current file→1. Kept as a fixed 3-slot ring rather than a
        timestamp-suffixed pile — this project makes no ``datetime.now()``
        call in business logic today, and an unbounded ``.bak-<ts>`` pile is
        exactly the on-disk litter its "no state file" philosophy avoids
        elsewhere (see ``wrappers.discover_managed``'s docstring).
        """
        if slot not in (1, 2, 3):
            raise CodeHelperError(f"invalid backup slot (must be 1, 2, or 3): {slot!r}")
        return self.codex_dir / f"config.toml.bak{slot}"

    def credentials_file(self) -> Path:
        """``~/.config/code-helper/credentials.json`` — the cached token store.

        A ``{provider_name: {profile_name: token}}`` JSON object: profiles are
        scoped to providers, never wrappers, because one provider may back many
        wrappers and the credential belongs to the provider. The first key is
        stored under the internal ``default`` profile. Owned and written only
        by ``services/secrets.py``. Pure arithmetic, no IO, no existence check —
        same contract as every other accessor here.

        This file is a CACHE of values the user typed at an install prompt, not
        a session with any provider and not the source of truth: each
        installed wrapper carries its own token baked in (``0o700``), so
        deleting this file does not break an installed wrapper — it only means
        the next ``add``/``--list-models`` will prompt for the token again.
        """
        return self.config_dir / "credentials.json"
