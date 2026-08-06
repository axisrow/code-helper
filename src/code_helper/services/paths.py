"""Pure-domain ``Paths`` object: the root configuration object of the project.

Every filesystem path the tool touches is resolved from a single injected
``home`` through ``Paths.from_home`` — just ``~/.local/bin`` (the XDG user-bin
dir where every generated wrapper script lives). This is the single source of
truth for resolved paths: no other module may hard-code ``~/.local/bin``
literals.

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
        """
        h = Path(home)
        return cls(bin_dir=h / ".local" / "bin")

    @classmethod
    def default(cls) -> Paths:
        """Production entry point: ``from_home(Path.home())``.

        A one-line thin wrapper with no alternate resolution path. Tests
        never call this — they inject ``tmp_path`` via :meth:`from_home`
        directly; that naming split is what makes the isolation provable.
        """
        return cls.from_home(Path.home())

    def script_for(self, name: str) -> Path:
        """Return the resolved path of the wrapper script named ``name``.

        Pure arithmetic (``bin_dir / name``) — no existence check, no IO.

        ``name`` must be a single path component. This is a STRUCTURAL guard,
        independent of :func:`code_helper.services.naming.validate_alias`:
        that function owns the human-facing rules and runs early, this one
        guarantees that no code path — including a future one that forgets to
        validate — can address a file outside ``bin_dir``. Duplication here is
        deliberate; the failure it prevents is writing an executable to an
        arbitrary filesystem location.

        ``Path(name).name != name`` is the whole check: it rejects
        ``"../../etc/passwd"``, ``"a/b"``, ``""``, ``"."`` and ``".."`` alike,
        without ``resolve()`` — so this stays pure arithmetic with no IO.

        Raises:
            CodeHelperError: ``name`` is not a single path component.
        """
        if name in ("", ".", "..") or Path(name).name != name:
            raise CodeHelperError(
                f"invalid wrapper name (must be a single path component): {name!r}"
            )
        return self.bin_dir / name
