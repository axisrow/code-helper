"""Crash-safe file-write boundary behind the ``backends/`` IO layer.

A single primitive, :func:`atomic_write`, performs every disk mutation the
tool will ever make. The atomicity recipe is temp-in-same-dir + ``os.fsync`` +
``os.replace``: a write goes to a temp file created as a *sibling* of the
destination, the temp is fsync-ed, then ``os.replace`` renames it atomically
over the destination (atomic on POSIX/macOS). An interrupted write — crash,
power loss, Ctrl-C, exception — therefore never leaves a half-written file at
the destination and never orphans a temp file.

Mode contract:

- ``mode=None`` → do NOT chmod; the destination inherits the tempfile's mode.
  ``tempfile.NamedTemporaryFile`` (via ``mkstemp``) creates at ``0o600``
  UMASK-INDEPENDENTLY, and ``os.replace`` preserves that onto the destination —
  so a ``mode=None`` write lands at ``0o600``, NOT a umask-governed mode.
  Secrets still pass an EXPLICIT ``0o700``/``0o600`` rather than relying on
  this, so a future tempfile change cannot silently widen them.
- ``mode=0o700`` (or any explicit int) → ``os.chmod(path, mode)`` AFTER the
  successful replace — the single mechanism by which a wrapper script that
  carries a secret token lands restricted.

The helper never prints or logs ``data``: tokens pass through it unchanged.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

__all__ = ["atomic_write", "read_text_or_none", "remove_file"]


def remove_file(path: str | Path) -> None:
    """Remove one path without following it or recursively deleting anything."""
    Path(path).unlink()


def atomic_write(path: str | Path, data: bytes | str, mode: int | None = None) -> None:
    """Write ``data`` to ``path`` crash-safely.

    Sequence (order is load-bearing):

    1. Coerce ``path`` to ``pathlib.Path`` (accepts ``str | Path``).
    2. ``path.parent.mkdir(parents=True, exist_ok=True)`` — the one non-atomic
       allowance, run BEFORE temp creation so the temp lands in the right dir.
    3. ``tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False)`` —
       temp is a *sibling* of the destination so ``os.replace`` is a
       same-filesystem rename (atomic on POSIX/macOS).
    4. Write ``data`` (``str`` encoded UTF-8), ``flush()``, ``os.fsync(fd)``
       (load-bearing durability call).
    5. Close the temp; if a real EMPTY directory sits at ``path`` (which
       ``os.replace`` cannot overwrite with a file), ``os.rmdir`` it first — an
       empty dir at a wholesale-write destination is corruption to clear. A
       NON-empty dir is refused (ENOTEMPTY → cleanup + re-raise): this generic
       primitive never recursively deletes a tree that may hold user data or a
       mount point. Symlinks are NOT followed/removed. Then
       ``os.replace(temp, path)`` (atomic overwrite).
    6. Set the destination mode AFTER replace (never the temp): an integer
       ``mode`` → ``os.chmod(path, mode)``; ``mode=None`` over an EXISTING file →
       restore that file's prior mode (captured from the lstat in step 5, since
       the replace leaves the temp's mode); ``mode=None`` on a fresh write → the
       temp default is kept.

    On ANY exception after temp creation: ``os.unlink(temp)`` cleanup then
    re-raise — the destination is never visible in a partial state and no
    orphaned temp survives. The original exception is never swallowed.

    Args:
        path: Destination file (``str | Path``); parents are created if absent.
        data: Payload (``bytes`` written verbatim, ``str`` encoded UTF-8).
        mode: ``None`` preserves the EXISTING file's mode when overwriting;
            for a fresh write the file gets the temp default. An integer
            (e.g. ``0o700`` for a wrapper carrying a token) is applied via
            ``os.chmod`` AFTER replace regardless.

    Returns:
        None — the observable side effect is the written file at ``path``.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = data.encode("utf-8") if isinstance(data, str) else data

    # Create the temp as a SIBLING of the destination so os.replace is a
    # same-filesystem atomic rename.
    tmp = tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False)
    tmp_name = tmp.name
    try:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())  # load-bearing durability call
    finally:
        # Close before replace (portable: Windows cannot replace an open file;
        # POSIX tolerates it but closing first is cleaner and symmetric).
        tmp.close()

    try:
        # os.replace CANNOT overwrite a directory with a file (raises
        # IsADirectoryError), so an EMPTY directory left at the destination — by
        # a botched write, a git-checkout collision, or manual tampering — would
        # crash every backend that routes here. Clear ONLY an empty directory,
        # and ONLY with os.rmdir (never shutil.rmtree): rmdir removes an empty
        # dir and REFUSES a non-empty one (ENOTEMPTY). That is deliberate — a
        # non-empty directory at a config-file path holds data this generic
        # write primitive must NOT recursively destroy (a mount point, or files
        # a user put there); it fails loudly into the cleanup below instead of
        # silently deleting a tree. lstat (does NOT follow symlinks) gates on a
        # REAL directory, so a symlink-to-dir is left for os.replace to overwrite
        # (the link itself, not its target); a regular file / FIFO / socket /
        # device is likewise left for os.replace.
        try:
            st = os.lstat(dest)
        except OSError:
            st = None  # dest absent (fresh write) or unstattable → let replace decide
        # mode=None means "preserve the existing file's mode". Capture it from
        # the SAME lstat BEFORE the replace: after os.replace the file has the
        # temp's mode (~0600), so without this an existing file's mode would be
        # silently narrowed. Only a real regular file has a mode worth
        # preserving; a dir/symlink/fresh-write has none (prior_mode stays None
        # → temp default).
        prior_mode = (
            stat.S_IMODE(st.st_mode)
            if (mode is None and st is not None and stat.S_ISREG(st.st_mode))
            else None
        )
        if st is not None and stat.S_ISDIR(st.st_mode):
            os.rmdir(dest)  # empty-only; ENOTEMPTY → cleanup + re-raise (no data loss)
        os.replace(tmp_name, str(dest))  # atomic overwrite on POSIX/macOS
    except BaseException:
        # replace failed → unlink the temp so no orphan survives, and the
        # destination is untouched (old file is the only visible state).
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    if mode is not None:
        # chmod the DESTINATION after a successful replace — never the temp.
        # A crash between replace and chmod leaves a correctly-replaced file
        # with the old perms, not a half-applied state.
        os.chmod(str(dest), mode)
    elif prior_mode is not None:
        # mode=None over an existing file → restore its prior mode (the temp's
        # mode would otherwise stick).
        os.chmod(str(dest), prior_mode)


def read_text_or_none(path: str | Path) -> str | None:
    """Read ``path`` as UTF-8 text, returning ``None`` on any read failure.

    The read-side companion to :func:`atomic_write`: the same modules that
    route every write through one primitive used to each carry their own copy
    of this "never raises" read (``wrappers._read_text_or_none`` and
    ``codex_default._read_text_or_none``), differing only in whether
    ``FileNotFoundError`` was listed separately — even though it is already an
    ``OSError`` subclass. One function here closes that drift.

    ``None`` covers: the file does not exist, is unreadable (permissions), is
    a dangling symlink, or is not decodable as UTF-8 (a binary, a truncated
    UTF-8 sequence). Callers (the idempotence check, the ownership guard, the
    TOML pre-check) all treat ``None`` as "safe-refuse / not ours" — the same
    posture :func:`is_managed` already uses for unreadable files.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
