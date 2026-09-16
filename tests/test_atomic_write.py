"""Pin the crash-safe ``atomic_write`` helper's contract.

Atomic (never partial): a write via temp + fsync + ``os.replace`` leaves the
destination byte-exact, and a forced exception between temp-create and
replace leaves NO partial destination, NO orphaned temp, and a pre-existing
destination byte-for-byte preserved. The temp is a sibling of the destination
(same-filesystem atomic rename), and ``os.fsync`` is called strictly before
``os.replace``.

Mode param: ``mode=0o600`` produces a destination whose ``stat.S_IMODE`` is
exactly ``0o600``; ``mode=None`` on a FRESH write keeps the temp default (no
chmod); ``mode=None`` on an OVERWRITE preserves the existing file's mode
(0600 stays 0600, 0644 stays 0644); an explicit ``mode`` always wins.

Cross-cutting: the helper never emits ``data`` via print/stdout/stderr
(secrets discipline — tokens pass through it), and it creates missing parent
directories.

All dests live under ``tmp_path`` (the autouse ``_isolate_home`` fixture
already redirects ``HOME`` to a tmp sandbox).
"""

from __future__ import annotations

import builtins
import os
import stat
import sys

import pytest

import codehelper.backends._atomic as atomic_mod
from codehelper.backends._atomic import atomic_write


class _Boom(RuntimeError):
    """Sentinel exception so ``pytest.raises`` proves the helper re-raised."""


# --------------------------------------------------------------------------- #
# round-trip integrity
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_roundtrip_bytes(tmp_path):
    dest = tmp_path / "cfg.toml"
    atomic_write(dest, b"hello")
    assert dest.read_bytes() == b"hello"


@pytest.mark.unit
def test_atomic_write_roundtrip_str(tmp_path):
    """str payload is encoded UTF-8 on disk."""
    dest = tmp_path / "cfg.toml"
    atomic_write(dest, "héllo")
    assert dest.read_bytes() == "héllo".encode()


# --------------------------------------------------------------------------- #
# atomicity on exception (no pre-existing destination)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_failure_leaves_no_partial_and_no_temp(tmp_path):
    """A forced os.replace failure leaves no dest file and no temp."""
    dest = tmp_path / "cfg.toml"

    def boom(*_args, **_kwargs):
        raise _Boom("forced replace failure")

    monkeypatch_os_replace(boom)
    try:
        with pytest.raises(_Boom):
            atomic_write(dest, b"payload")
    finally:
        restore_os_replace()

    assert not dest.exists(), "destination must not appear in a partial state"

    leftovers = os.listdir(tmp_path)
    assert "cfg.toml" not in leftovers, "partial dest leaked"
    assert leftovers == [], f"orphaned temp survived: {leftovers}"


# --------------------------------------------------------------------------- #
# atomicity on exception (pre-existing destination preserved)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_failure_preserves_pre_existing_destination(tmp_path):
    """On a failed overwrite, the pre-existing destination is unchanged."""
    dest = tmp_path / "cfg.toml"
    prior = b"ORIGINAL-CONFIG"
    dest.write_bytes(prior)

    def boom(*_args, **_kwargs):
        raise _Boom("forced replace failure")

    monkeypatch_os_replace(boom)
    try:
        with pytest.raises(_Boom):
            atomic_write(dest, b"NEW-BUT-SHOULD-NOT-LAND")
    finally:
        restore_os_replace()

    assert dest.read_bytes() == prior, "pre-existing destination was corrupted"


# --------------------------------------------------------------------------- #
# temp is a sibling of the destination (same-filesystem rename)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_temp_is_sibling_of_destination(tmp_path):
    dest = tmp_path / "nested" / "cfg.toml"
    dest.parent.mkdir(parents=True, exist_ok=True)

    seen_dirs = []
    real_ntf = atomic_mod.tempfile.NamedTemporaryFile

    def recording_ntf(*args, **kwargs):
        seen_dirs.append(kwargs.get("dir"))
        return real_ntf(*args, **kwargs)

    atomic_mod.tempfile.NamedTemporaryFile = recording_ntf
    try:
        atomic_write(dest, b"x")
    finally:
        atomic_mod.tempfile.NamedTemporaryFile = real_ntf

    assert seen_dirs, "NamedTemporaryFile was not called"
    assert str(dest.parent) in seen_dirs, (
        f"temp dir must be the destination parent; got {seen_dirs}"
    )


# --------------------------------------------------------------------------- #
# fsync called strictly before os.replace (load-bearing order)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_fsync_before_replace_order(tmp_path):
    dest = tmp_path / "cfg.toml"
    order: list[str] = []

    real_fsync = atomic_mod.os.fsync
    real_replace = atomic_mod.os.replace

    def fsync_recorder(fd):
        order.append("fsync")
        return real_fsync(fd)

    def replace_recorder(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    atomic_mod.os.fsync = fsync_recorder
    atomic_mod.os.replace = replace_recorder
    try:
        atomic_write(dest, b"x")
    finally:
        atomic_mod.os.fsync = real_fsync
        atomic_mod.os.replace = real_replace

    assert order == ["fsync", "replace"], (
        f"fsync must precede replace; observed order={order}"
    )


# --------------------------------------------------------------------------- #
# mode=None does NOT call os.chmod
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_mode_none_fresh_write_does_not_chmod(tmp_path):
    dest = tmp_path / "cfg.toml"  # does not exist → fresh write
    chmod_calls: list[tuple] = []

    real_chmod = atomic_mod.os.chmod
    atomic_mod.os.chmod = lambda *a, **k: chmod_calls.append((a, k))
    try:
        atomic_write(dest, b"x", mode=None)
    finally:
        atomic_mod.os.chmod = real_chmod

    assert chmod_calls == [], f"mode=None must not chmod; saw calls={chmod_calls}"


# --------------------------------------------------------------------------- #
# mode=0o600 produces exactly 0600 on the destination
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_mode_0600_exact_permissions(tmp_path):
    dest = tmp_path / "secret.yml"
    atomic_write(dest, b"token: secret", mode=0o600)
    got = stat.S_IMODE(os.stat(dest).st_mode)
    assert got == 0o600, f"expected 0o600, got {oct(got)}"


# --------------------------------------------------------------------------- #
# overwrite with mode=None preserves pre-existing mode
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_overwrite_preserves_pre_existing_0600(tmp_path):
    dest = tmp_path / "cfg.toml"
    atomic_write(dest, b"first", mode=0o600)
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600

    atomic_write(dest, b"second", mode=None)
    assert dest.read_bytes() == b"second"
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600, (
        "mode=None overwrite must preserve pre-existing 0600"
    )


@pytest.mark.unit
def test_atomic_write_overwrite_preserves_pre_existing_0644(tmp_path):
    """Rewriting a 0644 file with mode=None keeps it 0644, not ~0600.

    This is the case os.replace + no-chmod gets wrong: the temp file is
    ~0600, so an existing 0644 file would be silently narrowed. atomic_write
    restores the prior 0644.
    """
    dest = tmp_path / "config.toml"
    dest.write_bytes(b"original")
    os.chmod(dest, 0o644)
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o644

    atomic_write(dest, b"patched", mode=None)
    assert dest.read_bytes() == b"patched"
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o644, (
        "mode=None overwrite of a 0644 file must stay 0644, not narrow to the temp default"
    )


@pytest.mark.unit
def test_atomic_write_explicit_mode_wins_over_existing(tmp_path):
    """An explicit mode overrides the existing file's mode."""
    dest = tmp_path / "cfg.toml"
    dest.write_bytes(b"x")
    os.chmod(dest, 0o644)
    atomic_write(dest, b"y", mode=0o600)  # explicit → force 0600
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600


# --------------------------------------------------------------------------- #
# Secrets discipline — data never emitted via print/stdout/stderr
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_never_emits_data_via_stdio(tmp_path):
    """The helper never prints/writes data to stdout/stderr.

    Patches builtins.print, sys.stdout.write, sys.stderr.write; asserts none
    were called with any argument containing the secret. Also asserts the
    helper module does not import logging. The file itself DOES contain the
    secret (the discipline is about logging, not about the file).
    """
    secret = b"SECRETVALUE-d0-not-leak"
    dest = tmp_path / "secret.yml"

    print_calls: list[tuple] = []
    stdout_calls: list[tuple] = []
    stderr_calls: list[tuple] = []

    real_print = builtins.print
    real_stdout_write = sys.stdout.write
    real_stderr_write = sys.stderr.write

    def spy_print(*args, **kwargs):
        print_calls.append((args, kwargs))
        return None

    def spy_stdout_write(s):
        stdout_calls.append((s,))
        return len(s)

    def spy_stderr_write(s):
        stderr_calls.append((s,))
        return len(s)

    monkeypatch_stdio(spy_print, spy_stdout_write, spy_stderr_write)
    try:
        atomic_write(dest, secret, mode=0o600)
    finally:
        restore_stdio(real_print, real_stdout_write, real_stderr_write)

    assert dest.read_bytes() == secret

    for args in print_calls:
        assert not any("SECRETVALUE" in str(a) for a in args[0]), (
            f"print leaked secret: {args}"
        )
    for (s,) in stdout_calls:
        assert "SECRETVALUE" not in s, f"stdout leaked secret: {s!r}"
    for (s,) in stderr_calls:
        assert "SECRETVALUE" not in s, f"stderr leaked secret: {s!r}"

    assert not hasattr(atomic_mod, "logging"), (
        "_atomic.py must not import logging; secrets pass through it"
    )


# --------------------------------------------------------------------------- #
# Directory creation — the one non-atomic allowance
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_atomic_write_creates_missing_parent_dirs(tmp_path):
    dest = tmp_path / "a" / "b" / "c" / "cfg.toml"
    atomic_write(dest, b"x")
    assert dest.read_bytes() == b"x"
    assert dest.parent.is_dir(), "parent directories were not created"


@pytest.mark.unit
def test_atomic_write_clears_an_empty_directory_at_the_dest(tmp_path):
    """An EMPTY directory at the dest is cleared so os.replace can land the file."""
    dest = tmp_path / "cfg.toml"
    dest.mkdir()  # empty directory

    atomic_write(dest, b"canonical")  # must not raise

    assert dest.is_file()
    assert dest.read_bytes() == b"canonical"


@pytest.mark.unit
def test_atomic_write_refuses_to_delete_a_non_empty_directory(tmp_path):
    """A NON-empty dir at the dest is NOT recursively deleted — fail loud, no data loss."""
    dest = tmp_path / "cfg.toml"
    dest.mkdir()
    (dest / "precious.txt").write_text("do not delete")

    with pytest.raises(OSError):  # os.rmdir on a non-empty dir → OSError (ENOTEMPTY)
        atomic_write(dest, b"canonical")

    assert dest.is_dir()
    assert (dest / "precious.txt").read_text() == "do not delete"
    leftovers = [p for p in os.listdir(tmp_path) if p != "cfg.toml"]
    assert leftovers == [], f"orphaned temp survived: {leftovers}"


@pytest.mark.unit
def test_atomic_write_over_symlink_does_not_clear_target(tmp_path):
    """The dir-clear must NOT follow a symlink and touch its target."""
    dest = tmp_path / "cfg.toml"
    target = tmp_path / "real_dir"
    target.mkdir()
    (target / "keep.txt").write_text("precious")
    os.symlink(target, dest)

    atomic_write(dest, b"canonical")  # must not raise, must not touch target/

    assert dest.is_file()
    assert not dest.is_symlink()
    assert dest.read_bytes() == b"canonical"
    assert (target / "keep.txt").read_text() == "precious"  # target untouched


# --------------------------------------------------------------------------- #
# Local helpers — module-namespace monkeypatching (intercepts the helper's
# exact os.replace / tempfile.NamedTemporaryFile calls).
# --------------------------------------------------------------------------- #
_REAL_REPLACE = atomic_mod.os.replace


def monkeypatch_os_replace(fn):
    atomic_mod.os.replace = fn


def restore_os_replace():
    atomic_mod.os.replace = _REAL_REPLACE


def monkeypatch_stdio(print_fn, stdout_fn, stderr_fn):
    builtins.print = print_fn
    sys.stdout.write = stdout_fn
    sys.stderr.write = stderr_fn


def restore_stdio(real_print, real_stdout_write, real_stderr_write):
    builtins.print = real_print
    sys.stdout.write = real_stdout_write
    sys.stderr.write = real_stderr_write
