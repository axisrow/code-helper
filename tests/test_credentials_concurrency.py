"""Empirical answer to issue #17: does concurrent access to
``credentials.json`` lose an update?

``save_credential``/``rename_profile``/``invalidate_cached_credential`` are
all read-modify-write: read the whole file via ``load_credentials``, mutate
an in-memory dict, write the whole file back via ``atomic_write``.
``atomic_write`` (temp-sibling + fsync + ``os.replace``) is crash-safe — an
interrupted write never leaves a half-written file — but on its own it
provides no inter-thread/inter-process serialization of that read-modify-
write window. Four PR #16 review cycles raised this and were triaged SKIP
purely from reading the code, never from a reproduction.

This file settles it empirically, per issue #17's explicit instructions:

1. ``test_save_credential_race_reproduces_a_lost_update_before_the_fix``
   proves the race is real by construction (a forced barrier inside the
   read-write window), against a *bypassed*-lock version of the write path —
   i.e. it demonstrates what issue #17 describes, on the code shape issue #17
   describes, independent of whatever fix lands in this same PR.
2. ``test_many_concurrent_saves_reproduced_lost_updates_before_the_fix``
   is the same demonstration without any injected delay — plain
   GIL-scheduled thread interleaving reliably lost most of 16 concurrent
   writes on this unlocked path (reproduced on every run while writing this
   test, not an occasional flake).
3. The remaining tests exercise the SHIPPED fix — ``secrets._locked_update``,
   an ``fcntl.flock`` held for the read-modify-write window, now wrapping all
   three writers — and prove it closes the window: the same concurrent
   scenarios now preserve every write.

Verdict: the race was real and reproduced immediately under ordinary
scheduling, not just under an adversarially forced interleaving. This is a
**fix**, not a close-with-test — see the ``_locked_update`` docstring in
``secrets.py`` and the ``services/secrets.py`` bullet in ``CLAUDE.md``.
"""

from __future__ import annotations

import json
import threading

import pytest
from conftest import paths_from_home as _paths

from codehelper.backends._atomic import atomic_write
from codehelper.services import secrets as secrets_module
from codehelper.services.paths import Paths
from codehelper.services.secrets import (
    DEFAULT_PROFILE,
    invalidate_cached_credential,
    load_credentials,
    rename_profile,
    save_credential,
    seed_default_profile,
)


def _save_credential_without_lock(
    paths: Paths, provider_name: str, token: str, profile_name: str = DEFAULT_PROFILE
) -> None:
    """A copy of ``save_credential``'s pre-#17 body — no ``_locked_update``.

    Used only to demonstrate the race issue #17 describes on the exact code
    shape it describes, so the "before" half of this file's verdict does not
    depend on being able to see the old git revision.
    """
    if not token:
        return
    data = load_credentials(paths)
    data.setdefault(provider_name, {})[profile_name] = token
    atomic_write(
        paths.credentials_file(),
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )


@pytest.mark.unit
def test_save_credential_race_reproduces_a_lost_update_before_the_fix(tmp_path):
    """Force the read-modify-write window open with a barrier on the
    UNLOCKED write path: two threads both read the pre-state before either
    writes, so the second write silently discards the first — a real lost
    update, not a corrupted file. This is the "yes, the window is real" half
    of issue #17's empirical question, demonstrated independent of the fix.
    """
    paths = _paths(tmp_path)
    _save_credential_without_lock(paths, "seed", "keep-me")

    barrier = threading.Barrier(2)
    real_load = load_credentials

    def _load_then_wait(p: Paths):
        data = real_load(p)
        barrier.wait(timeout=5)
        return data

    errors: list[BaseException] = []

    def _save(provider: str, token: str, load_fn) -> None:
        try:
            if not token:
                return
            data = load_fn(paths)
            data.setdefault(provider, {})[DEFAULT_PROFILE] = token
            atomic_write(
                paths.credentials_file(),
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                mode=0o600,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic only
            errors.append(exc)

    t1 = threading.Thread(target=_save, args=("litellm", "sk-litellm", _load_then_wait))
    t2 = threading.Thread(target=_save, args=("zai", "sk-zai", _load_then_wait))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"unexpected exception(s): {errors}"

    final = load_credentials(paths)
    assert final.get("seed", {}).get(DEFAULT_PROFILE) == "keep-me"

    survivors = {
        name for name in ("litellm", "zai") if final.get(name, {}).get(DEFAULT_PROFILE)
    }
    assert len(survivors) == 1, (
        "expected the forced interleaving on the UNLOCKED write path to "
        f"lose exactly one concurrent write, got: {final}"
    )


@pytest.mark.unit
def test_many_concurrent_saves_reproduced_lost_updates_before_the_fix(tmp_path):
    """Realistic case, unlocked path: N threads, no injected delay. Measured
    (not predicted) result: ordinary GIL-scheduled thread interleaving on an
    uncontended local filesystem reliably lost most of 16 concurrent writes,
    every time this was run while diagnosing issue #17 — not an occasional
    flake. This is the empirical justification for the fix now shipped in
    ``secrets._locked_update`` rather than a fourth SKIP verdict.
    """
    paths = _paths(tmp_path)
    n = 16
    providers = [f"provider-{i}" for i in range(n)]

    threads = [
        threading.Thread(
            target=_save_credential_without_lock,
            args=(paths, p, f"token-for-{p}"),
        )
        for p in providers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    final = load_credentials(paths)
    survived = [
        p
        for p in providers
        if final.get(p, {}).get(DEFAULT_PROFILE) == f"token-for-{p}"
    ]

    assert len(survived) < n, (
        "expected ordinary concurrent writes on the UNLOCKED path to lose "
        f"at least one update, but all {n} survived: {final}"
    )


@pytest.mark.unit
def test_locked_save_credential_survives_the_same_forced_interleaving(tmp_path):
    """The fixed path: two threads racing ``save_credential`` for different
    providers, through the real (locked) code. No injected delay or barrier
    is needed — the lock alone serializes the two read-modify-write cycles,
    so the second thread's ``load_credentials`` call only runs after the
    first thread has already released the lock and completed its write.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "seed", "keep-me")

    def _save(provider: str, token: str) -> None:
        save_credential(paths, provider, token)

    providers = {"litellm": "sk-litellm", "zai": "sk-zai"}
    threads = [
        threading.Thread(target=_save, args=(name, token))
        for name, token in providers.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    final = load_credentials(paths)
    assert final.get("seed", {}).get(DEFAULT_PROFILE) == "keep-me"
    for name, token in providers.items():
        assert final.get(name, {}).get(DEFAULT_PROFILE) == token, (
            f"lock failed to prevent a lost update for {name}: {final}"
        )


@pytest.mark.unit
def test_many_concurrent_locked_saves_lose_nothing(tmp_path):
    """The fixed counterpart of the unlocked N-thread test above: same 16
    concurrent providers, real ``save_credential``, no injected delay — every
    write must survive now that the read-modify-write window is serialized.
    """
    paths = _paths(tmp_path)
    n = 16
    providers = [f"provider-{i}" for i in range(n)]

    def _save(provider: str) -> None:
        save_credential(paths, provider, f"token-for-{provider}")

    threads = [threading.Thread(target=_save, args=(p,)) for p in providers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    final = load_credentials(paths)
    missing = [
        p
        for p in providers
        if final.get(p, {}).get(DEFAULT_PROFILE) != f"token-for-{p}"
    ]
    assert missing == [], f"lock failed to prevent lost update(s): {missing} in {final}"


@pytest.mark.unit
def test_locked_rename_and_invalidate_do_not_race(tmp_path):
    """The other two writers, concurrently, through the real (locked) code
    path: ``rename_profile`` and ``invalidate_cached_credential`` racing each
    other must both take full effect — issue #17 names only
    ``save_credential`` and ``invalidate_cached_credential``; ``rename_profile``
    is the same read-modify-write shape and goes through the same lock.
    """
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-old", profile_name="work")
    save_credential(paths, "zai", "sk-zai")

    t1 = threading.Thread(
        target=rename_profile, args=(paths, "litellm", "work", "renamed")
    )
    t2 = threading.Thread(target=invalidate_cached_credential, args=(paths, "zai"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    final = load_credentials(paths)
    assert final.get("litellm", {}).get("renamed") == "sk-old"
    assert "zai" not in final


def _seed_default_profile_check_outside_lock(
    paths: Paths, provider_name: str, token: str
) -> bool:
    """A copy of ``seed_default_profile``'s pre-fix body — the existence
    check (``profile_names``) runs OUTSIDE ``_locked_update``, only the
    write is locked. Used only to demonstrate the TOCTOU window on the exact
    code shape it existed in, independent of the shipped fix.
    """
    if not token or secrets_module.profile_names(paths, provider_name):
        return False
    save_credential(paths, provider_name, token, DEFAULT_PROFILE)
    return True


@pytest.mark.unit
def test_seed_default_profile_race_loses_an_update_on_the_unlocked_path(tmp_path):
    """Force the TOCTOU window open with a barrier on the UNLOCKED
    check-then-write path: two threads both see "no profile yet" before
    either writes, both pass the guard, and the second ``save_credential``
    call (itself correctly serialized by the lock) still unconditionally
    overwrites — it never re-checks whether a profile appeared in the
    meantime. This contradicts the docstring's "an existing provider profile
    is never overwritten," demonstrated independent of the shipped fix (see
    ``test_seed_default_profile_never_loses_a_concurrent_update`` for the
    fixed counterpart).
    """
    paths = _paths(tmp_path)
    barrier = threading.Barrier(2)
    real_profile_names = secrets_module.profile_names

    def _profile_names_then_wait(p: Paths, provider_name: str):
        result = real_profile_names(p, provider_name)
        barrier.wait(timeout=5)
        return result

    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def _seed(label: str, token: str) -> None:
        try:
            secrets_module.profile_names = _profile_names_then_wait
            results[label] = _seed_default_profile_check_outside_lock(
                paths, "ollama-direct", token
            )
        except BaseException as exc:  # pragma: no cover - diagnostic only
            errors.append(exc)
        finally:
            secrets_module.profile_names = real_profile_names

    t1 = threading.Thread(target=_seed, args=("A", "token-A"))
    t2 = threading.Thread(target=_seed, args=("B", "token-B"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"unexpected exception(s): {errors}"
    assert results == {"A": True, "B": True}, (
        "expected the forced interleaving on the UNLOCKED check-then-write "
        f"path to make BOTH concurrent seed calls believe they were the "
        f"legitimate seeder, got: {results}"
    )

    final = load_credentials(paths)
    assert final.get("ollama-direct", {}).get(DEFAULT_PROFILE) in (
        "token-A",
        "token-B",
    ), (
        "expected exactly one of the two tokens to survive (the other "
        f"silently overwritten), got: {final}"
    )


@pytest.mark.unit
def test_seed_default_profile_never_loses_a_concurrent_update(tmp_path):
    """The fixed path: many threads racing ``seed_default_profile`` for the
    SAME provider, through the real (locked) code — the existence check and
    the write now share one ``_locked_update`` window, so only the first
    caller to acquire the lock ever sees "no profile yet"; every later
    caller re-checks under the same lock and backs off instead of
    clobbering. Exactly one call must report success, and its token must be
    the one that survives.
    """
    paths = _paths(tmp_path)
    n = 8
    tokens = [f"token-{i}" for i in range(n)]
    results: dict[str, bool] = {}

    def _seed(token: str) -> None:
        results[token] = seed_default_profile(paths, "ollama-direct", token)

    threads = [threading.Thread(target=_seed, args=(t,)) for t in tokens]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    winners = [token for token, ok in results.items() if ok]
    assert len(winners) == 1, (
        f"expected exactly one concurrent seed call to win, got: {results}"
    )

    final = load_credentials(paths)
    assert final.get("ollama-direct", {}).get(DEFAULT_PROFILE) == winners[0]


@pytest.mark.unit
def test_locked_update_falls_back_when_the_lock_cannot_be_acquired(
    tmp_path, monkeypatch
):
    """``_locked_update`` must never raise or hang when locking is
    unavailable — mirrors this module's never-raises contract elsewhere
    (``load_credentials``, ``invalidate_cached_credential``). Since #108 the
    acquisition lives in ``backends._atomic.best_effort_lock``, so the
    injected failure is a ``file_lock`` raising ``OSError`` — the same seam
    ``state._locked_update``'s degradation test patches. A normal,
    single-threaded ``save_credential`` must still succeed.
    """
    import codehelper.backends._atomic as atomic_mod

    def _boom(*_args, **_kwargs):
        raise OSError("no locking here")

    monkeypatch.setattr(atomic_mod, "file_lock", _boom)
    paths = _paths(tmp_path)
    save_credential(paths, "litellm", "sk-1")
    assert load_credentials(paths) == {"litellm": {DEFAULT_PROFILE: "sk-1"}}


@pytest.mark.unit
def test_locked_update_falls_back_when_the_lock_file_cannot_be_opened(tmp_path):
    """A lock file that cannot be opened must degrade to no locking rather
    than propagate — the write itself must still succeed, exactly as it did
    before issue #17's fix. The failure shape is real, not mocked: a
    DIRECTORY sitting on the ``credentials.json.lock`` name makes
    ``file_lock``'s open raise (the same shape ``test_state_concurrency``'s
    ``_break_locking`` uses). Only the ``.lock``-suffixed path is blocked;
    ``credentials.json`` in the same (writable) directory is unaffected.
    """
    paths = _paths(tmp_path)
    lock_path = paths.credentials_file().with_suffix(
        paths.credentials_file().suffix + ".lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.mkdir()

    save_credential(paths, "litellm", "sk-1")
    assert load_credentials(paths) == {"litellm": {DEFAULT_PROFILE: "sk-1"}}
