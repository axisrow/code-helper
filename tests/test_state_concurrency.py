"""Does concurrent access to ``state.json`` lose an update?

``state.py`` deliberately shipped WITHOUT locking, and said so in its module
docstring: "A race here loses a pre-selection at worst, never a token."
That reasoning was sound for what the file held — the active token-profile
pointer and the default-wrapper map, both trivially re-set by the user from
the UI that wrote them.

``set_saved_proxy`` broke that premise. The saved proxy address is the ONLY
copy: ``proxy off`` blanks the live values in ``settings.json`` and banks the
address here, so losing this entry to a concurrent writer means ``proxy on``
has nothing to restore and the address is gone for good. The cost of a lost
update stopped being "re-pick a profile" and became data loss.

These tests follow ``test_credentials_concurrency.py``'s structure for the
same reason it exists: settle the question empirically rather than by reading
the code. The first proves the race is real against the unlocked shape; the
rest exercise the shipped fix (``state._locked_update``).
"""

from __future__ import annotations

import json
import threading

import pytest
from conftest import paths_from_home as _paths

from codehelper.backends._atomic import atomic_write
from codehelper.errors import CodeHelperError
from codehelper.services import state as state_module
from codehelper.services.paths import Paths
from codehelper.services.state import (
    active_selection,
    default_wrapper,
    load_state,
    saved_proxy,
    set_active_selection,
    set_default_wrapper,
    set_saved_proxy,
)

_URL = "http://127.0.0.1:8118"


def _set_saved_proxy_without_lock(paths: Paths, url: str, barrier=None) -> None:
    """``set_saved_proxy`` as it was BEFORE the fix: load, (pause), write.

    The barrier forces the interleaving the unlocked shape permits, so the
    demonstration does not depend on scheduler luck.
    """
    state = load_state(paths)
    if barrier is not None:
        barrier.wait(timeout=5)
    state["proxy"] = {"url": url}
    atomic_write(paths.state_file(), json.dumps(state))


def _set_active_without_lock(paths: Paths, provider: str, profile: str, barrier=None):
    state = load_state(paths)
    if barrier is not None:
        barrier.wait(timeout=5)
    state.pop("active_provider", None)
    state.pop("active_profiles", None)
    state["active"] = {"provider": provider, "profile": profile}
    atomic_write(paths.state_file(), json.dumps(state))


@pytest.mark.unit
def test_unlocked_state_writes_lose_the_saved_proxy(tmp_path):
    """The race, proven by construction on the pre-fix shape: a profile
    switch that loaded before the proxy was banked writes its stale snapshot
    back on top, and the address is gone."""
    paths = _paths(tmp_path)
    barrier = threading.Barrier(2)

    threads = [
        threading.Thread(
            target=_set_saved_proxy_without_lock, args=(paths, _URL, barrier)
        ),
        threading.Thread(
            target=_set_active_without_lock, args=(paths, "zai", "default", barrier)
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    state = load_state(paths)
    lost = "proxy" not in state or "active" not in state
    assert lost, (
        "expected the UNLOCKED read-modify-write shape to lose one of the two "
        f"concurrent updates, but both survived: {state}"
    )


@pytest.mark.unit
def test_locked_writers_preserve_both_updates(tmp_path):
    """The shipped fix: the same two writers through the real (locked) code.
    Both updates survive — the lock serialises the read-modify-write cycles
    instead of letting them interleave."""
    paths = _paths(tmp_path)

    threads = [
        threading.Thread(target=set_saved_proxy, args=(paths, _URL)),
        threading.Thread(target=set_active_selection, args=(paths, "zai", "default")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert saved_proxy(paths) == _URL
    assert active_selection(paths) == ("zai", "default")


@pytest.mark.unit
def test_many_concurrent_locked_writers_lose_nothing(tmp_path):
    """Every writer in the module, racing at once, with no injected delay."""
    paths = _paths(tmp_path)
    agents = [f"agent-{i}" for i in range(8)]

    threads = [
        threading.Thread(target=set_saved_proxy, args=(paths, _URL)),
        threading.Thread(target=set_active_selection, args=(paths, "zai", "default")),
        *(
            threading.Thread(target=set_default_wrapper, args=(paths, a, f"w-{a}"))
            for a in agents
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert saved_proxy(paths) == _URL
    assert active_selection(paths) == ("zai", "default")
    for agent in agents:
        assert default_wrapper(paths, agent) == f"w-{agent}"


@pytest.mark.unit
def test_an_exception_inside_the_locked_block_propagates_once(tmp_path):
    """The lock's best-effort acquisition must not swallow the CALLER's
    exceptions. Wrapping the `yield` in `try/except OSError` did exactly that
    — it caught an OSError raised by the body, yielded a second time, and
    crashed with "generator didn't stop after throw()" while silently running
    the caller's block twice. Caught by `test_wrappers.py`'s injected-failure
    test, pinned here."""
    paths = _paths(tmp_path)
    entered = []

    with pytest.raises(OSError, match="from the body"):
        with state_module._locked_update(paths):
            entered.append(1)
            raise OSError("from the body")

    assert entered == [1], "the locked block must run exactly once"


def _break_locking(paths: Paths) -> None:
    """Make lock acquisition fail while the state write itself still works.

    A directory where the ``.lock`` file belongs: ``file_lock`` cannot open
    it, but ``state.json`` in the same (writable) directory is unaffected.
    This is the shape that makes fail-open observable — an unwritable config
    dir would fail the WRITE too, so it proves nothing about the lock.
    """
    lock_path = paths.state_file().with_suffix(paths.state_file().suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.mkdir(exist_ok=True)


@pytest.mark.unit
def test_the_saved_proxy_refuses_to_write_unserialized(tmp_path):
    """The proxy address is the only copy of a value the user cannot re-pick,
    so writing it without serialization is worse than not writing it: a
    silently-lost update looks like success. This writer fails closed."""
    paths = _paths(tmp_path)
    _break_locking(paths)

    with pytest.raises(CodeHelperError, match="could not be locked"):
        set_saved_proxy(paths, _URL)

    assert saved_proxy(paths) is None


@pytest.mark.unit
def test_pre_selection_writers_still_degrade_to_unlocked(tmp_path):
    """The other writers hold re-pickable UI state and a never-raises
    contract — turning an unavailable lock into a crash would break
    `add`/`edit-token` over a profile pointer the user can simply set again.
    They keep the best-effort posture the module shipped with."""
    paths = _paths(tmp_path)
    _break_locking(paths)

    set_active_selection(paths, "zai", "default")
    set_default_wrapper(paths, "claude", "glm")

    assert active_selection(paths) == ("zai", "default")
    assert default_wrapper(paths, "claude") == "glm"


@pytest.mark.unit
def test_the_lock_never_raises_when_it_cannot_be_taken(tmp_path, monkeypatch):
    """Same never-raises contract as ``secrets._locked_update``: a lock that
    cannot be acquired degrades to the unlocked path rather than turning a
    pre-selection write into a crash."""

    def _boom(*_args, **_kwargs):
        raise OSError("no locking here")

    monkeypatch.setattr(state_module, "file_lock", _boom)

    set_active_selection(paths := _paths(tmp_path), "zai", "default")

    assert active_selection(paths) == ("zai", "default")
