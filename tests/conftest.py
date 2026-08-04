"""Project-wide pytest configuration and shared fixtures.

The autouse ``_isolate_home`` fixture is the project's "do not corrupt the
developer's real files" ideology made testable: it points ``HOME`` at a
per-test temporary directory. Every test — unit or integration — gets this
isolation with zero opt-in (``autouse=True``). A buggy test still must not
write to the developer's real ``~/.local/bin``.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Isolate EVERY test from the real ``$HOME``.

    Sets ``HOME`` to a per-test ``tmp_path``. Unlike the archived project's
    conftest, no subdirectory is pre-created here — ``atomic_write`` creates
    ``~/.local/bin`` itself on first write, so there is nothing to seed.
    Yields the isolated home path so tests that want to assert against it may
    request the fixture explicitly.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path
