"""Project-wide pytest configuration, fixtures, and shared test helpers.

The autouse ``_isolate_home`` fixture is the project's "do not corrupt the
developer's real files" ideology made testable: it points ``HOME`` at a
per-test temporary directory. Every test — unit or integration — gets this
isolation with zero opt-in (``autouse=True``). A buggy test still must not
write to the developer's real ``~/.local/bin``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from codehelper.services.paths import Paths


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


# --- shared file helpers -----------------------------------------------------


def paths_from_home(tmp_path: Path) -> Paths:
    """``Paths`` bound to the isolated test HOME (what ``_isolate_home`` set)."""
    return Paths.from_home(tmp_path)


def write_settings(paths: Paths, data: dict) -> None:
    """Seed ``~/.claude/settings.json`` with a full payload.

    The one primitive behind the former per-file ``_write_settings`` copies
    (test_cli_proxy / test_cli_switch / test_doctor). Payload shaping — env
    only vs env+hooks — is the caller's policy and stays at the call site.
    """
    paths.claude_dir.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(json.dumps(data), encoding="utf-8")


# --- the select_from_menu test double ----------------------------------------


def ctx_window_zero(text: str) -> str | None:
    """Decide hook: the scripted "no declaration" answer for the
    context-window ask (issue #83). Tests that don't care about the window
    still must not hang on it, and it must not consume a scripted step."""
    if text.startswith("Context window for"):
        return "0"
    return None


class RecordingSelect:
    """A ``select_from_menu`` double: records rendered prompts, plays back
    scripted answers.

    Each call renders the prompt (the main screen's header is a lazy
    callable) into ``prompts``, then answers. ``decide`` sees the rendered
    text first and may return the answer for that prompt — or raise —
    without consuming a scripted step; ``None`` falls through to the
    ``answers`` iterator.
    """

    def __init__(
        self,
        *answers: str,
        decide: Callable[[str], str | None] | None = None,
    ) -> None:
        self._answers = iter(answers)
        self._decide = decide
        self.prompts: list[str] = []

    def __call__(
        self,
        _items: object,
        *,
        prompt: str | Callable[[], str] = "",
        **_kwargs: object,
    ) -> str:
        text = prompt() if callable(prompt) else prompt
        self.prompts.append(text)
        decided = self._decide(text) if self._decide else None
        return decided if decided is not None else next(self._answers)


@pytest.fixture
def recording_select(monkeypatch):
    """Install a :class:`RecordingSelect` as ``cli.menu.select_from_menu``.

    Returns the instance so the test can assert on ``.prompts``.
    """

    def _install(*answers: str, decide=None) -> RecordingSelect:
        select = RecordingSelect(*answers, decide=decide)
        monkeypatch.setattr("codehelper.cli.menu.select_from_menu", select)
        return select

    return _install
