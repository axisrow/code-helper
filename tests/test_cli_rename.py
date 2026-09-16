"""Tests for the ``rename`` subcommand (issue #95): CLI parity for the TUI's
``e`` — the wrapper alias move and the profile rename with its marker cascade.

Follows the ``tests/test_cli_disable.py`` pattern: ``main([...])`` through the
real dispatch, HOME isolated by the autouse fixture.
"""

from __future__ import annotations

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths
from codehelper.services.secrets import credential_for
from codehelper.services.wrappers import is_installed


@pytest.mark.integration
def test_rename_wrapper_moves_the_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert main(["rename", "wrapper", "glm", "glm2"]) == 0

    paths = Paths.from_home(tmp_path)
    assert is_installed(paths, "glm2")
    assert not is_installed(paths, "glm")


@pytest.mark.integration
def test_rename_wrapper_collision_fails_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0

    assert main(["rename", "wrapper", "glm", "glm"]) == 1
    assert "already exists" in capsys.readouterr().err


@pytest.mark.integration
def test_rename_profile_moves_the_cache_and_marker(tmp_path, monkeypatch):
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    save_credential(paths, "zai", "sk-env", "work")
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm", "--profile", "work"]) == 0

    assert main(["rename", "profile", "zai", "work", "personal"]) == 0

    assert credential_for(paths, "zai", "work") == ""
    assert credential_for(paths, "zai", "personal") == "sk-env"
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert "profile=personal" in body.split("\n")[1]


@pytest.mark.integration
def test_rename_profile_collision_fails_clean(tmp_path, monkeypatch, capsys):
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    save_credential(paths, "zai", "sk-env", "work")
    save_credential(paths, "zai", "sk-other", "personal")
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm", "--profile", "work"]) == 0

    assert main(["rename", "profile", "zai", "work", "personal"]) == 1
    assert "invalid new profile name" in capsys.readouterr().err


@pytest.mark.integration
def test_rename_profile_needs_three_args(tmp_path, capsys):
    assert main(["rename", "profile", "zai", "work"]) == 1
    assert "rename profile takes" in capsys.readouterr().err


@pytest.mark.integration
def test_rename_wrapper_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-env")
    assert main(["add", "glm"]) == 0
    paths = Paths.from_home(tmp_path)

    assert main(["--dry-run", "rename", "wrapper", "glm", "glm2"]) == 0

    assert is_installed(paths, "glm")
    assert not is_installed(paths, "glm2")
