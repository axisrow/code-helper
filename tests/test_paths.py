"""Prove the injectable ``Paths`` object: pure arithmetic + HOME isolation.

SC-1: ``Paths.from_home(home)`` resolves ``bin_dir`` under one injected home.
SC-2: a unit test using ``Paths.from_home(tmp_path)`` round-trips the resolved
path AND provably never touches the developer's real ``$HOME``.

``REAL_HOME`` is captured at module import time — BEFORE pytest's autouse
``_isolate_home`` fixture swaps ``HOME`` — so the test can prove the resolved
path does not live under the developer's real home, even though ``from_home``
is pure and creates nothing.
"""

import dataclasses
import os
from pathlib import Path

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.paths import Paths

# Captured at import time, before _isolate_home redirects HOME.
REAL_HOME = Path(os.environ["HOME"])


@pytest.mark.unit
def test_from_home_resolves_bin_dir_under_injected_home(tmp_path):
    p = Paths.from_home(tmp_path)
    assert p.bin_dir == tmp_path / ".local" / "bin"


@pytest.mark.unit
def test_from_home_accepts_str_and_path(tmp_path):
    assert Paths.from_home(str(tmp_path)) == Paths.from_home(tmp_path)


@pytest.mark.unit
def test_from_home_is_pure_no_fs_effects(tmp_path):
    """``from_home`` adds nothing to ``tmp_path``'s contents."""
    before = set(tmp_path.iterdir())
    Paths.from_home(tmp_path)
    after = set(tmp_path.iterdir())
    assert before == after


@pytest.mark.unit
def test_paths_is_frozen(tmp_path):
    p = Paths.from_home(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.bin_dir = tmp_path


@pytest.mark.unit
def test_default_returns_from_home_of_path_home(tmp_path):
    """Under the autouse fixture, ``Path.home()`` is the tmp dir."""
    assert Paths.default() == Paths.from_home(Path.home())


@pytest.mark.unit
def test_from_home_never_references_real_home(tmp_path):
    """No resolved path prefixes the developer's REAL_HOME."""
    real_home_str = str(REAL_HOME)
    p = Paths.from_home(tmp_path)
    assert not str(p.bin_dir).startswith(real_home_str), (
        f"resolved path {p.bin_dir} leaks under the developer's real home"
    )


@pytest.mark.unit
def test_script_for_resolves_under_bin_dir(tmp_path):
    p = Paths.from_home(tmp_path)
    assert p.script_for("deepseek") == tmp_path / ".local" / "bin" / "deepseek"


@pytest.mark.unit
def test_from_home_resolves_config_dir_under_injected_home(tmp_path):
    p = Paths.from_home(tmp_path)
    assert p.config_dir == tmp_path / ".config" / "code-helper"


@pytest.mark.unit
def test_config_dir_never_references_real_home(tmp_path):
    """Same HOME-isolation guarantee as bin_dir."""
    real_home_str = str(REAL_HOME)
    p = Paths.from_home(tmp_path)
    assert not str(p.config_dir).startswith(real_home_str)


@pytest.mark.unit
def test_credentials_file_resolves_under_config_dir(tmp_path):
    p = Paths.from_home(tmp_path)
    assert (
        p.credentials_file()
        == tmp_path / ".config" / "code-helper" / "credentials.json"
    )


@pytest.mark.unit
def test_config_dir_resolution_is_pure_no_fs_effects(tmp_path):
    """Resolving config_dir creates nothing — directory creation is the write
    boundary's job (``backends/_atomic.py``), not ``Paths``."""
    before = set(tmp_path.iterdir())
    Paths.from_home(tmp_path).credentials_file()
    after = set(tmp_path.iterdir())
    assert before == after


@pytest.mark.unit
def test_from_home_resolves_claude_dir_under_injected_home(tmp_path):
    p = Paths.from_home(tmp_path)
    assert p.claude_dir == tmp_path / ".claude"


@pytest.mark.unit
def test_claude_dir_never_references_real_home(tmp_path):
    """Same HOME-isolation guarantee as bin_dir/config_dir."""
    real_home_str = str(REAL_HOME)
    p = Paths.from_home(tmp_path)
    assert not str(p.claude_dir).startswith(real_home_str)


@pytest.mark.unit
def test_claude_settings_resolves_under_claude_dir(tmp_path):
    p = Paths.from_home(tmp_path)
    assert p.claude_settings() == tmp_path / ".claude" / "settings.json"


@pytest.mark.unit
def test_claude_settings_resolution_is_pure_no_fs_effects(tmp_path):
    before = set(tmp_path.iterdir())
    Paths.from_home(tmp_path).claude_settings()
    after = set(tmp_path.iterdir())
    assert before == after


@pytest.mark.unit
@pytest.mark.parametrize("slot", [1, 2, 3])
def test_claude_settings_backup_resolves_under_claude_dir(tmp_path, slot):
    p = Paths.from_home(tmp_path)
    assert p.claude_settings_backup(slot) == (
        tmp_path / ".claude" / f"settings.json.bak{slot}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("slot", [0, 4, -1, 99])
def test_claude_settings_backup_rejects_an_invalid_slot(tmp_path, slot):
    p = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="invalid backup slot"):
        p.claude_settings_backup(slot)
