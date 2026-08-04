"""Prove the autouse HOME-isolation fixture protects the developer's real
``~/.local/bin``.

``REAL_HOME`` is captured at module import time — BEFORE pytest's autouse
``_isolate_home`` fixture swaps ``HOME`` to a tmp dir — so the test can
compare the post-isolation ``$HOME`` against the real home and assert the
developer's actual ``~/.local/bin/test_marker`` is never created.
"""

import os
from pathlib import Path

import pytest

# Captured at import time, before _isolate_home redirects HOME.
REAL_HOME = Path(os.environ["HOME"])


@pytest.mark.unit
def test_home_isolated_to_tmp(_isolate_home):
    """HOME points at the per-test tmp_path."""
    assert Path(os.environ["HOME"]) == _isolate_home


@pytest.mark.unit
def test_real_local_bin_not_touched(_isolate_home):
    """A marker written to ``$HOME/.local/bin`` lands in the sandbox, not real ``~``.

    The developer's REAL ``~/.local/bin/test_marker`` must NOT exist after the
    test writes its marker — the load-bearing assertion that the autouse
    fixture actually protects the developer's real files.
    """
    bin_dir = Path(os.environ["HOME"]) / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    marker = bin_dir / "test_marker"
    marker.write_text("isolation probe")
    # Landed in the per-test sandbox.
    assert (_isolate_home / ".local" / "bin" / "test_marker").exists()
    # Did NOT leak into the developer's real home.
    assert not (REAL_HOME / ".local" / "bin" / "test_marker").exists()
