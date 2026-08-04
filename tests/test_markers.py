"""Test that the two tier markers resolve via ``pytest --markers``.

If a marker were unregistered, ``--strict-markers`` (set in ``addopts``) would
turn a typo'd ``@pytest.mark.unnit`` into a hard collection error. This test
fails loud if either of unit/integration is missing from the registry, which
is the early-warning signal that marker discipline has regressed.

The subprocess runs with the REAL home restored (see ``_subprocess_env``) so
the child Python can resolve user site-packages (pytest pulls in dependencies
that on macOS live in ``~/Library/Python/3.x/lib/python/site-packages``,
located relative to ``HOME``).
"""

import os
import subprocess
import sys

import pytest

# Captured at import time, before _isolate_home redirects HOME.
REAL_HOME = os.environ["HOME"]


def _subprocess_env() -> dict[str, str]:
    """Build a subprocess env with the REAL home restored."""
    env = dict(os.environ)
    env["HOME"] = REAL_HOME
    return env


@pytest.mark.unit
def test_markers_registered():
    """Both tier markers (unit/integration) appear in --markers."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    for marker in ("unit", "integration"):
        assert f"@pytest.mark.{marker}" in result.stdout, (
            f"marker '{marker}' missing from pytest --markers output"
        )
