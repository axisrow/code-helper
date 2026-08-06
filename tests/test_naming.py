"""Tests for wrapper-name (alias) validation.

Two independent gates are pinned here:

- :func:`validate_alias` — the semantic rule set, with actionable messages.
- :meth:`Paths.script_for` — the structural guard, tested WITHOUT going
  through ``validate_alias`` (see ``test_script_for_*``). That separation is
  the point: if a future code path forgets to validate, escaping ``bin_dir``
  must still be impossible.
"""

from __future__ import annotations

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.naming import (
    MAX_ALIAS_LENGTH,
    RESERVED_ALIASES,
    validate_alias,
)
from code_helper.services.paths import Paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    [
        "glm",
        "glm-codex",
        "deepseek",
        "glm-ollama",
        "my.wrapper_1",
        "a",
        "9lives",
        "x" * MAX_ALIAS_LENGTH,
    ],
)
def test_valid_aliases_pass_through(alias):
    assert validate_alias(alias) == alias


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["", "   ", "\t"])
def test_empty_alias_rejected(alias):
    with pytest.raises(CodeHelperError, match="must not be empty"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", [".", ".."])
def test_dot_aliases_rejected(alias):
    # `bin_dir / "."` is the directory itself — a write there is not a wrapper.
    with pytest.raises(CodeHelperError, match="must not be"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    ["../../etc/passwd", "a/b", "/abs", "sub/dir/name", "a\\b"],
)
def test_path_separators_rejected(alias):
    # The headline vulnerability: these used to reach `bin_dir / name` intact.
    with pytest.raises(CodeHelperError, match="path separator"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["-rf", "--force", "-"])
def test_leading_dash_rejected(alias):
    with pytest.raises(CodeHelperError, match="must not start with"):
        validate_alias(alias)


@pytest.mark.unit
def test_too_long_alias_rejected():
    with pytest.raises(CodeHelperError, match="too long"):
        validate_alias("x" * (MAX_ALIAS_LENGTH + 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias",
    [
        "with space",
        "tab\there",
        "new\nline",
        "nul\x00byte",
        "semi;colon",
        "dollar$sign",
        "back`tick",
        "pipe|char",
        "star*",
        "кириллица",
        ".hidden",
        "_leading",
    ],
)
def test_unsafe_characters_rejected(alias):
    with pytest.raises(CodeHelperError, match="may only contain"):
        validate_alias(alias)


@pytest.mark.unit
@pytest.mark.parametrize("alias", sorted(RESERVED_ALIASES))
def test_reserved_names_rejected(alias):
    """An alias equal to an agent binary would exec itself forever.

    `~/.local/bin/claude` is also a real symlink on a normal install, so this
    additionally prevents clobbering a working Claude Code entry point.
    """
    with pytest.raises(CodeHelperError, match="reserved name"):
        validate_alias(alias)


# --------------------------------------------------------------------------- #
# Paths.script_for — the structural guard, independent of validate_alias
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["../../etc/passwd", "a/b", "", ".", "..", "sub/dir"],
)
def test_script_for_rejects_names_that_are_not_one_component(tmp_path, name):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="invalid wrapper name"):
        paths.script_for(name)


@pytest.mark.unit
def test_script_for_never_escapes_bin_dir(tmp_path):
    """The guarantee, stated positively: whatever gets through stays inside."""
    paths = Paths.from_home(tmp_path)
    for name in ["glm", "glm-codex", "a.b_c-d"]:
        assert paths.script_for(name).parent == paths.bin_dir


@pytest.mark.unit
def test_script_for_still_does_no_io(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.script_for("glm")
    assert not paths.bin_dir.exists()
