"""Tests for user-defined agent persistence (``services/agents.py``).

Mirrors ``test_secrets.py``'s shape: never-raises degradation on a bad file,
round-trip through the real write path, and the security-critical rejects
(reserved names, binary collisions) that guard the unquoted shell
interpolation of ``agent.binary``.
"""

from __future__ import annotations

import json

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.agents import (
    add_user_agent,
    all_agents,
    get_agent,
    load_user_agents,
)
from code_helper.services.model import AGENTS, ConfigShape
from code_helper.services.paths import Paths


def _paths(tmp_path):
    return Paths.from_home(tmp_path)


def _write_agents_file(paths, text):
    path = paths.agents_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# load_user_agents — never raises
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_user_agents_missing_file_is_empty(tmp_path):
    assert load_user_agents(_paths(tmp_path)) == ()


@pytest.mark.unit
def test_load_user_agents_malformed_json_is_empty(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(paths, "{not valid json")
    assert load_user_agents(paths) == ()


@pytest.mark.unit
def test_load_user_agents_non_dict_payload_is_empty(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(paths, '["foo", "bar"]')
    assert load_user_agents(paths) == ()


@pytest.mark.unit
def test_load_user_agents_missing_agents_key_is_empty(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(paths, json.dumps({"other": []}))
    assert load_user_agents(paths) == ()


@pytest.mark.unit
def test_load_user_agents_skips_malformed_entries(tmp_path):
    """A stray bad entry must not cost the user the rest of the file."""
    paths = _paths(tmp_path)
    _write_agents_file(
        paths,
        json.dumps(
            {
                "agents": [
                    {"name": "good-agent", "binary": "good-agent", "description": "ok"},
                    {"name": ""},  # empty name
                    "not-a-dict",
                    {"binary": "no-name"},  # missing name
                    {"name": "bad binary", "binary": "Not Valid!"},  # fails regex
                ]
            }
        ),
    )
    agents = load_user_agents(paths)
    assert [a.name for a in agents] == ["good-agent"]


@pytest.mark.unit
def test_load_user_agents_skips_duplicate_names(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(
        paths,
        json.dumps(
            {
                "agents": [
                    {"name": "dup", "binary": "dup", "description": "first"},
                    {"name": "dup", "binary": "dup2", "description": "second"},
                ]
            }
        ),
    )
    agents = load_user_agents(paths)
    assert len(agents) == 1
    assert agents[0].description == "first"


@pytest.mark.unit
def test_load_user_agents_defaults_binary_to_name(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(paths, json.dumps({"agents": [{"name": "onlyname"}]}))
    agents = load_user_agents(paths)
    assert agents[0].binary == "onlyname"


@pytest.mark.unit
def test_load_user_agents_are_ollama_launch_only(tmp_path):
    paths = _paths(tmp_path)
    _write_agents_file(paths, json.dumps({"agents": [{"name": "myagent"}]}))
    agents = load_user_agents(paths)
    assert agents[0].shapes == frozenset({ConfigShape.OLLAMA_LAUNCH})


# --------------------------------------------------------------------------- #
# all_agents / get_agent — merge order and lookup
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_all_agents_puts_built_ins_first(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "myagent")
    agents = all_agents(paths)
    assert agents[: len(AGENTS)] == AGENTS
    assert agents[-1].name == "myagent"


@pytest.mark.unit
def test_get_agent_finds_a_user_agent(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "myagent", description="My Agent")
    agent = get_agent(paths, "myagent")
    assert agent.name == "myagent"
    assert agent.binary == "myagent"
    assert agent.description == "My Agent"


@pytest.mark.unit
def test_get_agent_still_finds_a_built_in(tmp_path):
    paths = _paths(tmp_path)
    agent = get_agent(paths, "claude")
    assert agent.name == "claude"


@pytest.mark.unit
def test_get_agent_unknown_name_raises_and_lists_known(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "myagent")
    with pytest.raises(CodeHelperError, match="myagent"):
        get_agent(paths, "nope")


# --------------------------------------------------------------------------- #
# add_user_agent — validation and persistence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_add_user_agent_round_trips(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "myagent", "mybin", "My Agent")
    agents = load_user_agents(paths)
    assert len(agents) == 1
    assert agents[0].name == "myagent"
    assert agents[0].binary == "mybin"
    assert agents[0].description == "My Agent"


@pytest.mark.unit
def test_add_user_agent_persists_across_reads(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "first")
    add_user_agent(paths, "second")
    names = {a.name for a in load_user_agents(paths)}
    assert names == {"first", "second"}


@pytest.mark.unit
def test_add_user_agent_rejects_a_built_in_name(tmp_path):
    """Silently shadowing `claude` would change what every claude wrapper
    execs — this must be a hard reject, never an override."""
    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError):
        add_user_agent(paths, "claude")


@pytest.mark.unit
def test_add_user_agent_rejects_a_built_in_binary_as_the_binary(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError, match="reserved"):
        add_user_agent(paths, "myagent", "codex")


@pytest.mark.unit
def test_add_user_agent_rejects_code_helper_itself(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError, match="reserved"):
        add_user_agent(paths, "code-helper")


@pytest.mark.unit
def test_add_user_agent_rejects_duplicate_user_agent_name(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "myagent")
    with pytest.raises(CodeHelperError, match="already exists"):
        add_user_agent(paths, "myagent")


@pytest.mark.unit
def test_add_user_agent_rejects_duplicate_binary_across_user_agents(tmp_path):
    paths = _paths(tmp_path)
    add_user_agent(paths, "first", "shared-bin")
    with pytest.raises(CodeHelperError, match="already exists"):
        add_user_agent(paths, "second", "shared-bin")


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_name", ["Has-Upper", "-leading-dash", "has space", "has/slash"]
)
def test_add_user_agent_rejects_invalid_binary_pattern(tmp_path, bad_name):
    """The exact same gate the built-in registry itself is checked against —
    `binary` is interpolated UNQUOTED into a generated script, so this is a
    shell-injection guard, not a cosmetic validation."""
    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError):
        add_user_agent(paths, bad_name, bad_name)


@pytest.mark.unit
def test_add_user_agent_rejects_empty_name(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(CodeHelperError):
        add_user_agent(paths, "")


@pytest.mark.unit
def test_add_user_agent_returns_the_created_agent(tmp_path):
    paths = _paths(tmp_path)
    agent = add_user_agent(paths, "myagent", "mybin", "desc")
    assert agent.name == "myagent"
    assert agent.binary == "mybin"
    assert agent.shapes == frozenset({ConfigShape.OLLAMA_LAUNCH})
