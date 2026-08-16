"""Tests for ``services/claude_settings.py`` — the ``switch`` command.

Like ``config.toml`` in ``test_codex_default.py``, ``settings.json`` here is
ALWAYS a foreign file by definition — it belongs to Claude Code, not
code-helper — so this module patches specific ``env`` keys and must leave
everything else (foreign env keys, every top-level key) byte-for-byte
untouched. The single most important test in this file is
``test_apply_switch_preserves_foreign_env_keys`` — losing a user's
``HTTPS_PROXY``/``NO_PROXY`` would break their network.
"""

from __future__ import annotations

import json
import re
import stat

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.claude_settings import (
    MANAGED_ENV_KEYS,
    SettingsPatch,
    apply_switch,
    current_switch,
    diff_preview,
    patch_settings,
    read_settings,
    resolve_switch_patch,
    restore_settings,
)
from code_helper.services.model import (
    ConfigShape,
    Provider,
    get_agent,
    get_provider,
    with_base_url,
)
from code_helper.services.paths import Paths
from code_helper.services.render import _render_anthropic_env
from code_helper.services.spec import TierModels, WrapperSpec

ZAI = get_provider("zai")
OLLAMA = get_provider("ollama")
LITELLM = get_provider("litellm")
NATIVE = get_provider("native")

_FOREIGN_ENV = {
    "HTTPS_PROXY": "http://127.0.0.1:8118",
    "HTTP_PROXY": "http://127.0.0.1:8118",
    "NO_PROXY": "localhost,127.0.0.1,::1,z.ai,.z.ai",
    "no_proxy": "localhost,127.0.0.1,::1,z.ai,.z.ai",
    "IS_DEMO": "1",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
}

_FOREIGN_TOP_LEVEL = {
    "permissions": {"allow": ["Bash(git *)"]},
    "hooks": {"PreToolUse": []},
    "statusLine": {"type": "command", "command": "echo hi"},
    "model": "opusplan",
    "enabledPlugins": ["etopro-plugins"],
    "extraKnownMarketplaces": {},
    "alwaysThinkingEnabled": True,
    "autoMode": {"soft_deny": []},
}


def _write(paths: Paths, data: dict) -> None:
    paths.claude_settings().parent.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read(paths: Paths) -> dict:
    return json.loads(paths.claude_settings().read_text(encoding="utf-8"))


def _confirm_no(_path, _preview):
    return False


# --------------------------------------------------------------------------- #
# resolve_switch_patch — pure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_resolve_switch_patch_zai():
    patch = resolve_switch_patch(
        ZAI, tier_models=TierModels.uniform("glm-5.2"), token="sk-test"
    )
    assert patch.provider_name == "zai"
    assert patch.env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert patch.env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    assert patch.env["ANTHROPIC_API_KEY"] == ""
    assert patch.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.2"
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in patch.env
    assert not patch.is_reset


@pytest.mark.unit
def test_resolve_switch_patch_includes_subagent_model_when_given():
    patch = resolve_switch_patch(
        ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        subagent_model="glm-5.2",
    )
    assert patch.env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2"


@pytest.mark.unit
def test_resolve_switch_patch_native_ignores_models_and_token():
    """A reset patch is env=={} no matter what the caller passes — the CLI
    layer is expected to collect neither for `switch native`, but even if it
    did, nothing here may leak them into the patch."""
    patch = resolve_switch_patch(
        NATIVE,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-should-never-appear",
    )
    assert patch.env == {}
    assert patch.is_reset


@pytest.mark.unit
def test_resolve_switch_patch_base_url_matches_the_renderer():
    """switch and a generated wrapper for the same provider must agree on
    ANTHROPIC_BASE_URL — both derive it via render.anthropic_base_url."""
    litellm = with_base_url(LITELLM, "https://litellm.example.com/v1")
    patch = resolve_switch_patch(
        litellm, tier_models=TierModels.uniform("glm-5.2"), token="sk-x"
    )
    spec = WrapperSpec(
        alias="x",
        agent=get_agent("claude"),
        provider=litellm,
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.2",
        tier_models=TierModels.uniform("glm-5.2"),
    )
    rendered = _render_anthropic_env(spec, "sk-x")
    assert f"ANTHROPIC_BASE_URL='{patch.env['ANTHROPIC_BASE_URL']}'" in rendered


@pytest.mark.unit
def test_resolve_switch_patch_required_base_url_refused():
    with pytest.raises(CodeHelperError, match="requires a base URL"):
        resolve_switch_patch(LITELLM, tier_models=TierModels.uniform("x"), token="sk-x")


@pytest.mark.unit
def test_resolve_switch_patch_missing_model_refused():
    with pytest.raises(CodeHelperError, match="a model is required"):
        resolve_switch_patch(ZAI, tier_models=None, token="sk-x")


@pytest.mark.unit
def test_resolve_switch_patch_non_switchable_provider_refused():
    bad = Provider(
        name="wrapper-only",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
    )
    with pytest.raises(CodeHelperError, match="cannot be switched to live"):
        resolve_switch_patch(bad, tier_models=TierModels.uniform("x"), token="")


@pytest.mark.unit
def test_managed_keys_match_renderer():
    """The closed MANAGED_ENV_KEYS list must match, exactly, every env var
    _render_anthropic_env can ever emit for a fully-populated spec — the
    lockstep guard between the two `ANTHROPIC_*` mechanisms."""
    spec = WrapperSpec(
        alias="x",
        agent=get_agent("claude"),
        provider=ZAI,
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.2",
        tier_models=TierModels.uniform("glm-5.2"),
        subagent_model="glm-5.2",  # force every optional line to render
    )
    rendered = _render_anthropic_env(spec, "sk-x")
    exported = set(re.findall(r"^export ([A-Z_]+)=", rendered, re.MULTILINE))
    assert exported == set(MANAGED_ENV_KEYS)


# --------------------------------------------------------------------------- #
# patch_settings — pure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_patch_settings_preserves_foreign_env_keys():
    original = {"env": dict(_FOREIGN_ENV)}
    patch = SettingsPatch(provider_name="zai", env={"ANTHROPIC_BASE_URL": "https://x"})
    patched = patch_settings(original, patch)
    for key, value in _FOREIGN_ENV.items():
        assert patched["env"][key] == value


@pytest.mark.unit
def test_patch_settings_preserves_top_level_keys():
    original = {"env": {}, **_FOREIGN_TOP_LEVEL}
    patch = SettingsPatch(provider_name="zai", env={"ANTHROPIC_BASE_URL": "https://x"})
    patched = patch_settings(original, patch)
    for key, value in _FOREIGN_TOP_LEVEL.items():
        assert patched[key] == value


@pytest.mark.unit
def test_patch_settings_does_not_mutate_the_original():
    original = {"env": {"HTTPS_PROXY": "http://x"}}
    patch = SettingsPatch(provider_name="zai", env={"ANTHROPIC_BASE_URL": "https://x"})
    patch_settings(original, patch)
    assert original == {"env": {"HTTPS_PROXY": "http://x"}}


@pytest.mark.unit
def test_patch_settings_reset_removes_only_managed_keys():
    original = {
        "env": {
            **_FOREIGN_ENV,
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_AUTH_TOKEN": "sk-old",
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.2",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.2",
        }
    }
    patched = patch_settings(original, SettingsPatch(provider_name="native", env={}))
    assert patched["env"] == _FOREIGN_ENV


@pytest.mark.unit
def test_patch_settings_reset_removes_empty_env_block():
    original = {"env": {"ANTHROPIC_BASE_URL": "https://x"}}
    patched = patch_settings(original, SettingsPatch(provider_name="native", env={}))
    assert "env" not in patched


@pytest.mark.unit
def test_patch_settings_replaces_a_previous_switch():
    """zai -> ollama must not leave zai's CLAUDE_CODE_SUBAGENT_MODEL behind."""
    original = {
        "env": {
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_AUTH_TOKEN": "sk-old",
            "CLAUDE_CODE_SUBAGENT_MODEL": "glm-5.2",
        }
    }
    new_patch = SettingsPatch(
        provider_name="ollama",
        env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:11434",
            "ANTHROPIC_AUTH_TOKEN": "ollama",
        },
    )
    patched = patch_settings(original, new_patch)
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in patched["env"]
    assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"


# --------------------------------------------------------------------------- #
# read_settings — IO
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_read_settings_missing_file_reads_as_empty(tmp_path):
    paths = Paths.from_home(tmp_path)
    raw, parsed = read_settings(paths)
    assert raw == ""
    assert parsed == {}


@pytest.mark.unit
def test_read_settings_corrupt_json_refused(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True)
    paths.claude_settings().write_text("{not json", encoding="utf-8")
    with pytest.raises(CodeHelperError, match="not valid JSON"):
        read_settings(paths)


@pytest.mark.unit
def test_read_settings_non_object_json_refused(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True)
    paths.claude_settings().write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CodeHelperError, match="JSON object"):
        read_settings(paths)


# --------------------------------------------------------------------------- #
# apply_switch — IO
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_apply_switch_preserves_foreign_env_keys(tmp_path):
    """THE most important test in this file: HTTPS_PROXY/NO_PROXY etc. must
    survive a real switch, or this feature breaks the user's network."""
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    result = _read(paths)
    for key, value in _FOREIGN_ENV.items():
        assert result["env"][key] == value


@pytest.mark.unit
def test_apply_switch_preserves_top_level_keys(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {}, **_FOREIGN_TOP_LEVEL})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    result = _read(paths)
    for key, value in _FOREIGN_TOP_LEVEL.items():
        assert result[key] == value


@pytest.mark.unit
def test_apply_switch_creates_a_missing_settings_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert not paths.claude_settings().exists()

    changed = apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    assert changed is True
    assert paths.claude_settings().exists()
    assert _read(paths)["env"]["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"


@pytest.mark.unit
def test_apply_switch_corrupt_json_refused_and_file_unchanged(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True)
    paths.claude_settings().write_text("{not json", encoding="utf-8")
    before = paths.claude_settings().read_bytes()

    with pytest.raises(CodeHelperError, match="not valid JSON"):
        apply_switch(
            paths,
            provider=ZAI,
            tier_models=TierModels.uniform("glm-5.2"),
            token="sk-test",
            force=True,
        )

    assert paths.claude_settings().read_bytes() == before


@pytest.mark.unit
def test_apply_switch_unreadable_file_refused_not_treated_as_missing(tmp_path):
    # Regression: read_settings's `raw is None` branch treats a MISSING file
    # and an UNREADABLE one (permission-denied, binary/non-UTF-8) the same
    # way — "", {} — the fresh-install case. With --force that silently
    # replaces a permission-denied-but-real settings.json (root-owned, a
    # read-only mount, ...) with a brand-new two-key file, exactly the data
    # loss the module's own docstring says read_settings refuses to allow.
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True)
    settings = paths.claude_settings()
    settings.write_bytes(b"\xff\xfe\x00\x01not valid utf-8 or json")
    before = settings.read_bytes()

    with pytest.raises(CodeHelperError):
        apply_switch(
            paths,
            provider=ZAI,
            tier_models=TierModels.uniform("glm-5.2"),
            token="sk-test",
            force=True,
        )

    assert settings.read_bytes() == before


@pytest.mark.unit
def test_apply_switch_dry_run_writes_nothing(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    before = paths.claude_settings().read_bytes()

    called = []
    changed = apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        dry_run=True,
        confirm=lambda *a: called.append(a) or True,
    )

    assert changed is True
    assert paths.claude_settings().read_bytes() == before
    assert not list(paths.claude_dir.glob("settings.json.bak*"))
    assert called == []  # dry-run never prompts


@pytest.mark.unit
def test_apply_switch_dry_run_redacts_the_token(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {}})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-secret-token-value",
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "sk-secret-token-value" not in out


@pytest.mark.unit
def test_apply_switch_dry_run_redacts_the_previous_token_too(tmp_path, capsys):
    # Regression: `_redacted_preview` used to replace only the NEW token
    # being written. Switching AWAY from a provider that already left its
    # own token in settings.json therefore printed that OLD credential
    # verbatim in the confirm/--dry-run diff — a real leak into stdout/logs,
    # not merely a cosmetic gap.
    paths = Paths.from_home(tmp_path)
    litellm = with_base_url(LITELLM, "https://litellm.example.com/v1")
    old_patch = resolve_switch_patch(
        litellm,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-OLD-litellm-secret",
    )
    _write(paths, {"env": dict(old_patch.env)})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-NEW-zai-secret",
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "sk-OLD-litellm-secret" not in out
    assert "sk-NEW-zai-secret" not in out


@pytest.mark.unit
def test_apply_switch_confirm_refused_leaves_file_untouched(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    before = paths.claude_settings().read_bytes()

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        apply_switch(
            paths,
            provider=ZAI,
            tier_models=TierModels.uniform("glm-5.2"),
            token="sk-test",
            confirm=_confirm_no,
        )

    assert paths.claude_settings().read_bytes() == before
    assert not list(paths.claude_dir.glob("settings.json.bak*"))


@pytest.mark.unit
def test_apply_switch_force_skips_confirm(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {}})

    changed = apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
        confirm=_confirm_no,  # must be bypassed by force
    )
    assert changed is True


@pytest.mark.unit
def test_apply_switch_refuses_when_file_changed_since_it_was_read(tmp_path):
    # Regression: apply_switch reads settings.json, computes a patch off that
    # snapshot, then (after confirm) writes unconditionally — a second
    # concurrent switch (or a hand-edit) landing in between was silently
    # overwritten with no recheck. `confirm` is the one hook that runs AFTER
    # the read and BEFORE the write, so it doubles here as a way to simulate
    # a concurrent editor winning the race.
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})

    def _concurrent_editor(_path, _preview):
        # Someone else (a second `switch`, or the user by hand) changes the
        # file after apply_switch already read it but before it writes.
        _write(paths, {"env": {**_FOREIGN_ENV, "IS_DEMO": "0"}})
        return True

    with pytest.raises(CodeHelperError, match="changed since it was read"):
        apply_switch(
            paths,
            provider=ZAI,
            tier_models=TierModels.uniform("glm-5.2"),
            token="sk-test",
            confirm=_concurrent_editor,
        )

    # The concurrent editor's write must survive — apply_switch must NOT
    # have clobbered it with a patch computed off the stale snapshot.
    assert _read(paths)["env"]["IS_DEMO"] == "0"


@pytest.mark.unit
def test_apply_switch_is_idempotent_and_consumes_no_backup_slot(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    assert paths.claude_settings_backup(1).exists()
    bak1_after_first = paths.claude_settings_backup(1).read_bytes()

    second = apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    assert second is False
    # A no-op must not rotate the ring — a real switch already sits in bak1;
    # a no-op "switch" must not push it out to bak2/lose it.
    assert paths.claude_settings_backup(1).read_bytes() == bak1_after_first
    assert not paths.claude_settings_backup(2).exists()


@pytest.mark.unit
def test_apply_switch_file_mode_is_0600(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {}})

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    mode = stat.S_IMODE(paths.claude_settings().stat().st_mode)
    assert mode == 0o600


@pytest.mark.unit
def test_apply_switch_native_reset_clears_managed_keys(tmp_path):
    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    assert _read(paths)["env"].get("ANTHROPIC_BASE_URL")

    changed = apply_switch(paths, provider=NATIVE, force=True)

    assert changed is True
    result = _read(paths)
    assert "env" not in result or not (
        set(result.get("env", {})) & set(MANAGED_ENV_KEYS)
    )


@pytest.mark.unit
def test_apply_switch_backup_ring_rotates(tmp_path):
    paths = Paths.from_home(tmp_path)
    for model in ("glm-5.2", "glm-5-turbo", "glm-4.7"):
        apply_switch(
            paths,
            provider=ZAI,
            tier_models=TierModels.uniform(model),
            token=f"sk-{model}",
            force=True,
        )
    # After 3 switches (2 rotations beyond the first write), bak1 holds the
    # state right before the LAST switch (glm-5-turbo's), bak2 the one before
    # that (glm-5.2's).
    bak1 = json.loads(paths.claude_settings_backup(1).read_text())
    bak2 = json.loads(paths.claude_settings_backup(2).read_text())
    assert bak1["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5-turbo"
    assert bak2["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.2"


@pytest.mark.unit
def test_apply_switch_incompatible_provider_leaves_file_untouched(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    before = paths.claude_settings().read_bytes()
    bad = Provider(
        name="wrapper-only",
        shapes=frozenset({ConfigShape.ANTHROPIC_ENV}),
        base_url="http://x",
    )
    with pytest.raises(CodeHelperError, match="cannot be switched to live"):
        apply_switch(
            paths, provider=bad, tier_models=TierModels.uniform("x"), force=True
        )
    assert paths.claude_settings().read_bytes() == before


# --------------------------------------------------------------------------- #
# current_switch
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_current_switch_missing_file_is_none(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert current_switch(paths) is None


@pytest.mark.unit
def test_current_switch_corrupt_file_is_none_not_raise(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True)
    paths.claude_settings().write_text("{not json", encoding="utf-8")
    assert current_switch(paths) is None


@pytest.mark.unit
def test_current_switch_roundtrips_after_a_switch_and_a_reset(tmp_path):
    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    assert current_switch(paths) == "zai"

    apply_switch(paths, provider=NATIVE, force=True)
    assert current_switch(paths) is None


@pytest.mark.unit
def test_current_switch_none_for_a_hand_configured_endpoint(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(
        paths,
        {"env": {"ANTHROPIC_BASE_URL": "https://not-a-registered-provider.example"}},
    )
    assert current_switch(paths) is None


# --------------------------------------------------------------------------- #
# restore_settings — IO
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_restore_settings_no_backup_raises(tmp_path):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="no backup found"):
        restore_settings(paths, slot=1, force=True)


@pytest.mark.unit
def test_restore_settings_round_trips_including_foreign_keys(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV), **_FOREIGN_TOP_LEVEL})
    before = _read(paths)

    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    assert _read(paths) != before

    changed = restore_settings(paths, slot=1, force=True)

    assert changed is True
    assert _read(paths) == before


@pytest.mark.unit
def test_restore_settings_dry_run_writes_nothing(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    before = paths.claude_settings().read_bytes()

    restore_settings(paths, slot=1, dry_run=True)

    assert paths.claude_settings().read_bytes() == before


@pytest.mark.unit
def test_restore_settings_confirm_refused_leaves_file_untouched(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    before = paths.claude_settings().read_bytes()

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        restore_settings(paths, slot=1, confirm=_confirm_no)

    assert paths.claude_settings().read_bytes() == before


@pytest.mark.unit
def test_restore_settings_dry_run_redacts_credentials(tmp_path, capsys):
    # Regression: restore_settings built its preview with plain diff_preview,
    # unlike apply_switch's _redacted_preview — printing both the CURRENT and
    # the BACKUP ANTHROPIC_AUTH_TOKEN verbatim into --dry-run / confirm
    # output.
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-BACKUP-token-value",
        force=True,
    )
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-CURRENT-token-value",
        force=True,
    )

    restore_settings(paths, slot=1, dry_run=True)

    out = capsys.readouterr().out
    assert "sk-BACKUP-token-value" not in out
    assert "sk-CURRENT-token-value" not in out


@pytest.mark.unit
def test_restore_settings_refuses_when_file_changed_since_it_was_read(tmp_path):
    # Regression: restore_settings read `current`, asked for confirmation,
    # then wrote `backup_body` unconditionally — a concurrent switch/restore
    # or hand-edit landing during the confirm prompt was silently clobbered,
    # the same race apply_switch already guards against.
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )

    def _concurrent_editor(_path, _preview):
        _write(paths, {"env": {**_FOREIGN_ENV, "IS_DEMO": "0"}})
        return True

    with pytest.raises(CodeHelperError, match="changed since it was read"):
        restore_settings(paths, slot=1, confirm=_concurrent_editor)

    assert _read(paths)["env"]["IS_DEMO"] == "0"


@pytest.mark.unit
def test_restore_settings_refuses_a_malformed_backup(tmp_path):
    # Regression: restore_settings wrote backup_body verbatim without ever
    # parsing it — a truncated/hand-corrupted .bakN file could be restored
    # straight into settings.json, leaving Claude Code unable to load its
    # config. Mirror read_settings's own JSON-object validation here.
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    paths.claude_settings_backup(1).write_text("{not json", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="not valid JSON"):
        restore_settings(paths, slot=1, force=True)

    assert _read(paths)["env"] == _FOREIGN_ENV


@pytest.mark.unit
def test_restore_settings_missing_slot_raises(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    with pytest.raises(CodeHelperError, match="no backup found"):
        restore_settings(paths, slot=2, force=True)


# --------------------------------------------------------------------------- #
# diff_preview
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_diff_preview_empty_when_identical():
    assert diff_preview("same\n", "same\n") == ""


@pytest.mark.unit
def test_diff_preview_shows_a_real_change():
    preview = diff_preview('{"a": 1}\n', '{"a": 2}\n')
    assert "-{" not in preview or "settings.json" in preview
    assert preview != ""
