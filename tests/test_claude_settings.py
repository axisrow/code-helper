"""Tests for ``services/claude_settings.py`` — the ``switch`` command.

Like ``config.toml`` in ``test_codex_default.py``, ``settings.json`` here is
ALWAYS a foreign file by definition — it belongs to Claude Code, not
codehelper — so this module patches specific ``env`` keys and must leave
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

from codehelper.errors import CodeHelperError
from codehelper.services.claude_settings import (
    MANAGED_ENV_KEYS,
    SettingsPatch,
    active_switch_env,
    apply_switch,
    current_switch,
    diff_preview,
    matches_switch_spec,
    patch_settings,
    read_settings,
    resolve_switch_patch,
    restore_settings,
)
from codehelper.services.model import (
    ConfigShape,
    Provider,
    get_agent,
    get_provider,
    with_base_url,
)
from codehelper.services.paths import Paths
from codehelper.services.render import _render_anthropic_env
from codehelper.services.spec import TierModels, WrapperSpec

ZAI = get_provider("zai")
OLLAMA = get_provider("ollama-direct")
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
def test_resolve_switch_patch_declares_known_context_window():
    """A model Claude Code cannot resolve would auto-compact at its 200k
    fallback — the switch declares the real window when the catalog knows it
    and every tier shares it (Claude Code docs: for a non-``claude-``, no-
    ``[1m]``, unresolvable ID the variable applies directly)."""
    patch = resolve_switch_patch(
        ZAI, tier_models=TierModels.uniform("glm-5.3"), token="sk-test"
    )
    assert patch.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"


@pytest.mark.unit
def test_resolve_switch_patch_skips_context_window_for_unknown_model():
    """No declaration beats a guessed window — an oversized claim overflows
    the real one mid-session."""
    patch = resolve_switch_patch(
        ZAI, tier_models=TierModels.uniform("glm-5-turbo"), token="sk-test"
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in patch.env


@pytest.mark.unit
def test_resolve_switch_patch_skips_context_window_for_mixed_tiers():
    """The variable declares ONE window for the session; tiers that disagree
    (here: an unknown haiku next to a 1M sonnet) must yield no declaration."""
    patch = resolve_switch_patch(
        ZAI,
        tier_models=TierModels(haiku="mystery-3b", sonnet="glm-5.3", opus="glm-5.3"),
        token="sk-test",
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in patch.env


@pytest.mark.unit
def test_resolve_switch_patch_native_ignores_models_and_token():
    """Native explicitly blanks every live key without leaking caller input.

    Claude Code's settings watcher applies environment updates to a running
    process. Removing a key from settings.json does not remove its already
    applied process value, so native must send an explicit empty value for
    each key codehelper previously set.

    Cold-start semantics (issue #62, measured 2026-09-19, Claude Code
    2.1.277/2.1.278): every managed key set to ``""`` behaves exactly like
    an absent key on a fresh launch — identical auth path (no "configured
    but invalid" mode; the same login-required outcome), identical alias
    resolution for ``--model sonnet``/``haiku``/``opus`` (same claude-* id
    with the alias key empty and with it absent, against a recording mock
    endpoint), no ``CLAUDE_CODE_MAX_CONTEXT_TOKENS=""`` parse error, and
    subagents spawn with the same resolved model under
    ``CLAUDE_CODE_SUBAGENT_MODEL=""``. Blanking is safe on cold start, not
    only on live reload; do not re-raise "empty vs unset" in review.
    """
    patch = resolve_switch_patch(
        NATIVE,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-should-never-appear",
    )
    assert patch.env == {key: "" for key in MANAGED_ENV_KEYS}
    assert patch.is_reset


@pytest.mark.unit
def test_resolve_switch_patch_native_blanks_only_keys_present_in_current_env():
    """Native must stay a no-op on a settings.json with no managed override.

    Blanking every MANAGED_ENV_KEYS key unconditionally would turn `switch
    native` on an already-native file into a write that adds keys nobody set
    — the opposite of "native means clear the override". Only keys already
    present in the live env need the explicit-empty treatment; a key that
    was never set has no stale process value to reset.
    """
    patch = resolve_switch_patch(
        NATIVE,
        tier_models=None,
        token="",
        current_env={"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic"},
    )
    assert patch.env == {"ANTHROPIC_BASE_URL": ""}


@pytest.mark.unit
def test_resolve_switch_patch_native_with_no_current_env_blanks_nothing():
    patch = resolve_switch_patch(NATIVE, tier_models=None, token="", current_env={})
    assert patch.env == {}
    assert not patch.is_reset


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


@pytest.mark.unit
def test_renderer_refuses_an_anthropic_env_spec_without_tiers():
    """The renderer-side twin of test_resolve_switch_patch_missing_model_refused:
    build_spec materializes uniform tiers for ANTHROPIC_ENV, so the renderer's
    own raise is unreachable through the front doors — pinned so a future
    second spec constructor that skips it fails here as a domain error,
    not as an AttributeError mid-render."""
    spec = WrapperSpec(
        alias="x",
        agent=get_agent("claude"),
        provider=ZAI,
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.2",
        tier_models=None,
    )
    with pytest.raises(CodeHelperError, match="no tier models"):
        _render_anthropic_env(spec, "sk-x")


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
def test_patch_settings_native_explicitly_blanks_only_managed_keys():
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
    patched = patch_settings(
        original,
        SettingsPatch(
            provider_name="native", env={key: "" for key in MANAGED_ENV_KEYS}
        ),
    )
    assert patched["env"] == {
        **_FOREIGN_ENV,
        **{key: "" for key in MANAGED_ENV_KEYS},
    }


@pytest.mark.unit
def test_patch_settings_native_keeps_empty_managed_keys_for_live_reset():
    original = {"env": {"ANTHROPIC_BASE_URL": "https://x"}}
    patched = patch_settings(
        original,
        SettingsPatch(
            provider_name="native", env={key: "" for key in MANAGED_ENV_KEYS}
        ),
    )
    assert patched["env"] == {key: "" for key in MANAGED_ENV_KEYS}


@pytest.mark.unit
def test_patch_settings_replaces_a_previous_switch():
    """zai -> ollama-direct must not leave zai's CLAUDE_CODE_SUBAGENT_MODEL behind."""
    original = {
        "env": {
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_AUTH_TOKEN": "sk-old",
            "CLAUDE_CODE_SUBAGENT_MODEL": "glm-5.2",
        }
    }
    new_patch = SettingsPatch(
        provider_name="ollama-direct",
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
def test_apply_switch_dry_run_writes_nothing(tmp_path):
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
def test_apply_switch_native_explicitly_blanks_managed_keys(tmp_path):
    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        subagent_model="glm-5.2",  # populate every managed key, not just base_url/token
        force=True,
    )
    assert _read(paths)["env"].get("ANTHROPIC_BASE_URL")

    changed = apply_switch(paths, provider=NATIVE, force=True)

    assert changed is True
    result = _read(paths)
    assert {key: result["env"].get(key) for key in MANAGED_ENV_KEYS} == {
        key: "" for key in MANAGED_ENV_KEYS
    }


@pytest.mark.unit
def test_apply_switch_native_on_a_pristine_file_is_a_no_op(tmp_path):
    """switch native must not touch a settings.json with no managed override.

    Blanking keys that were never set would add codehelper-managed entries
    to a file it never touched — the opposite of "native means clear the
    override".
    """
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": dict(_FOREIGN_ENV)})

    changed = apply_switch(paths, provider=NATIVE, force=True)

    assert changed is False
    assert _read(paths)["env"] == _FOREIGN_ENV


@pytest.mark.unit
def test_apply_switch_native_reset_prints_reset_message(tmp_path, capsys):
    """The write confirmation names the reset explicitly, not a generic write.

    ``SettingsPatch.is_reset`` exists to answer exactly this — "was this
    write a reset" — so it should drive the message the user sees, not sit
    unused outside tests.
    """
    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    capsys.readouterr()

    apply_switch(paths, provider=NATIVE, force=True)

    out = capsys.readouterr().out
    assert "reset to native" in out


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
    assert current_switch(paths) == "custom"


@pytest.mark.unit
def test_custom_litellm_target_is_not_native_and_matches_its_wrapper(tmp_path):
    """A runtime URL has no registry URL, but the chip can still read itself."""
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(LITELLM, "https://litellm.example.example")
    spec = WrapperSpec(
        alias="local-litellm",
        agent=get_agent("claude"),
        provider=provider,
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.2",
        tier_models=TierModels.uniform("glm-5.2"),
    )
    apply_switch(
        paths,
        provider=provider,
        tier_models=spec.tier_models,
        token="sk-test",
        force=True,
    )
    assert current_switch(paths) == "custom"
    assert matches_switch_spec(active_switch_env(paths), spec)


@pytest.mark.unit
def test_matches_switch_spec_distinguishes_same_provider_base_urls(tmp_path):
    """Claude chip readback must include the wrapper's resolved endpoint."""
    paths = Paths.from_home(tmp_path)
    first = with_base_url(LITELLM, "https://proxy-one.example")
    second = with_base_url(LITELLM, "https://proxy-two.example")
    spec = WrapperSpec(
        alias="proxy-two",
        agent=get_agent("claude"),
        provider=second,
        shape=ConfigShape.ANTHROPIC_ENV,
        model="glm-5.2",
        tier_models=TierModels.uniform("glm-5.2"),
    )
    apply_switch(
        paths,
        provider=first,
        tier_models=TierModels.uniform("glm-5.2"),
        token="sk-test",
        force=True,
    )
    assert not matches_switch_spec(active_switch_env(paths), spec)


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


# --------------------------------------------------------------------------- #
# The explicit context_window axis (issue #83) — switch mechanism + chip honesty
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_resolve_switch_patch_explicit_context_window_wins_over_the_catalog():
    """A recorded answer declares even though the catalog is silent — this is
    what `switch --from-wrapper` on a ctx-carrying wrapper rides."""
    patch = resolve_switch_patch(
        ZAI,
        tier_models=TierModels.uniform("mystery-3b"),
        token="sk-test",
        context_window=750_000,
    )
    assert patch.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "750000"

    # ...and overrides the catalog when they disagree.
    overridden = resolve_switch_patch(
        ZAI,
        tier_models=TierModels.uniform("glm-5.3"),
        token="sk-test",
        context_window=500_000,
    )
    assert overridden.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "500000"


@pytest.mark.unit
def test_resolve_switch_patch_ctx_zero_declares_nothing():
    """``0`` is an explicit "no declaration" — it suppresses the catalog's
    1M instead of declaring it."""
    patch = resolve_switch_patch(
        ZAI,
        tier_models=TierModels.uniform("glm-5.3"),
        token="sk-test",
        context_window=0,
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in patch.env


@pytest.mark.unit
def test_matches_switch_spec_honors_a_recorded_ctx():
    """CHIP HONESTY (the #80/#81 class): a live env carrying an explicit
    window matches the ctx-recording spec and ONLY it — deriving from the
    catalog here would leave the wrapper's own chip unset."""
    from codehelper.services.spec import build_spec

    spec = build_spec(
        agent="claude",
        provider=ZAI,
        model="mystery-3b",
        context_window=750_000,
    )
    patch = resolve_switch_patch(
        ZAI,
        tier_models=TierModels.uniform("mystery-3b"),
        token="sk-live",
        context_window=750_000,
    )
    assert matches_switch_spec(patch.env, spec) is True

    # A catalog-derived env (no window) must NOT match it.
    assert matches_switch_spec({"ANTHROPIC_AUTH_TOKEN": "sk-live"}, spec) is False


@pytest.mark.unit
def test_apply_switch_threads_context_window(tmp_path):
    from codehelper.services.paths import Paths
    from codehelper.services.spec import TierModels as TM

    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=ZAI,
        tier_models=TM.uniform("mystery-3b"),
        token="sk-test",
        context_window=750_000,
        force=True,
    )
    _, settings = read_settings(paths)
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "750000"
