"""Tests for ``services/wrappers.py`` — the generated Claude Code wrapper scripts.

A wrapper is not a shell alias string; it is a generated bash script (mode
0o700 for secret-backed wrappers, 0o755 for literal-token wrappers) that
exports ``ANTHROPIC_*`` envs and runs ``claude``. This mirrors the archived
``zai-codex-helper`` project's hand-written ``~/.local/bin/glm``, generalized
across providers via :class:`WrapperSpec`.
"""

from __future__ import annotations

import re
import stat

import pytest

from codehelper.errors import CodeHelperError
from codehelper.services.model import (
    ConfigShape,
    ModelListAPI,
    Provider,
    get_provider,
    with_auth,
    with_base_url,
)
from codehelper.services.paths import Paths
from codehelper.services.render import (
    _marker,
    anthropic_base_url,
    openai_toml_body,
    render_script,
)
from codehelper.services.spec import (
    WrapperSpec,
    build_spec,
    get_preset,
    spec_from_preset,
)
from codehelper.services.wrappers import (
    WRAPPERS,
    discover_managed,
    get_spec,
    install_wrapper,
    is_installed,
    is_managed,
    list_wrappers,
    spec_from_installed,
    token_from_installed,
    valid_default_wrapper,
)


def script_text(paths: Paths, name: str) -> str:
    return paths.script_for(name).read_text(encoding="utf-8")


def _rendered(preset: str, token: str = "", *, model: str | None = None) -> str:
    """Render a preset's body, optionally with a ``--model`` override.

    ``render_script`` no longer takes ``model_override`` — the model is
    resolved when the spec is built. This helper keeps the tests reading the
    way they did while going through the new path.
    """
    return render_script(
        spec_from_preset(get_preset(preset), model_override=model), token
    )


_LITERAL_TOKEN = "ollama"
_SECRET_TOKEN = "00000000000000000000000000000000.aaaaaaaaaaaaaaaa"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_registry_has_deepseek_ollama_glm_and_glm_ollama():
    assert {w.name for w in WRAPPERS} == {"deepseek-ollama", "glm", "glm-ollama"}


@pytest.mark.unit
def test_glm_ollama_uses_the_launcher_shape():
    spec = get_spec("glm-ollama")
    assert spec.shape is ConfigShape.OLLAMA_LAUNCH
    assert spec.agent.name == "claude"
    assert spec.model == "glm-5.2:cloud"
    assert spec.auth == "literal"  # no token — ollama launch authenticates itself


@pytest.mark.integration
def test_token_from_installed_recovers_a_managed_secret_wrapper(tmp_path):
    paths = Paths.from_home(tmp_path)
    token = "sk-existing"
    install_wrapper(paths, "glm", token=token)
    before = paths.script_for("glm").read_bytes()

    assert token_from_installed(paths, "glm", "zai") == token
    assert paths.script_for("glm").read_bytes() == before


@pytest.mark.integration
def test_token_from_installed_rejects_a_provider_mismatch(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-existing")

    assert token_from_installed(paths, "glm", "litellm") is None


@pytest.mark.integration
def test_token_from_installed_recovers_openai_toml_secret_wrapper(tmp_path):
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(
        agent="codex",
        provider=provider,
        model="gpt-4o",
        alias="lite",
    )
    install_wrapper(paths, spec, token="sk-existing")

    assert token_from_installed(paths, "lite", "litellm") == "sk-existing"


@pytest.mark.unit
def test_deepseek_ollama_is_literal_auth():
    spec = get_spec("deepseek-ollama")
    assert spec.auth == "literal"
    assert spec.auth_value == "ollama"
    assert spec.provider.base_url == "http://127.0.0.1:11434"
    # Same provider as glm-ollama, different shape — that is the whole point
    # of naming the axes.
    assert spec.provider.name == "ollama-direct"
    assert spec.shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_glm_is_secret_auth():
    spec = get_spec("glm")
    assert spec.auth == "secret"
    assert spec.token_env_var == "ZAI_API_KEY"
    assert spec.provider.base_url == "https://api.z.ai/api/anthropic"


@pytest.mark.unit
def test_glm_preset_targets_glm_5_3_in_every_tier():
    """The `glm` preset pins the current Z.ai flagship, uniform across tiers."""
    spec = get_spec("glm")
    tiers = spec.tier_models
    assert spec.model == "glm-5.3"
    assert (tiers.haiku, tiers.sonnet, tiers.opus) == ("glm-5.3", "glm-5.3", "glm-5.3")


@pytest.mark.unit
def test_get_spec_unknown_name_raises():
    with pytest.raises(CodeHelperError, match="unknown wrapper"):
        get_spec("nope")


# --------------------------------------------------------------------------- #
# render_script — pure, embeds the token + models, single-quotes everything.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_render_script_embeds_token_single_quoted():
    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in body
    assert "#!/bin/bash" in body


@pytest.mark.unit
def test_render_script_deepseek_ollama_has_endpoint_and_models():
    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    assert "ANTHROPIC_BASE_URL='http://127.0.0.1:11434'" in body
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL=" in body
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=" in body
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL=" in body
    assert "CLAUDE_CODE_SUBAGENT_MODEL=" in body
    # the env shape launches via --settings — see the dedicated tests below —
    # so the exec line must carry the flag and still forward "$@" verbatim
    exec_line = next(ln for ln in body.splitlines() if "claude --settings " in ln)
    assert exec_line.endswith('\' "$@"')


@pytest.mark.unit
def test_render_script_glm_has_no_subagent_model():
    """glm's spec has no subagent model — the line must be absent."""
    body = render_script(get_spec("glm"), _SECRET_TOKEN)
    assert "CLAUDE_CODE_SUBAGENT_MODEL=" not in body


@pytest.mark.unit
def test_render_script_is_executable_shebang():
    assert render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN).startswith(
        "#!/bin/bash"
    )


@pytest.mark.unit
def test_render_script_empties_anthropic_api_key():
    """A real ANTHROPIC_API_KEY inherited from the shell must not win over AUTH_TOKEN."""
    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    assert "export ANTHROPIC_API_KEY=" in body


@pytest.mark.unit
def test_render_script_model_override_replaces_all_tiers():
    body = _rendered("deepseek-ollama", _LITERAL_TOKEN, model="custom:tag")
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL='custom:tag'" in body
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL='custom:tag'" in body
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL='custom:tag'" in body
    assert "CLAUDE_CODE_SUBAGENT_MODEL='custom:tag'" in body


# --------------------------------------------------------------------------- #
# render_script — command shape (launch_command set): exec the provider's own
# launcher instead of exporting ANTHROPIC_* envs here.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_render_script_command_shape_launches_ollama():
    body = render_script(get_spec("glm-ollama"), "")
    assert body.startswith("#!/bin/bash")
    # `--` is required: without it, `ollama launch` parses the forwarded flags
    # itself and rejects any Claude-bound flag (e.g. `-p`) as an unknown
    # Ollama flag. For a claude-like agent the --settings payload rides in
    # the FORWARDED region — after that separator.
    assert "exec ollama launch claude --model 'glm-5.2:cloud' -- --settings " in body
    assert body.rstrip().endswith('\' "$@"')
    # the command shape does NOT export ANTHROPIC_* — ollama launch sets them;
    # the keys inside the --settings JSON are payload, not exports
    assert "export ANTHROPIC_BASE_URL" not in body
    assert "export ANTHROPIC_AUTH_TOKEN" not in body
    assert "export ANTHROPIC_API_KEY=" not in body
    # bare `claude "$@"` (env-var shape's exec line) must not also be present
    assert 'claude "$@"\n' not in body


@pytest.mark.unit
def test_render_script_command_shape_model_override():
    body = _rendered("glm-ollama", model="custom:tag")
    assert "--model 'custom:tag' -- --settings " in body
    assert "glm-5.2:cloud" not in body


@pytest.mark.unit
def test_render_script_command_shape_model_injection_is_neutralized():
    """--model is user input even in the command shape — must be single-quoted."""
    hostile = "x'; touch /tmp/pwned; echo '"
    body = _rendered("glm-ollama", model=hostile)
    assert f"--model '{hostile}'" not in body  # naive form would break out
    assert "'\"'\"'" in body  # escaped-quote sequence proves quoting engaged


# --------------------------------------------------------------------------- #
# render_script — the --settings payload. Claude Code (>= 2.0.1) applies every
# settings.json `env` entry INTO the process environment at startup, replacing
# the value inherited from the shell — an empty string included. Wrapper
# exports therefore lose to anything `switch` leaves in ~/.claude/settings.json
# (e.g. the `""` blanks of `switch native`), unless the wrapper ALSO carries
# its env at the command-line settings level, which sits ABOVE the user file.
# --------------------------------------------------------------------------- #


def _settings_payload(body: str) -> dict:
    """Parse the ``--settings '<json>'`` payload out of a rendered body."""
    import json

    found = re.search(r"--settings '(.*?)' \"\$@\"", body)
    assert found is not None, "no --settings payload in body"
    return json.loads(found.group(1).replace("'\"'\"'", "'"))


@pytest.mark.unit
def test_settings_payload_mirrors_the_exports():
    """Exports and --settings must carry the SAME env — one dict feeds both."""
    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    payload = _settings_payload(body)["env"]
    exports = dict(re.findall(r"^export (\w+)='(.*)'$", body, re.MULTILINE))
    # ANTHROPIC_API_KEY is rendered unquoted-empty; everything else is quoted
    exports["ANTHROPIC_API_KEY"] = ""
    assert payload == exports


@pytest.mark.unit
def test_settings_payload_overrides_what_switch_native_leaves_behind():
    """The payload's keys must be exactly the managed set a switch writes."""
    from codehelper.services.claude_settings import MANAGED_ENV_KEYS

    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    assert set(_settings_payload(body)["env"]) <= set(MANAGED_ENV_KEYS)
    # every managed key a blank `switch native` leaves behind must be
    # overridden, or the wrapper silently launches native
    for key in MANAGED_ENV_KEYS:
        if key == "CLAUDE_CODE_SUBAGENT_MODEL":
            continue  # deepseek-ollama sets it; glm deliberately does not
        assert key in body


@pytest.mark.unit
def test_settings_payload_survives_a_quote_in_the_token():
    """A token with a single quote must not break out of the quoting."""
    hostile = "tok'en"
    body = render_script(get_spec("glm"), hostile)
    payload = _settings_payload(body)
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == hostile


@pytest.mark.unit
def test_command_shape_forwards_settings_only_to_claude_like_agents():
    """--settings is a Claude Code flag: never forwarded to other agents."""
    from codehelper.services.model import get_agent

    # hermes is OLLAMA_LAUNCH-only, so the shape cannot resolve away from it
    spec = build_spec(
        agent=get_agent("hermes"), provider="ollama-direct", model="qwen3.5:9b"
    )
    body = render_script(spec, "")
    assert "--settings" not in body
    assert "exec ollama launch hermes --model 'qwen3.5:9b' -- \"$@\"" in body


@pytest.mark.unit
def test_command_shape_settings_payload_mirrors_ollama_injection():
    """The payload must say what `ollama launch` itself would inject."""
    body = render_script(get_spec("glm-ollama"), "")
    env = _settings_payload(body)["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ollama"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-5.2:cloud"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.2:cloud"


# --------------------------------------------------------------------------- #
# render_script — the context-window declaration. Claude Code cannot resolve a
# non-claude- model ID, so it assumes its 200k fallback and auto-compacts
# there; CLAUDE_CODE_MAX_CONTEXT_TOKENS (declared in both the exports and the
# --settings payload) makes compaction continue at the real window. Emission
# is catalog-driven and conditional: no declaration beats a guessed window.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_uniform_context_window_requires_every_model_known_and_equal():
    from codehelper.services.render import uniform_context_window

    assert uniform_context_window(["glm-5.3"]) == 1_000_000
    assert uniform_context_window(["glm-5.2", "glm-5.2:cloud"]) == 1_000_000
    # unknown model, mixed windows, empty input: no declaration
    assert uniform_context_window(["mystery-3b"]) is None
    assert uniform_context_window(["glm-5.3", "mystery-3b"]) is None
    assert uniform_context_window([]) is None


@pytest.mark.unit
def test_env_shape_declares_context_window_for_known_model():
    body = render_script(get_spec("deepseek-ollama"), _LITERAL_TOKEN)
    assert "export CLAUDE_CODE_MAX_CONTEXT_TOKENS='1000000'" in body
    assert _settings_payload(body)["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == (
        "1000000"
    )


@pytest.mark.unit
def test_env_shape_omits_context_window_for_unknown_model():
    body = _rendered("deepseek-ollama", _LITERAL_TOKEN, model="custom:tag")
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in body


@pytest.mark.unit
def test_command_shape_declares_context_window_for_known_model():
    body = render_script(get_spec("glm-ollama"), "")
    assert _settings_payload(body)["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == (
        "1000000"
    )


# --------------------------------------------------------------------------- #
# render_script — shell-injection defense (single-quoting), on EVERY channel.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_render_script_token_injection_is_neutralized():
    """A token containing a single quote cannot break out of the quoted value.

    The raw hostile string, if copied verbatim inside a single-quoted shell
    value, would close the quote early at the first ``'`` and let
    ``; touch /tmp/pwned; echo`` run as separate shell commands. Proving
    safety means proving the value is NEVER embedded as a bare ``'...'`` —
    i.e. the escaped POSIX form (each ``'`` replaced by ``'"'"'``) is present,
    and the naive/vulnerable form (the raw token between two plain quotes) is
    absent.
    """
    hostile = "x'; touch /tmp/pwned; echo '"
    body = render_script(get_spec("deepseek-ollama"), hostile)
    naive_vulnerable_form = f"ANTHROPIC_AUTH_TOKEN='{hostile}'"
    assert naive_vulnerable_form not in body
    assert "'\"'\"'" in body  # the escaped-quote sequence proves quoting engaged


@pytest.mark.unit
def test_render_script_model_override_injection_is_neutralized():
    """--model is now user-controlled input; it must be quoted just like a token."""
    hostile = "x'; touch /tmp/pwned; echo '"
    body = _rendered("deepseek-ollama", _LITERAL_TOKEN, model=hostile)
    assert "'\"'\"'" in body


# --------------------------------------------------------------------------- #
# install_wrapper — writes an executable, idempotent, always overwrites by name.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_install_wrapper_writes_executable_script(tmp_path):
    paths = Paths.from_home(tmp_path)

    wrote = install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    assert wrote is True
    script = paths.script_for("deepseek-ollama")
    assert script.exists()
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode & stat.S_IXUSR, f"script not executable: {oct(mode)}"
    body = script.read_text(encoding="utf-8")
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in body


@pytest.mark.integration
def test_install_wrapper_idempotent(tmp_path):
    paths = Paths.from_home(tmp_path)

    first = install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)
    second = install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    assert first is True
    assert second is False  # already present, identical → no rewrite


@pytest.mark.integration
def test_install_wrapper_dry_run_writes_nothing(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)

    wrote = install_wrapper(
        paths, "deepseek-ollama", token=_LITERAL_TOKEN, dry_run=True
    )

    assert wrote is True
    assert not paths.script_for("deepseek-ollama").exists()
    out = capsys.readouterr().out
    assert str(paths.script_for("deepseek-ollama")) in out


@pytest.mark.integration
def test_install_wrapper_unknown_name_raises_before_any_write(tmp_path):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="unknown wrapper"):
        install_wrapper(paths, "nope", token="x")
    assert not paths.bin_dir.exists()


@pytest.mark.integration
def test_install_wrapper_refuses_to_clobber_a_foreign_file(tmp_path):
    """A file we did not write is NOT silently replaced.

    This inverts the project's earlier "overwrites any existing file"
    behaviour. That was safe while names came from a closed registry; with
    user-chosen aliases the same code path could destroy an unrelated
    executable on PATH (``~/.local/bin/claude`` is a real symlink on a normal
    install). Presets are not exempt — the guard keys off the file, not the
    name's origin.
    """
    paths = Paths.from_home(tmp_path)
    existing = "#!/bin/bash\necho my own deepseek\n"
    script = paths.script_for("deepseek-ollama")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(existing, encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    assert script.read_text(encoding="utf-8") == existing  # untouched


@pytest.mark.integration
def test_install_wrapper_force_overwrites_a_foreign_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek-ollama")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho mine\n", encoding="utf-8")

    assert (
        install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN, force=True)
        is True
    )
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in script.read_text()


@pytest.mark.integration
def test_install_wrapper_replaces_its_own_earlier_output(tmp_path):
    """The guard must not get in the way of the normal update path."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    def _explode(_path):  # pragma: no cover - must not be reached
        raise AssertionError("must not ask before replacing our own wrapper")

    wrote = install_wrapper(
        paths, "deepseek-ollama", token="rotated-token", confirm=_explode
    )
    assert wrote is True
    assert "ANTHROPIC_AUTH_TOKEN='rotated-token'" in script_text(
        paths, "deepseek-ollama"
    )


@pytest.mark.integration
def test_install_wrapper_non_interactive_fails_fast_instead_of_blocking(tmp_path):
    """confirm=None means "no way to ask" — must error, never read stdin.

    This is the guarantee that a scripted/CI run cannot hang waiting on input.
    """
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek-ollama")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho mine\n", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="--force"):
        install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN, confirm=None)


@pytest.mark.integration
def test_install_wrapper_dry_run_never_asks_about_a_foreign_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek-ollama")
    script.parent.mkdir(parents=True, exist_ok=True)
    original = "#!/bin/bash\necho mine\n"
    script.write_text(original, encoding="utf-8")

    def _explode(_path):  # pragma: no cover - must not be reached
        raise AssertionError("dry-run must never prompt")

    assert (
        install_wrapper(
            paths,
            "deepseek-ollama",
            token=_LITERAL_TOKEN,
            dry_run=True,
            confirm=_explode,
        )
        is True
    )
    assert script.read_text(encoding="utf-8") == original


@pytest.mark.integration
def test_is_managed_distinguishes_ours_from_foreign(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)
    assert is_managed(paths, "deepseek-ollama") is True

    foreign = paths.script_for("glm")
    foreign.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    assert is_installed(paths, "glm") is True  # exists...
    assert is_managed(paths, "glm") is False  # ...but not ours


@pytest.mark.integration
def test_is_managed_false_for_binary_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    paths.script_for("blob").write_bytes(b"\x7fELF\x00\x01\x02binary")
    assert is_managed(paths, "blob") is False


@pytest.mark.integration
def test_discover_managed_finds_ad_hoc_wrappers(tmp_path):
    """A wrapper built from the axes has no preset — the marker is its only record."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(
        paths, build_spec(agent="codex", provider="ollama-direct", model="qwen3.5:9b")
    )
    paths.script_for("stranger").write_text("#!/bin/sh\n", encoding="utf-8")

    assert discover_managed(paths) == ["qwen3.5-codex"]


@pytest.mark.integration
def test_spec_from_installed_reconstructs_a_user_agent_wrapper(tmp_path):
    """A wrapper created for a user-defined agent records that agent's name in
    its marker. Reconstruction must resolve it through the MERGED registry
    (built-ins + user agents), not the built-in-only lookup — otherwise the
    wrapper is invisible to the TUI's list/remove/token actions."""
    from codehelper.services.agents import add_user_agent, get_agent

    paths = Paths.from_home(tmp_path)
    add_user_agent(paths, "myagent", "myagent", "a real CLI")
    agent = get_agent(paths, "myagent")
    spec = build_spec(agent=agent, provider="ollama-direct", model="qwen3.5:9b")
    install_wrapper(paths, spec)

    rebuilt = spec_from_installed(paths, spec.alias)
    assert rebuilt is not None
    assert rebuilt.agent.name == "myagent"
    assert rebuilt.provider.name == "ollama-direct"


@pytest.mark.integration
def test_reserved_named_managed_wrapper_stays_discoverable_and_removable(tmp_path):
    """A wrapper whose name became reserved (e.g. ``opencode`` after the agent
    registry grew) must remain discoverable and removable. Reservation gates
    NEW creation (``validate_alias`` at the install boundary); it must never
    strand a pre-existing managed file on PATH with no way to list or remove
    it."""
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    paths.script_for("opencode").write_text(
        "# codehelper: managed wrapper\n#!/bin/sh\n", encoding="utf-8"
    )
    assert is_managed(paths, "opencode") is True
    assert "opencode" in discover_managed(paths)
    assert remove_wrapper(paths, "opencode") is True
    assert not paths.script_for("opencode").exists()


# --------------------------------------------------------------------------- #
# valid_default_wrapper — staleness cross-check (issue #28)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_valid_default_wrapper_none_when_unset(tmp_path):
    """No saved alias — the common case before the user picks a default."""
    paths = Paths.from_home(tmp_path)
    assert valid_default_wrapper(paths, "claude") is None


@pytest.mark.integration
def test_valid_default_wrapper_none_for_uninstalled_preset(tmp_path):
    """A preset name alone must not create a selectable default ghost."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    set_default_wrapper(paths, "claude", "glm")
    assert valid_default_wrapper(paths, "claude") is None


@pytest.mark.integration
def test_valid_default_wrapper_returns_a_managed_alias(tmp_path):
    """An ad-hoc managed wrapper (no preset) is live via ``discover_managed``."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    install_wrapper(
        paths, build_spec(agent="codex", provider="ollama-direct", model="qwen3.5:9b")
    )
    set_default_wrapper(paths, "codex", "qwen3.5-codex")
    assert valid_default_wrapper(paths, "codex") == "qwen3.5-codex"


@pytest.mark.integration
def test_valid_default_wrapper_none_for_a_stale_alias(tmp_path):
    """A saved alias that is neither a preset nor on disk is stale — readers
    must fall back to None instead of highlighting a ghost wrapper."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    set_default_wrapper(paths, "claude", "ghost")
    assert valid_default_wrapper(paths, "claude") is None


@pytest.mark.integration
def test_valid_default_wrapper_none_for_an_unmanaged_file(tmp_path):
    """A bare file on disk without our marker is NOT a managed wrapper — the
    live check goes through ``discover_managed`` (which requires the marker),
    not bare ``script_for(name).exists()``."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    paths.script_for("ghost").write_text("#!/bin/sh\n", encoding="utf-8")
    set_default_wrapper(paths, "claude", "ghost")
    assert valid_default_wrapper(paths, "claude") is None


@pytest.mark.integration
def test_valid_default_wrapper_rejects_a_wrong_agent_preset(tmp_path):
    """A preset alias belongs to a specific agent — ``valid_default_wrapper``
    for a different agent must not return it, or a per-agent consumer would
    apply a wrapper that launches the wrong agent/config shape."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    set_default_wrapper(paths, "codex", "glm")  # glm is a claude preset
    assert valid_default_wrapper(paths, "codex") is None


@pytest.mark.integration
def test_valid_default_wrapper_installed_wrapper_wins_over_same_named_preset(tmp_path):
    """A managed wrapper on disk takes precedence over a same-named preset —
    the on-disk wrapper's agent is authoritative, not the preset's. Otherwise a
    claude consumer could be handed a wrapper that actually launches codex."""
    from codehelper.services.spec import build_spec
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    # Install a codex wrapper named "glm" (collides with the claude preset).
    install_wrapper(
        paths,
        build_spec(
            agent="codex", provider="ollama-direct", model="qwen3.5:9b", alias="glm"
        ),
    )
    set_default_wrapper(paths, "claude", "glm")
    assert (
        valid_default_wrapper(paths, "claude") is None
    )  # on-disk glm is codex, not claude
    set_default_wrapper(paths, "codex", "glm")
    assert valid_default_wrapper(paths, "codex") == "glm"  # on-disk glm IS codex


@pytest.mark.integration
def test_valid_default_wrapper_none_for_a_malformed_alias(tmp_path):
    """A hand-edited/corrupt state entry with a path separator must degrade to
    None, never raise — the read path is contractually non-raising."""
    from codehelper.services.state import set_default_wrapper

    paths = Paths.from_home(tmp_path)
    set_default_wrapper(paths, "claude", "../x")
    assert valid_default_wrapper(paths, "claude") is None


@pytest.mark.integration
def test_install_wrapper_updates_after_token_rotation(tmp_path):
    """Re-installing with a new token rewrites the script."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token=_SECRET_TOKEN)

    new_token = "11111111111111111111111111111111.bbbbbbbbbbbbbbbb"
    wrote = install_wrapper(paths, "glm", token=new_token)

    assert wrote is True
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert new_token in body


@pytest.mark.integration
def test_install_wrapper_secret_mode_is_owner_only(tmp_path):
    """A secret-backed wrapper carries a token in plain text — must be 0o700."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token=_SECRET_TOKEN)

    mode = stat.S_IMODE(paths.script_for("glm").stat().st_mode)
    assert mode & stat.S_IXUSR, "owner must be able to execute"
    assert not (mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)), "no group bits"
    assert not (mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)), "no other bits"


@pytest.mark.integration
def test_install_wrapper_literal_mode_is_normal_executable(tmp_path):
    """A literal-token wrapper (no real secret) gets a normal 0o755."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    mode = stat.S_IMODE(paths.script_for("deepseek-ollama").stat().st_mode)
    assert mode == 0o755


@pytest.mark.integration
def test_install_wrapper_command_shape_writes_and_idempotent(tmp_path):
    """Command-shape install writes a script; a byte-identical re-install is a no-op.

    Mode (0o755 for literal auth) and the rendered body are pinned by the
    literal-mode and command-shape render unit tests respectively — this test
    only pins the install path itself (write happens, file exists, idempotent).
    """
    paths = Paths.from_home(tmp_path)

    assert install_wrapper(paths, "glm-ollama", token="") is True
    assert paths.script_for("glm-ollama").exists()
    assert install_wrapper(paths, "glm-ollama", token="") is False


# --------------------------------------------------------------------------- #
# is_installed / list_wrappers — plain existence, no ownership tracking.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_is_installed_true_after_install(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    assert is_installed(paths, "deepseek-ollama") is True


@pytest.mark.integration
def test_is_installed_false_when_absent(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert is_installed(paths, "deepseek-ollama") is False


@pytest.mark.integration
def test_is_installed_true_for_any_existing_file(tmp_path):
    """Existence alone is enough — no marker/ownership check."""
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek-ollama")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho someone else's\n", encoding="utf-8")

    assert is_installed(paths, "deepseek-ollama") is True


@pytest.mark.integration
def test_list_wrappers_reports_installed_and_not_installed(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek-ollama", token=_LITERAL_TOKEN)

    lines = []
    list_wrappers(paths, print_fn=lines.append)
    output = "\n".join(lines)

    assert "deepseek-ollama" in output and "installed" in output
    assert "glm" in output and "not installed" in output


# --------------------------------------------------------------------------- #
# CLI: `add` / `list` through main([...]).
# --------------------------------------------------------------------------- #

from codehelper.__main__ import main  # noqa: E402


@pytest.mark.integration
def test_cli_add_deepseek_ollama_creates_script(tmp_path):
    assert main(["add", "deepseek-ollama"]) == 0

    paths = Paths.from_home(tmp_path)
    assert paths.script_for("deepseek-ollama").exists()
    assert paths.script_for("deepseek-ollama").stat().st_mode & stat.S_IXUSR


@pytest.mark.integration
def test_cli_add_glm_reads_token_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", _SECRET_TOKEN)

    assert main(["add", "glm"]) == 0

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert _SECRET_TOKEN in body


@pytest.mark.integration
def test_cli_add_with_model_override(tmp_path):
    assert (
        main(
            [
                "add",
                "deepseek-ollama",
                "--model",
                "custom-model:tag",
                "--context-window",
                "none",
            ]
        )
        == 0
    )

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("deepseek-ollama").read_text(encoding="utf-8")
    assert "custom-model:tag" in body


@pytest.mark.integration
def test_cli_list_shows_registry(tmp_path, capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "deepseek-ollama" in out
    assert "glm" in out


@pytest.mark.integration
def test_cli_unknown_wrapper_name_exits_1(tmp_path, capsys):
    code = main(["add", "nope"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "unknown wrapper" in err


@pytest.mark.integration
def test_cli_dry_run_add_does_not_write(tmp_path):
    assert main(["--dry-run", "add", "deepseek-ollama"]) == 0
    paths = Paths.from_home(tmp_path)
    assert not paths.script_for("deepseek-ollama").exists()


# --------------------------------------------------------------------------- #
# OPENAI_TOML shape — codex via a Codex profile (~/.codex/<alias>.config.toml)
# plus a model catalog, launched as `codex --profile <alias>`.
# --------------------------------------------------------------------------- #


def _toml_spec(model: str = "glm-5.2:cloud", alias: str = "glm-5-codex") -> WrapperSpec:
    return build_spec(agent="codex", provider="ollama-direct", model=model, alias=alias)


@pytest.mark.unit
def test_render_openai_toml_wrapper_runs_codex_with_profile():
    """The wrapper is a one-line dispatch to `codex --profile <alias>`."""
    spec = _toml_spec()
    body = render_script(spec, "")
    assert body.startswith("#!/bin/bash")
    assert "exec codex --profile 'glm-5-codex' \"$@\"" in body
    # No ANTHROPIC_* env and no launcher — the profile carries the config.
    assert "ANTHROPIC_" not in body
    assert "ollama launch" not in body


@pytest.mark.unit
def test_render_openai_toml_wrapper_alias_is_quoted():
    """The alias is user-chosen, so it is single-quoted — defence in depth on
    top of ``validate_alias``'s allow-list (which already excludes quotes and
    shell metacharacters). The model picker / ``--alias`` can name a wrapper
    with ``.``/``-`` (e.g. ``glm-5-codex``), which the quoting wraps harmlessly.
    """
    spec = _toml_spec(alias="glm-5-codex")
    body = render_script(spec, "")
    assert "exec codex --profile 'glm-5-codex' \"$@\"" in body


@pytest.mark.unit
def test_openai_toml_body_carries_marker_model_base_url_wire_api():
    """The profile body matches the contract in issue #7, with /v1/ derived."""
    spec = _toml_spec(model="glm-5.2:cloud")
    body = openai_toml_body(spec)
    # Marker on line 1 — what the ownership guard keys off.
    assert body.startswith("# codehelper: managed wrapper")
    assert 'model = "glm-5.2:cloud"' in body  # `:` and `.` => must be quoted
    # Data-driven: table key + model_provider derive from spec.provider.name,
    # not a hardcoded "ollama-launch" — that is what makes a second
    # OpenAI-compatible provider a PROVIDERS entry rather than a renderer edit.
    assert 'model_provider = "ollama-direct"' in body
    # No catalog: an empty base_instructions would silently replace Codex's
    # real system prompt. glm-5.2:cloud resolves in MODEL_CONTEXT_WINDOWS, so
    # the window rides along as a plain config.toml key instead.
    assert "model_catalog_json" not in body
    assert "model_context_window = 1000000" in body
    assert "[model_providers.ollama-direct]" in body
    # The display name is the provider's description (falling back to its name).
    assert 'name = "local Ollama daemon"' in body
    # base_url derives /v1/ from the provider's Anthropic-root base_url.
    assert 'base_url = "http://127.0.0.1:11434/v1/"' in body
    assert 'wire_api = "responses"' in body


@pytest.mark.unit
def test_openai_toml_body_omits_context_window_for_unknown_model():
    """An unrecognised model gets NO ``model_context_window`` — never a guess.

    Codex falls through to its own ``model_info_from_slug`` fallback, which
    carries the real bundled system prompt (unlike the removed catalog).
    """
    spec = _toml_spec(model="some-unknown-model")
    body = openai_toml_body(spec)
    assert "model_context_window" not in body
    assert "model_catalog_json" not in body


@pytest.mark.unit
def test_openai_toml_body_quoted_model_with_colon_and_dot():
    """A model like `glm-5.2:cloud` MUST be in quotes or Codex rejects the TOML."""
    spec = _toml_spec(model="glm-5.2:cloud")
    body = openai_toml_body(spec)
    assert 'model = "glm-5.2:cloud"' in body
    # A double quote in the model would be escaped, not break the string.
    weird = 'we"ird'
    spec2 = _toml_spec(model=weird)
    body2 = openai_toml_body(spec2)
    assert 'model = "we\\"ird"' in body2


@pytest.mark.unit
def test_openai_toml_wrapper_body_is_unchanged_for_a_literal_provider():
    """GOLDEN: byte-exact output for a non-secret (``ollama-direct``, literal)
    provider.

    A literal string, not a re-render of the same function — proving the
    function equals itself proves nothing. This is what pins the OPENAI_TOML
    wrapper's structure across the env_key/export change added for a secret
    provider (e.g. a runtime-``base_url`` LiteLLM proxy): for
    ``ollama-direct`` the output MUST stay exactly what it already is, or
    every previously-installed ``codex × ollama-direct`` wrapper stops
    matching ``_decide``'s SKIP path and gets silently rewritten on the next
    ``add``.
    """
    spec = _toml_spec(model="glm-5.2:cloud", alias="glm-5-codex")
    body = render_script(spec, "")
    assert body == (
        "#!/bin/bash\n"
        "# codehelper: managed wrapper (agent=codex, provider=ollama-direct, "
        "shape=openai-toml)\n"
        "exec codex --profile 'glm-5-codex' \"$@\"\n"
    )


@pytest.mark.unit
def test_openai_toml_profile_has_no_env_key_for_a_literal_provider():
    """GOLDEN: no env_key line for a non-secret provider (ollama-direct)."""
    spec = _toml_spec(model="glm-5.2:cloud", alias="glm-5-codex")
    body = openai_toml_body(spec)
    assert "env_key" not in body


@pytest.mark.unit
def test_marker_records_auth_only_for_a_secret_wrapper():
    """issue #81: the marker carries the wrapper's effective auth mode, but
    ONLY when it is secret — a literal marker stays byte-identical to the
    pre-#81 form, so every already-installed literal wrapper keeps matching
    `_decide`'s SKIP path instead of being rewritten on the next add."""
    secret = build_spec(
        agent="claude",
        provider=with_auth(get_provider("ollama-direct"), want_secret=True),
        model="m",
        alias="s",
    )
    literal = build_spec(agent="claude", provider="ollama-direct", model="m", alias="l")
    assert "auth=secret" in render_script(secret, "tok")
    assert "auth=" not in render_script(literal, "")


def _secret_toml_provider(name: str = "secret-openai") -> Provider:
    return Provider(
        name=name,
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.secret.invalid/v1",
        auth="secret",
        token_env_var="SECRET_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="chat",
    )


@pytest.mark.unit
def test_openai_toml_wrapper_exports_the_token_for_a_secret_provider():
    spec = build_spec(
        agent="codex", provider=_secret_toml_provider(), model="m", alias="x"
    )
    body = render_script(spec, "tok")
    export_idx = body.index("export SECRET_API_KEY='tok'")
    exec_idx = body.index("exec codex")
    assert export_idx < exec_idx  # the export must precede exec


@pytest.mark.unit
def test_openai_toml_profile_carries_env_key_for_a_secret_provider():
    spec = build_spec(
        agent="codex", provider=_secret_toml_provider(), model="m", alias="x"
    )
    body = openai_toml_body(spec)
    assert 'env_key = "SECRET_API_KEY"' in body


@pytest.mark.unit
def test_secret_openai_toml_token_with_a_quote_is_shell_safe():
    """Injection regression on the NEW channel: token in an OPENAI_TOML wrapper."""
    spec = build_spec(
        agent="codex", provider=_secret_toml_provider(), model="m", alias="x"
    )
    evil_token = "a'; rm -rf /tmp/pwned; echo '"
    body = render_script(spec, evil_token)
    assert "export SECRET_API_KEY='a'\"'\"'; rm -rf /tmp/pwned; echo '\"'\"''" in body


@pytest.mark.unit
def test_an_existing_ollama_wrapper_reinstalls_as_a_no_op(tmp_path):
    """The install-time proof, not just the render-time one.

    Installs `codex × ollama-direct`, reinstalls the identical spec, and
    asserts `install_wrapper` returns False (`_decide`'s SKIP path) — this is
    the thing the golden byte tests above exist to protect: a structural
    change to the renderer that DOES change ollama-direct's output would turn
    this into a silent rewrite instead of a no-op.
    """
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec(model="glm-5.2:cloud", alias="glm-5-codex")
    assert install_wrapper(paths, spec, token="") is True
    assert install_wrapper(paths, spec, token="") is False


@pytest.mark.unit
def test_openai_toml_body_never_carries_base_instructions_or_catalog_path():
    """REGRESSION: no ``base_instructions``/catalog line, in any form.

    Codex's ``ModelInfo::get_model_instructions`` returns ``base_instructions``
    verbatim as the session's real system prompt when no ``model_messages``
    template is set — an empty synthesized value is not "no override", it IS
    the (empty) instructions, and a matched catalog entry never falls back to
    Codex's own bundled prompt (``used_fallback_model_metadata`` is only set
    for an UNMATCHED slug). The old catalog wrote exactly this. Pin that the
    profile carries no path to any such catalog and the string never appears.
    """
    spec = _toml_spec(model="glm-5.2:cloud")
    body = openai_toml_body(spec)
    assert "model_catalog_json" not in body
    assert "base_instructions" not in body
    assert ".model.json" not in body


@pytest.mark.integration
def test_install_openai_toml_writes_two_files_no_catalog(tmp_path):
    """REGRESSION: install writes only the wrapper + TOML profile — no
    ``<alias>.model.json`` catalog (see ``openai_toml_body`` for why: an
    empty synthesized ``base_instructions`` would silently replace Codex's
    real system prompt for every wrapper reading it).
    """
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    assert install_wrapper(paths, spec) is True

    assert paths.script_for("glm-5-codex").exists()
    assert paths.codex_config_for("glm-5-codex").exists()
    assert not paths.codex_catalog_for("glm-5-codex").exists()
    # The wrapper is executable; the config is owner-only (no token,
    # but it is our generated config — 0o600, not 0o755).
    assert stat.S_IMODE(paths.script_for("glm-5-codex").stat().st_mode) & stat.S_IXUSR
    config_mode = stat.S_IMODE(paths.codex_config_for("glm-5-codex").stat().st_mode)
    assert config_mode == 0o600


@pytest.mark.integration
def test_install_openai_toml_is_idempotent_across_two_files(tmp_path):
    """A byte-identical re-install of both files is a no-op."""
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    first = install_wrapper(paths, spec)
    second = install_wrapper(paths, spec)

    assert first is True
    assert second is False


@pytest.mark.integration
def test_install_openai_toml_warns_about_a_stale_catalog_from_a_prior_version(
    tmp_path,
    capsys,
):
    """MIGRATION: a marker-owned ``<alias>.model.json`` left by a previous
    version of this tool (before the catalog write was removed) is NOT deleted
    on re-install under the SAME alias — it is only warned about and left in
    place. The ``managed_by=codehelper`` marker proves the file was at some
    point written by this tool, not that a user has not hand-edited it since;
    an install must never destroy data it cannot prove is byte-identical to
    what it wrote (and the old renderer that produced the canonical bytes no
    longer exists to compare against). The catalog is inert — nothing writes
    ``model_catalog_json`` any more — so leaving it costs nothing.
    """
    import json

    paths = Paths.from_home(tmp_path)
    spec = _toml_spec(model="glm-5.2:cloud", alias="glm-5-codex")
    install_wrapper(paths, spec)
    stale_catalog = paths.codex_catalog_for("glm-5-codex")
    stale_catalog.parent.mkdir(parents=True, exist_ok=True)
    stale_catalog.write_text(
        json.dumps({"version": 1, "managed_by": "codehelper", "models": []}),
        encoding="utf-8",
    )
    assert stale_catalog.exists()

    wrote = install_wrapper(paths, spec)

    assert wrote is False  # a byte-identical re-install is a no-op
    assert stale_catalog.exists()
    assert "stale codehelper-managed model catalog" in capsys.readouterr().err


@pytest.mark.integration
def test_install_openai_toml_leaves_a_foreign_catalog_alone(tmp_path, capsys):
    """A hand-curated catalog with no ``managed_by`` marker is left alone, and
    NOT even warned about — no install-time sweep touches any catalog,
    marker-owned or not (see
    ``test_install_openai_toml_warns_about_a_stale_catalog_from_a_prior_version``);
    only ``remove_wrapper`` ever deletes one, as the explicit removal action.
    """
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec(model="glm-5.2:cloud", alias="glm-5-codex")
    foreign_catalog = paths.codex_catalog_for("glm-5-codex")
    foreign_catalog.parent.mkdir(parents=True, exist_ok=True)
    foreign_catalog.write_text('{"hand": "curated"}', encoding="utf-8")

    install_wrapper(paths, spec)

    assert foreign_catalog.exists()
    assert foreign_catalog.read_text(encoding="utf-8") == '{"hand": "curated"}'
    assert "stale codehelper-managed model catalog" not in capsys.readouterr().err


@pytest.mark.integration
def test_install_openai_toml_refuses_foreign_config_file(tmp_path):
    """A foreign ~/.codex/<alias>.config.toml is not silently clobbered."""
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()
    config = paths.codex_config_for("glm-5-codex")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('# someone else\nmodel = "gpt-4o"\n', encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, spec)

    # Untouched — the guard fired before any write, including the wrapper.
    assert config.read_text(encoding="utf-8") == '# someone else\nmodel = "gpt-4o"\n'
    assert not paths.script_for("glm-5-codex").exists()


@pytest.mark.integration
def test_install_openai_toml_force_overwrites_foreign_config(tmp_path):
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()
    config = paths.codex_config_for("glm-5-codex")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("# foreign\n", encoding="utf-8")

    assert install_wrapper(paths, spec, force=True) is True
    assert config.read_text(encoding="utf-8").startswith("# codehelper:")


@pytest.mark.integration
def test_install_openai_toml_dry_run_writes_nothing_and_lists_two(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    assert install_wrapper(paths, spec, dry_run=True) is True

    assert not paths.script_for("glm-5-codex").exists()
    assert not paths.codex_config_for("glm-5-codex").exists()
    assert not paths.codex_catalog_for("glm-5-codex").exists()
    out = capsys.readouterr().out
    assert "would write" in out
    # Both intended writes are announced — no catalog.
    assert str(paths.script_for("glm-5-codex")) in out
    assert str(paths.codex_config_for("glm-5-codex")) in out
    assert str(paths.codex_catalog_for("glm-5-codex")) not in out


@pytest.mark.integration
def test_install_openai_toml_replacing_our_own_never_prompts(tmp_path):
    """A model bump rewrites all three without asking — add-as-update path."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, _toml_spec(model="glm-5:cloud"))

    def _explode(_path):  # pragma: no cover - must not be reached
        raise AssertionError("must not ask before replacing our own wrapper")

    wrote = install_wrapper(paths, _toml_spec(model="glm-5.2:cloud"), confirm=_explode)
    assert wrote is True
    config = paths.codex_config_for("glm-5-codex").read_text(encoding="utf-8")
    assert 'model = "glm-5.2:cloud"' in config


@pytest.mark.unit
def test_paths_codex_accessors_reject_non_single_component():
    """The same structural guard as script_for — no writing outside ~/.codex."""
    paths = Paths.from_home("/tmp/whatever")
    for bad in ("../../etc/passwd", "a/b", "", ".", ".."):
        with pytest.raises(CodeHelperError, match="single path component"):
            paths.codex_config_for(bad)
        with pytest.raises(CodeHelperError, match="single path component"):
            paths.codex_catalog_for(bad)


# --------------------------------------------------------------------------- #
# OPENAI_TOML extension point, ownership, recovery, and shape-switch — the
# round-2 review findings (F1/F2/F4/F1-own/F7/F8/F9).
# --------------------------------------------------------------------------- #


# A second OpenAI-compatible provider, deliberately NOT in the registry (like
# test_model._OPENAI_ONLY but non-secret so build_spec accepts it). This is the
# proof the renderer is data-driven: wiring a new provider reuses the shape with
# no renderer edit, and its own name/base_url/wire_api land in the profile.
_OPENAI_LITERAL = Provider(
    name="acme-openai",
    shapes=frozenset({ConfigShape.OPENAI_TOML}),
    base_url="https://api.acme.invalid/v1",
    auth="literal",
    auth_value="acme",
    model_list_api=ModelListAPI.OPENAI_V1,
    wire_api="chat",
    description="Acme OpenAI-compatible",
)


def _acme_spec(model: str = "acme-7b", alias: str = "acme-codex") -> WrapperSpec:
    return build_spec(agent="codex", provider=_OPENAI_LITERAL, model=model, alias=alias)


@pytest.mark.unit
def test_openai_toml_body_for_non_ollama_provider_pins_extension_point():
    """A second OpenAI-compatible provider reuses the renderer with NO edit.

    The profile carries the provider's OWN name/table/base_url/wire_api (F4:
    data-driven, not the hardcoded ``ollama``/``ollama-launch`` the original
    shipped), and a ``/v1``-suffixed ``base_url`` is NOT doubled (F1: the
    renderer appends ``/`` only, so ``.../v1`` -> ``.../v1/`` not ``.../v1/v1/``).
    """
    spec = _acme_spec(model="acme-7b")
    body = openai_toml_body(spec)
    assert 'model = "acme-7b"' in body
    assert 'model_provider = "acme-openai"' in body
    assert "[model_providers.acme-openai]" in body
    assert 'name = "Acme OpenAI-compatible"' in body
    # F1: /v1-suffixed base_url gets a trailing slash only — no /v1/v1/.
    assert 'base_url = "https://api.acme.invalid/v1/"' in body
    assert "/v1/v1/" not in body
    assert 'wire_api = "chat"' in body


@pytest.mark.unit
def test_openai_toml_body_gemini_keeps_full_openai_root():
    """Gemini's base_url IS the complete OpenAI root — /v1/ must NOT be appended.

    openai_base_url would otherwise rewrite
    https://generativelanguage.googleapis.com/v1beta/openai/ to a nonexistent
    .../openai/v1/ and every Codex request would 404. The provider declares
    base_url_is_openai_root=True (data, not a name check) and the renderer
    honours it.
    """
    spec = build_spec(
        agent="codex", provider="gemini", model="gemini-2.5-pro", alias="gem"
    )
    body = openai_toml_body(spec)
    assert 'model_provider = "gemini"' in body
    assert "[model_providers.gemini]" in body
    assert (
        'base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"' in body
    )
    assert "/v1beta/openai/v1/" not in body
    assert 'wire_api = "chat"' in body


@pytest.mark.unit
def test_openai_toml_body_deepseek_openai_bare_root_gets_v1():
    """deepseek-openai's base_url is a bare root — the renderer appends /v1/
    (NOT is_openai_root), and the secret provider carries env_key."""
    spec = build_spec(
        agent="codex",
        provider="deepseek-openai",
        model="deepseek-v4-flash",
        alias="ds",
    )
    body = openai_toml_body(spec)
    assert 'base_url = "https://api.deepseek.com/v1/"' in body
    assert 'wire_api = "responses"' in body
    assert 'env_key = "DEEPSEEK_API_KEY"' in body


@pytest.mark.unit
def test_openai_toml_body_quotes_wire_api():
    """wire_api is a quoted TOML string, never a bare token (F2).

    The original renderer wrote ``wire_api = responses`` unquoted; that is
    invalid TOML. Pin the quotes so a regression to a bare value is caught.
    """
    spec = _toml_spec()
    body = openai_toml_body(spec)
    assert 'wire_api = "responses"' in body
    # The bare form must NOT appear anywhere.
    assert "wire_api = responses" not in body


@pytest.mark.unit
def test_openai_toml_body_escapes_control_chars_and_stays_parseable():
    """A model with control characters is escaped, not emitted raw (F2).

    TOML basic strings forbid literal U+0000–U+001F (except tab); a raw newline
    in the model would make Codex reject the profile. The escapes must produce
    a profile tomllib can parse back to the original model.
    """
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        pytest.skip(
            "tomllib unavailable on 3.10; escape correctness is exercised by the body assertions"
        )

    model = "weird\tname\nx"
    spec = _toml_spec(model=model)
    body = openai_toml_body(spec)
    # No literal control chars inside the model string line.
    model_line = next(ln for ln in body.splitlines() if ln.startswith("model = "))
    assert "\n" not in model_line and "\t" not in model_line
    # The escaped profile round-trips through tomllib to the original model.
    parsed = tomllib.loads(body)
    assert parsed["model"] == model


@pytest.mark.integration
def test_install_openai_toml_ignores_a_foreign_catalog(tmp_path):
    """No renderer writes or checks a catalog any more, so a pre-existing
    hand-curated ``<alias>.model.json`` (no ``managed_by`` marker — not one
    this tool could have written) is neither read nor touched by install: it
    is not our file's business any more, foreign or otherwise.
    """
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()
    catalog = paths.codex_catalog_for("glm-5-codex")
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        '{"version": 1, "models": [{"id": "hand-curated", "context_window": 200000}]}',
        encoding="utf-8",
    )

    assert install_wrapper(paths, spec) is True

    # Untouched — a hand-curated context_window was not touched.
    assert "hand-curated" in catalog.read_text(encoding="utf-8")
    assert paths.script_for("glm-5-codex").exists()


@pytest.mark.integration
def test_install_openai_toml_refuses_handwritten_wrapper_without_marker(tmp_path):
    """A hand-written ``exec codex --profile`` script is NOT adopted as ours (F1-own).

    OPENAI_TOML is a new shape with no pre-marker legacy form, so the wrapper
    slot keys off the marker ALONE (no byte-identical-to-legacy clause). A
    third-party script that happens to dispatch ``codex --profile`` must still
    route to OVERWRITE_FOREIGN, or a user's hand-maintained dispatch would be
    silently replaced.
    """
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()
    script = paths.script_for("glm-5-codex")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        '#!/bin/bash\n# my own dispatch\nexec codex --profile glm-5-codex "$@"\n',
        encoding="utf-8",
    )

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, spec)

    assert "my own dispatch" in script.read_text(encoding="utf-8")


@pytest.mark.integration
def test_spec_from_installed_recovers_model_from_toml_profile(tmp_path):
    """edit-token reaches a codex × ollama-direct wrapper and recovers its model (F7).

    The OPENAI_TOML wrapper body carries no model — it lives in the sibling
    TOML profile. spec_from_installed must read it from there, else edit-token
    falls back to the preset path and silently reverts a ``--model`` choice
    (the very bug the installed-spec lookup exists to prevent).
    """
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud"))

    spec = spec_from_installed(paths, "glm-5-codex")
    assert spec is not None
    assert spec.model == "glm-5.2:cloud"
    assert spec.agent.name == "codex"
    assert spec.provider.name == "ollama-direct"
    assert spec.shape is ConfigShape.OPENAI_TOML


@pytest.mark.integration
def test_spec_from_installed_returns_none_when_profile_missing(tmp_path):
    """A marked wrapper whose sibling profile is gone recovers no model (F7).

    The model is unrecoverable, so the caller falls back to the preset path
    rather than raising — mirroring every other None return in
    spec_from_installed.
    """
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud"))
    # Strip the sibling profile away — simulate a user who deleted it.
    paths.codex_config_for("glm-5-codex").unlink()

    assert spec_from_installed(paths, "glm-5-codex") is None


# --------------------------------------------------------------------------- #
# spec_from_installed — round-trip base_url for a non-FIXED provider
#
# Rule: the FILE wins for anything but a FIXED base_url (REQUIRED/OVERRIDABLE
# have no legitimate registry value to fall back to, and edit-token must not
# be licence to silently change an unrelated setting — same reasoning as
# reading all three env-shape tiers instead of one).
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_spec_from_installed_recovers_base_url_anthropic_env(tmp_path):
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="claude", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    recovered = spec_from_installed(paths, "lm")
    assert recovered is not None
    # anthropic_base_url stripped the /v1 suffix when rendering, so recovery
    # reads back the bare root the export line now carries.
    assert recovered.provider.base_url == "http://h:4000"


@pytest.mark.integration
def test_spec_from_installed_never_raises_on_a_corrupted_base_url(tmp_path):
    """spec_from_installed's documented contract is that it never raises —
    a recovered base_url that fails validate_base_url (a hand-edited or
    truncated wrapper) must degrade to None, not propagate CodeHelperError
    uncaught to a caller like edit-token/add --alias (cycle-review round 2,
    finding R1).
    """
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="claude", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    script = paths.script_for("lm")
    body = script.read_text(encoding="utf-8")
    # Corrupt the recovered value with an embedded tab — a single-quoted
    # shell value validate_base_url rejects (embedded control byte), but
    # that _env_value's single-line regex still matches and hands back
    # unchanged, so with_base_url/validate_base_url is what has to catch it.
    hacked = body.replace(
        "export ANTHROPIC_BASE_URL='http://h:4000'",
        "export ANTHROPIC_BASE_URL='http://h:4000\tbad'",
    )
    assert hacked != body  # sanity: the replace actually matched
    script.write_text(hacked, encoding="utf-8")

    # Must return None, not raise.
    assert spec_from_installed(paths, "lm") is None


@pytest.mark.integration
def test_spec_from_installed_recovers_base_url_openai_toml(tmp_path):
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="codex", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    recovered = spec_from_installed(paths, "lm")
    assert recovered is not None
    # openai_base_url appended /v1/ when writing the profile.
    assert recovered.provider.base_url == "http://h:4000/v1/"


@pytest.mark.integration
def test_reinstalling_a_recovered_openai_toml_spec_is_byte_identical(tmp_path):
    """Pins openai_base_url's idempotence across a round-trip: without it,
    re-deriving /v1/ from an already-/v1/-suffixed value would double it."""
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="codex", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    recovered = spec_from_installed(paths, "lm")
    assert recovered is not None
    assert install_wrapper(paths, recovered, token="tok") is False  # no-op


# --- anthropic_base_url ------------------------------------------------------
#
# openai_base_url's mirror image: ANTHROPIC_BASE_URL must be the version-less
# root because Claude Code appends /v1/messages itself. A base_url still
# carrying a /v1 suffix (from --base-url, or round-tripped through
# _recover_base_url off an OPENAI_TOML profile where /v1/ was already
# appended) must have it stripped here, or the request becomes
# .../v1/v1/messages and 404s — the bug this test suite exists to catch.


@pytest.mark.unit
def test_anthropic_base_url_strips_a_v1_suffix():
    assert anthropic_base_url("http://h:4000/v1") == "http://h:4000"


@pytest.mark.unit
def test_anthropic_base_url_strips_a_trailing_v1_slash():
    assert anthropic_base_url("http://h:4000/v1/") == "http://h:4000"


@pytest.mark.unit
def test_anthropic_base_url_leaves_a_bare_root_alone():
    assert anthropic_base_url("http://h:4000") == "http://h:4000"


@pytest.mark.unit
def test_anthropic_base_url_is_idempotent():
    once = anthropic_base_url("http://h:4000/v1")
    assert anthropic_base_url(once) == once


@pytest.mark.unit
def test_anthropic_base_url_does_not_strip_a_v1_inside_the_path():
    assert anthropic_base_url("http://h:4000/v1/proxy") == "http://h:4000/v1/proxy"


@pytest.mark.unit
def test_anthropic_base_url_does_not_strip_a_host_ending_in_v1():
    assert anthropic_base_url("http://v1") == "http://v1"


@pytest.mark.integration
def test_reinstalling_a_recovered_anthropic_env_spec_is_byte_identical(tmp_path):
    """Mirror of test_reinstalling_a_recovered_openai_toml_spec_is_byte_identical
    for ANTHROPIC_ENV: without anthropic_base_url's strip being idempotent, the
    recover -> re-render loop would oscillate the file body on every
    edit-token."""
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="claude", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    recovered = spec_from_installed(paths, "lm")
    assert recovered is not None
    assert install_wrapper(paths, recovered, token="tok") is False  # no-op


@pytest.mark.integration
def test_spec_from_installed_falls_back_to_the_default_for_overridable(
    tmp_path, monkeypatch
):
    """OVERRIDABLE with nothing recovered from the file: the registry default
    stands rather than refusing — unlike REQUIRED, there IS a legitimate
    value to fall back to, so refusing would be needless.

    No shipped provider is OVERRIDABLE today, so this monkeypatches ollama's
    policy for the duration of the test — the same out-of-registry technique
    test_model.py's _RUNTIME_OVERRIDABLE uses, applied here at the install
    layer instead of the pure-function layer.
    """
    from dataclasses import replace

    import codehelper.services.model as model_mod
    from codehelper.services.model import BaseUrlPolicy

    ollama = get_provider("ollama-direct")
    patched = replace(ollama, base_url_policy=BaseUrlPolicy.OVERRIDABLE)
    monkeypatch.setattr(
        model_mod,
        "PROVIDERS",
        tuple(patched if p.name == "ollama-direct" else p for p in model_mod.PROVIDERS),
    )

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="claude", provider=patched, model="m", alias="ov")
    install_wrapper(paths, spec, token="")

    recovered = spec_from_installed(paths, "ov")
    assert recovered is not None
    assert recovered.provider.base_url == "http://127.0.0.1:11434"


# --------------------------------------------------------------------------- #
# spec_from_installed — the recorded auth override (issue #81)
#
# The marker records the wrapper's effective auth mode. Without it, a wrapper
# installed with `--auth secret` on an OVERRIDABLE provider (ollama-direct)
# reconstructs as the registry's literal default — and every consumer of the
# reconstruction (`switch --from-wrapper`, the TUI chipset, `edit-token`)
# then applies or edits the literal "ollama" credential instead of the
# account token embedded in the wrapper's own body.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_spec_from_installed_honors_a_recorded_auth_override(tmp_path):
    """A wrapper installed with `--auth secret` on ollama-direct must come
    back SECRET, with the embedded account token reachable through
    token_from_installed — not the registry's literal 'ollama' default."""
    paths = Paths.from_home(tmp_path)
    provider = with_auth(get_provider("ollama-direct"), want_secret=True)
    spec = build_spec(
        agent="claude", provider=provider, model="glm-5:cloud", alias="ollama-secure"
    )
    install_wrapper(paths, spec, token="sk-account-token")

    recovered = spec_from_installed(paths, "ollama-secure")
    assert recovered is not None
    assert recovered.auth == "secret"
    assert recovered.auth_value == ""  # no literal value lingers on it
    assert (
        token_from_installed(paths, "ollama-secure", "ollama-direct")
        == "sk-account-token"
    )


@pytest.mark.integration
def test_spec_from_installed_keeps_the_registry_default_when_the_marker_has_no_auth(
    tmp_path,
):
    """MIGRATION: a marker written before the auth field existed reconstructs
    exactly as before — the registry default stands. ollama-direct reads back
    literal; zai (registry-secret) reads back secret. Emulated by stripping
    the field from a freshly rendered marker."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="claude", provider="zai", model="glm-5.3", alias="gz")
    install_wrapper(paths, spec, token="sk-zai")
    script = paths.script_for("gz")
    body = script.read_text(encoding="utf-8")
    legacy_body = body.replace(", auth=secret", "")
    assert legacy_body != body  # sanity: the field was actually there
    script.write_text(legacy_body, encoding="utf-8")

    recovered = spec_from_installed(paths, "gz")
    assert recovered is not None
    assert recovered.auth == "secret"  # the registry default, as before #81's fix


@pytest.mark.integration
def test_spec_from_installed_never_raises_on_an_impossible_recorded_auth(
    tmp_path, monkeypatch
):
    """`auth=secret` on a provider whose registry auth is FIXED-literal (a
    hand-edited marker, or — as emulated here — a registry downgrade after
    the install) fails CLOSED — the same "records something this build
    cannot honour → None" answer as an unknown shape, never a
    wrong-credential spec. No shipped provider is FIXED-literal, so the
    downgrade is monkeypatched onto ollama-direct, the same out-of-registry
    technique test_model.py pins with_auth's own refusal with."""
    from dataclasses import replace

    import codehelper.services.model as model_mod
    from codehelper.services.model import AuthPolicy

    paths = Paths.from_home(tmp_path)
    provider = with_auth(get_provider("ollama-direct"), want_secret=True)
    spec = build_spec(agent="claude", provider=provider, model="m", alias="od")
    install_wrapper(paths, spec, token="sk-downgraded")

    downgraded = replace(get_provider("ollama-direct"), auth_policy=AuthPolicy.FIXED)
    monkeypatch.setattr(
        model_mod,
        "PROVIDERS",
        tuple(
            downgraded if p.name == "ollama-direct" else p for p in model_mod.PROVIDERS
        ),
    )

    # Must return None, not raise — and not hand back a literal spec either.
    assert spec_from_installed(paths, "od") is None


@pytest.mark.integration
def test_install_over_a_secret_override_wrapper_with_a_lost_profile_refuses(tmp_path):
    """The `_discards_only_secret` fallback must honour the marker's recorded
    auth (issue #81): an OPENAI_TOML secret wrapper whose sibling profile is
    gone still holds the only copy of its token — `spec_from_installed`
    returns None (no model to recover), and the marker-level fallback used to
    ask the REGISTRY, which answers "literal" for ollama-direct. A literal
    replacement at the same alias must refuse without --force, not silently
    destroy the token."""
    paths = Paths.from_home(tmp_path)
    provider = with_auth(get_provider("ollama-direct"), want_secret=True)
    secret = build_spec(agent="codex", provider=provider, model="m", alias="sec")
    install_wrapper(paths, secret, token="sk-only-copy")
    paths.codex_config_for("sec").unlink()  # lose the profile → no model to recover

    literal = build_spec(
        agent="codex", provider=get_provider("ollama-direct"), model="m", alias="sec"
    )
    with pytest.raises(CodeHelperError, match="only copy"):
        install_wrapper(paths, literal, token="")


@pytest.mark.integration
def test_spec_from_installed_resolves_a_marker_from_before_the_ollama_rename(
    tmp_path,
):
    """A wrapper marker written by an OLDER codehelper still says
    ``provider=ollama`` (before the ollama -> ollama-direct rename, forced by
    Codex CLI v0.150.1 reserving that name) — spec_from_installed must still
    resolve it to the CURRENT ``ollama-direct`` Provider object via
    get_provider_for_legacy_read, not treat the wrapper as foreign/corrupt.
    """
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="claude", provider="ollama-direct", model="m", alias="old")
    install_wrapper(paths, spec, token="")

    script = paths.script_for("old")
    body = script.read_text(encoding="utf-8")
    legacy_body = body.replace("provider=ollama-direct", "provider=ollama")
    assert legacy_body != body  # sanity: the replace actually matched
    script.write_text(legacy_body, encoding="utf-8")

    recovered = spec_from_installed(paths, "old")
    assert recovered is not None
    assert recovered.provider.name == "ollama-direct"


@pytest.mark.integration
def test_spec_from_installed_returns_none_when_a_required_provider_lost_its_url(
    tmp_path,
):
    """REQUIRED with no registry fallback: an unrecoverable base_url must
    refuse to reconstruct rather than hand back a spec with an empty one —
    edit-token would otherwise silently reinstall the wrapper pointed at
    nothing."""
    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="codex", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="tok")

    # Strip the base_url line from the profile — simulate a hand-edit.
    config_path = paths.codex_config_for("lm")
    body = config_path.read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in body.split("\n") if not line.startswith("base_url")
    )
    config_path.write_text(stripped, encoding="utf-8")

    assert spec_from_installed(paths, "lm") is None


@pytest.mark.integration
def test_spec_from_installed_ignores_a_file_base_url_for_a_fixed_provider(tmp_path):
    """FIXED: the registry wins even over a hand-edited file — there is no
    axis on which the file could carry a legitimate override, so honouring
    it would let a hand-edit silently redirect a wrapper's endpoint."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude", provider="ollama-direct", model="m", alias="ollama-t"
    )
    install_wrapper(paths, spec, token="")

    script = paths.script_for("ollama-t")
    body = script.read_text(encoding="utf-8")
    hacked = body.replace(
        "export ANTHROPIC_BASE_URL='http://127.0.0.1:11434'",
        "export ANTHROPIC_BASE_URL='http://hacked/v1'",
    )
    assert hacked != body  # sanity: the replace actually matched
    script.write_text(hacked, encoding="utf-8")

    recovered = spec_from_installed(paths, "ollama-t")
    assert recovered is not None
    assert recovered.provider.base_url == "http://127.0.0.1:11434"


@pytest.mark.integration
def test_edit_token_preserves_the_base_url(tmp_path, monkeypatch):
    """Rotating a litellm token via the edit-token CLI command must not touch
    the base_url the wrapper was installed with.

    Installation itself goes through the service layer directly
    (build_spec/with_base_url/install_wrapper) rather than ``add --base-url``
    — that CLI flag is wired in Part 2/2 (feat/base-url-cli); ``edit-token``
    is unconditional CLI here in Part 1/2 and needs no such flag, so it is
    the one piece of this round-trip this branch can actually exercise
    end-to-end.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    import getpass

    from codehelper.__main__ import main

    paths = Paths.from_home(tmp_path)
    provider = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    spec = build_spec(agent="claude", provider=provider, model="gpt-4o", alias="lm")
    install_wrapper(paths, spec, token="old-tok")

    monkeypatch.setattr(getpass, "getpass", lambda *_a, **_kw: "new-tok")
    assert main(["edit-token", "lm"]) == 0

    body = paths.script_for("lm").read_text(encoding="utf-8")
    assert "export ANTHROPIC_BASE_URL='http://h:4000'" in body
    assert "export ANTHROPIC_AUTH_TOKEN='new-tok'" in body


@pytest.mark.integration
def test_shape_switch_cleans_up_orphaned_openai_toml_siblings(tmp_path, capsys):
    """Reusing an alias for a non-OPENAI_TOML shape removes the old profile,
    and a stale catalog left by a PREVIOUS version of this tool (F8) is
    warned about but left in place.

    A previous OPENAI_TOML install leaves ``~/.codex/<alias>.config.toml``
    behind; the new wrapper no longer dispatches ``codex --profile <alias>``,
    so it is an orphan. The catalog is manufactured by hand — no renderer
    writes one any more — to prove a leftover from before this fix is never
    deleted: the ``managed_by`` marker alone does not prove it was not
    hand-edited since. Only the profile (marker-proven ours) is removed.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud", alias=alias))
    assert paths.codex_config_for(alias).exists()
    stale_catalog = paths.codex_catalog_for(alias)
    stale_catalog.write_text(
        '{"version": 1, "managed_by": "codehelper", "models": []}',
        encoding="utf-8",
    )
    assert stale_catalog.exists()

    # Switch the alias to the launcher shape (claude × ollama via ollama-launch).
    launcher_spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="glm-5:cloud",
        alias=alias,
        shape=ConfigShape.OLLAMA_LAUNCH,
    )
    assert install_wrapper(paths, launcher_spec) is True

    # The wrapper was rewritten; the orphaned PROFILE is gone, the stale
    # catalog is only warned about.
    assert paths.script_for(alias).exists()
    assert not paths.codex_config_for(alias).exists()
    assert stale_catalog.exists()
    assert "stale codehelper-managed model catalog" in capsys.readouterr().err


@pytest.mark.integration
def test_shape_switch_leaves_foreign_siblings(tmp_path):
    """A foreign profile under the alias is NOT removed on shape switch (F8).

    The cleanup proves authorship the same way the install guard does — the
    profile marker. A foreign profile (no marker) is left alone, exactly as
    the guard would refuse to overwrite it.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    # A foreign profile + catalog the user hand-curated, no marker.
    config = paths.codex_config_for(alias)
    catalog = paths.codex_catalog_for(alias)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('# my codex profile\nmodel = "gpt-4o"\n', encoding="utf-8")
    catalog.write_text('{"version": 1, "models": []}', encoding="utf-8")

    launcher_spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="glm-5:cloud",
        alias=alias,
        shape=ConfigShape.OLLAMA_LAUNCH,
    )
    # The launcher wrapper itself lands fine (the alias path was free); the
    # foreign siblings stay put.
    assert install_wrapper(paths, launcher_spec) is True
    assert (
        config.read_text(encoding="utf-8") == '# my codex profile\nmodel = "gpt-4o"\n'
    )
    assert catalog.read_text(encoding="utf-8") == '{"version": 1, "models": []}'


@pytest.mark.unit
def test_build_spec_allows_secret_auth_with_openai_toml():
    """secret + OPENAI_TOML is now VALID — this used to be F9's refusal.

    ``openai_toml_body`` writes ``env_key`` for a secret provider and
    ``_render_openai_toml`` exports the matching variable before ``exec``, so
    the shape can carry a token now. This is the enabling change for a
    runtime-``base_url`` provider (e.g. a LiteLLM proxy) that needs secret
    auth on Codex.
    """
    secret_provider = Provider(
        name="secret-openai",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.secret.invalid/v1",
        auth="secret",
        token_env_var="SECRET_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="chat",
    )
    spec = build_spec(agent="codex", provider=secret_provider, model="m")
    assert spec.auth == "secret"
    assert spec.shape is ConfigShape.OPENAI_TOML


# --------------------------------------------------------------------------- #
# Round-2 review findings (M1-M5): cleanup's own ownership guard, TOMLDecodeError
# leaking past spec_from_installed, SKIP-install cleanup gating, a self-marked
# catalog surviving a missing profile, and wire_api registry validation.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_shape_switch_cleanup_refuses_to_delete_a_foreign_catalog(tmp_path):
    """M1: a hand-curated catalog next to OUR profile survives a shape switch.

    _cleanup_openai_toml_siblings must gate the catalog unlink on the
    catalog's OWN ownership proof, not inherit it from the profile's marker —
    otherwise a foreign catalog sitting next to our profile is silently
    deleted with no --force and no prompt, exactly what the install-time guard
    exists to prevent.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud", alias=alias))
    assert paths.codex_config_for(alias).exists()

    # Replace ONLY the catalog with a foreign, hand-curated one — the profile
    # (and its marker) is untouched.
    catalog = paths.codex_catalog_for(alias)
    catalog.write_text(
        '{"version": 1, "models": [{"id": "hand-curated", "context_window": 200000}]}',
        encoding="utf-8",
    )

    launcher_spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="glm-5:cloud",
        alias=alias,
        shape=ConfigShape.OLLAMA_LAUNCH,
    )
    assert install_wrapper(paths, launcher_spec) is True

    # The wrapper switched shape; the foreign catalog survives untouched.
    assert paths.script_for(alias).exists()
    assert "hand-curated" in catalog.read_text(encoding="utf-8")
    # The profile WAS ours and is cleaned up independently.
    assert not paths.codex_config_for(alias).exists()


@pytest.mark.integration
def test_shape_switch_cleanup_leaves_a_legacy_catalog_with_no_self_marker(tmp_path):
    """A legacy catalog (no ``managed_by`` field — pre-migration, or someone
    else's file sitting next to our profile) is left as a harmless orphan on
    a shape switch, rather than deleted on the strength of the sibling
    profile's marker alone.

    Cleanup deliberately does NOT use ``_ownership_catalog``'s sibling-marker
    fallback (unlike the install-time overwrite guard, which does, and which
    a user can override with ``--force``): a delete has no such escape hatch,
    and "the profile next to this catalog is ours" cannot distinguish a
    catalog WE wrote before the ``managed_by`` field existed from a foreign
    one a user happened to drop next to our profile. Left behind rather than
    silently destroyed — the safe direction to be wrong in.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud", alias=alias))

    # Replace the catalog with a body carrying no self-marker — indistinguishable,
    # from the catalog's own JSON alone, from a foreign hand-curated file.
    catalog = paths.codex_catalog_for(alias)
    catalog.write_text(
        '{"version": 1, "models": [{"id": "glm-5.2:cloud", "context_window": 128000}]}',
        encoding="utf-8",
    )

    launcher_spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="glm-5:cloud",
        alias=alias,
        shape=ConfigShape.OLLAMA_LAUNCH,
    )
    assert install_wrapper(paths, launcher_spec) is True

    # The marked profile IS cleaned up; the unmarked catalog is left behind —
    # orphaned clutter, not data loss.
    assert not paths.codex_config_for(alias).exists()
    assert catalog.exists()


@pytest.mark.integration
def test_spec_from_installed_survives_corrupt_toml_profile(tmp_path):
    """M2: a hand-truncated/corrupt TOML profile degrades to None, never raises.

    spec_from_installed's documented contract is "never raises" — every caller
    (edit-token, discover_managed) relies on falling back to the preset path.
    tomllib.loads raises TOMLDecodeError (a ValueError) on malformed TOML; that
    must be caught, not left to escape as a raw traceback.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud", alias=alias))

    # Corrupt the profile — an unterminated string is invalid TOML.
    config = paths.codex_config_for(alias)
    corrupted = config.read_text(encoding="utf-8").replace(
        'model = "glm-5.2:cloud"', 'model = "glm-5.2:cloud'
    )
    config.write_text(corrupted, encoding="utf-8")

    # Must not raise — None is the documented "unrecoverable" outcome.
    assert spec_from_installed(paths, alias) is None


@pytest.mark.integration
def test_install_openai_toml_skip_still_cleans_orphaned_siblings_and_reports_it(
    tmp_path,
):
    """M3: a byte-identical (SKIP) wrapper re-install still cleans up siblings
    left over from an EARLIER OPENAI_TOML install, and that cleanup is
    reflected in the return value — not silently dropped because the wrapper
    slot itself didn't change.
    """
    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    install_wrapper(paths, _toml_spec(model="glm-5.2:cloud", alias=alias))
    assert paths.codex_config_for(alias).exists()

    # Switch shapes once (this itself cleans up — assert a clean starting
    # point by re-manufacturing an orphaned sibling by hand afterwards).
    launcher_spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="glm-5:cloud",
        alias=alias,
        shape=ConfigShape.OLLAMA_LAUNCH,
    )
    install_wrapper(paths, launcher_spec)
    assert not paths.codex_config_for(alias).exists()

    # Re-create an orphaned, OUR-marked profile by hand (simulating a leftover
    # from an install that predates this cleanup, or a partial failure) —
    # the wrapper slot for a second install of the SAME launcher_spec is now
    # byte-identical (SKIP).
    from codehelper.services.render import openai_toml_body

    stray_profile = paths.codex_config_for(alias)
    stray_profile.write_text(
        openai_toml_body(_toml_spec(alias=alias)),
        encoding="utf-8",
    )
    assert stray_profile.exists()

    wrote = install_wrapper(paths, launcher_spec)
    # The wrapper write itself is a no-op (SKIP), but the stray profile was
    # cleaned up — that must still be reported as a change.
    assert wrote is True
    assert not stray_profile.exists()


@pytest.mark.unit
def test_validate_registries_rejects_openai_toml_provider_without_wire_api():
    """M5: a provider declaring openai-toml with a bad wire_api fails at
    import/registry-validation time, not with a broken profile at codex
    runtime — the whole point of the shape being a data-driven extension
    point.
    """
    import codehelper.services.model as model_mod

    bad_provider = Provider(
        name="bad-openai",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.bad.invalid/v1",
        auth="literal",
        auth_value="x",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="",  # missing — must be "responses" or "chat"
    )
    original = model_mod.PROVIDERS
    model_mod.PROVIDERS = original + (bad_provider,)
    try:
        with pytest.raises(CodeHelperError, match="invalid wire_api"):
            model_mod._validate_registries()
    finally:
        # Restore WITHOUT importlib.reload: reloading would mint new class
        # objects for ConfigShape/Provider/etc, breaking identity comparisons
        # (`is`) any test running after this one relies on.
        model_mod.PROVIDERS = original


@pytest.mark.unit
def test_toml_profile_data_fallback_scopes_base_url_to_its_own_table(
    monkeypatch, tmp_path
):
    """The sub-3.11 fallback regex path must not attribute a sibling table's
    base_url to the matched table (F1 from PR #12's cycle-review round 1).

    The real ``tomllib.loads`` path naturally nests per table; the defensive
    fallback used three independent whole-file regex searches instead, so a
    profile with more than one ``[model_providers.*]`` table (reachable via a
    hand-edited or ``--force``-adopted legacy file) could hand the SECOND
    table's ``base_url`` to the FIRST table's name. Force the fallback branch
    by making the ``tomllib`` import inside ``_toml_profile_data`` raise
    ``ModuleNotFoundError``, regardless of the interpreter's real version.
    """
    import builtins

    import codehelper.services.wrappers as wrappers_mod

    real_import = builtins.__import__

    def _no_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("simulated: no tomllib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tomllib)

    paths = Paths.from_home(tmp_path)
    profile_path = paths.codex_config_for("glm-5-codex")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    # Two tables: the first (matched by the un-scoped regex first) has NO
    # base_url; the second carries one. A buggy whole-file search would still
    # find `base_url_found` (it's present somewhere in the file) and wrongly
    # attach it to the first table's name.
    profile_path.write_text(
        'model = "glm-5.2:cloud"\n'
        "\n"
        "[model_providers.first-table]\n"
        'name = "First"\n'
        'wire_api = "responses"\n'
        "\n"
        "[model_providers.second-table]\n"
        'name = "Second"\n'
        'base_url = "https://second.example.com/v1/"\n'
        'wire_api = "responses"\n',
        encoding="utf-8",
    )

    data = wrappers_mod._toml_profile_data(paths, "glm-5-codex")

    assert data is not None
    # The fallback only ever recovers ONE table (the first match) by design —
    # what matters is that when it does, it does NOT smuggle in a base_url
    # that belongs to a different table.
    assert "first-table" in data["model_providers"]
    assert data["model_providers"]["first-table"]["base_url"] is None


@pytest.mark.unit
def test_all_has_no_duplicate_entries():
    """``__all__`` is the module's public-API list; a duplicate is dead
    weight from an edit, never intentional (a name is exported once)."""
    import codehelper.services.wrappers as wrappers_mod

    assert len(wrappers_mod.__all__) == len(set(wrappers_mod.__all__))


@pytest.mark.unit
def test_describe_wrapper_marker_preserves_state_column_alignment():
    """The ``● `` default marker is a PREFIX to ``spec.name``, so the name
    field must shrink by the marker's width — otherwise the install-state
    column shifts right by two on the marked row and the three menus that
    share this format string drift apart on layout."""
    from codehelper.services.wrappers import describe_wrapper

    spec = build_spec(
        agent="claude", provider="ollama-direct", model="qwen3.5:9b", alias="glm"
    )
    unmarked = describe_wrapper(
        spec, installed=True, installed_word="installed", not_installed_word="missing"
    )
    marked = describe_wrapper(
        spec,
        installed=True,
        installed_word="installed",
        not_installed_word="missing",
        default=True,
    )
    assert unmarked.index("installed") == marked.index("installed")


@pytest.mark.integration
def test_remove_wrapper_removes_owned_siblings_and_clears_default(tmp_path):
    from codehelper.services.state import default_wrapper, set_default_wrapper
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)
    set_default_wrapper(paths, "codex", spec.alias)

    assert remove_wrapper(paths, spec.alias) is True
    assert not paths.script_for(spec.alias).exists()
    assert not paths.codex_config_for(spec.alias).exists()
    assert not paths.codex_catalog_for(spec.alias).exists()
    assert default_wrapper(paths, "codex") is None


@pytest.mark.integration
def test_remove_wrapper_asks_confirm_before_unlinking(tmp_path):
    """``confirm`` is called with the full target list before anything is
    touched; declining leaves every file in place."""
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    seen = []

    def decline(targets):
        seen.extend(targets)
        return False

    with pytest.raises(CodeHelperError, match="not confirmed"):
        remove_wrapper(paths, spec.alias, confirm=decline)
    assert paths.script_for(spec.alias) in seen
    assert paths.script_for(spec.alias).exists()

    def accept(targets):
        return True

    assert remove_wrapper(paths, spec.alias, confirm=accept) is True
    assert not paths.script_for(spec.alias).exists()


@pytest.mark.integration
def test_remove_wrapper_force_bypasses_confirm(tmp_path):
    """``force`` is "I already know what I'm removing" — it skips the
    prompt entirely, same contract as ``install_wrapper``'s ``force``."""
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    def explode(_targets):
        raise AssertionError("confirm must not be called when force=True")

    assert remove_wrapper(paths, spec.alias, force=True, confirm=explode) is True
    assert not paths.script_for(spec.alias).exists()


@pytest.mark.integration
def test_remove_wrapper_dry_run_never_asks_confirm(tmp_path):
    """Dry-run never mutates anything, so there is nothing to confirm."""
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    def explode(_targets):
        raise AssertionError("confirm must not be called under dry_run")

    assert remove_wrapper(paths, spec.alias, dry_run=True, confirm=explode) is True
    assert paths.script_for(spec.alias).exists()


@pytest.mark.integration
def test_remove_wrapper_refuses_foreign_file_without_force(tmp_path):
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    paths.bin_dir.mkdir(parents=True)
    paths.script_for("foreign").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(CodeHelperError, match="use --force"):
        remove_wrapper(paths, "foreign")
    assert remove_wrapper(paths, "foreign", force=True) is True


@pytest.mark.integration
def test_remove_wrapper_survives_a_failed_sibling_unlink(tmp_path, monkeypatch):
    """A mid-sequence unlink failure must not make the wrapper itself
    unrecoverable (siblings unlink before the wrapper executable, the
    ownership proof a retry needs), AND must not clear the default pointer
    for a wrapper that is still installed and still the user's default —
    the default is only cleared once every unlink has actually succeeded."""
    from pathlib import Path

    from codehelper.services.state import default_wrapper, set_default_wrapper
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)
    set_default_wrapper(paths, "codex", spec.alias)

    # No renderer writes a catalog any more; manufacture a legacy, marker-
    # owned one by hand so its unlink is still exercised — the same leftover
    # `remove_wrapper` (and only `remove_wrapper`) deletes on an explicit,
    # confirmed removal.
    catalog = paths.codex_catalog_for(spec.alias)
    catalog.write_text(
        '{"version": 1, "managed_by": "codehelper", "models": []}',
        encoding="utf-8",
    )
    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **kw):
        if self == catalog:
            raise OSError("simulated transient failure")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(CodeHelperError, match="failed to remove"):
        remove_wrapper(paths, spec.alias)

    # The wrapper itself must still be present — it is the ownership proof a
    # retry needs — and its default must be untouched: the removal did not
    # actually succeed, so the still-installed wrapper must not lose its
    # default out from under it.
    assert paths.script_for(spec.alias).exists()
    assert default_wrapper(paths, "codex") == spec.alias

    monkeypatch.undo()

    # A retry now succeeds and cleans up everything, including the sibling
    # that failed the first time, and only NOW clears the default.
    assert remove_wrapper(paths, spec.alias) is True
    assert not paths.script_for(spec.alias).exists()
    assert not paths.codex_config_for(spec.alias).exists()
    assert not catalog.exists()
    assert default_wrapper(paths, "codex") is None


@pytest.mark.integration
def test_remove_wrapper_reports_but_does_not_undo_a_failed_default_clear(
    tmp_path, monkeypatch
):
    """If every unlink succeeds but the follow-up state.json write fails, the
    files are gone regardless — that cannot be undone — so the failure must
    be reported (not swallowed as success) without claiming the deletion
    itself didn't happen."""
    import codehelper.services.state as state
    from codehelper.services.state import set_default_wrapper
    from codehelper.services.wrappers import remove_wrapper

    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)
    set_default_wrapper(paths, "codex", spec.alias)

    def flaky_write_state(paths_arg, state_dict):
        raise OSError("simulated disk-full failure")

    monkeypatch.setattr(state, "_write_state", flaky_write_state)

    with pytest.raises(CodeHelperError, match="failed to clear its default pointer"):
        remove_wrapper(paths, spec.alias)

    # The deletion itself is NOT undone — the files are genuinely gone.
    assert not paths.script_for(spec.alias).exists()
    assert not paths.codex_config_for(spec.alias).exists()
    assert not paths.codex_catalog_for(spec.alias).exists()


# --------------------------------------------------------------------------- #
# CLI: `remove` through main([...]) — confirm gate.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_cli_remove_refuses_off_a_tty_without_force(tmp_path, monkeypatch):
    """``confirm=None`` means "no way to ask" — must error, never read stdin,
    same off-a-TTY fail-fast contract as ``add``'s foreign-file guard."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    assert main(["remove", spec.alias]) == 1
    assert paths.script_for(spec.alias).exists()


@pytest.mark.integration
def test_cli_remove_force_bypasses_confirm_off_a_tty(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    assert main(["remove", spec.alias, "--force"]) == 0
    assert not paths.script_for(spec.alias).exists()


@pytest.mark.integration
def test_cli_remove_dry_run_never_prompts_and_never_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    paths = Paths.from_home(tmp_path)
    spec = build_spec(agent="codex", provider="ollama-direct", model="remove-me")
    install_wrapper(paths, spec)

    assert main(["--dry-run", "remove", spec.alias]) == 0
    assert paths.script_for(spec.alias).exists()


# --------------------------------------------------------------------------- #
# Issue #83: the explicit context_window axis — build_spec validation.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_spec_rejects_an_unusable_context_window():
    """A garbage explicit window refuses at build time, before anything
    interactive — the same early-refusal contract as a bad alias."""
    for bad in (-1, -5, 10_000_001, 99_999_999):
        with pytest.raises(CodeHelperError, match="unusable context window"):
            build_spec(
                agent="claude",
                provider="ollama-direct",
                model="mystery-3b",
                context_window=bad,
            )


@pytest.mark.unit
def test_build_spec_accepts_zero_and_positive_context_windows():
    """``0`` is a real answer (explicit no-declaration) and any sane positive
    count is accepted verbatim."""
    assert (
        build_spec(
            agent="claude",
            provider="ollama-direct",
            model="mystery-3b",
            context_window=0,
        ).context_window
        == 0
    )
    assert (
        build_spec(
            agent="claude",
            provider="ollama-direct",
            model="mystery-3b",
            context_window=500_000,
        ).context_window
        == 500_000
    )
    assert (
        build_spec(
            agent="claude", provider="ollama-direct", model="mystery-3b"
        ).context_window
        is None
    )


# --------------------------------------------------------------------------- #
# Issue #83: the explicit context_window axis — marker field + emission
# precedence (explicit wins; 0 suppresses even a catalog hit; without an
# explicit value every byte is unchanged).
# --------------------------------------------------------------------------- #


def _windowed_spec(model: str, window: int | None, agent: str = "claude"):
    return build_spec(
        agent=agent,
        provider="ollama-direct",
        model=model,
        context_window=window,
    )


@pytest.mark.unit
def test_marker_records_ctx_only_for_an_explicit_window():
    """``ctx=`` appears ONLY for an explicit answer — and it rides LAST in the
    marker so every existing marker's bytes (and all prefix-matching code)
    are untouched."""
    marker = _marker(_windowed_spec("mystery-3b", 500_000))
    assert marker.endswith("ctx=500000)")

    literal = _marker(_windowed_spec("mystery-3b", None))
    assert "ctx=" not in literal

    zero = _marker(_windowed_spec("glm-5.3", 0))
    assert "ctx=0)" in zero  # an explicit suppression is recorded too

    plain = _marker(_windowed_spec("glm-5.3", None))
    assert "ctx=" not in plain  # catalog-known: today's bytes, unchanged


@pytest.mark.unit
def test_render_without_an_explicit_window_is_byte_identical_to_catalog():
    """The golden rule from #82, restated for #83: a spec with no explicit
    window renders EXACTLY today's body — no SKIP-path churn for any
    already-installed wrapper."""
    plain = build_spec(agent="claude", provider="ollama-direct", model="glm-5.3")
    ctx_none = build_spec(
        agent="claude", provider="ollama-direct", model="glm-5.3", context_window=None
    )
    assert render_script(plain, _LITERAL_TOKEN) == render_script(
        ctx_none, _LITERAL_TOKEN
    )


@pytest.mark.unit
def test_render_explicit_window_wins_over_the_catalog():
    """An explicit answer overrides what the catalog would derive — in BOTH
    channels (the export AND the --settings payload; one dict feeds both)."""
    body = render_script(_windowed_spec("glm-5.3", 500_000), _LITERAL_TOKEN)
    assert "export CLAUDE_CODE_MAX_CONTEXT_TOKENS='500000'" in body
    assert _settings_payload(body)["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "500000"


@pytest.mark.unit
def test_render_ctx_zero_suppresses_a_catalog_declaration():
    """``0`` means the user chose 'no declaration' — the catalog's 1M must
    NOT leak into the session."""
    body = render_script(_windowed_spec("glm-5.3", 0), _LITERAL_TOKEN)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in body


@pytest.mark.unit
def test_render_explicit_window_for_an_unknown_model():
    """The whole point of #83: a model the catalog has never heard of gets a
    declaration when — and only when — the user supplied one."""
    body = render_script(_windowed_spec("mystery-3b", 1_000_000), _LITERAL_TOKEN)
    assert "export CLAUDE_CODE_MAX_CONTEXT_TOKENS='1000000'" in body


@pytest.mark.unit
def test_command_shape_honors_explicit_context_window():
    """The OLLAMA_LAUNCH --settings payload follows the same precedence."""
    from dataclasses import replace

    preset_spec = spec_from_preset(
        get_preset("glm-ollama"), model_override="mystery-3b"
    )
    spec = replace(preset_spec, context_window=2_000_000)
    body = render_script(spec, "")
    assert _settings_payload(body)["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == (
        "2000000"
    )


@pytest.mark.unit
def test_openai_toml_body_honors_explicit_context_window():
    """Codex's profile gets ``model_context_window`` from the explicit
    answer too — and nothing for an explicit suppression."""
    spec = build_spec(
        agent="codex",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery-codex",
        context_window=2_000_000,
    )
    assert "model_context_window = 2000000" in openai_toml_body(spec)

    suppressed = build_spec(
        agent="codex",
        provider="ollama-direct",
        model="glm-5.2:cloud",
        alias="mystery-codex",
        context_window=0,
    )
    assert "model_context_window" not in openai_toml_body(suppressed)


# --------------------------------------------------------------------------- #
# spec_from_installed — the recorded context window (issue #83)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_spec_from_installed_honors_a_recorded_ctx(tmp_path):
    """A wrapper whose marker records ctx= must come back with that explicit
    window — not a re-derivation: switch --from-wrapper, the chipset, and
    edit-token all consume this reconstruction."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery",
        context_window=750_000,
    )
    install_wrapper(paths, spec, token=_LITERAL_TOKEN)

    recovered = spec_from_installed(paths, "mystery")
    assert recovered is not None
    assert recovered.context_window == 750_000


@pytest.mark.integration
def test_reinstalling_a_recovered_ctx_spec_is_byte_identical(tmp_path):
    """The round-trip rule: install → reconstruct → re-render must reproduce
    the same bytes (ctx= included) — reinstall's `no changes` SKIP path
    depends on it."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery",
        context_window=750_000,
    )
    install_wrapper(paths, spec, token=_LITERAL_TOKEN)
    before = script_text(paths, "mystery")

    recovered = spec_from_installed(paths, "mystery")
    assert recovered is not None
    assert render_script(recovered, _LITERAL_TOKEN) == before


@pytest.mark.integration
def test_spec_from_installed_keeps_no_ctx_when_the_marker_has_none(tmp_path):
    """MIGRATION: a marker written before the ctx field existed reconstructs
    with context_window=None — the catalog derivation stands, exactly the
    pre-#83 behaviour. Emulated by stripping the field from a freshly
    rendered marker."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery",
        context_window=750_000,
    )
    install_wrapper(paths, spec, token=_LITERAL_TOKEN)
    script = paths.script_for("mystery")
    body = script.read_text(encoding="utf-8")
    legacy_body = body.replace(", ctx=750000", "")
    assert legacy_body != body  # sanity: the field was actually there
    script.write_text(legacy_body, encoding="utf-8")

    recovered = spec_from_installed(paths, "mystery")
    assert recovered is not None
    assert recovered.context_window is None


@pytest.mark.integration
def test_spec_from_installed_returns_none_on_garbage_ctx(tmp_path):
    """`ctx=abc` (unparseable) fails CLOSED — None, the same answer as any
    other unrecognised marker value, never a silently-windowless spec that
    looks healthy."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery",
        context_window=750_000,
    )
    install_wrapper(paths, spec, token=_LITERAL_TOKEN)
    script = paths.script_for("mystery")
    script.write_text(
        script.read_text(encoding="utf-8").replace("ctx=750000", "ctx=abc"),
        encoding="utf-8",
    )

    assert spec_from_installed(paths, "mystery") is None


@pytest.mark.integration
def test_spec_from_installed_returns_none_on_a_negative_ctx(tmp_path):
    """`ctx=-5` parses as an int but fails build_spec's range check — the
    same fail-closed None."""
    paths = Paths.from_home(tmp_path)
    spec = build_spec(
        agent="claude",
        provider="ollama-direct",
        model="mystery-3b",
        alias="mystery",
        context_window=750_000,
    )
    install_wrapper(paths, spec, token=_LITERAL_TOKEN)
    script = paths.script_for("mystery")
    script.write_text(
        script.read_text(encoding="utf-8").replace("ctx=750000", "ctx=-5"),
        encoding="utf-8",
    )

    assert spec_from_installed(paths, "mystery") is None
