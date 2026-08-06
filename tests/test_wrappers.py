"""Tests for ``services/wrappers.py`` — the generated Claude Code wrapper scripts.

A wrapper is not a shell alias string; it is a generated bash script (mode
0o700 for secret-backed wrappers, 0o755 for literal-token wrappers) that
exports ``ANTHROPIC_*`` envs and runs ``claude``. This mirrors the archived
``zai-codex-helper`` project's hand-written ``~/.local/bin/glm``, generalized
across providers via :class:`WrapperSpec`.
"""

from __future__ import annotations

import stat

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.model import ConfigShape
from code_helper.services.paths import Paths
from code_helper.services.render import (
    openai_catalog_body,
    openai_toml_body,
    render_script,
)
from code_helper.services.spec import (
    WrapperSpec,
    build_spec,
    get_preset,
    spec_from_preset,
)
from code_helper.services.wrappers import (
    WRAPPERS,
    discover_managed,
    get_spec,
    install_wrapper,
    is_installed,
    is_managed,
    list_wrappers,
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
def test_registry_has_deepseek_glm_and_glm_ollama():
    assert {w.name for w in WRAPPERS} == {"deepseek", "glm", "glm-ollama"}


@pytest.mark.unit
def test_glm_ollama_uses_the_launcher_shape():
    spec = get_spec("glm-ollama")
    assert spec.shape is ConfigShape.OLLAMA_LAUNCH
    assert spec.agent.name == "claude"
    assert spec.model == "glm-5.2:cloud"
    assert spec.auth == "literal"  # no token — ollama launch authenticates itself


@pytest.mark.unit
def test_deepseek_is_literal_auth():
    spec = get_spec("deepseek")
    assert spec.auth == "literal"
    assert spec.auth_value == "ollama"
    assert spec.provider.base_url == "http://127.0.0.1:11434"
    # Same provider as glm-ollama, different shape — that is the whole point
    # of naming the axes.
    assert spec.provider.name == "ollama"
    assert spec.shape is ConfigShape.ANTHROPIC_ENV


@pytest.mark.unit
def test_glm_is_secret_auth():
    spec = get_spec("glm")
    assert spec.auth == "secret"
    assert spec.token_env_var == "ZAI_API_KEY"
    assert spec.provider.base_url == "https://api.z.ai/api/anthropic"


@pytest.mark.unit
def test_glm_keeps_distinct_models_per_tier():
    """The reason TierModels exists — one --model could not express this."""
    tiers = get_spec("glm").tier_models
    assert (tiers.haiku, tiers.sonnet, tiers.opus) == (
        "glm-4.7",
        "glm-5-turbo",
        "glm-5.2[1m]",
    )


@pytest.mark.unit
def test_get_spec_unknown_name_raises():
    with pytest.raises(CodeHelperError, match="unknown wrapper"):
        get_spec("nope")


# --------------------------------------------------------------------------- #
# render_script — pure, embeds the token + models, single-quotes everything.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_render_script_embeds_token_single_quoted():
    body = render_script(get_spec("deepseek"), _LITERAL_TOKEN)
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in body
    assert "#!/bin/bash" in body


@pytest.mark.unit
def test_render_script_deepseek_has_endpoint_and_models():
    body = render_script(get_spec("deepseek"), _LITERAL_TOKEN)
    assert "ANTHROPIC_BASE_URL='http://127.0.0.1:11434'" in body
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL=" in body
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=" in body
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL=" in body
    assert "CLAUDE_CODE_SUBAGENT_MODEL=" in body
    assert 'claude "$@"' in body


@pytest.mark.unit
def test_render_script_glm_has_no_subagent_model():
    """glm's spec has no subagent model — the line must be absent."""
    body = render_script(get_spec("glm"), _SECRET_TOKEN)
    assert "CLAUDE_CODE_SUBAGENT_MODEL=" not in body


@pytest.mark.unit
def test_render_script_is_executable_shebang():
    assert render_script(get_spec("deepseek"), _LITERAL_TOKEN).startswith("#!/bin/bash")


@pytest.mark.unit
def test_render_script_empties_anthropic_api_key():
    """A real ANTHROPIC_API_KEY inherited from the shell must not win over AUTH_TOKEN."""
    body = render_script(get_spec("deepseek"), _LITERAL_TOKEN)
    assert "export ANTHROPIC_API_KEY=" in body


@pytest.mark.unit
def test_render_script_model_override_replaces_all_tiers():
    body = _rendered("deepseek", _LITERAL_TOKEN, model="custom:tag")
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
    # `--` is required: without it, `ollama launch` parses "$@" itself and
    # rejects any Claude-bound flag (e.g. `-p`) as an unknown Ollama flag.
    assert "exec ollama launch claude --model 'glm-5.2:cloud' -- \"$@\"" in body
    # the command shape does NOT export ANTHROPIC_* — ollama launch sets them
    assert "ANTHROPIC_BASE_URL" not in body
    assert "ANTHROPIC_AUTH_TOKEN" not in body
    assert "export ANTHROPIC_API_KEY=" not in body
    # bare `claude "$@"` (env-var shape's exec line) must not also be present
    assert 'claude "$@"\n' not in body


@pytest.mark.unit
def test_render_script_command_shape_model_override():
    body = _rendered("glm-ollama", model="custom:tag")
    assert "--model 'custom:tag' -- \"$@\"" in body
    assert "glm-5.2:cloud" not in body


@pytest.mark.unit
def test_render_script_command_shape_model_injection_is_neutralized():
    """--model is user input even in the command shape — must be single-quoted."""
    hostile = "x'; touch /tmp/pwned; echo '"
    body = _rendered("glm-ollama", model=hostile)
    assert f"--model '{hostile}'" not in body  # naive form would break out
    assert "'\"'\"'" in body  # escaped-quote sequence proves quoting engaged


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
    body = render_script(get_spec("deepseek"), hostile)
    naive_vulnerable_form = f"ANTHROPIC_AUTH_TOKEN='{hostile}'"
    assert naive_vulnerable_form not in body
    assert "'\"'\"'" in body  # the escaped-quote sequence proves quoting engaged


@pytest.mark.unit
def test_render_script_model_override_injection_is_neutralized():
    """--model is now user-controlled input; it must be quoted just like a token."""
    hostile = "x'; touch /tmp/pwned; echo '"
    body = _rendered("deepseek", _LITERAL_TOKEN, model=hostile)
    assert "'\"'\"'" in body


# --------------------------------------------------------------------------- #
# install_wrapper — writes an executable, idempotent, always overwrites by name.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_install_wrapper_writes_executable_script(tmp_path):
    paths = Paths.from_home(tmp_path)

    wrote = install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    assert wrote is True
    script = paths.script_for("deepseek")
    assert script.exists()
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode & stat.S_IXUSR, f"script not executable: {oct(mode)}"
    body = script.read_text(encoding="utf-8")
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in body


@pytest.mark.integration
def test_install_wrapper_idempotent(tmp_path):
    paths = Paths.from_home(tmp_path)

    first = install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)
    second = install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    assert first is True
    assert second is False  # already present, identical → no rewrite


@pytest.mark.integration
def test_install_wrapper_dry_run_writes_nothing(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)

    wrote = install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN, dry_run=True)

    assert wrote is True
    assert not paths.script_for("deepseek").exists()
    out = capsys.readouterr().out
    assert str(paths.script_for("deepseek")) in out


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
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(existing, encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    assert script.read_text(encoding="utf-8") == existing  # untouched


@pytest.mark.integration
def test_install_wrapper_force_overwrites_a_foreign_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho mine\n", encoding="utf-8")

    assert install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN, force=True) is True
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in script.read_text()


@pytest.mark.integration
def test_install_wrapper_replaces_its_own_earlier_output(tmp_path):
    """The guard must not get in the way of the normal update path."""
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    def _explode(_path):  # pragma: no cover - must not be reached
        raise AssertionError("must not ask before replacing our own wrapper")

    wrote = install_wrapper(paths, "deepseek", token="rotated-token", confirm=_explode)
    assert wrote is True
    assert "ANTHROPIC_AUTH_TOKEN='rotated-token'" in script_text(paths, "deepseek")


@pytest.mark.integration
def test_install_wrapper_non_interactive_fails_fast_instead_of_blocking(tmp_path):
    """confirm=None means "no way to ask" — must error, never read stdin.

    This is the guarantee that a scripted/CI run cannot hang waiting on input.
    """
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho mine\n", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="--force"):
        install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN, confirm=None)


@pytest.mark.integration
def test_install_wrapper_dry_run_never_asks_about_a_foreign_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    original = "#!/bin/bash\necho mine\n"
    script.write_text(original, encoding="utf-8")

    def _explode(_path):  # pragma: no cover - must not be reached
        raise AssertionError("dry-run must never prompt")

    assert (
        install_wrapper(
            paths, "deepseek", token=_LITERAL_TOKEN, dry_run=True, confirm=_explode
        )
        is True
    )
    assert script.read_text(encoding="utf-8") == original


@pytest.mark.integration
def test_is_managed_distinguishes_ours_from_foreign(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)
    assert is_managed(paths, "deepseek") is True

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
        paths, build_spec(agent="codex", provider="ollama", model="qwen3.5:9b")
    )
    paths.script_for("stranger").write_text("#!/bin/sh\n", encoding="utf-8")

    assert discover_managed(paths) == ["qwen3.5-codex"]


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
    install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    mode = stat.S_IMODE(paths.script_for("deepseek").stat().st_mode)
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
    install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    assert is_installed(paths, "deepseek") is True


@pytest.mark.integration
def test_is_installed_false_when_absent(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert is_installed(paths, "deepseek") is False


@pytest.mark.integration
def test_is_installed_true_for_any_existing_file(tmp_path):
    """Existence alone is enough — no marker/ownership check."""
    paths = Paths.from_home(tmp_path)
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\necho someone else's\n", encoding="utf-8")

    assert is_installed(paths, "deepseek") is True


@pytest.mark.integration
def test_list_wrappers_reports_installed_and_not_installed(tmp_path):
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    lines = []
    list_wrappers(paths, print_fn=lines.append)
    output = "\n".join(lines)

    assert "deepseek" in output and "installed" in output
    assert "glm" in output and "not installed" in output


# --------------------------------------------------------------------------- #
# CLI: `add` / `list` through main([...]).
# --------------------------------------------------------------------------- #

from code_helper.__main__ import main  # noqa: E402


@pytest.mark.integration
def test_cli_add_deepseek_creates_script(tmp_path):
    assert main(["add", "deepseek"]) == 0

    paths = Paths.from_home(tmp_path)
    assert paths.script_for("deepseek").exists()
    assert paths.script_for("deepseek").stat().st_mode & stat.S_IXUSR


@pytest.mark.integration
def test_cli_add_glm_reads_token_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", _SECRET_TOKEN)

    assert main(["add", "glm"]) == 0

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("glm").read_text(encoding="utf-8")
    assert _SECRET_TOKEN in body


@pytest.mark.integration
def test_cli_add_with_model_override(tmp_path):
    assert main(["add", "deepseek", "--model", "custom-model:tag"]) == 0

    paths = Paths.from_home(tmp_path)
    body = paths.script_for("deepseek").read_text(encoding="utf-8")
    assert "custom-model:tag" in body


@pytest.mark.integration
def test_cli_list_shows_registry(tmp_path, capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "deepseek" in out
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
    assert main(["--dry-run", "add", "deepseek"]) == 0
    paths = Paths.from_home(tmp_path)
    assert not paths.script_for("deepseek").exists()


# --------------------------------------------------------------------------- #
# OPENAI_TOML shape — codex via a Codex profile (~/.codex/<alias>.config.toml)
# plus a model catalog, launched as `codex --profile <alias>`.
# --------------------------------------------------------------------------- #


def _toml_spec(model: str = "glm-5.2:cloud", alias: str = "glm-5-codex") -> WrapperSpec:
    return build_spec(agent="codex", provider="ollama", model=model, alias=alias)


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
    body = openai_toml_body(spec, "/home/u/.codex/glm-5-codex.model.json")
    # Marker on line 1 — what the ownership guard keys off.
    assert body.startswith("# code-helper: managed wrapper")
    assert 'model = "glm-5.2:cloud"' in body  # `:` and `.` => must be quoted
    assert 'model_provider = "ollama-launch"' in body
    assert 'model_catalog_json = "/home/u/.codex/glm-5-codex.model.json"' in body
    assert "[model_providers.ollama-launch]" in body
    # base_url derives /v1/ from the provider's Anthropic-root base_url.
    assert 'base_url = "http://127.0.0.1:11434/v1/"' in body
    assert 'wire_api = "responses"' in body


@pytest.mark.unit
def test_openai_toml_body_quoted_model_with_colon_and_dot():
    """A model like `glm-5.2:cloud` MUST be in quotes or Codex rejects the TOML."""
    spec = _toml_spec(model="glm-5.2:cloud")
    body = openai_toml_body(spec, "/x.json")
    assert 'model = "glm-5.2:cloud"' in body
    # A double quote in the model would be escaped, not break the string.
    weird = 'we"ird'
    spec2 = _toml_spec(model=weird)
    body2 = openai_toml_body(spec2, "/x.json")
    assert 'model = "we\\"ird"' in body2


@pytest.mark.unit
def test_openai_catalog_body_has_context_window_for_unknown_model():
    """The catalog gives Codex a context window for models it does not know."""
    import json

    spec = _toml_spec(model="glm-5.2:cloud")
    payload = json.loads(openai_catalog_body(spec))
    assert payload["version"] == 1
    entry = payload["models"][0]
    assert entry["id"] == "glm-5.2:cloud"
    assert entry["context_window"] >= 1


@pytest.mark.integration
def test_install_openai_toml_writes_three_files(tmp_path):
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    assert install_wrapper(paths, spec) is True

    assert paths.script_for("glm-5-codex").exists()
    assert paths.codex_config_for("glm-5-codex").exists()
    assert paths.codex_catalog_for("glm-5-codex").exists()
    # The wrapper is executable; the config/catalog are owner-only (no token,
    # but they are our generated config — 0o600, not 0o755).
    assert stat.S_IMODE(paths.script_for("glm-5-codex").stat().st_mode) & stat.S_IXUSR
    config_mode = stat.S_IMODE(paths.codex_config_for("glm-5-codex").stat().st_mode)
    assert config_mode == 0o600
    catalog_mode = stat.S_IMODE(paths.codex_catalog_for("glm-5-codex").stat().st_mode)
    assert catalog_mode == 0o600


@pytest.mark.integration
def test_install_openai_toml_is_idempotent_across_three_files(tmp_path):
    """A byte-identical re-install of all three files is a no-op."""
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    first = install_wrapper(paths, spec)
    second = install_wrapper(paths, spec)

    assert first is True
    assert second is False


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
    assert config.read_text(encoding="utf-8").startswith("# code-helper:")


@pytest.mark.integration
def test_install_openai_toml_dry_run_writes_nothing_and_lists_three(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    spec = _toml_spec()

    assert install_wrapper(paths, spec, dry_run=True) is True

    assert not paths.script_for("glm-5-codex").exists()
    assert not paths.codex_config_for("glm-5-codex").exists()
    assert not paths.codex_catalog_for("glm-5-codex").exists()
    out = capsys.readouterr().out
    assert "would write" in out
    # All three intended writes are announced.
    assert str(paths.script_for("glm-5-codex")) in out
    assert str(paths.codex_config_for("glm-5-codex")) in out
    assert str(paths.codex_catalog_for("glm-5-codex")) in out


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
