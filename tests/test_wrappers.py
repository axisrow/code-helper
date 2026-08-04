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
from code_helper.services.paths import Paths
from code_helper.services.wrappers import (
    WRAPPERS,
    get_spec,
    install_wrapper,
    is_installed,
    list_wrappers,
    render_script,
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
def test_glm_ollama_is_command_shape():
    spec = get_spec("glm-ollama")
    assert spec.launch_command == "ollama launch claude --model {model}"
    assert spec.launch_model == "glm-5.2:cloud"
    assert spec.auth == "literal"  # no token — ollama launch authenticates itself
    assert spec.base_url == ""  # env-var fields unused by command shape


@pytest.mark.unit
def test_deepseek_is_literal_auth():
    spec = get_spec("deepseek")
    assert spec.auth == "literal"
    assert spec.auth_value == "ollama"
    assert spec.base_url == "http://127.0.0.1:11434"


@pytest.mark.unit
def test_glm_is_secret_auth():
    spec = get_spec("glm")
    assert spec.auth == "secret"
    assert spec.token_env_var == "ZAI_API_KEY"
    assert spec.base_url == "https://api.z.ai/api/anthropic"


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
    body = render_script(
        get_spec("deepseek"), _LITERAL_TOKEN, model_override="custom:tag"
    )
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
    assert "exec ollama launch claude --model 'glm-5.2:cloud' \"$@\"" in body
    # the command shape does NOT export ANTHROPIC_* — ollama launch sets them
    assert "ANTHROPIC_BASE_URL" not in body
    assert "ANTHROPIC_AUTH_TOKEN" not in body
    assert "export ANTHROPIC_API_KEY=" not in body
    # bare `claude "$@"` (env-var shape's exec line) must not also be present
    assert 'claude "$@"\n' not in body


@pytest.mark.unit
def test_render_script_command_shape_model_override():
    body = render_script(get_spec("glm-ollama"), "", model_override="custom:tag")
    assert "--model 'custom:tag'" in body
    assert "glm-5.2:cloud" not in body


@pytest.mark.unit
def test_render_script_command_shape_model_injection_is_neutralized():
    """--model is user input even in the command shape — must be single-quoted."""
    hostile = "x'; touch /tmp/pwned; echo '"
    body = render_script(get_spec("glm-ollama"), "", model_override=hostile)
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
    body = render_script(get_spec("deepseek"), _LITERAL_TOKEN, model_override=hostile)
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
def test_install_wrapper_overwrites_any_existing_file(tmp_path):
    """A pre-existing ~/.local/bin/deepseek (hand-written or not) is replaced."""
    paths = Paths.from_home(tmp_path)
    existing = "#!/bin/bash\necho my own deepseek\n"
    script = paths.script_for("deepseek")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(existing, encoding="utf-8")

    wrote = install_wrapper(paths, "deepseek", token=_LITERAL_TOKEN)

    assert wrote is True
    body = script.read_text(encoding="utf-8")
    assert body != existing
    assert f"ANTHROPIC_AUTH_TOKEN='{_LITERAL_TOKEN}'" in body


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
