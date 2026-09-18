"""Tests for ``services/doctor.py`` — the read-only health report.

Every test injects ``environ`` / ``launchctl_fn`` / ``fetch`` so nothing here
touches the real process environment, launchd, or the network (same seam
pattern as ``test_models_api.py``). File-touching cases are
``@pytest.mark.integration`` against a ``tmp_path`` HOME (autouse
``_isolate_home``); pure cases are ``@pytest.mark.unit``.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from codehelper.services import doctor, secrets
from codehelper.services.doctor import CheckRow, DoctorReport
from codehelper.services.model import get_provider
from codehelper.services.paths import Paths

FREELLMAPI = get_provider("freellmapi")


def _write_config(paths: Paths, text: str) -> None:
    cfg = paths.codex_main_config()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")


def _write_settings(paths: Paths, env: dict) -> None:
    path = paths.claude_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": env}), encoding="utf-8")


def _no_gui(_var: str) -> str:
    return ""


def _gui_set(_var: str) -> str:
    return "from-launchctl"


def _fetch_models(url: str, timeout: float, token: str) -> bytes:
    return b'{"data": [{"id": "auto"}, {"id": "fusion"}]}'


def _fetch_down(url: str, timeout: float, token: str) -> bytes:
    raise urllib.error.URLError("connection refused")


def _free_config(env_key_line: str = 'env_key = "FREELLMAPI_API_KEY"\n') -> str:
    return (
        'model = "auto"\n'
        'model_provider = "freellmapi"\n'
        "\n"
        "[model_providers.freellmapi]\n"
        'name = "FreeLLMAPI local proxy"\n'
        'base_url = "http://127.0.0.1:3002/v1/"\n'
        'wire_api = "responses"\n' + env_key_line
    )


def _litellm_config() -> str:
    """A REQUIRED-policy provider: the registry base_url is empty, the real
    address lives in the config.toml table (what `set-default --base-url`
    writes)."""
    return (
        'model_provider = "litellm"\n'
        "\n"
        "[model_providers.litellm]\n"
        'name = "LiteLLM proxy"\n'
        'base_url = "http://127.0.0.1:9911/v1/"\n'
        'wire_api = "responses"\n'
        'env_key = "LITELLM_API_KEY"\n'
    )


def _row(report: DoctorReport, name: str) -> CheckRow:
    return next(r for r in report.rows if r.name == name)


# ---------------------------------------------------------------------------
# config.toml: presence, parseability, native/custom providers
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_missing_config_is_a_neutral_answer(tmp_path):
    paths = Paths.from_home(tmp_path)
    report = doctor.run(paths, environ={}, launchctl_fn=_no_gui, fetch=_fetch_models)
    row = _row(report, "codex default")
    assert row.status == "ok"
    assert "not found" in row.detail
    # No live provider -> no probe row, no failure.
    assert not report.has_failure
    assert not any("probe" in r.name for r in report.rows)


@pytest.mark.integration
def test_unparseable_config_warns(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, "not toml ][")
    row = _row(doctor.run(paths, environ={}, launchctl_fn=_no_gui), "codex default")
    assert row.status == "warn"
    assert "TOML" in row.detail


@pytest.mark.integration
def test_native_provider_is_ok(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, 'model = "x"\n')
    row = _row(doctor.run(paths, environ={}, launchctl_fn=_no_gui), "codex default")
    assert row.status == "ok"
    assert "native" in row.detail


@pytest.mark.integration
def test_non_secret_provider_needs_no_token(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(
        paths,
        'model_provider = "ollama-direct"\n'
        "\n"
        "[model_providers.ollama-direct]\n"
        'base_url = "http://127.0.0.1:11434/v1/"\n'
        'wire_api = "responses"\n',
    )
    row = _row(
        doctor.run(paths, environ={}, launchctl_fn=_no_gui, fetch=_fetch_models),
        "codex default",
    )
    assert row.status == "ok"
    assert "no token" in row.detail


@pytest.mark.integration
def test_custom_provider_with_env_key_gets_its_variable_checked(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(
        paths,
        'model_provider = "my-proxy"\n'
        "\n"
        "[model_providers.my-proxy]\n"
        'base_url = "http://127.0.0.1:9000/v1/"\n'
        'wire_api = "responses"\n'
        'env_key = "MY_KEY"\n',
    )
    report = doctor.run(
        paths,
        environ={"MY_KEY": "x"},
        launchctl_fn=_no_gui,
    )
    assert _row(report, "codex default").status == "ok"
    assert _row(report, "MY_KEY").status == "ok"


# ---------------------------------------------------------------------------
# the env_key chain — the 401 bug class this doctor exists for
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_secret_provider_without_env_key_fails(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config(env_key_line=""))
    row = _row(
        doctor.run(paths, environ={}, launchctl_fn=_no_gui, fetch=_fetch_models),
        "codex default",
    )
    assert row.status == "fail"
    assert "401" in row.detail
    assert "set-default" in row.hint


@pytest.mark.integration
def test_mismatched_env_key_fails(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config('env_key = "WRONG_KEY"\n'))
    row = _row(
        doctor.run(paths, environ={}, launchctl_fn=_no_gui, fetch=_fetch_models),
        "codex default",
    )
    assert row.status == "fail"
    assert "registry" in row.detail


@pytest.mark.integration
def test_env_key_plus_env_var_is_clean(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    report = doctor.run(
        paths,
        environ={"FREELLMAPI_API_KEY": "tok-123456789"},
        launchctl_fn=_no_gui,
        fetch=_fetch_models,
    )
    assert _row(report, "codex default").status == "ok"
    source = _row(report, "FREELLMAPI_API_KEY")
    assert source.status == "ok"
    # Masked head+tail only — the raw value must not reach the report.
    assert "tok-****6789" in source.detail
    assert "tok-123456789" not in source.detail
    assert not report.has_failure


@pytest.mark.integration
def test_env_var_only_in_launchctl_warns(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    row = _row(
        doctor.run(paths, environ={}, launchctl_fn=_gui_set, fetch=_fetch_models),
        "FREELLMAPI_API_KEY",
    )
    assert row.status == "warn"
    assert "launchctl" in row.detail
    assert "zshrc" in row.hint


@pytest.mark.integration
def test_env_var_missing_everywhere_fails(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    row = _row(
        doctor.run(paths, environ={}, launchctl_fn=_no_gui, fetch=_fetch_models),
        "FREELLMAPI_API_KEY",
    )
    assert row.status == "fail"
    assert "Missing environment variable" in row.detail


@pytest.mark.integration
def test_env_cache_conflict_becomes_a_warning_row(tmp_path, monkeypatch):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    monkeypatch.setattr(
        secrets, "credential_for", lambda paths_, name, profile=None: "cached-value"
    )
    report = doctor.run(
        paths,
        environ={"FREELLMAPI_API_KEY": "env-value"},
        launchctl_fn=_no_gui,
        fetch=_fetch_models,
    )
    rows = [r for r in report.rows if r.name == "FREELLMAPI_API_KEY"]
    assert [r.status for r in rows] == ["ok", "warn"]
    assert "differs from the cached" in rows[1].detail


# ---------------------------------------------------------------------------
# claude settings coherence
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_claude_settings_absent_is_native_ok(tmp_path):
    paths = Paths.from_home(tmp_path)
    row = _row(doctor.run(paths, environ={}, launchctl_fn=_no_gui), "claude settings")
    assert row.status == "ok"


@pytest.mark.integration
def test_claude_base_url_without_token_fails(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_settings(paths, {"ANTHROPIC_BASE_URL": "https://api.example.com"})
    row = _row(doctor.run(paths, environ={}, launchctl_fn=_no_gui), "claude settings")
    assert row.status == "fail"
    assert "no key" in row.detail


@pytest.mark.integration
def test_claude_base_url_with_token_is_ok(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_settings(
        paths,
        {"ANTHROPIC_BASE_URL": "https://api.example.com", "ANTHROPIC_AUTH_TOKEN": "t"},
    )
    row = _row(doctor.run(paths, environ={}, launchctl_fn=_no_gui), "claude settings")
    assert row.status == "ok"
    assert "switched to" in row.detail


# ---------------------------------------------------------------------------
# live probe through models_api (fetch injected, never raises)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_probe_ok_lists_models(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    report = doctor.run(
        paths,
        environ={"FREELLMAPI_API_KEY": "tok"},
        launchctl_fn=_no_gui,
        fetch=_fetch_models,
    )
    row = _row(report, "freellmapi: probe")
    assert row.status == "ok"
    assert "2 models" in row.detail


@pytest.mark.integration
def test_probe_degradation_is_a_warning_not_an_error(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    report = doctor.run(
        paths,
        environ={"FREELLMAPI_API_KEY": "tok"},
        launchctl_fn=_no_gui,
        fetch=_fetch_down,
    )
    row = _row(report, "freellmapi: probe")
    assert row.status == "warn"
    assert "discovery unavailable" in row.detail
    # A WARN probe must not fail the report — only FAIL rows do.
    assert not report.has_failure


@pytest.mark.integration
def test_probe_carries_the_resolved_token(tmp_path):
    seen: dict = {}

    def fetch(url: str, timeout: float, token: str) -> bytes:
        seen["token"] = token
        return b'{"data": []}'

    paths = Paths.from_home(tmp_path)
    _write_config(paths, _free_config())
    doctor.run(
        paths,
        environ={"FREELLMAPI_API_KEY": "env-value"},
        launchctl_fn=_no_gui,
        fetch=fetch,
    )
    assert seen["token"] == "env-value"


@pytest.mark.integration
def test_probe_uses_the_base_url_recorded_in_config_toml(tmp_path):
    """The probe must test the endpoint codex will actually hit: for a
    REQUIRED-policy provider (litellm) the registry base_url is an empty
    placeholder and the real address lives in the config.toml table —
    probing the registry entry would WARN "no base URL configured" on a
    perfectly healthy setup."""
    seen: dict = {}

    def fetch(url: str, timeout: float, token: str) -> bytes:
        seen["url"] = url
        return b'{"data": [{"id": "m1"}]}'

    paths = Paths.from_home(tmp_path)
    _write_config(paths, _litellm_config())
    report = doctor.run(
        paths,
        environ={"LITELLM_API_KEY": "tok"},
        launchctl_fn=_no_gui,
        fetch=fetch,
    )
    row = _row(report, "litellm: probe")
    assert row.status == "ok"
    assert seen["url"] == "http://127.0.0.1:9911/v1/models"


@pytest.mark.integration
def test_env_cache_conflict_respects_the_provider_policy(tmp_path, monkeypatch):
    """A non-fixed policy never consults the default cache
    (env_cache_conflict's own gate): the "unset the env var to use the
    profile" remedy is FALSE for a REQUIRED-policy provider — unsetting
    would prompt, not select the profile — so no conflict row may appear."""
    paths = Paths.from_home(tmp_path)
    _write_config(paths, _litellm_config())
    monkeypatch.setattr(
        secrets, "credential_for", lambda paths_, name, profile=None: "cached-value"
    )
    report = doctor.run(
        paths,
        environ={"LITELLM_API_KEY": "env-value"},
        launchctl_fn=_no_gui,
        fetch=_fetch_models,
    )
    rows = [r for r in report.rows if r.name == "LITELLM_API_KEY"]
    assert [r.status for r in rows] == ["ok"]


# ---------------------------------------------------------------------------
# pure semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_has_failure_is_fail_only():
    assert DoctorReport((CheckRow("ok", "a", "d"),)).has_failure is False
    assert DoctorReport((CheckRow("warn", "a", "d"),)).has_failure is False
    assert DoctorReport((CheckRow("fail", "a", "d"),)).has_failure is True


@pytest.mark.unit
def test_launchctl_getenv_reads_as_empty_without_the_binary(monkeypatch):
    # A missing launchctl (non-macOS) is an answer, not an error — and the
    # unit seam proves the never-raise branch without spawning anything.
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    assert doctor._launchctl_getenv("CODEHELPER_NOT_SET_ANYWHERE_XYZ") == ""
