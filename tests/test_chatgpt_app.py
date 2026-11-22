from types import SimpleNamespace

import pytest
from conftest import write_settings

from codehelper.__main__ import main
from codehelper.services import chatgpt_app
from codehelper.services.paths import Paths

_URL = "http://localhost:8118"


@pytest.mark.unit
def test_running_status_uses_the_bundle_id(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="true\n")

    monkeypatch.setattr(chatgpt_app.subprocess, "run", run)
    assert chatgpt_app._is_running()
    assert commands == [
        ["/usr/bin/osascript", "-e", 'application id "com.openai.codex" is running']
    ]


@pytest.mark.integration
@pytest.mark.parametrize("proxy_off", [False, True])
def test_chatgpt_uses_shared_proxy_for_app_and_runtime(
    tmp_path, monkeypatch, capsys, proxy_off
):
    paths = Paths.from_home(tmp_path)
    write_settings(paths, {"env": {"HTTPS_PROXY": _URL}})
    if proxy_off:
        assert main(["proxy", "off", "--force"]) == 0
        capsys.readouterr()
    session = paths.codex_dir / "sessions" / "original.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("original history\n")
    before = paths.claude_settings().read_bytes()
    calls = []
    monkeypatch.setattr(chatgpt_app, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(chatgpt_app, "_is_running", lambda: False)
    monkeypatch.setattr(
        chatgpt_app.subprocess, "run", lambda args, **kwargs: calls.append(args)
    )

    assert main(["chatgpt", "--dry-run"]) == 0
    assert _URL in capsys.readouterr().out
    assert calls == []
    assert main(["chatgpt"]) == 0
    (command,) = calls
    assert command[:3] == ["/usr/bin/open", "-b", "com.openai.codex"]
    assert f"HTTPS_PROXY={_URL}" in command
    assert f"--proxy-server={_URL}" in command
    assert "NO_PROXY=localhost,127.0.0.1,::1" in command
    assert not any("CODEX_HOME=" in arg for arg in command)
    assert session.read_text() == "original history\n"
    assert paths.claude_settings().read_bytes() == before


@pytest.mark.integration
def test_chatgpt_does_not_restart_a_running_app(tmp_path, monkeypatch, capsys):
    write_settings(Paths.from_home(tmp_path), {"env": {"HTTPS_PROXY": _URL}})
    monkeypatch.setattr(chatgpt_app, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(chatgpt_app, "_is_running", lambda: True)

    def unexpected_launch(*args, **kwargs):
        pytest.fail("a running app must stay open")

    monkeypatch.setattr(chatgpt_app.subprocess, "run", unexpected_launch)
    assert main(["chatgpt"]) == 1
    assert (
        "quit it, then run codehelper chatgpt from Terminal" in capsys.readouterr().err
    )


@pytest.mark.integration
def test_chatgpt_rejects_credentials_without_exposing_them(tmp_path, capsys):
    write_settings(
        Paths.from_home(tmp_path),
        {"env": {"HTTPS_PROXY": "http://user:private-secret@localhost:8118"}},
    )
    assert main(["chatgpt", "--dry-run"]) == 1
    error = capsys.readouterr().err
    assert "without embedded credentials" in error
    assert "private-secret" not in error
