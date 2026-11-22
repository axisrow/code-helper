import json
from io import StringIO
from types import SimpleNamespace

import pytest
from conftest import write_settings

from codehelper.__main__ import main
from codehelper.services import claude_desktop
from codehelper.services.paths import Paths

_CONFIG = "claude_desktop_config.json"


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_desktop_restore_reloads_preferences_after_quit(tmp_path, monkeypatch, capsys):
    paths = Paths.from_home(tmp_path)
    normal = paths.claude_desktop_app_dir() / _CONFIG
    third_party = paths.claude_desktop_app_dir(third_party=True) / _CONFIG
    library = third_party.parent / "configLibrary"
    meta = library / "_meta.json"
    ollama_id = claude_desktop.OLLAMA_PROFILE_ID
    profile = library / f"{ollama_id}.json"
    other = {"id": "unrelated-profile", "name": "User Gateway"}
    _write_json(normal, {"deploymentMode": "3p", "keep": 42})
    _write_json(third_party, {"deploymentMode": "3p"})
    _write_json(
        meta,
        {
            "appliedId": ollama_id,
            "entries": [other, {"id": ollama_id, "name": "Ollama"}],
        },
    )
    _write_json(
        profile,
        {
            "inferenceProvider": "gateway",
            "inferenceGatewayBaseUrl": "http://127.0.0.1:11435",
            "inferenceGatewayApiKey": "preserve-existing-key",
            "disableDeploymentModeChooser": True,
            "autoModeEnabled": True,
            "preserved": True,
        },
    )
    before = normal.read_bytes()
    original_profile = profile.read_bytes()

    assert main(["desktop", "--restore", "--dry-run"]) == 0
    preview = capsys.readouterr().out
    assert "inferenceGatewayBaseUrl" in preview
    assert "preserve-existing-key" not in preview
    assert normal.read_bytes() == before
    assert profile.read_bytes() == original_profile

    quit_modes = []
    open_modes = []

    def quit_desktop():
        current = _read_json(normal)
        quit_modes.append(current["deploymentMode"])
        _write_json(normal, dict(current, last_session="saved at shutdown"))

    def open_desktop(_paths):
        open_modes.append(_read_json(normal)["deploymentMode"])

    monkeypatch.setattr(claude_desktop, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(claude_desktop, "_desktop_is_running", lambda: True)
    monkeypatch.setattr(claude_desktop, "_quit_desktop", quit_desktop)
    monkeypatch.setattr(claude_desktop, "_open_desktop", open_desktop)

    assert main(["desktop", "--restore", "--force"]) == 0
    assert _read_json(profile) == {
        "disableDeploymentModeChooser": False,
        "inferenceGatewayApiKey": "preserve-existing-key",
        "preserved": True,
    }
    assert _read_json(meta) == {"entries": [other]}
    assert _read_json(third_party)["deploymentMode"] == "1p"
    assert _read_json(normal)["keep"] == 42
    assert _read_json(normal)["last_session"] == "saved at shutdown"
    assert main(["desktop", "--restore", "--force"]) == 0
    assert quit_modes == ["3p"]
    assert open_modes == ["1p"]
    assert not paths.claude_settings().exists()


@pytest.mark.integration
def test_desktop_restore_leaves_a_different_gateway_in_place(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    normal = paths.claude_desktop_app_dir() / _CONFIG
    meta = (
        paths.claude_desktop_app_dir(third_party=True) / "configLibrary" / "_meta.json"
    )
    _write_json(normal, {"deploymentMode": "3p"})
    _write_json(meta, {"appliedId": "unrelated-profile"})
    before = normal.read_bytes()

    assert main(["desktop", "--restore", "--force"]) == 1
    assert "restore the active Desktop profile" in capsys.readouterr().err
    assert normal.read_bytes() == before


@pytest.mark.integration
def test_desktop_gateway_discovers_models_and_restores_native(
    tmp_path, monkeypatch, capsys
):
    paths = Paths.from_home(tmp_path)
    normal = paths.claude_desktop_app_dir() / _CONFIG
    third_party = paths.claude_desktop_app_dir(third_party=True) / _CONFIG
    library = third_party.parent / "configLibrary"
    meta = library / "_meta.json"
    profile = library / f"{claude_desktop.GATEWAY_PROFILE_ID}.json"
    other = {"id": "unrelated-profile", "name": "Other Gateway"}
    _write_json(normal, {"deploymentMode": "1p", "keep": 42})
    _write_json(meta, {"entries": [other]})
    write_settings(paths, {"model": "sonnet"})
    original_cli = paths.claude_settings().read_bytes()
    history = paths.claude_dir / "projects" / "session.jsonl"
    history.parent.mkdir()
    history.write_text('{"message":"original history"}\n')
    original_history = history.read_bytes()

    key = "private-gateway-key"
    arguments = ["desktop", "--gateway", "https://gateway.test/anthropic"]
    monkeypatch.setattr("sys.stdin", StringIO(f"{key}\n"))
    assert main([*arguments, "--token-stdin", "--dry-run"]) == 0
    assert key not in capsys.readouterr().out
    assert not profile.exists()

    quits = []
    launches = []

    def quit_desktop():
        current = _read_json(normal)
        quits.append(current["deploymentMode"])
        _write_json(normal, dict(current, saved_at_shutdown=True))

    monkeypatch.setattr(claude_desktop, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(claude_desktop, "_desktop_is_running", lambda: True)
    monkeypatch.setattr(claude_desktop, "_quit_desktop", quit_desktop)
    monkeypatch.setattr(
        claude_desktop,
        "_open_desktop",
        lambda _: launches.append(_read_json(normal)["deploymentMode"]),
    )
    monkeypatch.setattr("sys.stdin", StringIO(f"{key}\n"))
    assert main([*arguments, "--token-stdin", "--force"]) == 0
    assert key not in capsys.readouterr().out
    assert profile.stat().st_mode & 0o777 == 0o600
    assert _read_json(profile) == {
        "inferenceProvider": "gateway",
        "inferenceGatewayBaseUrl": "https://gateway.test/anthropic",
        "inferenceGatewayApiKey": key,
        "inferenceGatewayAuthScheme": "bearer",
        "deploymentDisplayName": "Codehelper Gateway",
        "modelDiscoveryEnabled": True,
    }
    assert claude_desktop.desktop_api_host(paths) == "gateway.test"
    assert _read_json(third_party)["deploymentMode"] == "3p"
    assert _read_json(normal)["saved_at_shutdown"] is True
    assert _read_json(meta)["entries"][0] == other

    assert main([*arguments, "--force"]) == 0
    assert quits == ["1p"]
    assert _read_json(profile)["inferenceGatewayApiKey"] == key
    assert len(_read_json(meta)["entries"]) == 2

    assert main(["desktop", "--restore", "--force"]) == 0
    assert _read_json(normal) == {
        "deploymentMode": "1p",
        "keep": 42,
        "saved_at_shutdown": True,
    }
    assert _read_json(third_party)["deploymentMode"] == "1p"
    assert "appliedId" not in _read_json(meta)
    assert _read_json(meta)["entries"][0] == other
    assert _read_json(profile)["inferenceGatewayApiKey"] == key
    assert paths.claude_settings().read_bytes() == original_cli
    assert history.read_bytes() == original_history
    assert quits == ["1p", "3p"]
    assert launches == ["3p", "3p", "1p"]


@pytest.mark.integration
def test_desktop_gateway_rolls_back_a_partial_profile_change(
    tmp_path, monkeypatch, capsys
):
    paths = Paths.from_home(tmp_path)
    normal = paths.claude_desktop_app_dir() / _CONFIG
    third_party = paths.claude_desktop_app_dir(third_party=True) / _CONFIG
    library = third_party.parent / "configLibrary"
    meta = library / "_meta.json"
    profile = library / f"{claude_desktop.GATEWAY_PROFILE_ID}.json"
    _write_json(normal, {"deploymentMode": "1p"})
    _write_json(third_party, {"deploymentMode": "1p", "keep": 123})
    _write_json(meta, {"entries": [{"id": "other-profile"}]})
    before = {path: path.read_bytes() for path in (normal, third_party, meta)}
    write = claude_desktop.DesktopEdit.write

    def fail_last(edit):
        if edit.path == normal:
            raise OSError("cannot update native deployment mode")
        write(edit)

    monkeypatch.setattr(claude_desktop, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(claude_desktop, "_desktop_is_running", lambda: False)
    monkeypatch.setattr(claude_desktop.DesktopEdit, "write", fail_last)
    monkeypatch.setattr(
        claude_desktop, "_open_desktop", lambda _: pytest.fail("partial profile opened")
    )

    assert main(["desktop", "--gateway", "http://localhost:11435", "--force"]) == 1
    assert "Claude Desktop profile change failed" in capsys.readouterr().err
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not profile.exists()


@pytest.mark.integration
def test_desktop_gateway_rejects_credentials_in_url(tmp_path, capsys):
    assert (
        main(
            [
                "desktop",
                "--gateway",
                "https://user:private-key@gateway.test",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "private-key" not in capsys.readouterr().err
    assert not Paths.from_home(tmp_path).claude_desktop_app_dir().exists()
