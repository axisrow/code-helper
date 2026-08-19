"""Tests for ``codehelper proxy`` — the CLI seam over ``services/proxy.py``.

Everything runs through ``main([...])`` so the argparse wiring is exercised
too, mirroring ``test_cli_switch.py``. The service-level behaviour (key
precedence, the ownership boundary against ``switch``) lives in
``test_proxy.py``; what is tested here is the handler's own logic: verb
resolution, the save-before-blank ordering that makes ``on`` able to restore
an address, and the refusal to guess one when none exists.
"""

from __future__ import annotations

import json

import pytest

from codehelper.__main__ import main
from codehelper.services.paths import Paths
from codehelper.services.state import saved_proxy

_URL = "http://127.0.0.1:8118"


def _write_settings(tmp_path, env: dict) -> None:
    paths = Paths.from_home(tmp_path)
    paths.claude_dir.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(
        json.dumps({"env": env, "hooks": {"PreToolUse": []}}), encoding="utf-8"
    )


def _env(tmp_path) -> dict:
    return json.loads(
        Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")
    )["env"]


@pytest.mark.integration
def test_off_then_on_restores_the_address_through_state(tmp_path):
    """The round trip the whole feature exists for: off blanks the values, so
    the address has to survive somewhere else for on to put it back."""
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL})

    assert main(["proxy", "off", "--force"]) == 0
    assert _env(tmp_path)["HTTPS_PROXY"] == ""
    assert saved_proxy(Paths.from_home(tmp_path)) == _URL

    assert main(["proxy", "on", "--force"]) == 0
    assert _env(tmp_path)["HTTPS_PROXY"] == _URL


@pytest.mark.integration
def test_toggle_flips_whichever_way_the_file_currently_points(tmp_path):
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL})

    main(["proxy", "toggle", "--force"])
    assert _env(tmp_path)["HTTPS_PROXY"] == ""

    main(["proxy", "toggle", "--force"])
    assert _env(tmp_path)["HTTPS_PROXY"] == _URL


@pytest.mark.integration
def test_on_without_any_known_address_refuses_rather_than_guessing(tmp_path, capsys):
    _write_settings(tmp_path, {"IS_DEMO": "1"})

    assert main(["proxy", "on"]) == 1
    assert "no proxy address configured" in capsys.readouterr().err


@pytest.mark.integration
def test_url_sets_the_address_and_turns_the_proxy_on(tmp_path):
    _write_settings(tmp_path, {"IS_DEMO": "1"})

    assert main(["proxy", "--url", _URL, "--force"]) == 0

    env = _env(tmp_path)
    assert env["HTTPS_PROXY"] == _URL
    assert env["https_proxy"] == _URL
    assert env["IS_DEMO"] == "1"
    assert saved_proxy(Paths.from_home(tmp_path)) == _URL


@pytest.mark.integration
def test_url_together_with_off_is_rejected_as_contradictory(tmp_path, capsys):
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL})

    assert main(["proxy", "off", "--url", _URL]) == 1
    assert "cannot go with" in capsys.readouterr().err


@pytest.mark.integration
def test_an_invalid_url_is_rejected_by_the_cli(tmp_path, capsys):
    _write_settings(tmp_path, {"IS_DEMO": "1"})

    assert main(["proxy", "--url", "127.0.0.1:8118"]) == 1
    assert "scheme" in capsys.readouterr().err


@pytest.mark.integration
def test_no_proxy_edits_both_spellings_without_touching_the_proxy(tmp_path):
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL})

    assert main(["proxy", "--no-proxy", "localhost,.corp", "--force"]) == 0

    env = _env(tmp_path)
    assert env["NO_PROXY"] == "localhost,.corp"
    assert env["no_proxy"] == "localhost,.corp"
    assert env["HTTPS_PROXY"] == _URL


@pytest.mark.integration
def test_a_bare_proxy_command_reports_status_and_writes_nothing(tmp_path, capsys):
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL, "NO_PROXY": "localhost"})
    before = Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")

    assert main(["proxy"]) == 0

    out = capsys.readouterr().out
    assert "on" in out and _URL in out
    assert "NO_PROXY: localhost" in out
    assert (
        Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")
        == before
    )


@pytest.mark.integration
def test_status_masks_a_password_in_the_proxy_url(tmp_path, capsys):
    _write_settings(tmp_path, {"HTTPS_PROXY": "http://bob:hunter2@proxy:8118"})

    main(["proxy", "--status"])

    assert "hunter2" not in capsys.readouterr().out


@pytest.mark.integration
def test_dry_run_off_neither_writes_the_file_nor_saves_the_address(tmp_path):
    """A dry run must not leave state behind either — saving the address is
    part of the operation, not a free side effect."""
    _write_settings(tmp_path, {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL})
    before = Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")

    assert main(["proxy", "off", "--dry-run"]) == 0

    assert (
        Paths.from_home(tmp_path).claude_settings().read_text(encoding="utf-8")
        == before
    )
    assert saved_proxy(Paths.from_home(tmp_path)) is None
