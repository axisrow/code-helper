"""Tests for ``services/proxy.py`` — the ``proxy`` on/off toggle.

Two modules now patch ``~/.claude/settings.json``: ``claude_settings`` owns
the ``ANTHROPIC_*`` backend keys, this one owns the proxy keys. The most
important tests here are the two that pin that boundary from both sides —
``test_proxy_toggle_leaves_the_anthropic_keys_alone`` and
``test_switch_native_does_not_disable_the_proxy`` — because a regression in
either silently breaks something the user never asked to change: their
network, or their model backend.

The second theme is the precedence chain. Claude Code uses "the first one
that's set in the order ``https_proxy``, ``HTTPS_PROXY``, ``http_proxy``,
``HTTP_PROXY``", so an "off" that misses the lowercase spellings produces a
toggle that reports off while traffic still goes through the proxy.
"""

from __future__ import annotations

import json
import stat

import pytest

from codehelper.errors import CodeHelperError
from codehelper.services.claude_settings import MANAGED_ENV_KEYS, apply_switch
from codehelper.services.model import get_provider
from codehelper.services.paths import Paths
from codehelper.services.proxy import (
    NO_PROXY_ENV_KEYS,
    PROXY_ENV_KEYS,
    apply_proxy,
    proxy_status,
    redact_proxy_url,
    resolve_proxy_patch,
    validate_proxy_url,
)
from codehelper.services.state import saved_proxy, set_saved_proxy

_URL = "http://127.0.0.1:8118"

_ANTHROPIC_ENV = {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "sk-secret-token-value",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.6",
}

_FOREIGN_TOP_LEVEL = {
    "permissions": {"allow": ["Bash(git *)"]},
    "hooks": {"PreToolUse": []},
    "model": "opusplan",
    "autoMode": {"soft_deny": []},
}


def _write(paths: Paths, data: dict) -> None:
    paths.claude_settings().parent.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read(paths: Paths) -> dict:
    return json.loads(paths.claude_settings().read_text(encoding="utf-8"))


def _enabled_settings() -> dict:
    return {
        "env": {
            "NO_PROXY": "localhost,.z.ai",
            "no_proxy": "localhost,.z.ai",
            "HTTPS_PROXY": _URL,
            "HTTP_PROXY": _URL,
            **_ANTHROPIC_ENV,
        },
        **_FOREIGN_TOP_LEVEL,
    }


def _confirm_no(_path, _preview):
    return False


# --------------------------------------------------------------------------- #
# validate_proxy_url — pure, runs before any write
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_validate_accepts_http_and_https():
    assert validate_proxy_url(_URL) == _URL
    assert validate_proxy_url("https://proxy.corp:8080") == "https://proxy.corp:8080"


@pytest.mark.unit
def test_validate_rejects_a_url_with_no_scheme():
    """Claude Code stops launch on a proxy URL it cannot parse, so a toggle
    that accepted this would leave the agent unable to start."""
    with pytest.raises(CodeHelperError, match="no usable scheme"):
        validate_proxy_url("127.0.0.1:8118")


@pytest.mark.unit
def test_validate_rejects_socks_by_name():
    with pytest.raises(CodeHelperError, match="does not support"):
        validate_proxy_url("socks5://127.0.0.1:1080")


@pytest.mark.unit
def test_validate_rejects_a_url_with_no_host():
    with pytest.raises(CodeHelperError, match="names no host"):
        validate_proxy_url("http://")


@pytest.mark.unit
def test_validate_rejects_empty():
    with pytest.raises(CodeHelperError, match="empty"):
        validate_proxy_url("   ")


# --------------------------------------------------------------------------- #
# resolve_proxy_patch — pure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_turning_on_writes_every_key_in_the_precedence_chain():
    """A stale lowercase key would outrank the new uppercase one, so "on"
    writes the whole chain rather than only what is already present."""
    patch = resolve_proxy_patch(url=_URL, current_env={})
    assert patch == dict.fromkeys(PROXY_ENV_KEYS, _URL)


@pytest.mark.unit
def test_turning_off_blanks_the_lowercase_spellings_too():
    env = {"https_proxy": _URL, "HTTPS_PROXY": _URL, "HTTP_PROXY": _URL}
    patch = resolve_proxy_patch(url="", current_env=env)
    assert patch["https_proxy"] == ""
    assert patch["HTTPS_PROXY"] == ""
    assert patch["HTTP_PROXY"] == ""


@pytest.mark.unit
def test_turning_off_does_not_add_keys_that_were_never_set():
    """Off on a file with only the canonical pair must not sprout lowercase
    entries nobody wrote — the same reasoning as ``switch native``."""
    patch = resolve_proxy_patch(
        url="", current_env={"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL}
    )
    assert set(patch) == {"HTTPS_PROXY", "HTTP_PROXY"}


@pytest.mark.unit
def test_a_no_proxy_edit_leaves_the_proxy_chain_untouched():
    patch = resolve_proxy_patch(url=None, no_proxy="a,b", current_env={})
    assert set(patch) == set(NO_PROXY_ENV_KEYS)


@pytest.mark.unit
def test_no_proxy_is_written_to_both_spellings_with_one_value():
    """Two spellings holding DIFFERENT lists is the genuinely harmful case;
    they are always written together."""
    patch = resolve_proxy_patch(url=None, no_proxy="localhost,.corp", current_env={})
    assert patch == {"NO_PROXY": "localhost,.corp", "no_proxy": "localhost,.corp"}


# --------------------------------------------------------------------------- #
# The ownership boundary against claude_settings
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_two_owners_key_sets_are_disjoint():
    """The central invariant of a file with two owners, asserted rather than
    documented. Adding a proxy key to MANAGED_ENV_KEYS would leave BOTH
    modules' own verifications passing, and surface only as "my proxy turns
    itself off whenever I switch models" — a bug that reads as a Claude Code
    quirk rather than a codehelper one."""
    from codehelper.services.proxy import _OWNED_ENV_KEYS

    assert not set(MANAGED_ENV_KEYS) & set(_OWNED_ENV_KEYS)


@pytest.mark.unit
def test_a_foreign_credential_is_redacted_in_the_preview(tmp_path, capsys):
    """A unified diff carries CONTEXT lines, so owning only the proxy keys
    does not stop this command printing the OTHER owner's token."""
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_proxy(paths, url="", dry_run=True)

    assert "sk-secret-token-value" not in capsys.readouterr().out


@pytest.mark.unit
def test_proxy_toggle_leaves_the_anthropic_keys_alone(tmp_path):
    """THE boundary test from this side: turning the proxy off must not
    disturb the backend ``switch`` selected."""
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_proxy(paths, url="", force=True)

    env = _read(paths)["env"]
    for key in MANAGED_ENV_KEYS:
        if key in _ANTHROPIC_ENV:
            assert env[key] == _ANTHROPIC_ENV[key]


@pytest.mark.unit
def test_switch_native_does_not_disable_the_proxy(tmp_path):
    """THE boundary test from the other side: this is why the proxy keys are
    NOT in ``MANAGED_ENV_KEYS`` — ``switch native`` blanks everything it
    owns, and owning the proxy would mean wiping it on every backend change."""
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_switch(paths, provider=get_provider("native"), force=True)

    env = _read(paths)["env"]
    assert env["HTTPS_PROXY"] == _URL
    assert env["HTTP_PROXY"] == _URL
    assert env["NO_PROXY"] == "localhost,.z.ai"


@pytest.mark.unit
def test_toggling_preserves_no_proxy_and_every_top_level_key(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_proxy(paths, url="", force=True)
    apply_proxy(paths, url=_URL, force=True)

    data = _read(paths)
    assert data["env"]["NO_PROXY"] == "localhost,.z.ai"
    assert data["env"]["no_proxy"] == "localhost,.z.ai"
    for key, value in _FOREIGN_TOP_LEVEL.items():
        assert data[key] == value


# --------------------------------------------------------------------------- #
# apply_proxy — IO
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_off_blanks_values_rather_than_removing_keys(tmp_path):
    """Removing a key does not unset an already-applied value in a running
    Claude Code process; an empty string does."""
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_proxy(paths, url="", force=True)

    env = _read(paths)["env"]
    assert "HTTPS_PROXY" in env
    assert env["HTTPS_PROXY"] == ""


@pytest.mark.unit
def test_status_reports_the_key_claude_code_would_actually_use(tmp_path):
    """Lowercase wins the precedence chain, so status must report it — not
    whichever spelling happens to be written last."""
    paths = Paths.from_home(tmp_path)
    _write(
        paths,
        {"env": {"https_proxy": "http://lower:1", "HTTPS_PROXY": "http://upper:2"}},
    )

    assert proxy_status(paths).url == "http://lower:1"


@pytest.mark.unit
def test_status_never_raises_on_a_corrupt_file(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.claude_settings().parent.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text("{not json", encoding="utf-8")

    status = proxy_status(paths)
    assert status.enabled is False
    assert status.url is None


@pytest.mark.unit
def test_status_falls_back_to_the_saved_address_when_off(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"HTTPS_PROXY": "", "HTTP_PROXY": ""}})
    set_saved_proxy(paths, _URL)

    status = proxy_status(paths)
    assert status.enabled is False
    assert status.restorable_url == _URL


@pytest.mark.unit
def test_a_refused_write_does_not_bank_the_address(tmp_path):
    """Banking the address before the write is confirmed leaves state.json
    pointing at an address that was never applied — the next `proxy on`
    would then enable an endpoint the user explicitly declined."""
    from codehelper.services.proxy import set_proxy_state

    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"IS_DEMO": "1"}})

    with pytest.raises(CodeHelperError):
        set_proxy_state(paths, url=_URL, confirm=_confirm_no)

    assert saved_proxy(paths) is None


@pytest.mark.unit
def test_a_concurrent_change_does_not_bank_a_stale_address(tmp_path, monkeypatch):
    """`off` reads the live address, banks it, then writes. If another writer
    switches the proxy in between, the settings write is refused — so the
    proxy is never left off while pointing at a superseded endpoint."""
    from codehelper.services.proxy import set_proxy_state

    paths = Paths.from_home(tmp_path)
    _write(
        paths,
        {"env": {"HTTPS_PROXY": "http://old:8118", "HTTP_PROXY": "http://old:8118"}},
    )

    # The race window is between apply_proxy's own read and its locked write —
    # `confirm` is called inside exactly that window, so switching the file
    # here reproduces a concurrent writer landing mid-operation.
    def _confirm_then_meddle(_path, _preview):
        _write(
            paths,
            {
                "env": {
                    "HTTPS_PROXY": "http://new:9999",
                    "HTTP_PROXY": "http://new:9999",
                }
            },
        )
        return True

    with pytest.raises(CodeHelperError, match="changed since it was read"):
        set_proxy_state(paths, action="off", confirm=_confirm_then_meddle)

    # The settings write is rejected, so the proxy stays ON at whatever the
    # concurrent writer set — no half-applied state where the live proxy is
    # off but pointing somewhere unexpected.
    assert _read(paths)["env"]["HTTPS_PROXY"] == "http://new:9999"
    assert proxy_status(paths).enabled is True

    # The bank may hold the superseded address (it is written before the
    # settings patch, so `off` never destroys an unbanked value). That is the
    # deliberate trade: a stale SPARE beats a lost address. It is also
    # unreachable while a live address exists — `restorable_url` prefers the
    # live file — and the next successful `off` overwrites it.
    assert proxy_status(paths).restorable_url == "http://new:9999"


@pytest.mark.unit
def test_the_state_file_is_tightened_when_it_holds_a_proxy_password(tmp_path):
    """`atomic_write` preserves an existing file's mode, so a state.json that
    already sat at 0644 would keep a basic-auth proxy password world-readable
    — the exact exposure settings.json is written 0600 to avoid."""
    import os
    import stat as stat_mod

    paths = Paths.from_home(tmp_path)
    state = paths.state_file()
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text('{"active": {"provider": "zai", "profile": "default"}}')
    os.chmod(state, 0o644)

    set_saved_proxy(paths, "http://bob:hunter2@proxy.corp:8118")

    assert stat_mod.S_IMODE(state.stat().st_mode) == 0o600


@pytest.mark.unit
def test_turning_off_a_file_with_no_proxy_keys_is_a_no_op(tmp_path):
    """Blanking keys nobody ever set adds entries and burns a backup slot for
    nothing — the same reasoning as `switch native`'s current_env handling."""
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"IS_DEMO": "1"}})

    assert apply_proxy(paths, url="", force=True) is False

    assert _read(paths)["env"] == {"IS_DEMO": "1"}
    assert not paths.claude_settings_backup(1).exists()


@pytest.mark.unit
def test_on_recovers_the_address_from_the_settings_backup(tmp_path):
    """`off` blanks the live values and banks the address in state.json. If
    that bank write never landed — disk full, a lost update, a hand-cleared
    state.json — the address would be unrecoverable, because settings.json no
    longer holds it. It IS still in the backup `off` just rotated, so `on`
    reads there rather than giving up."""
    from codehelper.services.proxy import set_proxy_state

    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL}})

    set_proxy_state(paths, action="off", force=True)
    # Simulate the bank write never having landed.
    paths.state_file().unlink(missing_ok=True)
    assert saved_proxy(paths) is None
    assert proxy_status(paths).url is None

    set_proxy_state(paths, action="on", force=True)

    assert _read(paths)["env"]["HTTPS_PROXY"] == _URL


@pytest.mark.unit
def test_status_offers_the_backup_address_when_state_was_lost(tmp_path):
    """`proxy` (status) must not report "no address configured" while the
    address is sitting one file over — that is what sends a user hand-editing
    settings.json, which this command exists to avoid."""
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL}})

    apply_proxy(paths, url="", force=True)
    paths.state_file().unlink(missing_ok=True)

    assert proxy_status(paths).restorable_url == _URL


@pytest.mark.unit
def test_off_refuses_when_the_address_cannot_be_banked(tmp_path):
    """`off` erases the live address, so it must not run at all unless the
    address can first be stored somewhere durable. Blanking anyway and
    leaning on the settings backup does not work: the backup ring rotates on
    the very next settings write, and the address is gone with `off` having
    reported success. Refusing keeps the proxy on — a visibly unchanged
    state the user can act on — instead of a silent loss."""
    from codehelper.services.proxy import set_proxy_state

    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"HTTPS_PROXY": _URL, "HTTP_PROXY": _URL}})
    # Make lock acquisition fail while the state write itself would work.
    lock_path = paths.state_file().with_suffix(paths.state_file().suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.mkdir(exist_ok=True)

    with pytest.raises(CodeHelperError, match="could not be locked"):
        set_proxy_state(paths, action="off", force=True)

    # Nothing happened: the proxy is still on, exactly as before the command.
    assert _read(paths)["env"]["HTTPS_PROXY"] == _URL
    assert proxy_status(paths).enabled is True


@pytest.mark.unit
def test_on_still_works_when_the_bank_is_unavailable(tmp_path):
    """Only `off` destroys the address. Turning the proxy ON writes it into
    settings.json, so a bank failure there costs nothing — the value is right
    there in the live file."""
    from codehelper.services.proxy import set_proxy_state

    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"IS_DEMO": "1"}})
    lock_path = paths.state_file().with_suffix(paths.state_file().suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.mkdir(exist_ok=True)

    assert set_proxy_state(paths, url=_URL, force=True) is True

    assert _read(paths)["env"]["HTTPS_PROXY"] == _URL


@pytest.mark.unit
def test_saved_proxy_round_trips_through_state(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert saved_proxy(paths) is None
    set_saved_proxy(paths, _URL)
    assert saved_proxy(paths) == _URL


@pytest.mark.unit
def test_repeated_off_is_a_no_op_and_consumes_no_backup_slot(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    assert apply_proxy(paths, url="", force=True) is True
    first_backup = paths.claude_settings_backup(1).read_text(encoding="utf-8")

    assert apply_proxy(paths, url="", force=True) is False
    assert paths.claude_settings_backup(1).read_text(encoding="utf-8") == first_backup


@pytest.mark.unit
def test_dry_run_writes_nothing(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())
    before = paths.claude_settings().read_text(encoding="utf-8")

    assert apply_proxy(paths, url="", dry_run=True) is True

    assert paths.claude_settings().read_text(encoding="utf-8") == before
    assert "would write" in capsys.readouterr().out


@pytest.mark.unit
def test_a_refused_confirmation_writes_nothing(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())
    before = paths.claude_settings().read_text(encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        apply_proxy(paths, url="", confirm=_confirm_no)

    assert paths.claude_settings().read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_an_invalid_url_is_rejected_before_the_file_is_touched(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())
    before = paths.claude_settings().read_text(encoding="utf-8")

    with pytest.raises(CodeHelperError):
        apply_proxy(paths, url="socks5://127.0.0.1:1080", force=True)

    assert paths.claude_settings().read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_a_corrupt_settings_file_is_refused_not_replaced(tmp_path):
    """Starting from ``{}`` would trade a recoverable file for a two-key one."""
    paths = Paths.from_home(tmp_path)
    paths.claude_settings().parent.mkdir(parents=True, exist_ok=True)
    paths.claude_settings().write_text("{not json", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="not valid JSON"):
        apply_proxy(paths, url=_URL, force=True)


@pytest.mark.unit
def test_the_written_file_is_0600(tmp_path):
    """A proxy URL may embed basic-auth credentials."""
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    apply_proxy(paths, url="", force=True)

    mode = stat.S_IMODE(paths.claude_settings().stat().st_mode)
    assert mode == 0o600


@pytest.mark.unit
def test_a_concurrent_change_is_refused_rather_than_clobbered(tmp_path):
    paths = Paths.from_home(tmp_path)
    _write(paths, _enabled_settings())

    def _confirm_then_meddle(_path, _preview):
        _write(paths, {"env": {"HTTPS_PROXY": "http://someone-else:9"}})
        return True

    with pytest.raises(CodeHelperError, match="changed since it was read"):
        apply_proxy(paths, url="", confirm=_confirm_then_meddle)


# --------------------------------------------------------------------------- #
# Credential redaction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_redact_masks_the_password_and_keeps_the_rest():
    assert redact_proxy_url("http://bob:hunter2@proxy:8118") == (
        "http://bob:***@proxy:8118"
    )


@pytest.mark.unit
def test_redact_leaves_a_credential_free_url_alone():
    assert redact_proxy_url(_URL) == _URL


@pytest.mark.unit
def test_a_password_never_reaches_the_preview(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _write(paths, {"env": {"HTTPS_PROXY": "http://bob:hunter2@proxy:8118"}})

    apply_proxy(paths, url="", dry_run=True)

    assert "hunter2" not in capsys.readouterr().out
