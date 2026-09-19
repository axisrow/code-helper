"""Tests for ``codehelper disable`` / ``enable`` (issue #89).

Everything runs through ``main([...])`` so the argparse wiring (``--yes``,
``--force``, ``--dry-run`` via the shared parents) is exercised too. The
disable contract pinned here: wrappers PHYSICALLY deleted, the provider
hidden from every choice surface, tags on informational surfaces, state
written ONLY after every removal succeeded, tokens and context_windows
kept.
"""

from __future__ import annotations

import pytest

from codehelper.__main__ import main
from codehelper.services.model import get_provider
from codehelper.services.paths import Paths
from codehelper.services.spec import build_spec
from codehelper.services.state import (
    active_selection,
    context_window,
    disabled_providers,
    set_active_selection,
    set_context_window,
)
from codehelper.services.wrappers import (
    install_wrapper,
    is_installed,
    removal_discards_only_secret,
)


def _install_ollama_wrappers(paths: Paths) -> None:
    install_wrapper(paths, "deepseek-ollama", token="ollama")
    install_wrapper(paths, "glm-ollama", token="ollama")


# --------------------------------------------------------------------------- #
# dry-run
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_disable_dry_run_writes_nothing_and_prompts_nothing(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _install_ollama_wrappers(paths)
    code = main(["disable", "ollama", "--dry-run"])  # legacy spelling
    assert code == 0
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "would disable ollama-direct" in out
    assert is_installed(paths, "deepseek-ollama")
    assert disabled_providers(paths) == frozenset()


# --------------------------------------------------------------------------- #
# disable — the full blast radius
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_disable_deletes_wrappers_and_hides_the_provider(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _install_ollama_wrappers(paths)
    set_active_selection(paths, "ollama", "default")  # legacy spelling pointer
    code = main(["disable", "ollama", "--yes"])
    assert code == 0

    assert not is_installed(paths, "deepseek-ollama")
    assert not is_installed(paths, "glm-ollama")
    assert disabled_providers(paths) == frozenset({"ollama-direct"})
    assert active_selection(paths) is None  # pointer named the provider
    out = capsys.readouterr().out
    assert "disabled ollama-direct (2 wrappers removed)" in out


@pytest.mark.integration
def test_disable_keeps_tokens_and_context_windows(tmp_path):
    paths = Paths.from_home(tmp_path)
    _install_ollama_wrappers(paths)
    set_context_window(paths, "qwen3.5:9b", 0)  # provider-agnostic record
    main(["disable", "ollama-direct", "--yes"])
    assert context_window(paths, "qwen3.5:9b") == 0


@pytest.mark.integration
def test_disable_with_no_wrappers_still_writes_state(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    code = main(["disable", "ollama-direct", "--yes"])
    assert code == 0
    assert disabled_providers(paths) == frozenset({"ollama-direct"})
    assert "disabled ollama-direct" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# disable — refusals
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_disable_refuses_native():
    assert main(["disable", "native", "--yes"]) == 1


@pytest.mark.integration
def test_disable_refuses_an_already_disabled_provider():
    main(["disable", "ollama-direct", "--yes"])
    assert main(["disable", "ollama", "--yes"]) == 1  # legacy spelling, same provider


@pytest.mark.integration
def test_disable_refuses_when_a_wrapper_cannot_be_removed(tmp_path):
    """One stuck managed wrapper refuses the WHOLE disable — half-removal
    contradicts "vanishes from everything". State is untouched so the user
    can retry with --force after reading the failure."""
    paths = Paths.from_home(tmp_path)
    _install_ollama_wrappers(paths)
    # A managed marker whose file cannot be unlinked: make its PARENT read-only.
    foreign = paths.bin_dir / "stuck"
    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    foreign.write_text(
        "#!/bin/sh\n# codehelper: managed wrapper provider=ollama-direct\n"
    )
    foreign.chmod(0o444)
    paths.bin_dir.chmod(0o555)
    try:
        assert main(["disable", "ollama-direct", "--yes"]) == 1
    finally:
        paths.bin_dir.chmod(0o755)
        foreign.chmod(0o644)
    assert disabled_providers(paths) == frozenset()
    assert is_installed(paths, "deepseek-ollama")  # untouched by the refusal


# --------------------------------------------------------------------------- #
# enable
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_enable_lifts_the_disable_and_does_not_restore_wrappers(tmp_path, capsys):
    paths = Paths.from_home(tmp_path)
    _install_ollama_wrappers(paths)
    main(["disable", "ollama-direct", "--yes"])
    assert not is_installed(paths, "deepseek-ollama")

    code = main(["enable", "ollama"])
    assert code == 0
    assert disabled_providers(paths) == frozenset()
    assert not is_installed(paths, "deepseek-ollama")  # NOT restored
    assert "codehelper add" in capsys.readouterr().out


@pytest.mark.integration
def test_enable_a_not_disabled_provider_refuses():
    assert main(["enable", "ollama-direct"]) == 1


# --------------------------------------------------------------------------- #
# filtering — the provider vanishes from choice surfaces
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_refuses_a_disabled_preset_and_provider(capsys):
    main(["disable", "ollama-direct", "--yes"])
    assert main(["add", "deepseek-ollama", "--dry-run"]) == 1
    assert "disabled" in capsys.readouterr().err
    assert (
        main(
            [
                "add",
                "--agent",
                "codex",
                "--provider",
                "ollama-direct",
                "--model",
                "qwen3.5:9b",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "disabled" in capsys.readouterr().err


@pytest.mark.integration
def test_switch_refuses_a_disabled_provider(capsys):
    main(["disable", "ollama-direct", "--yes"])
    assert main(["switch", "ollama-direct", "--model", "qwen3.5:9b", "--force"]) == 1
    assert "disabled" in capsys.readouterr().err


@pytest.mark.integration
def test_list_providers_shows_the_disabled_tag(capsys):
    main(["disable", "ollama-direct", "--yes"])
    main(["list", "providers"])
    out = capsys.readouterr().out
    assert "ollama-direct" in out
    assert "(disabled)" in out


@pytest.mark.integration
def test_tokens_hides_a_disabled_providers_rows(tmp_path):
    from codehelper.cli.parser import _token_view_rows
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    save_credential(paths, "ollama-direct", "sk-ollama")
    save_credential(paths, "zai", "sk-zai")
    main(["disable", "ollama-direct", "--yes"])
    cache_rows, env_rows = _token_view_rows(paths)
    assert all(row[0] != "ollama-direct" for row in cache_rows)
    assert any(row[0] == "zai" for row in cache_rows)  # untouched provider stays
    assert all("OLLAMA" not in env_var for env_var, _value in env_rows)


@pytest.mark.integration
def test_enable_makes_add_work_again():
    main(["disable", "ollama-direct", "--yes"])
    main(["enable", "ollama-direct"])
    # Wrappers are NOT restored (see the test above) — but `add` works again.
    assert main(["add", "deepseek-ollama", "--dry-run"]) == 0


# --------------------------------------------------------------------------- #
# round-1 review fixes (cycle 1): live-config warning + only-copy token guard
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_disable_warns_when_claude_live_config_still_names_the_provider(
    tmp_path, capsys
):
    """Disable hides the provider from choice surfaces by design, but a live
    `switch` config is state — it keeps aiming claude at the retired backend.
    Disable must SAY so (with the documented `switch native` remedy), not
    leave the user auth-failing with no visible cause."""
    from codehelper.services.claude_settings import apply_switch
    from codehelper.services.model import with_auth
    from codehelper.services.secrets import save_credential
    from codehelper.services.spec import TierModels

    paths = Paths.from_home(tmp_path)
    apply_switch(
        paths,
        provider=with_auth(get_provider("zai"), want_secret=True),
        tier_models=TierModels.uniform("glm-5.3"),
        token="sk-live",
        subagent_model=None,
        context_window=None,
        force=True,
    )
    install_wrapper(paths, "glm", token="sk-live")
    save_credential(paths, "zai", "sk-live")  # cached → the only-copy guard passes
    assert main(["disable", "zai", "--yes"]) == 0
    err = capsys.readouterr().err
    assert "zai" in err
    assert "switch native" in err


@pytest.mark.integration
def test_disable_warns_when_codex_default_still_names_the_provider(tmp_path, capsys):
    from codehelper.services.codex_default import apply_set_default
    from codehelper.services.model import get_agent
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    apply_set_default(
        paths,
        agent=get_agent("codex"),
        provider=get_provider("deepseek-openai"),
        model="deepseek-v4-flash",
        force=True,
    )
    install_wrapper(
        paths,
        build_spec(agent="codex", provider="deepseek-openai", model="qwen3.5:9b"),
        token="sk-ds",
    )
    save_credential(paths, "deepseek-openai", "sk-ds")
    assert main(["disable", "deepseek-openai", "--yes"]) == 0
    err = capsys.readouterr().err
    assert "deepseek-openai" in err
    assert "set-default --restore" in err


@pytest.mark.integration
def test_disable_refuses_to_delete_the_only_copy_of_a_token(tmp_path, capsys):
    """A secret wrapper whose token was never cached (env-sourced at install,
    cache invalidated) holds the ONLY durable copy — deletion loses it for
    good. The overwrite path already refuses exactly this; disable must ask
    the same question before unlinking, and --force is the explicit override.
    A dry run refuses too — it must never promise a disable that cannot run.
    """
    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-only-copy")  # zai, secret, NOT cached

    assert main(["disable", "zai", "--dry-run"]) == 1
    assert is_installed(paths, "glm")
    assert main(["disable", "zai", "--yes"]) == 1
    assert is_installed(paths, "glm")
    assert "--force" in capsys.readouterr().err
    assert removal_discards_only_secret(paths, "glm") is True

    assert main(["disable", "zai", "--yes", "--force"]) == 0
    assert not is_installed(paths, "glm")


@pytest.mark.integration
def test_disable_proceeds_when_the_token_is_cached(tmp_path):
    """The documented promise — cached tokens survive disable — is what makes
    the guard narrow: a wrapper whose token IS in credentials.json deletes
    without any extra flag."""
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-cached")
    save_credential(paths, "zai", "sk-cached")
    assert removal_discards_only_secret(paths, "glm") is False
    assert main(["disable", "zai", "--yes"]) == 0
    assert not is_installed(paths, "glm")


# --------------------------------------------------------------------------- #
# round-2 review fixes (cycle 2): gate the other switch resolvers + edit-token
# --------------------------------------------------------------------------- #


def _switch_request(**overrides):
    from typing import Any

    from codehelper.cli.requests import SwitchRequest

    fields: dict[str, Any] = dict(
        provider=None,
        from_wrapper=None,
        model=None,
        haiku=None,
        sonnet=None,
        opus=None,
        subagent_model=None,
        base_url=None,
        auth=None,
        profile=None,
        restore=False,
        slot=None,
        status=False,
        dry_run=False,
        force=True,
        debug=False,
        from_preset=None,
    )
    fields.update(overrides)
    return SwitchRequest(**fields)


@pytest.mark.integration
def test_switch_from_preset_and_wrapper_refuse_a_disabled_provider(tmp_path):
    """The flags path has the disabled gate; the preset and wrapper resolvers
    are reached through TUI chip presses (which filter) — but they resolve
    registry/installed data that OUTLIVES the disable (preset specs, or a
    wrapper resurrected by a future path), so both carry the gate themselves.
    State is set WITHOUT the deletion step so the wrapper file exists and the
    wrapper resolver's missing-file guard cannot mask the gate."""
    from codehelper.cli.parser import _handle_switch
    from codehelper.errors import CodeHelperError
    from codehelper.services.state import set_provider_disabled

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-z")
    set_provider_disabled(paths, "zai", True, storage_names=frozenset({"zai"}))

    with pytest.raises(CodeHelperError, match="disabled"):
        _handle_switch(_switch_request(from_preset="glm"))
    with pytest.raises(CodeHelperError, match="disabled"):
        _handle_switch(_switch_request(from_wrapper="glm"))


@pytest.mark.integration
def test_edit_token_refuses_a_disabled_provider(tmp_path):
    """`edit-token glm` on a disabled provider must fail BEFORE any prompt or
    install — otherwise it recreates a wrapper `add` correctly refuses, a
    silent exception to "vanishes from every choice surface"."""
    from codehelper.cli.parser import _handle_edit_token
    from codehelper.cli.requests import EditTokenRequest
    from codehelper.errors import CodeHelperError
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-old")
    save_credential(paths, "zai", "sk-old")  # so disable passes the only-copy guard
    assert main(["disable", "zai", "--yes"]) == 0

    req = EditTokenRequest(
        name="glm",
        profile=None,
        profile_token="sk-new",
        profile_rename_from=None,
        profile_rename_to=None,
        dry_run=False,
        debug=False,
    )
    with pytest.raises(CodeHelperError, match="disabled"):
        _handle_edit_token(req)
    assert not is_installed(paths, "glm")


@pytest.mark.integration
def test_edit_token_picker_hides_a_disabled_providers_wrappers(tmp_path, monkeypatch):
    import codehelper.cli.menu as menu
    from codehelper.cli.parser import _edit_token_spec

    paths = Paths.from_home(tmp_path)
    install_wrapper(paths, "glm", token="sk-a")
    install_wrapper(paths, "bai", token="sk-l")
    from codehelper.services.secrets import save_credential

    save_credential(paths, "zai", "sk-a")  # so disable passes the only-copy guard
    captured: dict = {}

    def _capture(items, **_kwargs):
        captured["values"] = [item[0] for item in items if isinstance(item, tuple)]
        raise menu.MenuCancelled(False)  # soft cancel — picker's own back gesture

    monkeypatch.setattr(menu, "select_from_menu", _capture)

    assert _edit_token_spec(paths, None) is None
    assert "glm" in captured["values"]

    main(["disable", "zai", "--yes"])
    assert _edit_token_spec(paths, None) is None
    assert "glm" not in captured["values"]
    assert "bai" in captured["values"]  # untouched provider stays


@pytest.mark.integration
def test_enable_dry_run_refuses_a_not_disabled_provider(capsys):
    """A dry run must never promise what the real run refuses — disable's
    dry run already refuses exactly what the real run would refuse."""
    assert main(["enable", "litellm", "--dry-run"]) == 1
    assert "not disabled" in capsys.readouterr().err
