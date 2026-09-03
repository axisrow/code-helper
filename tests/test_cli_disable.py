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
from codehelper.services.paths import Paths
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
def test_disable_refuses_native(tmp_path):
    assert main(["disable", "native", "--yes"]) == 1


@pytest.mark.integration
def test_disable_refuses_an_already_disabled_provider(tmp_path):
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
def test_enable_a_not_disabled_provider_refuses(tmp_path):
    assert main(["enable", "ollama-direct"]) == 1


# --------------------------------------------------------------------------- #
# filtering — the provider vanishes from choice surfaces
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_add_refuses_a_disabled_preset_and_provider(tmp_path, capsys):
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
def test_switch_refuses_a_disabled_provider(tmp_path, capsys):
    main(["disable", "ollama-direct", "--yes"])
    assert main(["switch", "ollama-direct", "--model", "qwen3.5:9b", "--force"]) == 1
    assert "disabled" in capsys.readouterr().err


@pytest.mark.integration
def test_list_providers_shows_the_disabled_tag(tmp_path, capsys):
    main(["disable", "ollama-direct", "--yes"])
    main(["list", "providers"])
    out = capsys.readouterr().out
    assert "ollama-direct" in out
    assert "(disabled)" in out


@pytest.mark.integration
def test_tokens_hides_a_disabled_providers_rows(tmp_path, capsys):
    from codehelper.cli.parser import _token_view_rows
    from codehelper.services.secrets import save_credential

    paths = Paths.from_home(tmp_path)
    save_credential(paths, "ollama-direct", "default", "sk-ollama")
    save_credential(paths, "zai", "default", "sk-zai")
    main(["disable", "ollama-direct", "--yes"])
    cache_rows, env_rows = _token_view_rows(paths)
    assert all(row[0] != "ollama-direct" for row in cache_rows)
    assert any(row[0] == "zai" for row in cache_rows)  # untouched provider stays
    assert all("OLLAMA" not in env_var for env_var, _value in env_rows)


@pytest.mark.integration
def test_enable_makes_add_work_again(tmp_path):
    main(["disable", "ollama-direct", "--yes"])
    main(["enable", "ollama-direct"])
    # Wrappers are NOT restored (see the test above) — but `add` works again.
    assert main(["add", "deepseek-ollama", "--dry-run"]) == 0
