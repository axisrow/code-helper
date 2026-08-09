"""Tests for the ``tui`` CLI subcommand (and the bare ``code-helper`` default).

Follows the ``tests/test_edit_token.py`` "CLI through main([...])" pattern:
HOME is patched by the autouse ``_isolate_home`` fixture, so
``Paths.from_home(tmp_path)`` resolves to the same directory the handler
writes to via ``Paths.default()``. The fake ``select_from_menu`` is patched
on ``code_helper.cli.menu`` (not imported at module scope by ``tui.py`` /
``parser.py``) so the patch lands — same trick as
``test_edit_token_no_name_uses_menu``.

Menu choices are returned by VALUE now that ``select_from_menu`` renders
``(value, label)`` pairs — the fake below returns whatever the caller asked
for regardless of label, exactly like the real thing.

``add``'s flow is two menu picks (wrapper, then model) rather than one pick +
a bare ``input()`` — see ``cli/tui.py``'s ``_run_add``. Every
``_menu_sequence`` that drives ``add`` therefore lists both choices; use
``"__default__"`` to keep the spec's default model, or ``"__custom__"`` plus a
patched ``builtins.input`` to supply one.
"""

from __future__ import annotations

import pytest

from code_helper.__main__ import main
from code_helper.services.paths import Paths

_NEW_TOKEN = "11111111111111111111111111111111.bbbbbbbbbbbbbbbb"


def _menu_sequence(monkeypatch, answers):
    """Patch select_from_menu to return successive `answers` on each call."""
    it = iter(answers)

    def _fake(items, **_kw):
        return next(it)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _fake)


@pytest.mark.integration
def test_tui_list_dispatches_to_handler(tmp_path, monkeypatch, capsys):
    _menu_sequence(monkeypatch, ["list", "quit"])

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out
    assert "glm" in out


@pytest.mark.integration
def test_bare_invocation_opens_tui(tmp_path, monkeypatch, capsys):
    _menu_sequence(monkeypatch, ["list", "quit"])

    assert main([]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out
    assert "glm" in out


@pytest.mark.integration
def test_tui_add_installs_same_as_cli(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "__default__", "quit"])

    assert main(["tui"]) == 0

    tui_body = (
        Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    )

    # Compare against a clean CLI install in a second, separate HOME.
    other_home = tmp_path.parent / f"{tmp_path.name}-cli-control"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    assert main(["add", "deepseek"]) == 0
    cli_body = (
        Paths.from_home(other_home).script_for("deepseek").read_text(encoding="utf-8")
    )

    assert tui_body == cli_body


@pytest.mark.integration
def test_tui_add_model_override_reaches_handler(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "__custom__", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "my-model")

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    assert "my-model" in body


@pytest.mark.integration
def test_tui_add_custom_model_empty_input_means_default(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "__custom__", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "   ")

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    assert "deepseek-v4-flash:0731-cloud" in body


@pytest.mark.integration
def test_tui_add_default_model_never_calls_input(tmp_path, monkeypatch):
    # The common case (keep the default model) must not prompt at all.
    _menu_sequence(monkeypatch, ["add", "deepseek", "__default__", "quit"])

    def _boom(_prompt):
        raise AssertionError("input() must not be called for __default__")

    monkeypatch.setattr("builtins.input", _boom)

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("deepseek").read_text(encoding="utf-8")
    assert "deepseek-v4-flash:0731-cloud" in body


@pytest.mark.integration
def test_tui_edit_token_delegates(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ZAI_API_KEY", "00000000000000000000000000000000.aaaaaaaaaaaaaaaa"
    )
    assert main(["add", "glm"]) == 0
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    _menu_sequence(monkeypatch, ["edit-token", "glm", "quit"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: _NEW_TOKEN)

    assert main(["tui"]) == 0

    body = Paths.from_home(tmp_path).script_for("glm").read_text(encoding="utf-8")
    assert _NEW_TOKEN in body


@pytest.mark.integration
def test_tui_dry_run_flag_from_cli_still_respected(tmp_path, monkeypatch):
    # There is no menu toggle anymore — `--dry-run` passed on the command
    # line before `tui` must still prevent the write.
    _menu_sequence(monkeypatch, ["add", "deepseek", "__default__", "quit"])

    assert main(["--dry-run", "tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_loops_after_command(tmp_path, monkeypatch, capsys):
    # After `list` the menu reappears and `add` runs — proof of the loop.
    _menu_sequence(monkeypatch, ["list", "add", "deepseek", "__default__", "quit"])

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out  # `list` output
    assert Paths.from_home(tmp_path).script_for("deepseek").exists()  # `add` ran


@pytest.mark.integration
def test_tui_error_returns_to_menu(tmp_path, monkeypatch, capsys):
    from code_helper.errors import CodeHelperError

    def _boom(**_kw):
        raise CodeHelperError("simulated failure")

    monkeypatch.setattr("code_helper.services.secrets.resolve_token", _boom)
    _menu_sequence(monkeypatch, ["add", "glm", "__default__", "quit"])

    assert main(["tui"]) == 0

    err = capsys.readouterr().err
    assert "error:" in err
    assert "simulated failure" in err
    # The error did not exit the loop: `quit` was reached, and nothing was written.
    assert not Paths.from_home(tmp_path).script_for("glm").exists()


@pytest.mark.integration
def test_tui_soft_cancel_on_main_menu_exits(tmp_path, monkeypatch, capsys):
    # Esc/q (a soft MenuCancelled) on the MAIN menu means "leave the TUI".
    from code_helper.cli.menu import MenuCancelled

    def _cancel(items, **_kw):
        raise MenuCancelled(hard=False)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _cancel)

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_soft_cancel_in_submenu_returns_to_main_menu(tmp_path, monkeypatch, capsys):
    # Esc/q inside `add`'s wrapper picker returns to the main menu instead of
    # exiting the whole TUI — proven by `list` running afterwards.
    # Sequence: main menu -> "add"; wrapper picker -> soft cancel (back);
    # main menu again -> "list"; main menu again -> "quit".
    from code_helper.cli.menu import MenuCancelled

    _CANCEL = object()
    sequence = iter(["add", _CANCEL, "list", "quit"])

    def _driver(items, **_kw):
        nxt = next(sequence)
        if nxt is _CANCEL:
            raise MenuCancelled(hard=False)
        return nxt

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _driver)

    assert main(["tui"]) == 0

    out = capsys.readouterr().out
    assert "deepseek" in out  # `list` ran after returning from the cancelled `add`
    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_hard_cancel_exits_from_submenu_depth(tmp_path, monkeypatch, capsys):
    # Ctrl-C (a hard MenuCancelled) from INSIDE a sub-menu exits the whole
    # TUI immediately, not just one level.
    from code_helper.cli.menu import MenuCancelled

    sequence = iter(["add"])

    def _driver(items, **_kw):
        try:
            return next(sequence)
        except StopIteration:
            raise MenuCancelled(hard=True) from None

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _driver)

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_hard_cancel_inside_edit_token_picker_exits_tui(tmp_path, monkeypatch):
    # `_handle_edit_token`'s own wrapper picker (parser.py) re-raises a hard
    # MenuCancelled instead of swallowing it — this proves that reaches all
    # the way up through `_run` to the TUI's top-level catch and exits
    # cleanly, rather than being treated as "command finished, show a pause".
    from code_helper.cli.menu import MenuCancelled

    sequence = iter(["edit-token"])

    def _driver(items, **_kw):
        try:
            return next(sequence)
        except StopIteration:
            raise MenuCancelled(hard=True) from None

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _driver)

    assert main(["tui"]) == 0


@pytest.mark.integration
def test_tui_ctrl_c_during_custom_model_input_returns_to_menu(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "__custom__", "list", "quit"])

    def _interrupt(_prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", _interrupt)

    assert main(["tui"]) == 0

    # add was aborted by Ctrl-C during the model prompt — nothing written —
    # and the loop kept going (`list` afterwards proves it did not crash).
    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_settings_toggles_debug_and_returns(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["settings", "debug", "__back__", "quit"])

    assert main(["tui"]) == 0


@pytest.mark.integration
def test_tui_add_back_from_wrapper_picker_returns_to_main_menu(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "__back__", "quit"])

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_add_back_from_model_picker_returns_to_main_menu(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["add", "deepseek", "__back__", "quit"])

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


@pytest.mark.integration
def test_tui_quit_returns_zero(tmp_path, monkeypatch):
    _menu_sequence(monkeypatch, ["quit"])

    assert main(["tui"]) == 0

    assert not Paths.from_home(tmp_path).script_for("deepseek").exists()


# --------------------------------------------------------------------------- #
# `new` — the constructor flow (agent → provider → model → alias)
# --------------------------------------------------------------------------- #


def _fake_models(monkeypatch, *names, error=None):
    """Stub the model listing so the TUI never touches the network."""
    import code_helper.services.models_api as api
    from code_helper.services.models_api import ModelListResult

    monkeypatch.setattr(
        api,
        "list_models",
        lambda provider, **kw: ModelListResult(tuple(names), "url", error),
    )


@pytest.mark.integration
def test_tui_new_builds_a_wrapper_from_the_axes(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "glm-5:cloud")
    _menu_sequence(monkeypatch, ["new", "codex", "ollama", "glm-5:cloud", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": "")  # accept the default alias

    assert main(["tui"]) == 0

    paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"
    body = paths.script_for(alias).read_text()
    # codex × ollama resolves to OPENAI_TOML by priority now.
    assert f"exec codex --profile '{alias}' \"$@\"" in body
    assert paths.codex_config_for(alias).exists()
    assert paths.codex_catalog_for(alias).exists()


@pytest.mark.integration
def test_tui_new_matches_the_cli_byte_for_byte(tmp_path, monkeypatch):
    """The 1-to-1 contract, for the constructor path as well as presets.

    Extends to all three OPENAI_TOML files: the wrapper, the TOML profile, and
    the model catalog must each match between the TUI and CLI installs.
    """
    _fake_models(monkeypatch, "glm-5:cloud")
    _menu_sequence(monkeypatch, ["new", "codex", "ollama", "glm-5:cloud", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": "")
    main(["tui"])
    tui_paths = Paths.from_home(tmp_path)
    alias = "glm-5-codex"

    other_home = tmp_path / "other"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    main(["add", "--agent", "codex", "--provider", "ollama", "--model", "glm-5:cloud"])
    cli_paths = Paths.from_home(other_home)

    # The wrapper and the catalog carry no absolute paths, so they must be
    # byte-identical. The TOML profile embeds an absolute path to the catalog
    # (`model_catalog_json`), which differs between the two homes — compare it
    # modulo that path.
    assert (
        tui_paths.script_for(alias).read_text()
        == cli_paths.script_for(alias).read_text()
    )
    assert (
        tui_paths.codex_catalog_for(alias).read_text()
        == cli_paths.codex_catalog_for(alias).read_text()
    )
    tui_toml = (
        tui_paths.codex_config_for(alias)
        .read_text()
        .replace(str(tui_paths.codex_dir), "")
    )
    cli_toml = (
        cli_paths.codex_config_for(alias)
        .read_text()
        .replace(str(cli_paths.codex_dir), "")
    )
    assert tui_toml == cli_toml


@pytest.mark.integration
def test_tui_new_typed_alias_wins(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "glm-5:cloud")
    _menu_sequence(monkeypatch, ["new", "codex", "ollama", "glm-5:cloud", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": "mycodex")

    main(["tui"])
    assert Paths.from_home(tmp_path).script_for("mycodex").exists()


@pytest.mark.integration
def test_tui_new_offers_manual_entry_when_the_daemon_is_down(tmp_path, monkeypatch):
    """A missing daemon must degrade to typing a model, not block the flow."""
    _fake_models(monkeypatch, error="could not reach ollama")
    _menu_sequence(monkeypatch, ["new", "codex", "ollama", "__custom__", "quit"])
    typed = iter(["qwen3.5:9b", ""])  # model, then accept the default alias
    monkeypatch.setattr("builtins.input", lambda _p="": next(typed))

    assert main(["tui"]) == 0
    assert Paths.from_home(tmp_path).script_for("qwen3.5-codex").exists()


@pytest.mark.integration
def test_tui_new_only_offers_compatible_providers(tmp_path, monkeypatch):
    """codex cannot use z.ai, so z.ai must not even appear in its picker."""
    _fake_models(monkeypatch, "m")
    seen: list[list] = []

    def _fake(items, **_kw):
        seen.append([value for value, _label in items])
        # agent, provider, model, then quit
        return {0: "new", 1: "codex", 2: "ollama", 3: "m"}.get(len(seen) - 1, "quit")

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _fake)
    monkeypatch.setattr("builtins.input", lambda _p="": "")
    main(["tui"])

    provider_menu = seen[2]
    assert "ollama" in provider_menu
    assert "zai" not in provider_menu


@pytest.mark.integration
def test_tui_new_back_from_each_step_returns_to_menu(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "m")
    for sequence in (
        ["new", "__back__", "quit"],
        ["new", "codex", "__back__", "quit"],
        ["new", "codex", "ollama", "__back__", "quit"],
    ):
        _menu_sequence(monkeypatch, sequence)
        assert main(["tui"]) == 0
        assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_tui_new_ctrl_c_at_alias_prompt_returns_to_menu(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "glm-5:cloud")
    _menu_sequence(monkeypatch, ["new", "codex", "ollama", "glm-5:cloud", "quit"])

    def _interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)

    assert main(["tui"]) == 0
    assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_tui_preset_still_works_after_a_new_run(tmp_path, monkeypatch):
    """The loop reuses one Namespace — `new` must not leave `add` in axis mode."""
    _fake_models(monkeypatch, "glm-5:cloud")
    _menu_sequence(
        monkeypatch,
        [
            "new",
            "codex",
            "ollama",
            "glm-5:cloud",
            "add",
            "glm-ollama",
            "__default__",
            "quit",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda _p="": "")

    assert main(["tui"]) == 0
    paths = Paths.from_home(tmp_path)
    assert paths.script_for("glm-5-codex").exists()
    assert paths.script_for("glm-ollama").exists()


# --------------------------------------------------------------------------- #
# `new` — runtime base_url prompt (litellm)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_tui_new_asks_for_the_url_before_listing_models(tmp_path, monkeypatch):
    """The order is load-bearing: list_models must see the typed URL, not an
    empty one — pinned by RECORDING what list_models actually received."""
    import code_helper.services.models_api as api
    from code_helper.services.models_api import ModelListResult

    seen = {}

    def _recording_list_models(provider, **_kw):
        seen["base_url"] = provider.base_url
        return ModelListResult(("gpt-4o",), "url", None)

    monkeypatch.setattr(api, "list_models", _recording_list_models)
    _menu_sequence(monkeypatch, ["new", "claude", "litellm", "gpt-4o", "quit"])
    typed = iter(["http://h:4000/v1", ""])  # base URL, then accept default alias
    monkeypatch.setattr("builtins.input", lambda _p="": next(typed))
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")

    assert main(["tui"]) == 0
    assert seen["base_url"] == "http://h:4000/v1"


@pytest.mark.integration
def test_tui_new_ignores_a_cached_token_for_a_runtime_address_provider(
    tmp_path, monkeypatch
):
    """The TUI's model picker must NOT hand a cached token to a REQUIRED-policy
    provider (litellm) — its ``base_url`` can point anywhere the user types,
    so a token cached under the provider name alone must not follow it there.
    See ``token_for_discovery``'s ``base_url_policy`` gate in secrets.py."""
    import code_helper.services.models_api as api
    import code_helper.services.secrets as secrets
    from code_helper.services.models_api import ModelListResult
    from code_helper.services.paths import Paths as _Paths

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    secrets.save_credential(_Paths.from_home(tmp_path), "litellm", "sk-cached")

    seen = {}

    def _recording_list_models(provider, *, token=""):
        seen["token"] = token
        return ModelListResult(("gpt-4o",), "url", None)

    monkeypatch.setattr(api, "list_models", _recording_list_models)
    _menu_sequence(monkeypatch, ["new", "claude", "litellm", "gpt-4o", "quit"])
    typed = iter(["http://h:4000/v1", ""])  # base URL, then accept default alias
    monkeypatch.setattr("builtins.input", lambda _p="": next(typed))

    assert main(["tui"]) == 0
    assert seen["token"] == ""


@pytest.mark.integration
def test_tui_new_litellm_matches_the_cli_byte_for_byte(tmp_path, monkeypatch):
    """The 1-to-1 contract for the new --base-url flag, codex × litellm."""
    _fake_models(monkeypatch, "gpt-4o")
    _menu_sequence(monkeypatch, ["new", "codex", "litellm", "gpt-4o", "quit"])
    typed = iter(["http://h:4000/v1", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(typed))
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    main(["tui"])
    tui_paths = Paths.from_home(tmp_path)
    alias = "gpt-4o-codex"

    other_home = tmp_path / "other"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    main(
        [
            "add",
            "--agent",
            "codex",
            "--provider",
            "litellm",
            "--base-url",
            "http://h:4000/v1",
            "--model",
            "gpt-4o",
        ]
    )
    cli_paths = Paths.from_home(other_home)

    assert (
        tui_paths.script_for(alias).read_text()
        == cli_paths.script_for(alias).read_text()
    )
    assert (
        tui_paths.codex_catalog_for(alias).read_text()
        == cli_paths.codex_catalog_for(alias).read_text()
    )
    tui_toml = (
        tui_paths.codex_config_for(alias)
        .read_text()
        .replace(str(tui_paths.codex_dir), "")
    )
    cli_toml = (
        cli_paths.codex_config_for(alias)
        .read_text()
        .replace(str(cli_paths.codex_dir), "")
    )
    assert tui_toml == cli_toml


@pytest.mark.integration
def test_tui_new_rejects_a_bad_base_url_and_returns_to_menu(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "gpt-4o")
    _menu_sequence(monkeypatch, ["new", "claude", "litellm", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": "ftp://bad")

    assert main(["tui"]) == 0
    assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_tui_new_empty_base_url_cancels(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "gpt-4o")
    _menu_sequence(monkeypatch, ["new", "claude", "litellm", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": "")

    assert main(["tui"]) == 0
    assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_tui_new_ctrl_c_at_the_base_url_prompt_returns_to_menu(tmp_path, monkeypatch):
    _fake_models(monkeypatch, "gpt-4o")
    _menu_sequence(monkeypatch, ["new", "claude", "litellm", "quit"])

    def _interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)

    assert main(["tui"]) == 0
    assert not Paths.from_home(tmp_path).bin_dir.exists()


@pytest.mark.integration
def test_tui_add_clears_base_url_between_runs(tmp_path, monkeypatch):
    """The loop reuses one Namespace — a litellm `new` run must not leave
    args.base_url set for a later preset `add` (which rejects --base-url)."""
    _fake_models(monkeypatch, "gpt-4o")
    _menu_sequence(
        monkeypatch,
        [
            "new",
            "claude",
            "litellm",
            "gpt-4o",
            "add",
            "deepseek",
            "__default__",
            "quit",
        ],
    )
    typed = iter(["http://h:4000/v1", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(typed))
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")

    assert main(["tui"]) == 0
    paths = Paths.from_home(tmp_path)
    assert paths.script_for("gpt-4o-claude").exists()
    assert paths.script_for("deepseek").exists()


@pytest.mark.unit
def test_hint_never_promises_more_digits_than_the_menu_assigns():
    """Regression: a 21-item menu advertised "1-21" but only 1-9 worked.

    Caught only on a real terminal — every other TUI test injects
    select_from_menu and never renders a hint against a long list.
    """
    from code_helper.cli.menu import MAX_DIGIT_ITEMS
    from code_helper.cli.tui import _hint

    assert f"1-{MAX_DIGIT_ITEMS}" in _hint(21, exit_word="назад")
    assert "1-21" not in _hint(21, exit_word="назад")
    assert "1-3" in _hint(3, exit_word="назад")  # short menus unaffected


@pytest.mark.integration
def test_tui_new_long_model_list_is_capped_and_says_so(tmp_path, monkeypatch):
    """All models must stay reachable: the extras move behind manual entry."""
    from code_helper.cli.menu import MAX_DIGIT_ITEMS

    _fake_models(monkeypatch, *[f"m{i}" for i in range(20)])
    seen: list[list[str]] = []
    answers = iter(["new", "codex", "ollama", "__back__", "quit"])

    def _fake(items, **_kw):
        seen.append([label for _v, label in items])
        return next(answers)

    monkeypatch.setattr("code_helper.cli.menu.select_from_menu", _fake)
    main(["tui"])

    model_menu = seen[3]
    # MAX_DIGIT_ITEMS models + "manual" + "back"
    assert len(model_menu) == MAX_DIGIT_ITEMS + 2
    assert any("ещё 11" in label for label in model_menu)
