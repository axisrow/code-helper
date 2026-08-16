"""Tests for ``services/codex_default.py`` — the ``set-default`` command.

Unlike every other write path in this project, ``config.toml`` here is
ALWAYS a foreign file by definition — it belongs to Codex, not code-helper —
so the guard shape is different from ``wrappers.py``'s ownership marker: this
module PATCHES specific keys/tables and must leave everything else
byte-for-byte untouched, rather than deciding whole-file skip/write/refuse.
"""

from __future__ import annotations

import pytest

from code_helper.errors import CodeHelperError
from code_helper.services.codex_default import (
    DefaultPatch,
    _config_backup_slots,
    _rotate_backups,
    apply_set_default,
    clear_default,
    current_default,
    patch_config_toml,
    resolve_default_patch,
    restore_default,
)
from code_helper.services.model import ConfigShape, Provider, get_agent, get_provider
from code_helper.services.paths import Paths

CODEX = get_agent("codex")
CLAUDE = get_agent("claude")
OLLAMA = get_provider("ollama")


def _patch(**overrides) -> DefaultPatch:
    base = dict(
        model="glm-5.2:cloud",
        provider_table="ollama",
        display_name="local Ollama daemon",
        base_url="http://127.0.0.1:11434/v1/",
        wire_api="responses",
        catalog_json="/home/user/.codex/model.json",
    )
    base.update(overrides)
    return DefaultPatch(**base)


# ---------------------------------------------------------------------------
# Pure patcher: patch_config_toml
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_patch_from_scratch_when_file_absent():
    result = patch_config_toml("", _patch())
    assert 'model = "glm-5.2:cloud"' in result
    assert 'model_provider = "ollama"' in result
    assert 'model_catalog_json = "/home/user/.codex/model.json"' in result
    assert "[model_providers.ollama]" in result
    assert 'name = "local Ollama daemon"' in result
    assert 'base_url = "http://127.0.0.1:11434/v1/"' in result
    assert 'wire_api = "responses"' in result


@pytest.mark.unit
def test_patch_adds_missing_keys_without_disturbing_unrelated_content():
    original = 'some_other_key = "x"\n\n[projects."/repo"]\ntrust_level = "trusted"\n'
    result = patch_config_toml(original, _patch())
    assert 'some_other_key = "x"' in result
    assert '[projects."/repo"]' in result
    assert 'trust_level = "trusted"' in result
    # The unrelated table's content must be untouched verbatim.
    assert '[projects."/repo"]\ntrust_level = "trusted"' in result


@pytest.mark.unit
def test_patch_replaces_existing_top_level_keys_in_place_not_duplicated():
    original = (
        "# a comment above\n"
        'model = "old-model"\n'
        'model_provider = "old-provider"\n'
        'model_catalog_json = "/old/path.json"\n'
        "# a comment below\n"
    )
    result = patch_config_toml(original, _patch())
    assert result.count("model = ") == 1
    assert result.count("model_provider = ") == 1
    assert result.count("model_catalog_json = ") == 1
    assert '"old-model"' not in result
    assert "# a comment above" in result
    assert "# a comment below" in result


@pytest.mark.unit
def test_patch_replaces_existing_model_providers_table_wholesale():
    original = (
        "[model_providers.ollama]\n"
        'name = "Old Name"\n'
        'base_url = "http://old:1234/v1/"\n'
        'wire_api = "chat"\n'
    )
    result = patch_config_toml(original, _patch())
    assert result.count("[model_providers.ollama]") == 1
    assert "Old Name" not in result
    assert 'name = "local Ollama daemon"' in result


@pytest.mark.unit
def test_patch_does_not_touch_sibling_model_providers_table():
    original = (
        "[model_providers.other]\n"
        'name = "Other"\n'
        'base_url = "http://other/v1/"\n'
        'wire_api = "chat"\n'
        "\n"
        "[model_providers.ollama]\n"
        'name = "Old Name"\n'
        'base_url = "http://old:1234/v1/"\n'
        'wire_api = "chat"\n'
    )
    result = patch_config_toml(original, _patch())
    assert "[model_providers.other]" in result
    assert 'name = "Other"' in result
    assert 'base_url = "http://other/v1/"' in result


@pytest.mark.unit
def test_patch_does_not_swallow_network_table_placed_right_after():
    original = (
        "[model_providers.ollama]\n"
        'name = "Old Name"\n'
        'base_url = "http://old:1234/v1/"\n'
        'wire_api = "chat"\n'
        "\n"
        "[network]\n"
        'proxy_url = "http://proxy:8080"\n'
    )
    result = patch_config_toml(original, _patch())
    assert "[network]" in result
    assert 'proxy_url = "http://proxy:8080"' in result


@pytest.mark.unit
def test_patch_is_idempotent():
    original = 'some_other_key = "x"\n\n[projects."/repo"]\ntrust_level = "trusted"\n'
    once = patch_config_toml(original, _patch())
    twice = patch_config_toml(once, _patch())
    assert once == twice


@pytest.mark.unit
def test_patch_preserves_a_realistic_multi_table_file_byte_for_byte_elsewhere():
    original = (
        "# top comment\n"
        "some_flag = true\n"
        "\n"
        '[projects."/Users/me/repo-a"]\n'
        'trust_level = "trusted"\n'
        "\n"
        '[projects."/Users/me/repo-b"]\n'
        'trust_level = "untrusted"\n'
        "\n"
        "[mcp_servers.foo]\n"
        'command = "foo-mcp"\n'
        "args = []\n"
        "\n"
        "[hooks]\n"
        'on_start = "echo hi"\n'
    )
    result = patch_config_toml(original, _patch())
    for untouched in (
        "# top comment",
        "some_flag = true",
        '[projects."/Users/me/repo-a"]',
        'trust_level = "trusted"',
        '[projects."/Users/me/repo-b"]',
        'trust_level = "untrusted"',
        "[mcp_servers.foo]",
        'command = "foo-mcp"',
        "args = []",
        "[hooks]",
        'on_start = "echo hi"',
    ):
        assert untouched in result


@pytest.mark.unit
def test_patch_appends_table_with_exactly_one_blank_line_regardless_of_trailing_newlines():
    """The EOF-append branch (:func:`_patch_model_providers_table`) must

    normalize to exactly one blank-line separator no matter how many trailing
    newlines the text it receives already has — a prior version left a
    two-blank-line tail untouched for 3+ trailing newlines, since neither of
    its two sequential ``endswith`` checks fired. Exercised directly (not via
    the full ``patch_config_toml`` two-step pipeline) so the assertion is
    about this one function's normalization, independent of what
    ``_patch_top_level`` prepends first.
    """
    from code_helper.services.codex_default import _patch_model_providers_table

    for trailing in ("", "\n", "\n\n", "\n\n\n", "\n\n\n\n"):
        original = f"some_other_key = 1{trailing}" if trailing else ""
        result = _patch_model_providers_table(original, _patch())
        expected_sep = "\n\n" if trailing else ""
        prefix = f"some_other_key = 1{expected_sep}" if trailing else ""
        assert result.startswith(prefix + "[model_providers.ollama]"), (
            trailing,
            result,
        )


@pytest.mark.unit
def test_patch_handles_a_model_value_with_a_control_character():
    """A ``--model``/``--catalog-json`` value containing a control character

    (which ``toml_string`` encodes as ``\\uXXXX``) must not crash the
    in-place-replacement path: ``re.subn`` treats a plain string replacement
    as a backreference TEMPLATE and ``\\u`` is not a valid escape in one, so a
    prior version raised ``re.error`` instead of producing valid TOML.
    """
    original = 'model = "old"\n'
    patch = _patch(model="glm\x01bad")
    result = patch_config_toml(original, patch)
    assert 'model = "glm\\u0001bad"' in result


@pytest.mark.unit
def test_patch_does_not_splice_into_an_unindented_multi_line_array():
    """A top-level multi-line array whose continuation lines start at column

    0 with ``[`` (no leading indentation) must not be mistaken for a table
    header — the managed keys must land AFTER the array closes, not spliced
    into the middle of it, and the result must be valid TOML.
    """
    tomllib = pytest.importorskip("tomllib")
    original = (
        'some_array = [\n[1, 2],\n[3, 4],\n]\n\n[model_providers.other]\nname = "x"\n'
    )
    patch = _patch()
    result = patch_config_toml(original, patch)

    # The array body is untouched — nothing was inserted between its `[` and
    # its first element.
    assert "some_array = [\n[1, 2],\n[3, 4],\n]\n" in result
    # The managed keys landed after the array closed, not inside it.
    assert result.index("]\n") < result.index('model = "glm-5.2:cloud"')
    # And the whole thing parses as valid TOML.
    tomllib.loads(result)


# ---------------------------------------------------------------------------
# resolve_default_patch — axis validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_default_patch_rejects_claude():
    # claude declares no OPENAI_TOML shape at all — resolve_shape's own
    # "cannot use shape" message (it DOES share OTHER shapes with ollama,
    # just not this one), the same error `add --shape openai-toml` would hit.
    with pytest.raises(CodeHelperError, match="cannot use shape openai-toml"):
        resolve_default_patch(CLAUDE, OLLAMA, "glm-5.2:cloud", "/x/model.json")


@pytest.mark.unit
def test_resolve_default_patch_rejects_missing_wire_api():
    bad_provider = Provider(
        name="ollama",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="http://127.0.0.1:11434",
        wire_api="",  # invalid — bypasses model._validate_registries since
        # this object is constructed directly, not through PROVIDERS.
    )
    with pytest.raises(CodeHelperError, match="invalid wire_api"):
        resolve_default_patch(CODEX, bad_provider, "glm-5.2:cloud", "/x/model.json")


@pytest.mark.unit
def test_resolve_default_patch_ok_for_codex_ollama():
    result = resolve_default_patch(CODEX, OLLAMA, "glm-5.2:cloud", "/x/model.json")
    assert result.model == "glm-5.2:cloud"
    assert result.provider_table == "ollama"
    assert result.base_url == "http://127.0.0.1:11434/v1/"
    assert result.wire_api == "responses"


# ---------------------------------------------------------------------------
# Orchestrator: apply_set_default
# ---------------------------------------------------------------------------


def _install(paths, *, model="glm-5.2:cloud", force=True, **kw):
    return apply_set_default(
        paths, agent=CODEX, provider=OLLAMA, model=model, force=force, **kw
    )


@pytest.mark.integration
def test_set_default_dry_run_writes_nothing(tmp_path):
    paths = Paths.from_home(tmp_path)
    wrote = _install(paths, dry_run=True)
    assert wrote is True  # would-write is still "there's a change"
    assert not paths.codex_main_config().exists()
    assert not paths.codex_main_config_backup(1).exists()
    assert not (paths.codex_dir / "model.json").exists()


@pytest.mark.integration
def test_set_default_dry_run_never_prompts_or_refuses_without_force(tmp_path):
    """``--dry-run`` alone (no ``--force``, ``confirm=None``, off-a-TTY

    behavior) must preview, not raise — same contract ``add``/``edit-token``
    already pin for ``wrappers._install_plan`` (dry_run checked BEFORE any
    confirm/force gate). A prior version of ``apply_set_default`` checked the
    confirm/force gate first, so this exact call raised instead of previewing.
    """
    paths = Paths.from_home(tmp_path)
    wrote = apply_set_default(
        paths, agent=CODEX, provider=OLLAMA, model="glm-5.2:cloud", dry_run=True
    )
    assert wrote is True
    assert not paths.codex_main_config().exists()
    assert not paths.codex_main_config_backup(1).exists()
    assert not (paths.codex_dir / "model.json").exists()


@pytest.mark.integration
def test_set_default_catalog_dry_run_never_refuses_a_foreign_catalog(tmp_path):
    """Same contract as above, for the catalog's own ownership guard: a

    hand-curated (foreign) ``model.json`` must not turn ``--dry-run`` into a
    refusal.
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, force=True)  # config side already a no-op below

    catalog_path = paths.codex_dir / "model.json"
    catalog_path.write_text('{"hand": "curated"}', encoding="utf-8")

    wrote = apply_set_default(
        paths, agent=CODEX, provider=OLLAMA, model="glm-5.2:cloud", dry_run=True
    )
    assert wrote is True
    assert catalog_path.read_text(encoding="utf-8") == '{"hand": "curated"}'


@pytest.mark.integration
def test_set_default_creates_backup_containing_original_bytes(tmp_path):
    paths = Paths.from_home(tmp_path)
    original = 'some_other_key = "x"\n'
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(original, encoding="utf-8")

    _install(paths)

    assert paths.codex_main_config_backup(1).read_text(encoding="utf-8") == original
    assert 'model = "glm-5.2:cloud"' in paths.codex_main_config().read_text(
        encoding="utf-8"
    )


@pytest.mark.integration
def test_set_default_second_noop_run_does_not_recreate_backup(tmp_path):
    paths = Paths.from_home(tmp_path)
    original = 'some_other_key = "x"\n'
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(original, encoding="utf-8")

    _install(paths)
    backup_after_first = paths.codex_main_config_backup(1).read_text(encoding="utf-8")

    wrote = _install(paths)  # identical args — true no-op on the config side
    assert (
        paths.codex_main_config_backup(1).read_text(encoding="utf-8")
        == backup_after_first
    )
    # The catalog write is also idempotent by this point, so overall no-op.
    assert wrote is False


@pytest.mark.integration
def test_set_default_second_different_run_rotates_backup_one_step(tmp_path):
    paths = Paths.from_home(tmp_path)
    original = 'some_other_key = "x"\n'
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(original, encoding="utf-8")

    _install(paths, model="glm-5.2:cloud")
    after_first = paths.codex_main_config().read_text(encoding="utf-8")

    _install(paths, model="glm-5.3:cloud")
    assert paths.codex_main_config_backup(1).read_text(encoding="utf-8") == after_first
    assert "glm-5.3:cloud" in paths.codex_main_config().read_text(encoding="utf-8")


@pytest.mark.integration
def test_set_default_three_runs_rotate_all_three_slots(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text("", encoding="utf-8")

    _install(paths, model="model-a")
    after_a = paths.codex_main_config().read_text(encoding="utf-8")
    _install(paths, model="model-b")
    after_b = paths.codex_main_config().read_text(encoding="utf-8")
    _install(paths, model="model-c")

    assert paths.codex_main_config_backup(1).read_text(encoding="utf-8") == after_b
    assert paths.codex_main_config_backup(2).read_text(encoding="utf-8") == after_a


@pytest.mark.integration
def test_set_default_refuses_without_force_off_tty(tmp_path):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="--force"):
        _install(paths, force=False)
    assert not paths.codex_main_config().exists()


@pytest.mark.integration
def test_set_default_force_writes_on_non_tty(tmp_path):
    paths = Paths.from_home(tmp_path)
    wrote = _install(paths, force=True)
    assert wrote is True
    assert paths.codex_main_config().exists()


@pytest.mark.integration
def test_set_default_writes_catalog_and_refuses_a_foreign_one(tmp_path):
    paths = Paths.from_home(tmp_path)
    # First establish the config side with --force so it is already a no-op
    # on the next call — isolates the assertion to the CATALOG refusal path.
    _install(paths, force=True)

    catalog_path = paths.codex_dir / "model.json"
    catalog_path.write_text('{"hand": "curated"}', encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        _install(paths, force=False)
    assert catalog_path.read_text(encoding="utf-8") == '{"hand": "curated"}'

    _install(paths, force=True)
    assert "hand" not in catalog_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_set_default_leaves_config_untouched_when_catalog_write_is_refused(
    tmp_path,
):
    """If the catalog write refuses (foreign catalog, no ``--force``),

    ``config.toml`` must be left exactly as it was before this call — no
    half-applied state where the config already points at
    ``model_catalog_json`` but the catalog itself was never updated. The
    catalog is the referenced artifact, so it must be written/verified
    BEFORE config.toml is patched, not after.
    """
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    original_config = 'some_other_key = "x"\n'
    paths.codex_main_config().write_text(original_config, encoding="utf-8")

    catalog_path = paths.codex_dir / "model.json"
    catalog_path.write_text('{"hand": "curated"}', encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing to overwrite"):
        _install(paths, force=False)

    # Neither file changed, and no backup was rotated — the config write
    # never happened because the catalog it depends on was refused first.
    assert paths.codex_main_config().read_text(encoding="utf-8") == original_config
    assert catalog_path.read_text(encoding="utf-8") == '{"hand": "curated"}'
    assert not paths.codex_main_config_backup(1).exists()


@pytest.mark.integration
def test_set_default_catalog_overwrite_of_managed_catalog_requires_confirm(tmp_path):
    """A SECOND ``set-default`` overwriting our OWN previously-managed

    catalog (the common "change the default model" case, not a foreign file)
    must still go through the confirm/force gate — round 1 only gated
    FOREIGN catalogs, so a normal model switch silently clobbered the
    previous default's catalog with zero confirmation and zero backup.
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, model="glm-4.7", force=True)
    catalog_path = paths.codex_dir / "model.json"
    assert "glm-4.7" in catalog_path.read_text(encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        _install(paths, model="glm-5.2:cloud", force=False)
    # Refused before any write — the previous catalog survives untouched.
    assert "glm-4.7" in catalog_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_set_default_restore_also_restores_the_paired_catalog(tmp_path):
    """``--restore`` must undo BOTH halves of a ``set-default`` — config.toml

    AND the catalog it references — or a restored config can end up pointing
    at ``model_catalog_json`` while the catalog itself still describes the
    model the restore was supposed to move away from.
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, model="glm-4.7", force=True)
    catalog_path = paths.codex_dir / "model.json"
    assert "glm-4.7" in catalog_path.read_text(encoding="utf-8")

    _install(paths, model="glm-5.2:cloud", force=True)
    assert "glm-5.2" in catalog_path.read_text(encoding="utf-8")
    assert "glm-4.7" not in catalog_path.read_text(encoding="utf-8")

    restore_default(paths, slot=1, force=True)

    config_text = paths.codex_main_config().read_text(encoding="utf-8")
    catalog_text = catalog_path.read_text(encoding="utf-8")
    assert "glm-4.7" in config_text
    assert "glm-5.2" not in config_text
    # The catalog must match what the restored config now claims — NOT still
    # describe the model set-default was just undone away from.
    assert "glm-4.7" in catalog_text
    assert "glm-5.2" not in catalog_text


@pytest.mark.integration
def test_set_default_restore_declining_catalog_leaves_both_files_untouched(
    tmp_path,
):
    """A declined CATALOG confirm during ``--restore`` must not leave

    config.toml already restored — both files must be gated BEFORE either is
    written, or a declined catalog restore leaves config.toml pointing at
    ``model_catalog_json`` while the catalog itself still describes the
    model the restore was supposed to move away from (and, since
    ``restore_default`` does not itself back up ``current`` before
    overwriting, the pre-restore config would be unrecoverable).
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, model="glm-4.7", force=True)
    _install(paths, model="glm-5.2:cloud", force=True)

    config_before = paths.codex_main_config().read_text(encoding="utf-8")
    catalog_path = paths.codex_dir / "model.json"
    catalog_before = catalog_path.read_text(encoding="utf-8")

    calls = []

    def _accept_config_decline_catalog(path, preview):
        calls.append(path)
        return path == paths.codex_main_config()

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        restore_default(paths, slot=1, confirm=_accept_config_decline_catalog)

    # BOTH confirms were asked (proves the gate-both-first ordering), and
    # NEITHER file was written despite the config confirm being accepted.
    assert len(calls) == 2
    assert paths.codex_main_config().read_text(encoding="utf-8") == config_before
    assert catalog_path.read_text(encoding="utf-8") == catalog_before


@pytest.mark.integration
def test_set_default_declining_config_after_accepting_catalog_leaves_both_untouched(
    tmp_path,
):
    """A declined CONFIG confirm during ``set-default`` must not leave the

    catalog already overwritten — both writes must be gated before either is
    committed, or a declined config confirm leaves the (unpatched, still
    live) config.toml referencing a catalog that was already changed to
    describe a different model.
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, model="glm-4.7", force=True)

    config_before = paths.codex_main_config().read_text(encoding="utf-8")
    catalog_path = paths.codex_dir / "model.json"
    catalog_before = catalog_path.read_text(encoding="utf-8")

    calls = []

    def _accept_catalog_decline_config(path, preview):
        calls.append(path)
        return path == catalog_path

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        _install(
            paths,
            model="glm-5.2:cloud",
            force=False,
            confirm=_accept_catalog_decline_config,
        )

    assert len(calls) == 2
    assert paths.codex_main_config().read_text(encoding="utf-8") == config_before
    assert catalog_path.read_text(encoding="utf-8") == catalog_before


@pytest.mark.integration
def test_set_default_catalog_confirm_prompt_shows_a_real_diff(tmp_path):
    """The foreign-catalog confirm callback must see an actual preview of

    what is about to change (a unified diff of old vs. new catalog content),
    not an empty string — otherwise a user overwriting a hand-curated
    ``model.json`` has no way to know what they're agreeing to.
    """
    paths = Paths.from_home(tmp_path)
    _install(paths, force=True)

    catalog_path = paths.codex_dir / "model.json"
    catalog_path.write_text('{"hand": "curated"}', encoding="utf-8")

    seen_previews = []

    def _capture_confirm(path, preview):
        seen_previews.append(preview)
        return True

    _install(paths, force=False, confirm=_capture_confirm)
    assert len(seen_previews) == 1
    assert seen_previews[0] != ""
    assert "hand" in seen_previews[0]


@pytest.mark.unit
def test_rotate_backups_archives_the_passed_in_content_not_a_fresh_disk_read(
    tmp_path,
):
    """``_rotate_backups`` must archive the ``current`` text the caller

    already read/diffed/confirmed, not re-read ``config.toml`` from disk —
    re-reading would open a TOCTOU gap where a file changed between the
    caller's read and this call gets silently archived (and then overwritten
    by ``patched``, which was derived from the OLDER read) with no trace of
    the interim change in any backup slot.
    """
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    # On-disk content differs from what the (hypothetical) caller already
    # read earlier and is passing in as `current`.
    paths.codex_main_config().write_text("on_disk = true\n", encoding="utf-8")

    _rotate_backups(
        _config_backup_slots(paths), current="already_read_by_caller = true\n"
    )

    assert (
        paths.codex_main_config_backup(1).read_text(encoding="utf-8")
        == "already_read_by_caller = true\n"
    )


@pytest.mark.unit
def test_set_default_refuses_outright_without_tomllib(tmp_path, monkeypatch):
    """Without ``tomllib`` importable, ``set-default`` must refuse outright

    with a clear message rather than silently degrading its verification —
    unlike ``wrappers.py``'s no-op-without-tomllib precedent, which is safe
    only because that module owns the files it verifies wholesale.
    ``set-default`` regex-patches a foreign, hand-maintained file, so the
    ``tomllib``-based structural check is the only net catching a corrupted
    patch before it's written; skipping it silently here is unacceptable.
    ``tomllib`` is stdlib from Python 3.11 on (this project's floor — see
    ``pyproject.toml``'s ``requires-python``), so this simulates an
    interpreter below that floor via ``builtins.__import__`` rather than
    relying on the test runner's actual Python version.
    """
    import builtins

    from code_helper.services import codex_default

    real_import = builtins.__import__

    def _no_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("No module named 'tomllib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tomllib)

    with pytest.raises(CodeHelperError, match="requires Python 3.11"):
        codex_default._require_tomllib()


@pytest.mark.integration
def test_set_default_refuses_unparseable_existing_config(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    del tomllib  # only used to gate the test on py3.11+
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text("this is not [valid toml", encoding="utf-8")

    with pytest.raises(CodeHelperError, match="does not parse as valid TOML"):
        _install(paths, force=True)


@pytest.mark.integration
def test_set_default_refuses_when_a_managed_key_name_appears_inside_a_string_value(
    tmp_path,
):
    """The regex patcher's line-anchored key search can match a line that

    merely LOOKS like a managed key inside an unrelated multi-line string
    value. ``_verify_patch_applied``'s structural (not just value) check must
    catch it and refuse before any write, rather than silently mutating
    unrelated content — reproduced directly against ``patch_config_toml``:
    the earlier occurrence (inside the string) consumes the single
    in-place-replacement slot, corrupting the string while the REAL top-level
    key is left unpatched.
    """
    tomllib = pytest.importorskip("tomllib")
    del tomllib  # only used to gate the test on py3.11+
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(
        'notes = """\nmodel_catalog_json = "fake, do not use"\n"""\n',
        encoding="utf-8",
    )

    with pytest.raises(CodeHelperError, match="code-helper bug"):
        _install(paths, force=True)
    # Nothing was written — the refusal happens before any atomic_write.
    assert not paths.codex_main_config_backup(1).exists()


@pytest.mark.integration
def test_set_default_refuses_cleanly_when_model_providers_is_an_array_of_tables(
    tmp_path,
):
    """``model_providers`` as ``[[model_providers]]`` (an array-of-tables) is

    syntactically valid TOML but not the ``[model_providers.<name>]`` shape
    this patcher understands. ``_verify_patch_applied`` must still refuse
    cleanly with ``CodeHelperError`` (the documented fail-clean contract) —
    not crash with an uncaught ``AttributeError`` from calling ``.get()`` on a
    list, the way ``data.get("model_providers", {}).get(patch.provider_table,
    {})`` did before this fix.
    """
    tomllib = pytest.importorskip("tomllib")
    del tomllib  # only used to gate the test on py3.11+
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(
        '[[model_providers]]\nname = "weird"\n', encoding="utf-8"
    )

    with pytest.raises(CodeHelperError, match="code-helper bug"):
        _install(paths, force=True)
    # Nothing was written — the refusal happens before any atomic_write.
    assert not paths.codex_main_config_backup(1).exists()


# ---------------------------------------------------------------------------
# restore_default
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_restore_puts_back_the_original(tmp_path):
    paths = Paths.from_home(tmp_path)
    original = 'some_other_key = "x"\n'
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text(original, encoding="utf-8")

    _install(paths)
    restored = restore_default(paths, slot=1, force=True)
    assert restored is True
    assert paths.codex_main_config().read_text(encoding="utf-8") == original


@pytest.mark.integration
def test_restore_specific_slot(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text("", encoding="utf-8")

    _install(paths, model="model-a")
    after_a = paths.codex_main_config().read_text(encoding="utf-8")
    _install(paths, model="model-b")  # bak1 now == after_a

    restore_default(paths, slot=1, force=True)
    assert paths.codex_main_config().read_text(encoding="utf-8") == after_a


@pytest.mark.integration
def test_restore_fails_when_no_backup_exists(tmp_path):
    paths = Paths.from_home(tmp_path)
    with pytest.raises(CodeHelperError, match="no backup found"):
        restore_default(paths, slot=1, force=True)


@pytest.mark.integration
def test_restore_respects_dry_run(tmp_path):
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text('original_key = "x"\n', encoding="utf-8")
    _install(paths)

    current_before = paths.codex_main_config().read_text(encoding="utf-8")
    restore_default(paths, slot=1, force=True, dry_run=True)
    assert paths.codex_main_config().read_text(encoding="utf-8") == current_before


@pytest.mark.integration
def test_restore_dry_run_never_prompts_or_refuses_without_force(tmp_path):
    """Same ``--dry-run``-before-confirm contract as ``apply_set_default``:

    ``--restore --dry-run`` with no ``--force`` and no confirm callback must
    preview, not raise.
    """
    paths = Paths.from_home(tmp_path)
    paths.codex_main_config().parent.mkdir(parents=True, exist_ok=True)
    paths.codex_main_config().write_text('original_key = "x"\n', encoding="utf-8")
    _install(paths)

    current_before = paths.codex_main_config().read_text(encoding="utf-8")
    wrote = restore_default(paths, slot=1, dry_run=True)
    assert wrote is True
    assert paths.codex_main_config().read_text(encoding="utf-8") == current_before


@pytest.mark.integration
def test_restore_invalid_slot_raises():
    with pytest.raises(CodeHelperError, match="invalid backup slot"):
        Paths.from_home("/tmp").codex_main_config_backup(4)


# ---------------------------------------------------------------------------
# CLI handler: --slot only makes sense together with --restore
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_slot_without_restore_is_rejected(tmp_path, monkeypatch, capsys):
    """``--slot`` is only ever read inside the ``--restore`` branch of the

    handler — passing it alongside ``--agent``/``--provider``/``--model``
    (i.e. without ``--restore``) must raise loudly instead of silently
    having no effect.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from code_helper.__main__ import main

    assert (
        main(
            [
                "set-default",
                "--agent",
                "codex",
                "--provider",
                "ollama",
                "--model",
                "glm-5.2:cloud",
                "--slot",
                "2",
            ]
        )
        == 1
    )
    assert "--slot only applies together with --restore" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Service layer: resolve_default_patch refuses a REQUIRED provider with no
# base_url (the CLI --base-url flag itself is Part 2/2 — see feat/base-url-cli)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_default_patch_refuses_a_required_provider_with_no_base_url():
    """The service-layer invariant added alongside build_spec's: a
    REQUIRED-policy provider (its base_url is the user's own server) must
    never resolve a patch with an unresolved empty base_url, independent of
    whether any CLI entry point has grown a --base-url flag yet.
    """
    with pytest.raises(CodeHelperError, match="requires a base URL"):
        resolve_default_patch(CODEX, get_provider("litellm"), "gpt-4o", "/x/model.json")


@pytest.mark.unit
def test_resolve_default_patch_uses_the_substituted_provider():
    """A unit-level pin: resolve_default_patch itself is provider-agnostic —
    it is the CALLER's job (cli/parser.py) to substitute base_url in first."""
    from code_helper.services.model import with_base_url

    litellm = with_base_url(get_provider("litellm"), "http://h:4000/v1")
    result = resolve_default_patch(CODEX, litellm, "gpt-4o", "/x/model.json")
    assert result.base_url == "http://h:4000/v1/"


# ---------------------------------------------------------------------------
# current_default — the read-only "what is applied" probe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_current_default_reads_the_applied_provider(tmp_path):
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('model = "x"\nmodel_provider = "ollama"\n', encoding="utf-8")
    assert current_default(paths) == "ollama"


@pytest.mark.unit
def test_current_default_is_none_when_nothing_is_applied(tmp_path):
    """Missing file, empty file, and a file with no model_provider all mean
    the same thing: no override is in effect."""
    paths = Paths.from_home(tmp_path)
    assert current_default(paths) is None

    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("", encoding="utf-8")
    assert current_default(paths) is None

    config.write_text('approval_policy = "on-request"\n', encoding="utf-8")
    assert current_default(paths) is None


@pytest.mark.unit
def test_current_default_never_raises_on_a_corrupt_file(tmp_path):
    """Same posture as claude_settings.current_switch and state.load_state:
    this is read on the optional UI path, where an unparseable file must
    degrade to "nothing applied" rather than crash the screen that reads it.
    """
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("this is not [[[ valid toml", encoding="utf-8")
    assert current_default(paths) is None


@pytest.mark.unit
def test_current_default_rejects_a_provider_not_in_the_registry(tmp_path):
    """A hand-written or since-removed provider name is not echoed back, so
    every non-None return is guaranteed to be a real PROVIDERS name."""
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('model_provider = "no-such-provider"\n', encoding="utf-8")
    assert current_default(paths) is None


# ---------------------------------------------------------------------------
# clear_default — codex's "native": remove the override, keep everything else
# ---------------------------------------------------------------------------


_FOREIGN_CONFIG = """\
# a comment the user wrote
approval_policy = "on-request"
model = "glm-5.2:cloud"
model_provider = "ollama"
model_catalog_json = "/home/user/.codex/model.json"

[model_providers.ollama]
name = "local Ollama daemon"
base_url = "http://127.0.0.1:11434/v1/"
wire_api = "responses"

[model_providers.mine]
name = "hand written"
base_url = "http://example"

[projects.'/home/user/work']
trust_level = "trusted"
"""


@pytest.mark.unit
def test_clear_default_removes_only_the_managed_region(tmp_path):
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_FOREIGN_CONFIG, encoding="utf-8")

    assert clear_default(paths, force=True) is True

    result = config.read_text(encoding="utf-8")
    # The managed region is gone... (matched as whole assignments, not bare
    # substrings — `model_provider` is also a prefix of the `model_providers`
    # table name the user's own entry below must keep.)
    assert "model_provider =" not in result
    assert "model_catalog_json =" not in result
    assert "model =" not in result
    assert "[model_providers.ollama]" not in result
    # ...and everything the user owns survives untouched.
    assert "# a comment the user wrote" in result
    assert 'approval_policy = "on-request"' in result
    assert "[model_providers.mine]" in result
    assert 'name = "hand written"' in result
    assert "[projects.'/home/user/work']" in result
    assert current_default(paths) is None


@pytest.mark.unit
def test_clear_default_does_not_erase_an_unrecognized_providers_model_config(tmp_path):
    # Regression: clear_default() looks up current_default(paths) only to
    # find a REGISTRY provider name for the [model_providers.<name>] table
    # to drop — but clear_config_toml() removes the `model`/`model_provider`/
    # `model_catalog_json` top-level keys UNCONDITIONALLY, regardless of
    # whether current_default() recognized the provider. A user who
    # hand-configured `model_provider = "mine"` (not in code-helper's
    # PROVIDERS registry) gets those keys silently deleted by `native`
    # anyway, even though code-helper never wrote them and current_default()
    # itself reports None (nothing of ours is applied).
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        _FOREIGN_CONFIG.replace('model_provider = "ollama"', 'model_provider = "mine"'),
        encoding="utf-8",
    )

    assert current_default(paths) is None  # "mine" is not a registry provider

    clear_default(paths, force=True)

    result = config.read_text(encoding="utf-8")
    assert 'model_provider = "mine"' in result
    assert 'model = "glm-5.2:cloud"' in result
    assert "model_catalog_json" in result


@pytest.mark.unit
def test_clear_default_is_a_no_op_when_nothing_is_applied(tmp_path):
    paths = Paths.from_home(tmp_path)
    assert clear_default(paths, force=True) is False

    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('approval_policy = "on-request"\n', encoding="utf-8")
    assert clear_default(paths, force=True) is False
    assert config.read_text(encoding="utf-8") == 'approval_policy = "on-request"\n'


@pytest.mark.unit
def test_clear_default_writes_a_backup(tmp_path):
    """A clear is exactly as recoverable as a set — same rotate_backups path."""
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_FOREIGN_CONFIG, encoding="utf-8")

    clear_default(paths, force=True)

    backup = paths.codex_main_config_backup(1)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == _FOREIGN_CONFIG


@pytest.mark.unit
def test_clear_default_refuses_without_confirmation(tmp_path):
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_FOREIGN_CONFIG, encoding="utf-8")

    with pytest.raises(CodeHelperError, match="refusing without confirmation"):
        clear_default(paths)
    assert config.read_text(encoding="utf-8") == _FOREIGN_CONFIG


@pytest.mark.unit
def test_clear_default_dry_run_never_writes(tmp_path):
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_FOREIGN_CONFIG, encoding="utf-8")

    assert clear_default(paths, dry_run=True) is True
    assert config.read_text(encoding="utf-8") == _FOREIGN_CONFIG
    assert not paths.codex_main_config_backup(1).exists()


@pytest.mark.unit
def test_clear_default_is_the_inverse_of_apply_set_default(tmp_path):
    """Round-trip: applying then clearing returns the file to content that
    carries no managed region, with the user's own keys still in place."""
    paths = Paths.from_home(tmp_path)
    config = paths.codex_main_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('approval_policy = "on-request"\n', encoding="utf-8")

    apply_set_default(
        paths, agent=CODEX, provider=OLLAMA, model="glm-5.2:cloud", force=True
    )
    assert current_default(paths) == "ollama"

    clear_default(paths, force=True)
    assert current_default(paths) is None
    result = config.read_text(encoding="utf-8")
    assert 'approval_policy = "on-request"' in result
    assert "[model_providers.ollama]" not in result
    assert "model_provider =" not in result
