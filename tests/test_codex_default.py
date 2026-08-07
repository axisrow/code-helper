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
    _rotate_backups,
    apply_set_default,
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

    _rotate_backups(paths, current="already_read_by_caller = true\n")

    assert (
        paths.codex_main_config_backup(1).read_text(encoding="utf-8")
        == "already_read_by_caller = true\n"
    )


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
