"""``set-default`` — patch Codex's OWN ``~/.codex/config.toml``, in place.

Every other module in this project bends over backwards to never touch an
agent's own configuration (see ``services/paths.py``'s ``codex_dir``
docstring, ``services/render.py``'s ``openai_toml_body``). That invariant is
deliberately broken here, and ONLY here: this module is the one explicit,
honestly-named command that patches Codex's *default* — what runs when the
user types a bare ``codex``, no wrapper, no ``--profile``.

Why this cannot reuse ``services/wrappers.py``'s install machinery
--------------------------------------------------------------------
``_FilePlan``/``_decide``/``_install_plan`` are built around files this tool
creates and owns *wholesale*: a marker on line 1/2 proves authorship, and the
whole body is compared byte-for-byte to decide skip/write/refuse. Codex's
``config.toml`` is the opposite kind of file — never created by this tool, and
never fully owned by it: a real one is tens of kilobytes of ``[projects.*]``
tables, MCP server entries, hooks, and comments that must survive completely
untouched. So this module owns a different primitive: a PATCH, not an
install — read the whole file, replace only specific top-level keys and one
named table, leave every other byte exactly where it was.

The patcher (:func:`patch_config_toml`) is pure regex/string manipulation, not
a round-trip TOML parser — this project is stdlib-only (no ``tomlkit`` in
``pyproject.toml``), and even a full TOML library would drop comments and
reorder tables on a naive read-modify-write. ``tomllib`` (py3.11+, read-only in
the stdlib) is used only as a verification net before and after the patch —
never to construct the output.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.model import Agent, ConfigShape, Provider, resolve_shape
from code_helper.services.paths import Paths
from code_helper.services.render import (
    CATALOG_MANAGED_BY_KEY,
    openai_base_url,
    openai_catalog_body,
    toml_string,
)
from code_helper.services.spec import build_spec
from code_helper.services.wrappers import _catalog_self_marked

__all__ = [
    "DefaultPatch",
    "resolve_default_patch",
    "patch_config_toml",
    "diff_preview",
    "apply_set_default",
    "restore_default",
]

#: The literal alias fed to build_spec purely to obtain a WrapperSpec to hand
#: to render.openai_catalog_body — never used as a path or file name (the
#: catalog path here comes from --catalog-json / the default below, not
#: Paths.codex_catalog_for). Passes validate_alias trivially.
_CATALOG_SPEC_ALIAS = "set-default"

#: Default location of the DEFAULT catalog, distinct from the per-alias
#: <alias>.model.json the OPENAI_TOML wrapper shape writes.
_DEFAULT_CATALOG_NAME = "model.json"


@dataclass(frozen=True)
class DefaultPatch:
    """The exact values ``set-default`` writes into ``config.toml``."""

    model: str
    #: ``spec.provider.name`` — the ``[model_providers.<name>]`` key. Also the
    #: value of the top-level ``model_provider`` key.
    provider_table: str
    #: Display ``name`` inside the ``[model_providers.X]`` table.
    display_name: str
    base_url: str
    wire_api: str
    #: ``model_catalog_json``'s value, as a string (already resolved).
    catalog_json: str


def resolve_default_patch(
    agent: Agent, provider: Provider, model: str, catalog_path: str
) -> DefaultPatch:
    """Resolve a :class:`DefaultPatch` from the axes. Pure, no IO.

    Reuses :func:`resolve_shape` — the SAME compatibility check ``add`` uses —
    so ``set-default`` cannot silently accept a pairing ``add`` would reject.
    For today's registry this only ever resolves for ``codex``: ``claude``
    declares no ``OPENAI_TOML`` shape, so it fails here with the same
    "no common configuration mechanism" message a bad ``add`` invocation gets,
    not a bespoke error.

    Raises:
        CodeHelperError: incompatible pairing, or (defense-in-depth, mirrors
            ``model._validate_registries``) a provider that resolves to
            OPENAI_TOML but carries no usable ``wire_api`` — reachable only if
            a future registry entry skips import-time validation somehow.
    """
    resolve_shape(agent, provider, preferred=ConfigShape.OPENAI_TOML)

    if provider.wire_api not in ("responses", "chat"):
        raise CodeHelperError(
            f"provider {provider.name!r} declares openai-toml but has invalid "
            f"wire_api {provider.wire_api!r} (must be 'responses' or 'chat')"
        )

    return DefaultPatch(
        model=model,
        provider_table=provider.name,
        display_name=provider.description or provider.name,
        base_url=openai_base_url(provider.base_url),
        wire_api=provider.wire_api,
        catalog_json=catalog_path,
    )


#: Top-level scalar keys this command manages, in the order a fresh insert
#: appends them (matches the issue's own example layout).
_TOP_LEVEL_KEYS = ("model", "model_provider", "model_catalog_json")

#: Matches the START of the first table header — `[table]` or `[[array]]` —
#: which is where the top-level section ends. MULTILINE, anchored to line
#: start so a `[` inside a string value or a comment never matches.
_FIRST_TABLE_RE = re.compile(r"^\[", re.MULTILINE)

#: Matches ANY table header line — used to find the end of a specific named
#: table's body (its content runs until the next one of these, or EOF).
_ANY_TABLE_HEADER_RE = re.compile(r"^\[.*$", re.MULTILINE)


def _patch_value_for(key: str, patch: DefaultPatch) -> str:
    if key == "model":
        return patch.model
    if key == "model_provider":
        return patch.provider_table
    if key == "model_catalog_json":
        return patch.catalog_json
    raise AssertionError(f"unknown top-level key: {key!r}")  # pragma: no cover


def _patch_top_level(original: str, patch: DefaultPatch) -> str:
    """Replace/insert the three top-level scalar keys. Pure, no IO.

    Operates ONLY on the slice before the first ``^[`` line (or the whole file
    if there is none) — a table with a colliding-looking body can never be
    touched by this step. Each key is handled independently: replaced in
    place via an anchored, single-line, ``count=1`` regex if present, else
    appended (in :data:`_TOP_LEVEL_KEYS` order, skipping keys that were
    already found) right before the top-level/table boundary.
    """
    boundary_match = _FIRST_TABLE_RE.search(original)
    boundary = boundary_match.start() if boundary_match else len(original)
    top = original[:boundary]
    rest = original[boundary:]

    missing: list[str] = []
    for key in _TOP_LEVEL_KEYS:
        value = _patch_value_for(key, patch)
        line_re = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        replacement = f'{key} = "{toml_string(value)}"'
        # A callable replacement, NOT a template string: re.subn treats a
        # string replacement as a backreference template and interprets any
        # ``\`` in it (e.g. the ``\uXXXX`` toml_string emits for a control
        # character in a user-supplied --model/--catalog-json value) as an
        # escape — ``\u`` is not a valid one, so re.error crashes instead of
        # patching. A lambda's return value is used verbatim, no re-parsing.
        new_top, count = line_re.subn(lambda m, r=replacement: r, top, count=1)
        if count:
            top = new_top
        else:
            missing.append(replacement)

    if missing:
        # If the top-level slice is non-empty and doesn't already end in a
        # newline, add one before inserting — never glue onto a partial line.
        if top and not top.endswith("\n"):
            top += "\n"
        top += "".join(f"{line}\n" for line in missing)

    return top + rest


def _render_model_providers_table(patch: DefaultPatch) -> str:
    """The ``[model_providers.<name>]`` block body, header included.

    ONE renderer for both "insert fresh" and "replace existing" call sites in
    :func:`_patch_model_providers_table`, so a first ``set-default`` and a
    later re-run of the same arguments can never drift in what they write —
    the property idempotency (below) depends on.
    """
    table = patch.provider_table
    return (
        f"[model_providers.{table}]\n"
        f'name = "{toml_string(patch.display_name)}"\n'
        f'base_url = "{toml_string(patch.base_url)}"\n'
        f'wire_api = "{toml_string(patch.wire_api)}"\n'
    )


def _patch_model_providers_table(original: str, patch: DefaultPatch) -> str:
    """Replace/insert ``[model_providers.<name>]`` wholesale. Pure, no IO.

    Table BOUNDARY = header line through the line right before the next
    ``^[...`` header (any table, not just another ``model_providers.*`` one)
    or EOF — which is what guarantees a sibling table immediately before or
    after (``[model_providers.other]``, ``[network]``, ...) is never touched.
    """
    table = patch.provider_table
    header_re = re.compile(
        rf"^\[model_providers\.{re.escape(table)}\]\s*$", re.MULTILINE
    )
    header_match = header_re.search(original)
    rendered = _render_model_providers_table(patch)

    if header_match is None:
        # Append at EOF with exactly one blank-line separator. Normalizing
        # via rstrip (rather than two sequential endswith checks) is also
        # CORRECT for 3+ trailing newlines: the previous two-``if`` form left
        # a two-blank-line tail untouched whenever the file already ended in
        # "\n\n\n" or more, since both endswith checks were already true and
        # neither branch fired.
        prefix = original.rstrip("\n") + "\n\n" if original else ""
        return prefix + rendered

    # Body runs from the end of the header line to the next table header (any
    # table) or EOF.
    body_start = header_match.end()
    if body_start < len(original) and original[body_start] == "\n":
        body_start += 1
    next_header = _ANY_TABLE_HEADER_RE.search(original, header_match.end() + 1)
    body_end = next_header.start() if next_header else len(original)

    return original[: header_match.start()] + rendered + original[body_end:]


def patch_config_toml(original: str, patch: DefaultPatch) -> str:
    """Return ``original`` with ``patch`` applied. Pure, no IO.

    Two independent steps — top-level scalar keys, then the
    ``[model_providers.X]`` table — each anchored/regex-based, never a
    round-trip TOML parse. Idempotent by construction: applying the same
    ``patch`` twice in a row yields byte-identical output both times, because
    both steps always ask "is the target already here?" before deciding
    replace-in-place vs. append/insert, and both render their inserted value
    through the exact same encoder used for a replacement.
    """
    with_keys = _patch_top_level(original, patch)
    return _patch_model_providers_table(with_keys, patch)


def diff_preview(original: str, patched: str, *, label: str = "config.toml") -> str:
    """Unified diff of ``original`` -> ``patched``, stdlib only.

    Empty string when the two are identical (the no-op case) — callers print
    this as-is under ``--dry-run`` and before an interactive confirm. ``label``
    names the file in the diff headers — defaults to ``config.toml`` (the
    original, only caller) but ``_write_catalog`` passes its own catalog path
    so the confirm prompt shows what is actually about to change there too,
    instead of an empty preview.
    """
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"{label} (current)",
            tofile=f"{label} (new)",
        )
    )


def _read_text_or_none(path: Path) -> str | None:
    """``path``'s text, or ``None`` if it does not exist or is not UTF-8.

    Never raises — matches the safe-refuse posture ``wrappers.py`` uses
    throughout: an undecodable file is not this module's problem to diagnose,
    it is the pre-check's ("does this parse as TOML at all?") problem to
    refuse.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return None


def _require_tomllib():
    """Import and return ``tomllib``, or refuse with a clear message.

    Unlike ``wrappers._model_from_toml_profile`` — which accepts a no-op on
    py3.10 because it verifies a file this tool owns and wrote wholesale —
    ``set-default`` regex-patches the user's own hand-maintained
    ``config.toml``. Its patcher has documented blind spots (matching a
    managed-key-shaped line inside an unrelated multi-line string); the
    ``tomllib``-based structural verification in ``_verify_patch_applied`` is
    the ONLY thing that catches those before a write. Skipping it silently on
    py3.10 would mean a corrupted foreign file could be written with zero
    runtime check, which is a materially worse failure mode here than for a
    file this tool owns and can safely regenerate. So ``set-default`` refuses
    outright on py3.10 rather than degrading its safety net quietly.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        raise CodeHelperError(
            "set-default requires Python 3.11+ (needs the stdlib tomllib "
            "module to safely verify a patch to config.toml before writing "
            "it) — this interpreter is older"
        ) from None
    return tomllib


def _verify_toml_or_refuse(text: str, *, context: str) -> None:
    """Structural sanity check via ``tomllib`` — refuses before any write if

    ``text`` does not parse as TOML at all. See :func:`_require_tomllib` for
    why this is mandatory (not best-effort) for ``set-default`` specifically.
    """
    tomllib = _require_tomllib()
    try:
        tomllib.loads(text)
    except ValueError as exc:
        raise CodeHelperError(
            f"{context} does not parse as valid TOML — refusing to patch a "
            f"file I cannot verify (fix it by hand, or restore a backup with "
            f"--restore): {exc}"
        ) from None


def _without_managed_region(data: dict, patch: DefaultPatch) -> dict:
    """``data`` (an already-parsed TOML dict) with every key this patcher
    manages removed, so what's left is exactly the content that must survive
    a patch untouched.
    """
    out = {k: v for k, v in data.items() if k not in _TOP_LEVEL_KEYS}
    providers = out.get("model_providers")
    if isinstance(providers, dict) and patch.provider_table in providers:
        providers = {k: v for k, v in providers.items() if k != patch.provider_table}
        if providers:
            out["model_providers"] = providers
        else:
            out.pop("model_providers", None)
    return out


def _verify_patch_applied(original: str, patched: str, patch: DefaultPatch) -> None:
    """Re-parse ORIGINAL and PATCHED and assert two things, before any write:

    1. the patched text says what we intended (the managed keys/table carry
       the new values);
    2. everything OUTSIDE the managed region parsed identically before and
       after — a structural (not just byte) proof that the patch did not
       corrupt unrelated content. This catches the case a value-only check
       cannot: the regex patcher's line-anchored key search can, in principle,
       match a line that merely LOOKS like ``model_catalog_json = "..."``
       inside an unrelated multi-line string value and "replace" it in place
       there instead of (or as well as) the real top-level key — a parse that
       still succeeds and can even carry the right top-level values while a
       DIFFERENT part of the file was silently mutated. Comparing everything
       else's parsed structure closes that gap generically, without having to
       anticipate every way the regexes could mismatch.

    The safety net for "the regex patcher missed a corner case in someone's
    real 24 KB file": if this fails, nothing has been written yet — the
    orchestrator calls this before any ``atomic_write``. Requires ``tomllib``
    (py3.11+) — see :func:`_require_tomllib` for why this is mandatory rather
    than best-effort for ``set-default``; ``_verify_toml_or_refuse`` already
    ran earlier in the same call and would have refused on py3.10, so by the
    time this function runs ``tomllib`` is guaranteed importable.
    """
    tomllib = _require_tomllib()
    try:
        data = tomllib.loads(patched)
    except ValueError as exc:
        raise CodeHelperError(
            f"internal error: the patched config.toml does not parse as TOML "
            f"— refusing to write; this is a code-helper bug, please report "
            f"it: {exc}"
        ) from None

    providers = data.get("model_providers", {})
    table = (
        providers.get(patch.provider_table, {}) if isinstance(providers, dict) else {}
    )
    expected = {
        "model": patch.model,
        "model_provider": patch.provider_table,
        "model_catalog_json": patch.catalog_json,
    }
    actual = {k: data.get(k) for k in expected}
    table_expected = {
        "name": patch.display_name,
        "base_url": patch.base_url,
        "wire_api": patch.wire_api,
    }
    table_actual = {k: table.get(k) for k in table_expected}
    if actual != expected or table_actual != table_expected:
        raise CodeHelperError(
            "internal error: the patched config.toml does not contain the "
            "expected values — refusing to write; this is a code-helper bug, "
            "please report it"
        )

    # original is already known-parseable (_verify_toml_or_refuse ran first);
    # an empty original (fresh install) has nothing to preserve.
    if original.strip():
        original_data = tomllib.loads(original)
        if _without_managed_region(original_data, patch) != _without_managed_region(
            data, patch
        ):
            raise CodeHelperError(
                "internal error: patching config.toml appears to have altered "
                "content outside the managed keys/table — refusing to write; "
                "this is a code-helper bug, please report it"
            )


def _rotate_backups(paths: Paths, *, current: str) -> None:
    """FIFO-rotate the 3 backup slots, then archive ``current`` into slot 1.

    2->3 (oldest lost), 1->2, current->1. Runs via atomic_write for every
    slot so the crash-safety guarantee is uniform across the whole rotation,
    not just the final config.toml write. A slot that doesn't exist is simply
    skipped (no error) — the ring degrades gracefully on a fresh install.

    ``current`` is passed in by the caller (the same ``original`` text it
    already read and diffed/confirmed against) rather than re-read from disk
    here: re-reading would both waste an I/O round-trip and open a TOCTOU gap
    — on a real interactive confirm, the file on disk could have changed
    between the earlier read and this call, and archiving that different,
    unreviewed content into slot 1 while writing ``patched`` (derived from the
    ORIGINAL read) would silently discard whatever changed in between.
    """
    slot3, slot2, slot1 = (paths.codex_main_config_backup(n) for n in (3, 2, 1))
    body2 = _read_text_or_none(slot2)
    if body2 is not None:
        atomic_write(slot3, body2, mode=None)
    body1 = _read_text_or_none(slot1)
    if body1 is not None:
        atomic_write(slot2, body1, mode=None)

    atomic_write(slot1, current, mode=None)


def _resolve_catalog_path(paths: Paths, catalog_json: str | None) -> Path:
    if catalog_json:
        return Path(catalog_json)
    return paths.codex_dir / _DEFAULT_CATALOG_NAME


def _write_catalog(
    patch: DefaultPatch,
    agent: Agent,
    provider: Provider,
    catalog_path: Path,
    *,
    dry_run: bool,
    force: bool,
    confirm,
) -> bool:
    """Write the default's model catalog, respecting its own ownership guard.

    Reuses the SAME structural ``managed_by`` proof
    (``wrappers._catalog_self_marked``) the per-alias OPENAI_TOML catalogs use
    — a hand-curated ``~/.codex/model.json`` is protected exactly like a
    hand-curated ``<alias>.model.json`` would be.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — same signature as
            :func:`apply_set_default`'s ``confirm``, called with a unified
            diff of the existing catalog against the new one (via
            :func:`diff_preview`) so a foreign-catalog overwrite prompt shows
            what is about to change, same as the config.toml patch prompt.
    """
    spec = build_spec(
        agent=agent,
        provider=provider,
        model=patch.model,
        alias=_CATALOG_SPEC_ALIAS,
        shape=ConfigShape.OPENAI_TOML,
    )
    body = openai_catalog_body(spec)

    existing = _read_text_or_none(catalog_path)
    if existing == body:
        return False  # already exactly this — no-op, idempotent

    foreign = existing is not None and not _catalog_self_marked(catalog_path)

    if dry_run:
        # --dry-run never prompts and never refuses — it only previews, same
        # ordering as the config.toml patch above.
        print(f"would write {catalog_path}")
        return True

    catalog_preview = diff_preview(existing or "", body, label=str(catalog_path))
    if (
        foreign
        and not force
        and not (confirm and confirm(catalog_path, catalog_preview))
    ):
        raise CodeHelperError(
            f"{catalog_path} exists and was not created by code-helper "
            f"(missing {CATALOG_MANAGED_BY_KEY!r} marker) — refusing to "
            f"overwrite (use --force)"
        )

    atomic_write(catalog_path, body, mode=None)
    print(f"wrote {catalog_path}")
    return True


def apply_set_default(
    paths: Paths,
    *,
    agent: Agent,
    provider: Provider,
    model: str,
    catalog_json: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Patch Codex's ``~/.codex/config.toml`` to default onto ``agent``/``provider``/``model``.

    Orchestrates the whole ``set-default`` flow: pre-check -> resolve patch ->
    apply -> post-check -> write the sibling catalog through its own
    ownership guard FIRST (it's the artifact config.toml's
    ``model_catalog_json`` key references, so it must exist/be correct before
    the reference is written) -> then, if there is an actual change, rotate
    backups, confirm/force-gate, write config.toml. Mirrors
    ``wrappers.install_wrapper``'s shape (paths spec, dry_run, force, confirm)
    so the CLI handler stays a thin shell.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — called ONLY when a
            real, visible write is about to happen and neither ``--force`` nor
            an existing no-op short-circuit applies (``preview`` is a diff for
            the config patch, or ``""`` for the catalog's whole-file write).
            ``None`` behaves like "always refuse" (the fail-fast-off-a-TTY
            default the CLI layer supplies).

    Returns:
        True if anything was written (or would be, under ``--dry-run``);
        False on a true no-op (config.toml and catalog both already match).

    Raises:
        CodeHelperError: incompatible agent/provider, unparseable existing
            config.toml, a patch that fails its own post-write verification,
            or a refused overwrite (no ``--force``/confirmation).
    """
    catalog_path = _resolve_catalog_path(paths, catalog_json)
    patch = resolve_default_patch(agent, provider, model, str(catalog_path))

    config_path = paths.codex_main_config()
    original = _read_text_or_none(config_path) or ""
    _verify_toml_or_refuse(original, context=str(config_path))

    patched = patch_config_toml(original, patch)
    _verify_patch_applied(original, patched, patch)

    config_changed = patched != original

    # Catalog FIRST, config SECOND: config.toml's model_catalog_json key
    # REFERENCES the catalog, so the referenced artifact must be established
    # before the reference is written. Writing config first and having the
    # catalog write fail/refuse afterward (foreign file without --force, or
    # an I/O error) would leave config.toml already committed and pointing at
    # a catalog that was never updated — a half-applied, inconsistent state
    # with no automatic rollback. Reversing the order means a catalog
    # failure leaves config.toml untouched (the pre-existing, still-consistent
    # state), and a catalog success followed by a config failure just leaves
    # an unreferenced-but-correct catalog file, which is harmless.
    catalog_wrote = _write_catalog(
        patch,
        agent,
        provider,
        catalog_path,
        dry_run=dry_run,
        force=force,
        confirm=confirm,
    )

    if config_changed:
        preview = diff_preview(original, patched)
        if dry_run:
            # --dry-run never prompts and never refuses — it only previews,
            # same contract as wrappers._install_plan (dry_run checked BEFORE
            # any confirm/force gate).
            print(preview or "(no textual change)")
            print(f"would write {config_path}")
        else:
            if not force and not (confirm and confirm(config_path, preview)):
                raise CodeHelperError(
                    f"about to patch {config_path} — refusing without "
                    f"confirmation (use --force, or re-run interactively)"
                )
            _rotate_backups(paths, current=original)
            atomic_write(config_path, patched, mode=None)
            print(f"wrote {config_path} (backup: {paths.codex_main_config_backup(1)})")
    else:
        print("no changes to config.toml")

    return config_changed or catalog_wrote


def restore_default(
    paths: Paths,
    *,
    slot: int = 1,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Restore ``~/.codex/config.toml`` from backup slot ``slot`` (1-3, 1=newest).

    Does NOT itself create a new backup slot — restoring is the "put it back"
    operation, not a fresh edit to archive. If the user runs ``set-default``
    again afterwards, THAT invocation backs up the just-restored state.

    Raises:
        CodeHelperError: no backup exists at ``slot``, or the overwrite is
            refused (no ``--force``/confirmation) — restoring can still
            destroy a DIFFERENT current config.toml if the user edited it (by
            hand, or via another tool) since the backup was taken.
    """
    backup_path = paths.codex_main_config_backup(slot)
    backup_body = _read_text_or_none(backup_path)
    if backup_body is None:
        raise CodeHelperError(f"no backup found at {backup_path}; nothing to restore")

    config_path = paths.codex_main_config()
    current = _read_text_or_none(config_path) or ""
    if current == backup_body:
        print("no changes")
        return False

    if dry_run:
        # --dry-run never prompts and never refuses — it only previews, same
        # ordering as apply_set_default.
        print(f"would restore {config_path} from {backup_path}")
        return True

    preview = diff_preview(current, backup_body)
    if not force and not (confirm and confirm(config_path, preview)):
        raise CodeHelperError(
            f"about to restore {config_path} from {backup_path} — refusing "
            f"without confirmation (use --force)"
        )

    atomic_write(config_path, backup_body, mode=None)
    print(f"restored {config_path} from {backup_path}")
    return True
