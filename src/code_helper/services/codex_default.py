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
reorder tables on a naive read-modify-write. ``tomllib`` (read-only in the
stdlib, but only from Python 3.11 on) is used only as a verification net
before and after the patch — never to construct the output. That verification
is mandatory, not best-effort (see :func:`_require_tomllib`), which is why
this project's ``requires-python`` floor is 3.11, not lower.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from code_helper.backends._atomic import atomic_write, read_text_or_none
from code_helper.backends._atomic import rotate_backups as _rotate_backups
from code_helper.errors import CodeHelperError
from code_helper.services.model import (
    Agent,
    BaseUrlPolicy,
    ConfigShape,
    Provider,
    resolve_shape,
)
from code_helper.services.paths import Paths
from code_helper.services.render import (
    CATALOG_MANAGED_BY_KEY,
    openai_base_url,
    openai_catalog_body,
    toml_string,
)
from code_helper.services.spec import build_spec
from code_helper.services.wrappers import _ownership_catalog_marker

__all__ = [
    "DefaultPatch",
    "resolve_default_patch",
    "patch_config_toml",
    "diff_preview",
    "apply_set_default",
    "clear_default",
    "current_default",
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

    # Same invariant as build_spec (services/spec.py): a REQUIRED-policy
    # provider (its address is the user's own server) must never reach a
    # renderer/patch with an unresolved empty base_url — that would silently
    # patch config.toml to point Codex at "/v1/". The caller is expected to
    # substitute a real address via model.with_base_url before calling here;
    # if nobody did, refuse rather than resolve a malformed patch.
    if provider.base_url_policy is BaseUrlPolicy.REQUIRED and not provider.base_url:
        raise CodeHelperError(
            f"provider {provider.name!r} requires a base URL — supply one "
            f"via model.with_base_url(provider, url) before resolve_default_patch"
        )

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

#: Matches the START of a line that COULD be a table header — `[table]` or
#: `[[array]]`. MULTILINE, anchored to line start so a `[` inside a string
#: value or a comment never matches. This is a CANDIDATE only: a line inside
#: an open multi-line array (`some_array = [\n[1, 2],\n...`) also starts with
#: `[` and would match here too, which is why callers walk candidates via
#: :func:`_first_real_table_boundary` rather than taking the first match
#: as-is — see that function's docstring for why depth-tracking is needed.
_FIRST_TABLE_RE = re.compile(r"^\[", re.MULTILINE)

#: Matches ANY table header line — used to find the end of a specific named
#: table's body (its content runs until the next one of these, or EOF).
_ANY_TABLE_HEADER_RE = re.compile(r"^\[.*$", re.MULTILINE)

#: A crude, line-oriented bracket counter — NOT a TOML lexer. Strips `#`
#: line-comments and `"..."`/`'...'` quoted spans (both single- and
#: triple-quoted, non-greedily) before counting `[`/`]`/`{`/`}`, so a
#: bracket character INSIDE a string or comment never perturbs the count.
#: Good enough to detect "is this line inside an open multi-line array or
#: inline table", which is all :func:`_first_real_table_boundary` needs —
#: it does not need to understand TOML values otherwise.
_STRING_OR_COMMENT_RE = re.compile(
    r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'|"(?:[^"\\]|\\.)*"|\'[^\']*\'|#[^\n]*'
)


def _first_real_table_boundary(text: str) -> int | None:
    """Index where the FIRST real top-level table header begins, or ``None``.

    A line starting with ``[`` is only a genuine table/array-of-tables header
    when it appears OUTSIDE any open bracket — a continuation line of a
    multi-line top-level array (e.g. ``some_array = [\\n[1, 2],\\n...``) also
    starts with ``[`` syntactically but is NOT a table boundary. Round-3
    review caught this: ``_patch_top_level`` used to take the first ``^\\[``
    match unconditionally, so an unindented multi-line array made it splice
    the managed keys into the middle of the array literal, producing invalid
    TOML — caught by ``_verify_patch_applied`` before any write (so nothing
    was ever corrupted), but with a confusing "internal error, this is a
    code-helper bug" message for what is actually a legitimate, if unusual,
    TOML layout. This walks line-by-line, tracking a running bracket depth
    (via :data:`_STRING_OR_COMMENT_RE` to ignore brackets inside strings/
    comments), and only accepts a ``^\\[`` candidate when the depth entering
    that line is 0.
    """
    depth = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        if depth == 0 and _FIRST_TABLE_RE.match(line):
            return pos
        stripped = _STRING_OR_COMMENT_RE.sub("", line)
        depth += stripped.count("[") + stripped.count("{")
        depth -= stripped.count("]") + stripped.count("}")
        depth = max(depth, 0)  # a stray closer must never go negative
        pos += len(line)
    return None


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

    Operates ONLY on the slice before the first REAL table header (see
    :func:`_first_real_table_boundary` — a line starting with ``[`` inside an
    open multi-line array/inline table does not count) or the whole file if
    there is none. A table with a colliding-looking body can never be
    touched by this step. Each key is handled independently: replaced in
    place via an anchored, single-line, ``count=1`` regex if present, else
    appended (in :data:`_TOP_LEVEL_KEYS` order, skipping keys that were
    already found) right before the top-level/table boundary.
    """
    boundary = _first_real_table_boundary(original)
    if boundary is None:
        boundary = len(original)
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

    # The whole table (header through the next table header, any table, or
    # EOF) is replaced wholesale by `rendered`, which includes its own header
    # line — so only the boundary AFTER the table body matters here.
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


def _require_tomllib():
    """Import and return ``tomllib``, or refuse with a clear message.

    ``tomllib`` entered the stdlib in Python 3.11 — that is *why* this whole
    project's ``requires-python`` floor is 3.11, not merely a coincidence.
    Unlike ``wrappers._toml_profile_data`` — which accepts a no-op when
    ``tomllib`` is unavailable, because it verifies a file this tool owns and
    wrote wholesale — ``set-default`` regex-patches the user's own
    hand-maintained ``config.toml``. Its patcher has documented blind spots
    (matching a managed-key-shaped line inside an unrelated multi-line
    string); the ``tomllib``-based structural verification in
    ``_verify_patch_applied`` is the ONLY thing that catches those before a
    write. Skipping it silently would mean a corrupted foreign file could be
    written with zero runtime check, which is a materially worse failure mode
    here than for a file this tool owns and can safely regenerate. So
    ``set-default`` refuses outright rather than degrading its safety net
    quietly — this function is a defensive backstop (a normal install already
    refuses below 3.11 via ``requires-python``), reachable only if someone
    installs the package with that check bypassed (e.g. ``pip install
    --ignore-requires-python``).
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
    (py3.11+, this project's floor — see :func:`_require_tomllib`) for why
    this is mandatory rather than best-effort for ``set-default``;
    ``_verify_toml_or_refuse`` already ran earlier in the same call and would
    have refused otherwise, so by the time this function runs ``tomllib`` is
    guaranteed importable.
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


def _config_backup_slots(paths: Paths) -> tuple[Path, Path, Path]:
    return (
        paths.codex_main_config_backup(3),
        paths.codex_main_config_backup(2),
        paths.codex_main_config_backup(1),
    )


def _catalog_backup_slots(catalog_path: Path) -> tuple[Path, Path, Path]:
    """The catalog's own 3-slot ring, next to the catalog itself.

    Not routed through ``Paths`` (unlike the config ring) because the catalog
    path is user-choosable via ``--catalog-json`` and need not live under
    ``~/.codex`` at all — the backups simply live beside whatever file the
    catalog actually is, ``<catalog>.bak1``/``.bak2``/``.bak3``.
    """
    return (
        catalog_path.with_name(catalog_path.name + ".bak3"),
        catalog_path.with_name(catalog_path.name + ".bak2"),
        catalog_path.with_name(catalog_path.name + ".bak1"),
    )


def _resolve_catalog_path(paths: Paths, catalog_json: str | None) -> Path:
    if catalog_json:
        return Path(catalog_json)
    return paths.codex_dir / _DEFAULT_CATALOG_NAME


@dataclass(frozen=True)
class _CatalogPlan:
    """What :func:`_gate_catalog_write` decided, ready for

    :func:`_commit_catalog_write` — the confirm/refuse decision and the
    write itself are split into two calls so :func:`apply_set_default` can
    confirm BOTH the catalog and the config.toml writes before committing
    EITHER one (see the ordering comment there).
    """

    catalog_path: Path
    body: str
    existing: str | None
    overwriting: bool
    #: True when there is nothing to do — ``body`` already matches
    #: ``existing`` exactly (idempotent no-op).
    no_op: bool


def _gate_catalog_write(
    patch: DefaultPatch,
    agent: Agent,
    provider: Provider,
    catalog_path: Path,
    *,
    dry_run: bool,
    force: bool,
    confirm,
) -> _CatalogPlan:
    """Decide whether the catalog write is allowed, WITHOUT writing anything.

    Reuses the SAME structural ``managed_by`` proof
    (``wrappers._ownership_catalog_marker``) the per-alias OPENAI_TOML catalogs use
    — a hand-curated ``~/.codex/model.json`` is protected exactly like a
    hand-curated ``<alias>.model.json`` would be. EVERY real overwrite of
    EXISTING content — foreign or our own previously-managed catalog — goes
    through the confirm/force gate, same posture as config.toml. A managed
    catalog is not exempt: it is exactly the common case (a second
    ``set-default`` changing the model), and silently clobbering it with no
    confirm was the actual gap — the confirm/force gate previously fired only
    for a FOREIGN catalog, so switching models normally overwrote the
    previous default's catalog with no prompt and no way back.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — same signature as
            :func:`apply_set_default`'s ``confirm``, called with a unified
            diff of the existing catalog against the new one (via
            :func:`diff_preview`) whenever existing content would be
            overwritten, so the prompt shows what is about to change, same as
            the config.toml patch prompt.

    Raises:
        CodeHelperError: the overwrite is refused (no ``--force``/confirm).
            Never raises under ``--dry-run`` — that path only previews.
    """
    spec = build_spec(
        agent=agent,
        provider=provider,
        model=patch.model,
        alias=_CATALOG_SPEC_ALIAS,
        shape=ConfigShape.OPENAI_TOML,
    )
    body = openai_catalog_body(spec)

    existing = read_text_or_none(catalog_path)
    if existing == body:
        return _CatalogPlan(catalog_path, body, existing, False, no_op=True)

    foreign = existing is not None and not _ownership_catalog_marker(catalog_path)
    overwriting_existing_content = existing is not None

    if dry_run:
        return _CatalogPlan(
            catalog_path, body, existing, overwriting_existing_content, no_op=False
        )

    catalog_preview = diff_preview(existing or "", body, label=str(catalog_path))
    if overwriting_existing_content and not force:
        if not (confirm and confirm(catalog_path, catalog_preview)):
            if foreign:
                raise CodeHelperError(
                    f"{catalog_path} exists and was not created by code-helper "
                    f"(missing {CATALOG_MANAGED_BY_KEY!r} marker) — refusing to "
                    f"overwrite (use --force)"
                )
            raise CodeHelperError(
                f"about to overwrite {catalog_path} — refusing without "
                f"confirmation (use --force, or re-run interactively)"
            )

    return _CatalogPlan(
        catalog_path, body, existing, overwriting_existing_content, no_op=False
    )


def _commit_catalog_write(plan: _CatalogPlan) -> bool:
    """The write-only half of the catalog flow — commits a plan already

    confirmed/force-gated by :func:`_gate_catalog_write`. Returns whether
    anything changed (``True`` unless ``plan.no_op``).
    """
    if plan.no_op:
        return False
    if plan.overwriting:
        assert plan.existing is not None  # implied by `overwriting`
        _rotate_backups(_catalog_backup_slots(plan.catalog_path), current=plan.existing)
    atomic_write(plan.catalog_path, plan.body, mode=None)
    print(f"wrote {plan.catalog_path}")
    return True


def clear_config_toml(original: str, provider_table: str | None) -> str:
    """``original`` with this command's managed region removed. Pure, no IO.

    The textual inverse of :func:`patch_config_toml`: drops the three
    :data:`_TOP_LEVEL_KEYS` lines and, when ``provider_table`` names one, the
    whole ``[model_providers.<name>]`` block. Everything else — comments,
    key order, unrelated tables, the user's own ``[model_providers.*]``
    entries — is left byte-for-byte alone, because this is the user's own
    hand-maintained file and only the region this tool wrote is ours to take
    back.

    ``provider_table`` is ``None`` when ``model_provider`` names a provider
    NOT in this tool's registry (see :func:`current_default`) — i.e. this
    file's ``model``/``model_provider``/``model_catalog_json`` were not
    necessarily ever written by ``set-default``. Removing them anyway would
    be exactly the kind of foreign-config damage this function's own
    docstring promises never to do, so ``None`` here means "not proven
    ours" and the top-level keys are left untouched too, not just the table.
    """
    if not provider_table:
        return original
    text = original
    for key in _TOP_LEVEL_KEYS:
        # Consume the trailing newline with the line so removal does not
        # leave a blank gap where the key used to be.
        text = re.sub(rf"^{re.escape(key)}\s*=.*(?:\n|$)", "", text, flags=re.MULTILINE)
    if provider_table:
        header = re.compile(
            rf"^\[model_providers\.{re.escape(provider_table)}\]\s*$", re.MULTILINE
        )
        match = header.search(text)
        if match:
            # The table body runs to the next table header, or to EOF.
            nxt = _ANY_TABLE_HEADER_RE.search(text, match.end())
            text = text[: match.start()] + text[nxt.start() if nxt else len(text) :]
    return text


def _verify_cleared(original: str, cleared: str, provider_table: str | None) -> None:
    """Refuse before writing if clearing removed more than the managed region.

    The mirror of :func:`_verify_patch_applied`, and mandatory for the same
    reason: these are line-anchored regexes running over the user's own file,
    so the only honest proof that nothing else moved is a structural one.
    Asserts the managed keys/table are gone from the result AND that
    everything outside the managed region parses identically before and
    after.
    """
    tomllib = _require_tomllib()
    try:
        data = tomllib.loads(cleared)
    except ValueError as exc:
        raise CodeHelperError(
            f"internal error: config.toml does not parse as TOML after "
            f"clearing — refusing to write; this is a code-helper bug, "
            f"please report it: {exc}"
        ) from None

    leftover = [key for key in _TOP_LEVEL_KEYS if key in data]
    providers = data.get("model_providers")
    if provider_table and isinstance(providers, dict) and provider_table in providers:
        leftover.append(f"model_providers.{provider_table}")
    if leftover:
        raise CodeHelperError(
            f"internal error: clearing config.toml left managed keys behind "
            f"({', '.join(leftover)}) — refusing to write; this is a "
            f"code-helper bug, please report it"
        )

    if not original.strip():
        return
    # A DefaultPatch is only needed here for its provider_table field — the
    # comparison is "everything outside the managed region", the same notion
    # _verify_patch_applied uses, so the helper is reused rather than
    # restating which keys are managed.
    probe = DefaultPatch(
        model="",
        provider_table=provider_table or "",
        display_name="",
        base_url="",
        wire_api="",
        catalog_json="",
    )
    if _without_managed_region(
        tomllib.loads(original), probe
    ) != _without_managed_region(data, probe):
        raise CodeHelperError(
            "internal error: clearing config.toml appears to have altered "
            "content outside the managed keys/table — refusing to write; "
            "this is a code-helper bug, please report it"
        )


def clear_default(
    paths: Paths,
    *,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Remove this tool's override from ``config.toml`` — codex's "native".

    The codex counterpart of ``switch``'s ``native`` provider: where claude's
    ``env_reset`` clears the managed ``env`` block from ``settings.json``,
    this drops the managed keys and provider table from ``config.toml``, so
    Codex falls back to whatever it did before ``set-default`` ever ran.

    Deliberately NOT ``restore_default``: restore rolls the file back to a
    backup SNAPSHOT, undoing unrelated hand-edits made since. This removes
    only the region this tool owns and leaves every other edit in place —
    the difference between "undo my last change" and "stop overriding".

    Takes the same write path as :func:`apply_set_default` — gate on
    confirm/force, rotate backups, ``atomic_write`` — so a clear is exactly
    as recoverable as a set.

    Returns:
        True if anything was written (or would be, under ``dry_run``);
        False when there was no managed region to remove.
    """
    config_path = paths.codex_main_config()
    original = read_text_or_none(config_path) or ""
    if not original.strip():
        print("no changes to config.toml")
        return False
    _verify_toml_or_refuse(original, context=str(config_path))

    cleared = clear_config_toml(original, current_default(paths))
    if cleared == original:
        print("no changes to config.toml")
        return False
    _verify_cleared(original, cleared, current_default(paths))

    preview = diff_preview(original, cleared)
    if dry_run:
        print(preview or "(no textual change)")
        print(f"would write {config_path}")
        return True

    if not force and not (confirm and confirm(config_path, preview)):
        raise CodeHelperError(
            f"about to patch {config_path} — refusing without confirmation "
            f"(use --force, or re-run interactively)"
        )

    _rotate_backups(_config_backup_slots(paths), current=original)
    atomic_write(config_path, cleared, mode=None)
    print(f"wrote {config_path} (backup: {paths.codex_main_config_backup(1)})")
    return True


def current_default(paths: Paths) -> str | None:
    """Which provider ``config.toml``'s ``model_provider`` names, or ``None``.

    The ``set-default`` counterpart of ``claude_settings.current_switch``, and
    it holds the same posture: read-only, **never raises** — a missing,
    unreadable, or unparseable file reads as "no override applied", mirroring
    ``state.load_state``. Deliberately derives the answer from the FILE
    ITSELF rather than from ``state.json``, for the reason spelled out in
    ``current_switch``'s docstring: a second source of truth for "what is
    active" goes stale the moment the user hand-edits the config.

    Unlike ``current_switch`` — which has to match a live ``base_url`` back to
    a provider and admits it cannot tell two providers sharing one address
    apart — this resolves by KEY NAME: ``_patch_top_level`` writes
    ``model_provider = "<provider.name>"`` verbatim, so the file already
    carries the provider's identity and no address-matching heuristic is
    needed. A name that matches no registry entry (hand-written, or from a
    provider since removed) reads as ``None`` rather than being echoed back,
    so every non-``None`` return is a real ``PROVIDERS`` name.

    Uses ``tomllib`` directly rather than ``_require_tomllib``: that helper
    refuses loudly because ``set-default`` is about to WRITE, and a missing
    verifier there would mean writing unverified. This function only reads,
    and its whole contract is to degrade to ``None`` instead of raising.
    """
    from code_helper.services.model import PROVIDERS

    text = read_text_or_none(paths.codex_main_config())
    if not text:
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py3.11+ is the floor
        return None
    try:
        data = tomllib.loads(text)
    except ValueError:
        return None
    name = data.get("model_provider")
    if not isinstance(name, str):
        return None
    return name if any(p.name == name for p in PROVIDERS) else None


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
    apply -> post-check -> GATE both the catalog write and the config.toml
    patch (confirm/force, raising on refusal) BEFORE committing EITHER one ->
    only once both gates pass, commit the catalog write then the config.toml
    write. Gating both up front — rather than writing the catalog as soon as
    its own gate passes, then separately gating config.toml — is what
    guarantees a refusal on either file leaves BOTH files untouched; the
    two-phase ``_gate_catalog_write``/``_commit_catalog_write`` split exists
    specifically to make this possible. Mirrors ``wrappers.install_wrapper``'s
    shape (paths spec, dry_run, force, confirm) so the CLI handler stays a
    thin shell.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — called ONLY when a
            real, visible write is about to happen and neither ``--force`` nor
            an existing no-op short-circuit applies (``preview`` is a unified
            diff — of the config patch, or of the catalog's old vs. new
            content — in both cases via :func:`diff_preview`). ``None``
            behaves like "always refuse" (the fail-fast-off-a-TTY default the
            CLI layer supplies).

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
    original = read_text_or_none(config_path) or ""
    _verify_toml_or_refuse(original, context=str(config_path))

    patched = patch_config_toml(original, patch)
    _verify_patch_applied(original, patched, patch)

    config_changed = patched != original

    # GATE both writes (confirm/force, raise on refusal) BEFORE committing
    # EITHER one — this is what guarantees a refusal on either file leaves
    # BOTH files completely untouched, no matter which order they're checked
    # in. An earlier version of this function wrote the catalog immediately
    # once its own gate passed, then gated config.toml separately — so a
    # catalog write that succeeded followed by a DECLINED config confirm left
    # the catalog already changed while the (unpatched) config.toml, in the
    # common case where the catalog path is unchanged across runs, still
    # actively referenced it: not "harmless and unreferenced" as a prior
    # comment here claimed, but a live config<->catalog mismatch. Gating both
    # first removes the whole class of "one file wrote, the other refused"
    # states.
    catalog_plan = _gate_catalog_write(
        patch,
        agent,
        provider,
        catalog_path,
        dry_run=dry_run,
        force=force,
        confirm=confirm,
    )

    preview = diff_preview(original, patched) if config_changed else ""
    if config_changed and not dry_run:
        if not force and not (confirm and confirm(config_path, preview)):
            raise CodeHelperError(
                f"about to patch {config_path} — refusing without "
                f"confirmation (use --force, or re-run interactively)"
            )

    # Both gates passed (or --dry-run, which never raises) — now commit.
    if dry_run:
        if not catalog_plan.no_op:
            print(f"would write {catalog_path}")
        if config_changed:
            print(preview or "(no textual change)")
            print(f"would write {config_path}")
        else:
            print("no changes to config.toml")
        return not catalog_plan.no_op or config_changed

    catalog_wrote = _commit_catalog_write(catalog_plan)

    if config_changed:
        _rotate_backups(_config_backup_slots(paths), current=original)
        atomic_write(config_path, patched, mode=None)
        print(f"wrote {config_path} (backup: {paths.codex_main_config_backup(1)})")
    else:
        print("no changes to config.toml")

    return config_changed or catalog_wrote


@dataclass(frozen=True)
class _RestorePlan:
    """Everything :func:`restore_default` needs, read once before any write.

    Reading both files up front is what lets confirmation happen for BOTH
    restores before EITHER is written — see :func:`_confirm_restore`.
    """

    config_path: Path
    backup_path: Path
    backup_body: str
    current: str
    catalog_path: Path
    catalog_backup_path: Path
    catalog_backup_body: str | None
    catalog_current: str | None

    @property
    def config_changed(self) -> bool:
        return self.current != self.backup_body

    @property
    def catalog_changed(self) -> bool:
        return (
            self.catalog_backup_body is not None
            and self.catalog_current != self.catalog_backup_body
        )


def _restore_plan(paths: Paths, *, slot: int, catalog_json: str | None) -> _RestorePlan:
    """Read the backup slot and the current files a restore would overwrite.

    Raises:
        CodeHelperError: no config backup exists at ``slot``. A missing
            CATALOG backup is not an error — only the config.toml restore is
            mandatory (see :func:`restore_default`'s docstring).
    """
    backup_path = paths.codex_main_config_backup(slot)
    backup_body = read_text_or_none(backup_path)
    if backup_body is None:
        raise CodeHelperError(f"no backup found at {backup_path}; nothing to restore")

    config_path = paths.codex_main_config()
    catalog_path = _resolve_catalog_path(paths, catalog_json)
    catalog_backup_path = _catalog_backup_slots(catalog_path)[3 - slot]
    return _RestorePlan(
        config_path=config_path,
        backup_path=backup_path,
        backup_body=backup_body,
        current=read_text_or_none(config_path) or "",
        catalog_path=catalog_path,
        catalog_backup_path=catalog_backup_path,
        catalog_backup_body=read_text_or_none(catalog_backup_path),
        catalog_current=read_text_or_none(catalog_path),
    )


def _confirm_restore(plan: _RestorePlan, *, force: bool, confirm) -> None:
    """Collect every confirmation a restore needs, before ANY file is written.

    Writing config first (as an earlier version of this function did) and then
    asking about the catalog meant a declined catalog confirm left config.toml
    ALREADY overwritten with ``backup_body`` — an inconsistent config<->catalog
    pairing (the exact thing the paired restore exists to avoid) with no
    rollback, since ``restore_default`` does not itself back up ``current``
    before overwriting it. Collecting every confirmation up front means a
    refusal on either file leaves BOTH files completely untouched.

    Raises:
        CodeHelperError: either overwrite is refused (no ``--force``, no
            confirmation).
    """
    if plan.config_changed:
        preview = diff_preview(plan.current, plan.backup_body)
        if not force and not (confirm and confirm(plan.config_path, preview)):
            raise CodeHelperError(
                f"about to restore {plan.config_path} from {plan.backup_path} — "
                f"refusing without confirmation (use --force)"
            )
    if plan.catalog_changed:
        assert plan.catalog_backup_body is not None  # implied by catalog_changed
        catalog_preview = diff_preview(
            plan.catalog_current or "",
            plan.catalog_backup_body,
            label=str(plan.catalog_path),
        )
        if not force and not (confirm and confirm(plan.catalog_path, catalog_preview)):
            raise CodeHelperError(
                f"about to restore {plan.catalog_path} from "
                f"{plan.catalog_backup_path} — refusing without confirmation "
                f"(use --force)"
            )


def restore_default(
    paths: Paths,
    *,
    slot: int = 1,
    catalog_json: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Restore ``~/.codex/config.toml`` (and its paired catalog) from slot ``slot``.

    Does NOT itself create a new backup slot — restoring is the "put it back"
    operation, not a fresh edit to archive. If the user runs ``set-default``
    again afterwards, THAT invocation backs up the just-restored state.

    Also restores the catalog's OWN backup at the same slot number, if one
    exists (:func:`_catalog_backup_slots`) — a restored config.toml pointing
    at ``model_catalog_json`` while the catalog itself still describes the
    model this call just moved AWAY from is exactly the inconsistent
    config<->catalog pairing this command exists to avoid. ``catalog_json``
    resolves the catalog path the SAME way :func:`apply_set_default` does
    (:func:`_resolve_catalog_path`) — pass the same value you passed to the
    ``set-default`` call being undone, or omit it to use the default
    ``~/.codex/model.json`` path. The catalog restore is best-effort: a
    missing catalog backup (e.g. the catalog was never actually changed, or
    this restores a state from before the catalog got its own backup ring)
    is not an error — only the config.toml restore is mandatory.

    Raises:
        CodeHelperError: no config backup exists at ``slot``, or the overwrite
            is refused (no ``--force``/confirmation) — restoring can still
            destroy a DIFFERENT current config.toml (or catalog) if either was
            edited (by hand, or via another tool) since the backup was taken.
    """
    plan = _restore_plan(paths, slot=slot, catalog_json=catalog_json)

    if not plan.config_changed and not plan.catalog_changed:
        print("no changes")
        return False

    if dry_run:
        # --dry-run never prompts and never refuses — it only previews, same
        # ordering as apply_set_default.
        if plan.config_changed:
            print(f"would restore {plan.config_path} from {plan.backup_path}")
        if plan.catalog_changed:
            print(f"would restore {plan.catalog_path} from {plan.catalog_backup_path}")
        return True

    _confirm_restore(plan, force=force, confirm=confirm)

    if plan.config_changed:
        atomic_write(plan.config_path, plan.backup_body, mode=None)
        print(f"restored {plan.config_path} from {plan.backup_path}")
    if plan.catalog_changed:
        assert plan.catalog_backup_body is not None  # implied by catalog_changed
        atomic_write(plan.catalog_path, plan.catalog_backup_body, mode=None)
        print(f"restored {plan.catalog_path} from {plan.catalog_backup_path}")

    return True
