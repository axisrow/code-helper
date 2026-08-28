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

from codehelper.backends._atomic import (
    atomic_write,
    file_lock,
    read_text_or_none,
)
from codehelper.backends._atomic import rotate_backups as _rotate_backups
from codehelper.errors import CodeHelperError
from codehelper.services.model import (
    Agent,
    BaseUrlPolicy,
    ConfigShape,
    Provider,
    resolve_shape,
)
from codehelper.services.paths import Paths
from codehelper.services.render import (
    openai_base_url,
    toml_string,
    uniform_context_window,
)

__all__ = [
    "DefaultPatch",
    "CODEX_RESERVED_PROVIDER_IDS",
    "resolve_default_patch",
    "patch_config_toml",
    "diff_preview",
    "apply_set_default",
    "clear_default",
    "current_default",
    "restore_default",
]


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
    #: ``model_context_window``'s value, or None to omit/remove the key.
    #: Never a guessed floor — see :data:`MODEL_CONTEXT_WINDOWS`: only a model
    #: this tool actually knows the real window for gets one declared, the
    #: same conditional-emission rule ``render.openai_toml_body`` and
    #: ``_render_anthropic_env`` use.
    context_window: int | None


#: Provider IDs Codex CLI itself treats as built-in and refuses to see
#: overridden in ``[model_providers.<id>]`` — NOT this project's own
#: reservation (contrast :data:`codehelper.services.naming.RESERVED_ALIASES`,
#: which reserves *wrapper* names in this tool's own ``~/.local/bin``
#: namespace). This list is Codex's, sourced from its own error message
#: ("model_providers contains reserved built-in provider IDs"); kept in sync
#: by hand since Codex does not expose it as a queryable API. Confirmed
#: reserved as of Codex CLI v0.150.1: "openai", "ollama" — the latter is why
#: this project's own ``ollama`` provider was renamed to ``ollama-direct``.
CODEX_RESERVED_PROVIDER_IDS = frozenset({"openai", "ollama"})


def resolve_default_patch(agent: Agent, provider: Provider, model: str) -> DefaultPatch:
    """Resolve a :class:`DefaultPatch` from the axes. Pure, no IO.

    Reuses :func:`resolve_shape` — the SAME compatibility check ``add`` uses —
    so ``set-default`` cannot silently accept a pairing ``add`` would reject.
    For today's registry this only ever resolves for ``codex``: every other
    agent declares no ``OPENAI_TOML`` shape. Two distinct failures follow from
    that, both surfaced by ``resolve_shape`` itself rather than a bespoke
    error here:

    - an agent with NO shape in common with ``provider`` at all (e.g.
      ``claude`` + ``zai`` is fine for ``add`` but has no ``OPENAI_TOML``
      story) fails with "no common configuration mechanism";
    - an agent that shares a DIFFERENT shape with ``provider`` — every
      ``ollama launch``-only agent (``opencode``, ``droid``, …) against
      ``ollama`` shares ``OLLAMA_LAUNCH`` but not ``OPENAI_TOML`` — fails with
      "cannot use shape openai-toml (available: ollama-launch)" instead, since
      ``preferred`` was supplied and is simply not among the shared shapes.

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

    if provider.name in CODEX_RESERVED_PROVIDER_IDS:
        raise CodeHelperError(
            f"provider {provider.name!r} is reserved by Codex CLI itself as a "
            f"built-in provider ID and cannot be used in "
            f"[model_providers.{provider.name}] — Codex will refuse to load "
            f"config.toml entirely. Rename this provider in codehelper's own "
            f"registry (services/model.py PROVIDERS)."
        )

    return DefaultPatch(
        model=model,
        provider_table=provider.name,
        display_name=provider.description or provider.name,
        base_url=openai_base_url(provider.base_url, provider.base_url_is_openai_root),
        wire_api=provider.wire_api,
        context_window=uniform_context_window([model]),
    )


#: Top-level scalar keys this command manages, in the order a fresh insert
#: appends them (matches the issue's own example layout).
#:
#: ``model_catalog_json`` is a REMOVAL-only key, kept here (rather than
#: dropped from the tuple) so a fresh ``set-default`` scrubs a stale line an
#: older version of this tool left behind — that older catalog carried an
#: empty ``base_instructions`` per entry, silently replacing Codex's real
#: system prompt (see ``render.openai_toml_body``). :func:`_patch_value_for`
#: always returns ``None`` for it, and :func:`_patch_top_level` treats
#: ``None`` as "this key must not appear" rather than a value to write.
#: ``_without_managed_region`` (below) already excludes every key in this
#: tuple from BOTH sides of its structural comparison, so a key going from
#: "present with a value" to "absent" is not read as unmanaged content
#: changing.
_TOP_LEVEL_KEYS = (
    "model",
    "model_provider",
    "model_catalog_json",
    "model_context_window",
)

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
    codehelper bug" message for what is actually a legitimate, if unusual,
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


class _Skip:
    """Sentinel class: "leave whatever line is already there completely alone".

    A dedicated class rather than a bare ``object()`` so a type checker can
    narrow ``isinstance(value, _Skip)`` away from the real ``str | int | None``
    payloads instead of collapsing the whole union to ``object``.
    """

    __slots__ = ()
    _instance: _Skip | None = None

    def __new__(cls) -> _Skip:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


#: Sentinel for :func:`_patch_value_for`: "leave whatever line is already
#: there completely alone". Used for ``model_context_window`` when this tool
#: does NOT know the model's real window: unlike ``model_catalog_json``
#: (removal-only, only ever written by this tool), a pre-existing
#: ``model_context_window`` in the user's own ``config.toml`` may be a manual
#: setting for a custom model — one this ``set-default`` knows nothing about
#: must never delete it.
_SKIP = _Skip()


def _patch_value_for(key: str, patch: DefaultPatch) -> str | int | None | _Skip:
    """The value ``key`` should carry, ``None`` to remove it, :data:`_SKIP` to
    leave it alone.

    ``model``/``model_provider`` always carry a string. ``model_catalog_json``
    is REMOVAL-only (see :data:`_TOP_LEVEL_KEYS`'s docstring) — always
    ``None``. ``model_context_window`` is ``patch.context_window`` when this
    tool knows the model's real window, :data:`_SKIP` otherwise — never a
    guessed floor, and never a removal (see :data:`_SKIP`'s docstring).
    """
    if key == "model":
        return patch.model
    if key == "model_provider":
        return patch.provider_table
    if key == "model_catalog_json":
        return None
    if key == "model_context_window":
        return patch.context_window if patch.context_window is not None else _SKIP
    raise AssertionError(f"unknown top-level key: {key!r}")  # pragma: no cover


def _patch_top_level(original: str, patch: DefaultPatch) -> str:
    """Replace/insert/remove the top-level scalar keys. Pure, no IO.

    Operates ONLY on the slice before the first REAL table header (see
    :func:`_first_real_table_boundary` — a line starting with ``[`` inside an
    open multi-line array/inline table does not count) or the whole file if
    there is none. A table with a colliding-looking body can never be
    touched by this step. Each key is handled independently:

    - :func:`_patch_value_for` returns ``None`` → the key must NOT appear.
      An existing line for it is deleted outright (consuming its trailing
      newline, mirroring :func:`clear_config_toml`'s removal regex) rather
      than replaced — this is how a stale ``model_catalog_json`` line left
      by an earlier version of this tool gets scrubbed, not merely
      overwritten with a new value.
    - it returns :data:`_SKIP` → the key is not this run's business at all:
      an existing line is left byte-for-byte alone.
    - a string value → quoted (``toml_string``), same as before.
    - an int value (``model_context_window`` only) → written BARE: the field
      is ``Option<i64>`` on Codex's side, and a quoted ``"1000000"`` would not
      deserialize as one.

    A present value (string or int) is replaced in place via an anchored,
    single-line, ``count=1`` regex if the key already exists, else appended
    (in :data:`_TOP_LEVEL_KEYS` order, skipping keys that were already found
    or that resolve to ``None``/:data:`_SKIP`) right before the
    top-level/table boundary.
    """
    boundary = _first_real_table_boundary(original)
    if boundary is None:
        boundary = len(original)
    top = original[:boundary]
    rest = original[boundary:]

    missing: list[str] = []
    for key in _TOP_LEVEL_KEYS:
        value = _patch_value_for(key, patch)
        line_re = re.compile(rf"^{re.escape(key)}\s*=.*(?:\n|$)", re.MULTILINE)

        if value is None:
            top, _ = line_re.subn("", top, count=1)
            continue
        if isinstance(value, _Skip):
            continue

        replacement = (
            f"{key} = {value}"
            if isinstance(value, int)
            else f'{key} = "{toml_string(value)}"'
        )
        no_newline_re = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        # A callable replacement, NOT a template string: re.subn treats a
        # string replacement as a backreference template and interprets any
        # ``\`` in it (e.g. the ``\uXXXX`` toml_string emits for a control
        # character in a user-supplied --model value) as an escape — ``\u``
        # is not a valid one, so re.error crashes instead of patching. A
        # lambda's return value is used verbatim, no re-parsing.
        new_top, count = no_newline_re.subn(lambda m, r=replacement: r, top, count=1)
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


def _remove_model_providers_table(text: str, table_name: str) -> str:
    """Remove ``[model_providers.<table_name>]`` (header through the next
    table header, or EOF) if present; a no-op otherwise. Pure, no IO.

    The one place this header/next-table-boundary removal is implemented —
    shared by :func:`clear_config_toml` (removing the table ``set-default``
    itself owns) and :func:`_strip_reserved_ollama_table` (removing a
    DIFFERENT, stale table left behind by an older codehelper), so the two
    unrelated reasons to drop a ``[model_providers.*]`` table can never drift
    on how a table's boundary is found.
    """
    header_re = re.compile(
        rf"^\[model_providers\.{re.escape(table_name)}\]\s*$", re.MULTILINE
    )
    match = header_re.search(text)
    if match is None:
        return text
    next_header = _ANY_TABLE_HEADER_RE.search(text, match.end())
    return (
        text[: match.start()]
        + text[next_header.start() if next_header else len(text) :]
    )


#: The literal table name of ONE retired provider ID this project itself
#: used to write into config.toml before Codex CLI v0.150.1 reserved it as
#: its own built-in provider ID (see :data:`CODEX_RESERVED_PROVIDER_IDS`).
#: A one-off compatibility shim, not a general mechanism — deliberately a
#: literal string, not derived from ``model.RETIRED_PROVIDER_NAMES``,
#: because THIS specific table is the one that makes Codex refuse to load
#: the file at all, which is a stronger claim than "renamed in our
#: registry." Safe to delete once the affected population has run a
#: ``set-default``/``clear-default`` at least once after upgrading.
_STALE_RESERVED_PROVIDER_TABLE = "ollama"


def _strip_reserved_ollama_table(original: str) -> str:
    """Remove a stale ``[model_providers.ollama]`` table, if present.

    Self-healing one-time migration: a config.toml written by an older
    codehelper (before the ``ollama`` -> ``ollama-direct`` rename) may still
    carry this table, which Codex CLI v0.150.1+ refuses to load AT ALL
    (reserved built-in provider ID) — Codex itself never gets far enough to
    run again and fix this, so the fix has to happen the next time
    codehelper itself touches the file, regardless of which provider this
    particular call is patching. A no-op when the table is absent (already
    migrated, or never present).
    """
    return _remove_model_providers_table(original, _STALE_RESERVED_PROVIDER_TABLE)


def patch_config_toml(original: str, patch: DefaultPatch) -> str:
    """Return ``original`` with ``patch`` applied. Pure, no IO.

    Three steps — strip a stale reserved-name table left behind by an older
    codehelper (:func:`_strip_reserved_ollama_table`, a one-time
    self-healing migration, independent of what ``patch`` itself targets),
    then the top-level scalar keys, then the ``[model_providers.X]`` table —
    each anchored/regex-based, never a round-trip TOML parse. Idempotent by
    construction: applying the same ``patch`` twice in a row yields
    byte-identical output both times, because every step always asks "is the
    target already here?" before deciding replace-in-place vs.
    append/insert, and both key/table steps render their inserted value
    through the exact same encoder used for a replacement.
    """
    stripped = _strip_reserved_ollama_table(original)
    with_keys = _patch_top_level(stripped, patch)
    return _patch_model_providers_table(with_keys, patch)


def diff_preview(original: str, patched: str, *, label: str = "config.toml") -> str:
    """Unified diff of ``original`` -> ``patched``, stdlib only.

    Empty string when the two are identical (the no-op case) — callers print
    this as-is under ``--dry-run`` and before an interactive confirm. ``label``
    names the file in the diff headers — defaults to ``config.toml``, the
    only caller.
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


def _without_managed_region(
    data: dict, patch: DefaultPatch, *, also_exclude: str | None = None
) -> dict:
    """``data`` (an already-parsed TOML dict) with every key this patcher
    manages removed, so what's left is exactly the content that must survive
    a patch untouched.

    ``also_exclude``, when given, drops one more ``[model_providers.*]``
    table from the comparison alongside ``patch.provider_table`` — for
    :func:`_verify_patch_applied` ONLY, whose :func:`patch_config_toml` may
    have stripped :data:`_STALE_RESERVED_PROVIDER_TABLE` regardless of what
    ``patch`` itself targets. :func:`_verify_cleared` never passes this:
    ``clear_config_toml`` never calls the stale-table strip, so there is
    nothing extra to exclude there — keeping that one-time migration concern
    out of this function's own, unconditional comparison logic.
    """
    out = {k: v for k, v in data.items() if k not in _TOP_LEVEL_KEYS}
    providers = out.get("model_providers")
    excluded_tables = {patch.provider_table} | (
        {also_exclude} if also_exclude else set()
    )
    if isinstance(providers, dict):
        providers = {k: v for k, v in providers.items() if k not in excluded_tables}
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
            f"— refusing to write; this is a codehelper bug, please report "
            f"it: {exc}"
        ) from None

    providers = data.get("model_providers", {})
    table = (
        providers.get(patch.provider_table, {}) if isinstance(providers, dict) else {}
    )
    expected = {
        "model": patch.model,
        "model_provider": patch.provider_table,
        "model_catalog_json": None,  # scrubbed, never written
    }
    # ``model_context_window`` is asserted only when this run WRITES it —
    # with an unknown model the patch leaves any existing (possibly
    # user-set) value alone, so there is nothing to assert it equals.
    if patch.context_window is not None:
        expected["model_context_window"] = patch.context_window
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
            "expected values — refusing to write; this is a codehelper bug, "
            "please report it"
        )

    # original is already known-parseable (_verify_toml_or_refuse ran first);
    # an empty original (fresh install) has nothing to preserve.
    if original.strip():
        original_data = tomllib.loads(original)
        before = _without_managed_region(
            original_data, patch, also_exclude=_STALE_RESERVED_PROVIDER_TABLE
        )
        after = _without_managed_region(
            data, patch, also_exclude=_STALE_RESERVED_PROVIDER_TABLE
        )
        if before != after:
            raise CodeHelperError(
                "internal error: patching config.toml appears to have altered "
                "content outside the managed keys/table — refusing to write; "
                "this is a codehelper bug, please report it"
            )


def _config_backup_slots(paths: Paths) -> tuple[Path, Path, Path]:
    return (
        paths.codex_main_config_backup(3),
        paths.codex_main_config_backup(2),
        paths.codex_main_config_backup(1),
    )


#: The keys ``clear_config_toml`` removes — everything in
#: :data:`_TOP_LEVEL_KEYS` EXCEPT ``model_context_window``: an existing
#: window may be the user's own manual setting for a custom model (this tool
#: only ever writes it when it knows the model's real window), and "codex
#: native" must not silently delete user configuration. A window WE wrote
#: stays behind after a clear as a harmless leftover — a stale declared
#: window is at worst a conservative compaction point, never the silent
#: data loss a removed manual one would be.
_CLEAR_KEYS = ("model", "model_provider", "model_catalog_json")


def clear_config_toml(original: str, provider_table: str | None) -> str:
    """``original`` with this command's managed region removed. Pure, no IO.

    The textual inverse of :func:`patch_config_toml`: drops the
    :data:`_CLEAR_KEYS` lines and, when ``provider_table`` names one, the
    whole ``[model_providers.<name>]`` block. Everything else — comments,
    key order, unrelated tables, the user's own ``[model_providers.*]``
    entries, a possibly-user-set ``model_context_window`` — is left
    byte-for-byte alone, because this is the user's own hand-maintained file
    and only the region this tool wrote is ours to take back.

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
    for key in _CLEAR_KEYS:
        # Consume the trailing newline with the line so removal does not
        # leave a blank gap where the key used to be.
        text = re.sub(rf"^{re.escape(key)}\s*=.*(?:\n|$)", "", text, flags=re.MULTILINE)
    return _remove_model_providers_table(text, provider_table)


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
            f"clearing — refusing to write; this is a codehelper bug, "
            f"please report it: {exc}"
        ) from None

    leftover = [key for key in _CLEAR_KEYS if key in data]
    providers = data.get("model_providers")
    if provider_table and isinstance(providers, dict) and provider_table in providers:
        leftover.append(f"model_providers.{provider_table}")
    if leftover:
        raise CodeHelperError(
            f"internal error: clearing config.toml left managed keys behind "
            f"({', '.join(leftover)}) — refusing to write; this is a "
            f"codehelper bug, please report it"
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
        context_window=None,
    )
    if _without_managed_region(
        tomllib.loads(original), probe
    ) != _without_managed_region(data, probe):
        raise CodeHelperError(
            "internal error: clearing config.toml appears to have altered "
            "content outside the managed keys/table — refusing to write; "
            "this is a codehelper bug, please report it"
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
    ``env_reset`` explicitly blanks managed ``env`` values in ``settings.json``
    so a live Claude Code session receives the reset,
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

    with file_lock(config_path):
        current_now = read_text_or_none(config_path) or ""
        if current_now != original:
            raise CodeHelperError(
                f"{config_path} changed since it was read — refusing to write "
                "a stale config snapshot; re-run to patch the current file"
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
    provider since removed) reads as ``None`` rather than being echoed back
    — EXCEPT a RETIRED provider name (``model.RETIRED_PROVIDER_NAMES``, e.g.
    ``"ollama"`` before it was renamed to ``"ollama-direct"``), which is
    still echoed back as-is: it names a config.toml this project itself
    wrote before the rename, and callers resolve it via
    ``model.get_provider_for_legacy_read`` rather than plain ``get_provider``.

    Uses ``tomllib`` directly rather than ``_require_tomllib``: that helper
    refuses loudly because ``set-default`` is about to WRITE, and a missing
    verifier there would mean writing unverified. This function only reads,
    and its whole contract is to degrade to ``None`` instead of raising.
    """
    from codehelper.services.model import get_provider_for_legacy_read

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
    try:
        get_provider_for_legacy_read(name)
    except CodeHelperError:
        return None
    return name


def apply_set_default(
    paths: Paths,
    *,
    agent: Agent,
    provider: Provider,
    model: str,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Patch Codex's ``~/.codex/config.toml`` to default onto ``agent``/``provider``/``model``.

    Orchestrates the whole ``set-default`` flow: resolve patch -> apply ->
    verify -> confirm/force gate (raising on refusal) -> commit. A single
    file — no paired catalog: an earlier version of this function also wrote
    ``~/.codex/model.json`` for the context window, but Codex's ``ModelInfo``
    requires a per-entry ``base_instructions`` string, and an empty
    synthesized one silently replaces Codex's real system prompt for the
    default session (see ``render.openai_toml_body``). The context window
    now rides in ``config.toml`` itself, as the plain ``model_context_window``
    key (:data:`_TOP_LEVEL_KEYS`), and only when :func:`resolve_default_patch`
    resolves one via :func:`~codehelper.services.render.uniform_context_window`
    — never a guessed floor, and never a removal either: an unrecognized
    model leaves an existing ``model_context_window`` line untouched, since
    it may be the user's own manual setting (see :data:`_SKIP`). A stale
    ``model_catalog_json`` line left by an older version of this tool is
    scrubbed by the same patch (see :data:`_TOP_LEVEL_KEYS`'s docstring), so
    re-running ``set-default`` after upgrading is itself the migration.
    Mirrors ``wrappers.install_wrapper``'s shape (paths spec, dry_run, force,
    confirm) so the CLI handler stays a thin shell.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — called ONLY when a
            real, visible write is about to happen and neither ``--force`` nor
            an existing no-op short-circuit applies (``preview`` is a unified
            diff via :func:`diff_preview`). ``None`` behaves like "always
            refuse" (the fail-fast-off-a-TTY default the CLI layer supplies).

    Returns:
        True if anything was written (or would be, under ``--dry-run``);
        False on a true no-op (config.toml already matches).

    Raises:
        CodeHelperError: incompatible agent/provider, unparseable existing
            config.toml, a patch that fails its own post-write verification,
            or a refused overwrite (no ``--force``/confirmation).
    """
    patch = resolve_default_patch(agent, provider, model)

    config_path = paths.codex_main_config()
    original = read_text_or_none(config_path) or ""
    _verify_toml_or_refuse(original, context=str(config_path))

    patched = patch_config_toml(original, patch)
    _verify_patch_applied(original, patched, patch)

    config_changed = patched != original
    if not config_changed:
        print("no changes to config.toml")
        return False

    preview = diff_preview(original, patched)
    if dry_run:
        print(preview or "(no textual change)")
        print(f"would write {config_path}")
        return True

    if not force and not (confirm and confirm(config_path, preview)):
        raise CodeHelperError(
            f"about to patch {config_path} — refusing without "
            f"confirmation (use --force, or re-run interactively)"
        )

    with file_lock(config_path):
        current_now = read_text_or_none(config_path) or ""
        if current_now != original:
            raise CodeHelperError(
                f"{config_path} changed since it was read — refusing to write "
                "a stale config snapshot; re-run to patch the current file"
            )
        _rotate_backups(_config_backup_slots(paths), current=original)
        atomic_write(config_path, patched, mode=None)
        print(f"wrote {config_path} (backup: {paths.codex_main_config_backup(1)})")

    return True


@dataclass(frozen=True)
class _RestorePlan:
    """Everything :func:`restore_default` needs, read once before any write."""

    config_path: Path
    backup_path: Path
    backup_body: str
    current: str

    @property
    def config_changed(self) -> bool:
        return self.current != self.backup_body


def _restore_plan(paths: Paths, *, slot: int) -> _RestorePlan:
    """Read the backup slot and the current file a restore would overwrite.

    Raises:
        CodeHelperError: no config backup exists at ``slot``.
    """
    backup_path = paths.codex_main_config_backup(slot)
    backup_body = read_text_or_none(backup_path)
    if backup_body is None:
        raise CodeHelperError(f"no backup found at {backup_path}; nothing to restore")

    config_path = paths.codex_main_config()
    return _RestorePlan(
        config_path=config_path,
        backup_path=backup_path,
        backup_body=backup_body,
        current=read_text_or_none(config_path) or "",
    )


def _confirm_restore(plan: _RestorePlan, *, force: bool, confirm) -> None:
    """Confirm the restore before ANY file is written.

    Raises:
        CodeHelperError: the overwrite is refused (no ``--force``, no
            confirmation).
    """
    if not plan.config_changed:
        return
    preview = diff_preview(plan.current, plan.backup_body)
    if not force and not (confirm and confirm(plan.config_path, preview)):
        raise CodeHelperError(
            f"about to restore {plan.config_path} from {plan.backup_path} — "
            f"refusing without confirmation (use --force)"
        )


def restore_default(
    paths: Paths,
    *,
    slot: int = 1,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Restore ``~/.codex/config.toml`` from slot ``slot``.

    Does NOT itself create a new backup slot — restoring is the "put it back"
    operation, not a fresh edit to archive. If the user runs ``set-default``
    again afterwards, THAT invocation backs up the just-restored state.

    No catalog to restore in step — ``apply_set_default`` no longer pairs a
    ``~/.codex/model.json`` write with the config.toml patch (see its
    docstring), so a restore here is a single-file operation.

    Raises:
        CodeHelperError: no config backup exists at ``slot``, or the overwrite
            is refused (no ``--force``/confirmation) — restoring can still
            destroy a DIFFERENT current config.toml if it was edited (by
            hand, or via another tool) since the backup was taken.
    """
    plan = _restore_plan(paths, slot=slot)

    if not plan.config_changed:
        print("no changes")
        return False

    if dry_run:
        # --dry-run never prompts and never refuses — it only previews, same
        # ordering as apply_set_default.
        print(f"would restore {plan.config_path} from {plan.backup_path}")
        return True

    _confirm_restore(plan, force=force, confirm=confirm)

    with file_lock(plan.config_path):
        current_config = read_text_or_none(plan.config_path) or ""
        if current_config != plan.current:
            raise CodeHelperError(
                f"{plan.config_path} changed since it was read — refusing to "
                "restore over a concurrent change; re-run to restore"
            )
        atomic_write(plan.config_path, plan.backup_body, mode=None)
        print(f"restored {plan.config_path} from {plan.backup_path}")

    return True
