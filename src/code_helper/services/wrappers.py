"""Generated-script lifecycle: render → install → describe.

A "wrapper" is a small bash script in ``~/.local/bin`` that points a coding
agent at a model backend. What a wrapper *is* now lives in
``services/spec.py`` (the resolved agent × provider × model combination) and
``services/render.py`` (how that becomes a script body); this module owns only
the lifecycle around it — writing the file, reporting what is installed, and
formatting listings.

The ``auth`` mode (carried by the provider) decides how sensitive the on-disk
script is:

- ``"literal"`` / ``"none"`` — no real credential in the file, mode ``0o755``.
- ``"secret"`` — a real key is embedded in plain text, mode ``0o700``
  (owner-only).

**Ownership.** Every generated script carries a marker comment
(``render.MARKER_PREFIX``) on its second line, and :func:`is_managed` reads it
back. This reverses the project's earlier "no ownership guard" position, and
the reason is that the position's precondition disappeared: overwriting by name
was safe while the set of names was closed and curated (``get_spec`` admitted
exactly three), but a user-chosen ``--alias`` can name anything on ``PATH`` —
including ``~/.local/bin/claude``, a real working symlink on a normal install.
Presets still overwrite freely; only a *foreign* file triggers the guard.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.model import (
    Agent,
    BaseUrlPolicy,
    ConfigShape,
    Provider,
    get_provider,
    with_base_url,
)
from code_helper.services.naming import validate_alias
from code_helper.services.paths import Paths
from code_helper.services.render import (
    CATALOG_MANAGED_BY_KEY,
    CATALOG_MANAGED_BY_VALUE,
    MARKER_PREFIX,
    render_legacy_script,
    render_script,
)
from code_helper.services.spec import (
    PRESETS,
    Preset,
    TierModels,
    WrapperSpec,
    build_spec,
    get_preset,
    preset_names,
    spec_from_preset,
    suggest_alias,
)

__all__ = [
    # re-exported so existing imports keep working
    "WrapperSpec",
    "TierModels",
    "Preset",
    "PRESETS",
    "Agent",
    "Provider",
    "build_spec",
    "get_preset",
    "preset_names",
    "spec_from_preset",
    "suggest_alias",
    "render_script",
    # lifecycle
    "install_wrapper",
    "is_installed",
    "is_managed",
    "spec_from_installed",
    "token_from_installed",
    "profile_from_installed",
    "list_wrappers",
    "describe_wrapper",
    "describe_all",
    "discover_managed",
    "valid_default_wrapper",
    "get_spec",
    "WRAPPERS",
]

#: Owner-only. A "secret" wrapper carries a real credential in plain text —
#: group/other must have NO bits. Anything else is 0o755, a normal executable.
_MODE_SECRET = 0o700
_MODE_LITERAL = 0o755


def get_spec(name: str) -> WrapperSpec:
    """Return the resolved spec for the preset called ``name``.

    Kept as the preset lookup so callers written against the old flat registry
    keep working.

    Raises:
        CodeHelperError: unknown name. Fails BEFORE any token is resolved or
            file touched — a typo must not trigger a secret prompt.
    """
    return spec_from_preset(get_preset(name))


#: Backwards-compatible view of the preset registry as resolved specs.
WRAPPERS: list[WrapperSpec] = [spec_from_preset(p) for p in PRESETS]


def _mode_for(spec: WrapperSpec) -> int:
    return _MODE_SECRET if spec.auth == "secret" else _MODE_LITERAL


def is_installed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` currently exists.

    Pure existence — says nothing about who wrote it. Use :func:`is_managed`
    for that. The name is no longer validated against a registry here: an
    ad-hoc alias has no preset, and ``script_for`` already refuses anything
    that is not a single path component.
    """
    return paths.script_for(name).exists()


def _marker_at(path: Path) -> bool:
    """True iff ``path`` carries our marker on its first or second line.

    Reads only the first couple of lines. Anything unreadable (a binary, a
    permission error, a dangling symlink) counts as *not* ours — the safe
    answer, since it makes the guard refuse rather than clobber. The single
    marker-sniff behind :func:`is_managed` (which resolves the path from a
    wrapper name) and :func:`_is_ours_marker_only` (which takes a raw path for
    the OPENAI_TOML siblings), so the read order and the "unreadable = not
    ours" exception tuple live in one place.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(2):
                line = fh.readline()
                if not line:
                    break
                if line.startswith(MARKER_PREFIX):
                    return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def is_managed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` exists AND carries our marker.

    Reads only the first couple of lines. Anything unreadable (a binary, a
    permission error, a dangling symlink) counts as *not* ours — the safe
    answer, since it makes the guard refuse rather than clobber.
    """
    return _marker_at(paths.script_for(name))


def _installed_marker_provider_is_secret(paths: Paths, name: str) -> bool:
    """True iff the installed wrapper's OWN marker names a secret-auth provider.

    A narrower, more failure-tolerant cousin of :func:`spec_from_installed`,
    used ONLY as the fallback in :func:`_discards_only_secret` (cycle-review
    finding, PR #12 round 3): the marker's ``agent=``/``provider=`` fields are
    parsed straight from the wrapper script and never depend on an
    ``OPENAI_TOML`` sibling profile existing or parsing — unlike
    ``spec_from_installed``, which additionally needs the profile to recover
    the MODEL and returns ``None`` (a "could not reconstruct a full spec"
    answer) the moment that profile is missing or corrupt. That ``None`` is
    correct for ``spec_from_installed``'s own contract, but
    ``_discards_only_secret`` was reading it as "not a secret, safe to
    replace" — silently destroying an ``OPENAI_TOML`` secret wrapper's only
    token the instant its profile sibling went missing, with no ``--force``
    needed. This function answers the one narrower question the guard
    actually needs — "was this a secret provider?" — from information that
    survives a missing/corrupt profile.

    Returns False (never raises) for a missing/unmarked/unrecognised file —
    the same fail-open-to-"not secret" default the guard already had, just no
    longer reachable via a corrupt profile specifically.
    """
    body = _read_text_or_none(paths.script_for(name))
    if body is None:
        return False
    marker = next(
        (ln for ln in body.split("\n")[:2] if ln.startswith(MARKER_PREFIX)), None
    )
    if marker is None:
        return False
    fields = dict(re.findall(r"(\w+)=([^,()\s]+)", marker[len(MARKER_PREFIX) :]))
    provider_name = fields.get("provider")
    if not provider_name:
        return False
    try:
        return get_provider(provider_name).auth == "secret"
    except CodeHelperError:
        return False


def spec_from_installed(paths: Paths, name: str) -> WrapperSpec | None:
    """Reconstruct the spec of an installed wrapper from its own marker line.

    The marker records ``agent``/``provider``/``shape``; the model comes back
    out of the rendered body. Together that is every axis, which is what lets
    ``edit-token`` rotate a credential WITHOUT re-expanding a preset from
    scratch — the bug that silently reverted a user's ``--model`` choice — and
    what lets it reach a wrapper built from the axes, which no preset lookup
    can resolve because no preset describes it.

    Returns None when the file is missing, unreadable, unmarked, or records
    something this version does not recognise: every caller must be able to
    fall back to the preset path, so this never raises.
    """
    body = _read_text_or_none(paths.script_for(name))
    if body is None:
        return None

    marker = next(
        (ln for ln in body.split("\n")[:2] if ln.startswith(MARKER_PREFIX)), None
    )
    if marker is None:
        # A markerless file predates the marker, so its axes are not recorded
        # anywhere — but if a preset of this name renders to the same body
        # (any model), that preset supplies them and the FILE supplies the
        # models. Without this, rotating a legacy install customized with
        # ``--model`` reverted it to the preset defaults: the very bug the
        # installed-spec lookup exists to prevent, just one release older.
        return _spec_from_legacy_body(name, body)

    fields = dict(re.findall(r"(\w+)=([^,()\s]+)", marker[len(MARKER_PREFIX) :]))

    # The OPENAI_TOML wrapper body embeds no model — it lives in the sibling
    # TOML profile — so recover it from there rather than the body. The other
    # shapes carry the model in the rendered script itself. Read the profile
    # ONCE here (rather than letting _model_from_toml_data and, further down,
    # _base_url_from_toml_data each trigger their own read-and-parse) —
    # toml_profile is threaded through both lookups below.
    shape_field = fields.get("shape")
    toml_profile = (
        _toml_profile_data(paths, name)
        if shape_field == ConfigShape.OPENAI_TOML.value
        else None
    )
    if shape_field == ConfigShape.OPENAI_TOML.value:
        model = _model_from_toml_data(toml_profile)
    else:
        model = _model_from_body(body)

    if not all((fields.get("agent"), fields.get("provider"), model)):
        return None

    # ALL tiers, not just the one that names the wrapper. A single model would
    # let build_spec synthesize uniform tiers, which silently rewrites the two
    # tiers it did not read — and ``glm``, whose three tiers genuinely differ,
    # is the whole reason presets still exist.
    tiers = _tiers_from_body(body)

    try:
        provider_obj = get_provider(fields["provider"])
    except CodeHelperError:
        return None

    # Round-trip rule: the FILE wins for anything but a FIXED base_url. This
    # is a direct consequence of edit-token's own contract just above ("rotate
    # a credential WITHOUT re-expanding a preset from scratch" — the same bug
    # class as silently reverting --model) applied to base_url: an installed
    # wrapper is a complete record of itself, and a credential rotation is not
    # licence to change an unrelated setting. For FIXED the registry wins
    # instead — there IS no user-supplied value on that axis, so the file
    # could not have recorded a legitimate override, only a hand-edit; siding
    # with the registry there is what lets a genuine address change (e.g.
    # z.ai moving domains) reach already-installed wrappers on the next
    # edit-token, exactly as the docstring above intends for the model.
    if provider_obj.base_url_policy is not BaseUrlPolicy.FIXED:
        if shape_field == ConfigShape.OPENAI_TOML.value:
            recovered_url = _base_url_from_toml_data(toml_profile, fields["provider"])
        else:
            recovered_url = _env_value(body, "ANTHROPIC_BASE_URL")
        if recovered_url:
            try:
                provider_obj = with_base_url(provider_obj, recovered_url)
            except CodeHelperError:
                # The recovered value came from a hand-edited or truncated
                # file (ANTHROPIC_BASE_URL line / TOML base_url), not from
                # our own renderer — validate_base_url can reject it (bad
                # scheme, control chars, ...). This function's whole contract
                # is that it never raises; a malformed recovered address is
                # the same "unrecoverable" outcome as a missing one, not an
                # exception for the caller (edit-token/add --alias) to catch.
                return None
        elif provider_obj.base_url_policy is BaseUrlPolicy.REQUIRED:
            # No registry fallback exists for REQUIRED, and the file didn't
            # carry one either (a truncated profile, a hand-edited wrapper) —
            # returning a spec with an empty base_url here would let
            # edit-token silently reinstall the wrapper pointed at nothing.
            # Refusing to reconstruct is the same fail-safe as the "not
            # all(...)" check above for a missing model.
            return None
        # else: OVERRIDABLE with nothing recovered — the registry default on
        # provider_obj (untouched by with_base_url) stands, which is correct:
        # there is a real default, so refusing here would be needless.

    try:
        return build_spec(
            agent=fields["agent"],
            provider=provider_obj,
            model=model,
            alias=name,
            # The RECORDED shape, not a re-derived one. `claude × ollama` is
            # compatible with both shapes and `resolve_shape` prefers the env
            # one, so dropping this turns a wrapper installed as
            # `ollama launch` into an `ANTHROPIC_*` block — not a changed
            # model but a changed mechanism.
            shape=ConfigShape(fields["shape"]) if fields.get("shape") else None,
            tier_models=tiers,
            subagent_model=_env_value(body, "CLAUDE_CODE_SUBAGENT_MODEL"),
            profile_name=unquote(fields["profile"]) if fields.get("profile") else None,
        )
    except (CodeHelperError, ValueError):
        # A marker naming an agent/provider/shape this build no longer knows,
        # or a pairing that is no longer valid (ValueError comes from the
        # ConfigShape lookup). Not our problem to resolve here.
        return None


def token_from_installed(paths: Paths, name: str, provider_name: str) -> str | None:
    """Recover a secret token from an installed wrapper we can fully trust.

    The wrapper is the durable source of truth when the profile cache is
    missing: ``add`` already embeds the selected token in the generated file.
    Do not scrape arbitrary files, though. ``spec_from_installed`` first proves
    that the file is one of our recognised wrappers and reconstructs its
    provider/shape; only then is the shape-specific export read.

    Returns ``None`` for a missing, foreign, malformed, non-secret, or
    provider-mismatched wrapper. The token value is intentionally kept inside
    this service and is never logged.
    """
    spec = spec_from_installed(paths, name)
    if spec is None or spec.auth != "secret" or spec.provider.name != provider_name:
        return None

    body = _read_text_or_none(paths.script_for(name))
    if body is None:
        return None
    if spec.shape is ConfigShape.ANTHROPIC_ENV:
        return _env_value(body, "ANTHROPIC_AUTH_TOKEN") or None
    if spec.shape is ConfigShape.OPENAI_TOML:
        return _env_value(body, spec.token_env_var) or None
    return None


def profile_from_installed(paths: Paths, name: str) -> str | None:
    """Return the token profile recorded by a managed wrapper, if any.

    Wrappers written before profiles were introduced have no such metadata, so
    ``None`` is normal and deliberately distinct from a profile named
    ``default``.
    """
    spec = spec_from_installed(paths, name)
    if spec is None:
        return None
    return spec.profile_name


def _spec_from_legacy_body(name: str, body: str) -> WrapperSpec | None:
    """Spec for a markerless wrapper, if a same-named preset explains it.

    The preset supplies the axes the missing marker would have recorded; the
    body supplies the models, so a legacy install customized with ``--model``
    keeps that choice. Returns None unless re-rendering the result reproduces
    the file's every non-token byte — the same proof-of-authorship the guard
    uses, so this recognises no file the guard would refuse.
    """
    try:
        preset = spec_from_preset(get_preset(name))
    except CodeHelperError:
        return None

    candidate = _respec_from_body(preset, body)
    if candidate is None:
        return None

    token = _env_value(body, "ANTHROPIC_AUTH_TOKEN") or ""
    if _strip_token(body) != _strip_token(render_legacy_script(candidate, token)):
        return None
    return candidate


def _env_value(body: str, var: str) -> str | None:
    """The single-quoted value exported to ``var`` in ``body``, or None.

    The renderer single-quotes every value, so recovery is unquoting rather
    than shell parsing; the doubled-quote escape is reversed to match.
    """
    found = re.search(rf"^export {var}='(.*)'$", body, re.MULTILINE)
    return found.group(1).replace("'\"'\"'", "'") if found else None


def _tiers_from_body(body: str) -> TierModels | None:
    """The per-tier models a rendered env-shape body carries, or None.

    None for the launch shape, which has no tiers — ``build_spec`` then does
    the right thing for that shape on its own.
    """
    haiku = _env_value(body, "ANTHROPIC_DEFAULT_HAIKU_MODEL")
    sonnet = _env_value(body, "ANTHROPIC_DEFAULT_SONNET_MODEL")
    opus = _env_value(body, "ANTHROPIC_DEFAULT_OPUS_MODEL")
    if None in (haiku, sonnet, opus):
        return None
    return TierModels(haiku=haiku, sonnet=sonnet, opus=opus)


def _model_from_body(body: str) -> str | None:
    """The model that NAMES a rendered wrapper, or None.

    The sonnet tier for the env shape (the mid tier is what a user means by
    "the model" when tiers differ) and the ``--model`` argument for the launch
    shape. This identifies the wrapper; it does not describe it — see
    :func:`_tiers_from_body` for the full env-shape configuration.

    Returns None for the ``OPENAI_TOML`` shape: its wrapper body is just
    ``exec codex --profile <alias> "$@"`` and embeds no model — the model lives
    in the sibling ``~/.codex/<alias>.config.toml`` profile, read by
    :func:`_model_from_toml_data`.
    """
    sonnet = _env_value(body, "ANTHROPIC_DEFAULT_SONNET_MODEL")
    if sonnet is not None:
        return sonnet
    found = re.search(r"--model '(.*?)' --", body)
    return found.group(1).replace("'\"'\"'", "'") if found else None


def _toml_unescape(value: str) -> str:
    """Reverse :func:`render.toml_string` for a TOML basic-string body.

    Only the escapes ``toml_string`` emits are handled (``\\\\``, ``\\"``,
    ``\\n``, ``\\r``, ``\\t``, ``\\uXXXX``) — the values we write back out are
    the only values this ever reads. Used to recover the ``model`` from a
    profile we wrote, so a model containing a quote or a newline round-trips
    instead of being read back with its escapes still literal.
    """
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
            if nxt == '"':
                out.append('"')
                i += 2
                continue
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "r":
                out.append("\r")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt == "u" and i + 5 < len(value) + 1:
                hexpart = value[i + 2 : i + 6]
                if len(hexpart) == 4 and all(
                    c in "0123456789abcdef" for c in hexpart.lower()
                ):
                    out.append(chr(int(hexpart, 16)))
                    i += 6
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _toml_profile_data(paths: Paths, alias: str) -> dict | None:
    """The parsed ``<alias>.config.toml`` profile, or ``None`` if unrecoverable.

    Called once by ``spec_from_installed`` per lookup against an OPENAI_TOML
    wrapper, with the result passed to both :func:`_model_from_toml_data` and
    :func:`_base_url_from_toml_data` — a single read and a single
    ``tomllib.loads`` serving both extractions, rather than each doing its
    own independent read-and-parse of the same file.

    ``tomllib`` (3.11+, this project's floor) parses it correctly; the
    ``ModuleNotFoundError`` branch below is a defensive fallback for an
    interpreter below that floor (reachable only if the package was installed
    with ``requires-python`` bypassed) — line regexes read the ``model`` and
    ``[model_providers.X].base_url`` keys this renderer is the only writer of,
    assembled into the same ``{"model": ..., "model_providers": {name: {...}}}``
    shape ``tomllib.loads`` would produce, so both extractors below can stay
    format-agnostic. Unreadable, absent, or UNPARSEABLE → None, so the caller
    falls back to the preset path rather than raising — this function's whole
    contract, inherited by :func:`spec_from_installed`, is that it never
    raises. A hand-edited or truncated profile (reachable the moment a user
    touches ``~/.codex`` by hand) makes ``tomllib.loads`` raise
    ``TOMLDecodeError``, a ``ValueError`` subclass and NOT a
    :class:`CodeHelperError` — left uncaught it used to escape all the way to
    ``edit-token``/``add --alias`` as a raw traceback, bypassing the ownership
    guard whose entire job is to handle a bad file in the way (and which
    ``--force`` could otherwise rescue).
    """
    profile = _read_text_or_none(paths.codex_config_for(alias))
    if profile is None:
        return None
    try:
        import tomllib  # py3.11+, this project's floor
    except ModuleNotFoundError:
        # Defensive fallback below the floor: this renderer is the only
        # writer of the profile, so a plain line-regex match per key is
        # sufficient — tomllib's validation is not needed for a file we
        # authored. A malformed line simply fails to match, which is the same
        # "unrecoverable → None" outcome the tomllib branch gives for a
        # ValueError.
        model_found = re.search(r'^model = "(.*)"$', profile, re.MULTILINE)
        table_found = re.search(r"^\[model_providers\.(\S+)\]$", profile, re.MULTILINE)
        base_url_found = None
        if table_found:
            # Scope the base_url search to THIS table's body — from the end of
            # its header to the next top-level `[...` header (any table) or
            # EOF — never the whole file. This renderer only ever writes one
            # [model_providers.X] table per profile, but a hand-edited or
            # legacy-adopted file (reachable via --force) could carry more
            # than one; an unscoped search would attribute a sibling table's
            # base_url to this one, exactly the ambiguity tomllib.loads does
            # not have because it naturally nests per table.
            table_body_start = table_found.end()
            next_header = re.search(r"^\[", profile[table_body_start:], re.MULTILINE)
            table_body_end = (
                table_body_start + next_header.start() if next_header else len(profile)
            )
            base_url_found = re.search(
                r'^base_url = "(.*)"$',
                profile[table_body_start:table_body_end],
                re.MULTILINE,
            )
        return {
            "model": _toml_unescape(model_found.group(1)) if model_found else None,
            "model_providers": {
                table_found.group(1): {
                    "base_url": _toml_unescape(base_url_found.group(1))
                    if base_url_found
                    else None
                }
            }
            if table_found
            else {},
        }

    try:
        return tomllib.loads(profile)
    except ValueError:
        # tomllib.TOMLDecodeError is a ValueError subclass — a hand-edited or
        # truncated profile, not this module's problem to resolve here.
        return None


def _model_from_toml_data(data: dict | None) -> str | None:
    """The ``model`` value from an already-parsed ``<alias>.config.toml`` profile.

    The OPENAI_TOML wrapper body carries no model (it only dispatches
    ``codex --profile <alias>``), so ``edit-token``/``spec_from_installed``
    cannot recover the model from the wrapper the way the other shapes do —
    the model is in the TOML profile written alongside it. Pure lookup, no IO
    of its own: the caller reads and parses the profile once via
    :func:`_toml_profile_data` and passes the result here (and to
    :func:`_base_url_from_toml_data`) rather than each doing its own read.
    """
    if data is None:
        return None
    model = data.get("model")
    return model if isinstance(model, str) and model else None


def _base_url_from_toml_data(data: dict | None, provider_name: str) -> str | None:
    """The ``base_url`` value from ``[model_providers.<provider_name>]``.

    Same "already-parsed data in, pure lookup out" shape as
    :func:`_model_from_toml_data` — only ever CALLED for a provider whose
    ``base_url_policy is not FIXED`` (see :func:`spec_from_installed`): for
    everything else the registry value is authoritative and this function is
    not consulted, which is what lets ``edit-token`` pick up a registry
    address change for a fixed provider while a runtime-``base_url`` provider
    (a self-hosted LiteLLM proxy) keeps exactly the address the user gave it
    at ``add`` time — rotating a token is not licence to change that.
    """
    if data is None:
        return None
    table = data.get("model_providers", {})
    provider_table = table.get(provider_name, {}) if isinstance(table, dict) else {}
    base_url = (
        provider_table.get("base_url") if isinstance(provider_table, dict) else None
    )
    return base_url if isinstance(base_url, str) and base_url else None


def _read_text_or_none(script: Path) -> str | None:
    """``script``'s text, or None if it is not decodable as UTF-8.

    The idempotence check runs BEFORE the ownership guard, so an undecodable
    file here must not raise — otherwise the guard it feeds never runs and a
    binary in the way aborts with a traceback instead of the guard's message
    (or its ``--force`` override). ``is_managed`` already treats unreadable as
    "not ours"; this keeps the earlier read consistent with it.
    """
    try:
        return script.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _is_ours(paths: Paths, spec: WrapperSpec, token: str) -> bool:
    """True iff the file at ``spec.alias`` is one we may replace unasked.

    Two ways to qualify:

    - it carries the marker (:func:`is_managed`) — the normal case; or
    - it is byte-identical to what the PREVIOUS, markerless release would
      have written for this same spec.

    The second clause is the migration path. The marker did not exist before
    this branch, so every already-installed wrapper lacks it, and without this
    the guard would classify the tool's own prior output as a third-party file
    — telling users it "was not created by code-helper" about a file it did
    create, and leaving ``edit-token`` (which has no ``--force``) with no way
    forward at all. It is deliberately an EXACT body match against a
    regenerated legacy render, not a heuristic: a file that differs by one
    byte from what we would have written is not ours, and still needs
    ``--force`` or a confirm.

    The token is compared structurally rather than by value: a legacy wrapper
    embeds whatever token it was installed with, and ``edit-token`` — the very
    command this migration path exists to unblock — is by definition called
    with a DIFFERENT one. Matching on the token would therefore fail in the
    single case that matters most. Everything else in the body must still
    match exactly.

    The comparison is made against a spec rebuilt from the FILE's own models
    rather than the caller's, so a legacy wrapper installed with ``--model``
    is recognised too. Without that, the check only ever matched preset
    defaults, and a customized legacy install dead-ended on a message telling
    the user to pass ``--force`` — a flag ``edit-token`` does not accept.
    Recognition still proves authorship: a body only matches if re-rendering
    it reproduces every non-token byte.
    """
    if is_managed(paths, spec.alias):
        return True
    existing = _read_text_or_none(paths.script_for(spec.alias))
    if existing is None:
        return False

    candidates = [spec]
    as_written = _respec_from_body(spec, existing)
    if as_written is not None:
        candidates.append(as_written)

    stripped = _strip_token(existing)
    return any(
        stripped == _strip_token(render_legacy_script(candidate, token))
        for candidate in candidates
    )


def _discards_only_secret(paths: Paths, spec: WrapperSpec, script: Path) -> bool:
    """True iff installing ``spec`` would destroy the only copy of a token.

    A ``secret``-auth wrapper carries its token nowhere but the generated
    script: there is no config file, and the provider's ``*_API_KEY`` env var
    is a possible *source*, not a guaranteed copy — a user who typed the token
    at the prompt has it nowhere else. Replacing such a file with a wrapper
    that never asked for a token therefore loses it irrecoverably, and
    ``--alias`` is what makes that reachable by a typo.

    Deliberately NOT "the body changed". Prompting on any content change would
    fire on every model bump and every token rotation — ``add``-as-update, the
    tool's primary use — and a prompt seen constantly is one users learn to
    dismiss, which would cost more safety than it buys. The two conditions are:

    - the OUTGOING file is a ``secret`` wrapper (something is at stake), and
    - the INCOMING one is not (nobody was asked for a replacement token).

    secret → secret is therefore silent on purpose: the old token does go away,
    but the user was just prompted for the new one, so it is a deliberate
    replacement rather than a silent loss.
    """
    if spec.auth == "secret":
        return False
    installed = spec_from_installed(paths, script.name)
    if installed is not None:
        return installed.auth == "secret"
    # spec_from_installed returned None. That is ambiguous on its own: it
    # means either "not one of our wrappers" (truly nothing at stake) OR
    # "an OPENAI_TOML wrapper whose sibling profile is missing/corrupt, so
    # the model could not be recovered" (a secret token IS still at stake —
    # the marker alone already proves the provider). Fall back to the
    # narrower, profile-independent check before concluding "not a secret".
    return _installed_marker_provider_is_secret(paths, script.name)


def _respec_from_body(spec: WrapperSpec, body: str) -> WrapperSpec | None:
    """``spec`` with the models the FILE actually carries, or None.

    Only the models are taken from the body — agent, provider, and shape stay
    the caller's, so this can widen *which model* counts as ours but never
    which agent or provider does.
    """
    tiers = _tiers_from_body(body)
    model = _model_from_body(body)
    if model is None:
        return None
    try:
        return build_spec(
            agent=spec.agent,
            provider=spec.provider,
            model=model,
            alias=spec.alias,
            shape=spec.shape,
            tier_models=tiers,
            subagent_model=_env_value(body, "CLAUDE_CODE_SUBAGENT_MODEL"),
        )
    except CodeHelperError:
        return None


def _strip_token(body: str) -> str:
    """``body`` with the embedded auth token blanked, for structural comparison."""
    return "\n".join(
        "export ANTHROPIC_AUTH_TOKEN="
        if line.startswith("export ANTHROPIC_AUTH_TOKEN=")
        else line
        for line in body.split("\n")
    )


def _is_ours_marker_only(path: Path) -> bool:
    """True iff ``path`` carries our marker on its first or second line.

    The OPENAI_TOML TOML profile carries the same marker comment the bash
    wrapper does, which is what lets the ownership guard recognise it as ours.
    That file has NO pre-marker legacy form (the shape is new), so unlike
    :func:`_is_ours` there is no byte-identical-to-legacy migration clause —
    the marker alone is the proof of authorship. Unreadable (binary, perms, a
    dangling symlink) counts as not ours, so the guard refuses rather than
    clobbers — the same safe answer :func:`is_managed` gives.
    """
    return _marker_at(path)


def _catalog_self_marked(path: Path) -> bool:
    """True iff ``path`` is JSON carrying our own ``managed_by`` field.

    The catalog's own proof of authorship (see ``render.CATALOG_MANAGED_BY_KEY``)
    — independent of any sibling file. A purely structural test
    (``{"version": 1, "models": [...]}`` alone) is too permissive: that shape
    is generic enough that a hand-curated or third-party catalog plausibly
    uses it too, and treating it as proof clobbers a researched
    ``context_window`` with the :data:`_DEFAULT_CONTEXT_WINDOW` floor — silent
    data loss without ``--force``. Requiring our specific marker field closes
    that. Unreadable/non-JSON/missing key → False, the same safe-refuse answer
    every other ownership check in this module gives.
    """
    body = _read_text_or_none(path)
    if body is None:
        return False
    try:
        import json

        data = json.loads(body)
    except ValueError:
        return False
    return (
        isinstance(data, dict)
        and data.get(CATALOG_MANAGED_BY_KEY) == CATALOG_MANAGED_BY_VALUE
    )


def _is_our_catalog(paths: Paths, alias: str) -> bool:
    """True iff the catalog for ``alias`` is one we wrote.

    Proven ONLY by the catalog's own ``managed_by`` field
    (:func:`_catalog_self_marked`) — no sibling-profile fallback. An earlier
    version also accepted "the sibling ``<alias>.config.toml`` profile carries
    our marker" as a second, migration-path proof (mirroring how
    :func:`_is_ours` accepts a byte-identical legacy render for the wrapper).
    That fallback could not distinguish a legacy catalog WE wrote (before the
    ``managed_by`` field existed) from a FOREIGN hand-curated catalog a user
    simply placed next to our already-installed, marker-carrying profile —
    both look identical to it: no ``managed_by``, sibling profile marked.
    Reachable on an ordinary, idempotent re-install (no ``--alias`` typo, no
    edge case): install once, hand-edit the catalog's ``context_window`` to a
    researched value, re-run the SAME install command — the profile
    byte-matches and is skipped, but the catalog no longer byte-matches, so
    :func:`_decide` re-checks ownership, the fallback fires, and the
    researched value is silently flattened back to
    :data:`_DEFAULT_CONTEXT_WINDOW` with no prompt and no ``--force`` (the
    catalog is classified "ours", so it never reaches the foreign-file guard
    at all). :func:`_cleanup_openai_toml_siblings` already chose the
    self-marker-only answer for the DELETE side of this exact ambiguity (see
    its docstring); this brings the OVERWRITE side in line rather than leaving
    it more permissive than a plain deletion. A catalog with no self-marker —
    legacy or foreign, no longer distinguished — now routes to
    ``OVERWRITE_FOREIGN`` like the other two slots, recoverable with
    ``--force`` same as any foreign file; the byte-identical idempotence case
    is handled earlier in :func:`_decide` (SKIP), so this only gates
    non-identical existing catalogs.
    """
    return _catalog_self_marked(paths.codex_catalog_for(alias))


class _FilePlan(NamedTuple):
    """One file a wrapper install writes, and how to treat an existing copy.

    The plan is the shape-driven answer to "which files does this install
    write?" — one entry for a single-shape wrapper, three for ``OPENAI_TOML``
    (wrapper + TOML profile + catalog). Pairing each file with its own
    ``managed_check`` and ``is_wrapper`` flag as data is what lets one
    orchestrator (:func:`_install_plan`) handle every shape: the per-file
    ownership predicate varies (marker+legacy for the wrapper, marker-only for
    the TOML profile, structural-JSON for the catalog) without the orchestrator
    knowing which shape it is looking at.
    """

    path: Path
    body: str
    mode: int
    managed_check: Callable[[Path], bool]
    # Only the bash wrapper can carry a secret, so only it participates in the
    # discard-only-secret check.
    is_wrapper: bool


class _Action(StrEnum):
    """What :func:`_decide` resolved for one plan entry."""

    SKIP = "skip"  # byte-identical file already installed — write nothing
    WRITE = "write"  # no existing file, or ours and not a secret-discard
    OVERWRITE_FOREIGN = "overwrite_foreign"  # an existing file that is not ours
    DISCARD_SECRET = "discard_secret"  # ours, but holds the only copy of a token


#: The refusal message for each guarding action. One place so the ``--force``
#: hint and the "refusing to …" wording cannot drift between code paths.
_REFUSAL: dict[_Action, str] = {
    _Action.OVERWRITE_FOREIGN: (
        "{path} exists and was not created by code-helper — "
        "refusing to overwrite (use --force)"
    ),
    _Action.DISCARD_SECRET: (
        "{path} holds a wrapper whose token exists nowhere else, "
        "and the replacement does not use one — refusing to discard "
        "the only copy of its token (use --force)"
    ),
}


def _decide(paths: Paths, spec: WrapperSpec, f: _FilePlan) -> _Action:
    """Resolve what an install would do with one plan entry.

    Reads the existing file once and runs that file's ownership check once,
    returning the action. The orchestrator consumes the action without
    re-reading or re-checking, which is what keeps a multi-file install from
    reading and re-rendering each file twice — the refusal pass and the write
    pass share the one decision.

    Order matters and mirrors the old single-file lifecycle: an identical file
    short-circuits to SKIP before the ownership check runs (a foreign file that
    *happens* to be byte-identical to what we would write needs no write and no
    prompt), and the discard-only-secret check runs only for the wrapper slot.
    """
    if not f.path.exists():
        return _Action.WRITE
    if _read_text_or_none(f.path) == f.body:
        return _Action.SKIP
    if not f.managed_check(f.path):
        return _Action.OVERWRITE_FOREIGN
    if f.is_wrapper and _discards_only_secret(paths, spec, f.path):
        return _Action.DISCARD_SECRET
    return _Action.WRITE


def _install_plan(
    paths: Paths,
    spec: WrapperSpec,
    plan: list[_FilePlan],
    *,
    dry_run: bool,
    force: bool,
    confirm: Callable[[Path], bool] | None,
) -> bool:
    """Write every file in ``plan``. Return True iff any changed (or would, dry-run).

    One orchestrator for every shape: a single-file wrapper passes a one-entry
    plan, ``OPENAI_TOML`` passes three. The per-file lifecycle — idempotence,
    the ownership guard, the discard-only-secret check, dry-run print or atomic
    write — lives here once, not triplicated per shape.

    Atomic w.r.t. the ownership guard: in a non-dry-run install every foreign
    / discard-only-secret refusal is decided BEFORE any file is written, so a
    refusal in one slot never leaves another slot's file half-written. Each
    file's fate is decided once by :func:`_decide` (which does the reads and
    checks); the dry-run, refusal, and write passes below only consume those
    decisions, so nothing is re-read.

    Raises:
        CodeHelperError: a foreign or secret-discard file is in the way and was
            not confirmed (``force`` / ``confirm``).
    """
    decisions = [(f, _decide(paths, spec, f)) for f in plan]

    if dry_run:
        wrote = False
        for f, action in decisions:
            if action is _Action.SKIP:
                continue
            if action is _Action.OVERWRITE_FOREIGN:
                print(f"would overwrite UNMANAGED file {f.path}")
            elif action is _Action.DISCARD_SECRET:
                print(f"would discard the only copy of {f.path}'s token")
            else:
                print(f"would write {f.path}")
            wrote = True
        return wrote

    # Resolve EVERY refusal before writing ANY file — a foreign file in one
    # slot never leaves another slot's file half-written.
    for f, action in decisions:
        if action in (_Action.OVERWRITE_FOREIGN, _Action.DISCARD_SECRET):
            if not force and (confirm is None or not confirm(f.path)):
                raise CodeHelperError(_REFUSAL[action].format(path=f.path))

    wrote = False
    for f, action in decisions:
        if action is _Action.SKIP:
            continue
        if action is _Action.OVERWRITE_FOREIGN:
            print(f"overwriting unmanaged file {f.path}")
        elif action is _Action.DISCARD_SECRET:
            print(f"discarding the only copy of {f.path}'s token")
        try:
            atomic_write(f.path, f.body, mode=f.mode)
        except OSError as exc:
            # A mid-sequence write failure (disk full, a read-only parent, a
            # broken symlink) surfaces as a clean CodeHelperError instead of a
            # raw traceback. The plan is ordered catalog → profile → wrapper so
            # the on-PATH executable lands LAST: a failure on a sibling never
            # leaves a broken wrapper pointing at a missing profile/catalog,
            # only harmless orphaned siblings the next idempotent install
            # repairs.
            raise CodeHelperError(f"failed to write {f.path}: {exc}") from exc
        print(f"wrote {f.path}")
        wrote = True
    return wrote


def _wrapper_plan(paths: Paths, spec: WrapperSpec, token: str) -> list[_FilePlan]:
    """The one-file plan every non-``OPENAI_TOML`` shape installs."""
    return [
        _FilePlan(
            paths.script_for(spec.alias),
            render_script(spec, token),
            _mode_for(spec),
            lambda _p: _is_ours(paths, spec, token),
            True,
        )
    ]


def _openai_toml_plan(paths: Paths, spec: WrapperSpec, token: str) -> list[_FilePlan]:
    """The three-file plan: model catalog + TOML profile + bash wrapper.

    Ordered so the on-PATH executable lands LAST: if a sibling write fails
    mid-sequence (disk full, permission), no invocable wrapper is left
    pointing at a missing profile/catalog — only harmless orphaned siblings
    that the next idempotent install repairs. The wrapper-first order this
    replaced could leave a broken executable on ``PATH`` during the failure
    window.

    Each slot carries its own ownership check — :func:`_is_ours_marker_only`
    for the wrapper AND the TOML profile (both carry the marker comment;
    OPENAI_TOML is a new shape with no pre-marker legacy form, so the
    legacy byte-match clause :func:`_is_ours` uses for the other shapes would
    only ever match a third-party hand-written ``exec codex --profile`` script
    and adopt it as ours), and :func:`_is_our_catalog` for the catalog (JSON
    cannot carry a comment marker, so authorship is proven by the sibling
    profile's marker). Modes: the wrapper follows :func:`_mode_for` (``0o700``
    for a secret, else ``0o755``); the profile and catalog are ``0o600``
    owner-only.
    """
    from code_helper.services.render import openai_catalog_body, openai_toml_body

    catalog_path = paths.codex_catalog_for(spec.alias)
    config_path = paths.codex_config_for(spec.alias)
    return [
        _FilePlan(
            catalog_path,
            openai_catalog_body(spec),
            0o600,
            lambda _p: _is_our_catalog(paths, spec.alias),
            False,
        ),
        _FilePlan(
            config_path,
            openai_toml_body(spec, str(catalog_path)),
            0o600,
            _is_ours_marker_only,
            False,
        ),
        _FilePlan(
            paths.script_for(spec.alias),
            render_script(spec, token),
            _mode_for(spec),
            _is_ours_marker_only,
            True,
        ),
    ]


def _cleanup_openai_toml_siblings(paths: Paths, alias: str, *, dry_run: bool) -> bool:
    """Remove the ``~/.codex/<alias>.config.toml`` + ``<alias>.model.json`` we wrote.

    Only meaningful when a PREVIOUS install under ``alias`` was OPENAI_TOML and
    the new one is not: the old profile/catalog no longer match anything the
    new wrapper dispatches to, so leaving them is silent clutter (and a stale
    catalog could mislead a later ``codex --profile <alias>`` if the alias is
    ever reused for OPENAI_TOML again).

    Each sibling is gated by ITS OWN ownership proof, independently —
    :func:`_is_ours_marker_only` for the profile, :func:`_catalog_self_marked`
    for the catalog. This is deliberately NOT "the profile's marker decides
    both": an earlier version inferred the catalog's fate from the profile
    alone, which meant a hand-curated catalog sitting next to OUR profile was
    deleted with no ``--force`` and no prompt — the exact thing the
    install-time ownership guard exists to prevent, just reached through a
    different door. A foreign profile or foreign catalog is left untouched,
    matching what the install guard would have refused to overwrite.

    Note this is stricter than :func:`_is_our_catalog` (the install-time
    check, which also accepts the sibling-marker fallback for a catalog
    written before :data:`render.CATALOG_MANAGED_BY_KEY` existed): a DELETE
    has no ``--force`` escape hatch the way an overwrite does, and the
    sibling-marker proof is fundamentally ambiguous for a delete — "the
    profile next to this catalog is ours" cannot distinguish a legacy catalog
    WE wrote from a foreign one a user happened to drop next to our profile.
    An install-time overwrite gated on that proof is at least reversible by
    restoring a backup; an unprompted delete is not, so cleanup only removes
    what the catalog can prove about ITSELF. A legacy catalog with no
    self-marker is deliberately left as a harmless orphan rather than risking
    a foreign file's silent deletion — the safe direction to be wrong in.

    Best-effort by design: called AFTER the wrapper install already succeeded
    (see ``install_wrapper``), so a failure here must never make a successful
    install look failed. An unlink failure is reported on stderr and skipped,
    not raised — unlike the install-time write path, where a failure aborts
    before anything user-visible has changed.

    Returns:
        True iff anything was removed (or, in dry-run, would be) — folded into
        ``install_wrapper``'s own return value so a run whose only effect was
        deleting orphaned siblings is not reported as "no changes".
    """
    import sys

    config_path = paths.codex_config_for(alias)
    catalog_path = paths.codex_catalog_for(alias)

    to_remove = [
        path
        for path, is_ours in (
            (catalog_path, _catalog_self_marked(catalog_path)),
            (config_path, _is_ours_marker_only(config_path)),
        )
        if path.exists() and is_ours
    ]

    changed = False
    for path in to_remove:
        if dry_run:
            print(f"would remove orphaned sibling {path}")
            changed = True
            continue
        try:
            path.unlink()
        except OSError as exc:
            print(
                f"warning: failed to remove orphaned sibling {path}: {exc}",
                file=sys.stderr,
            )
            continue
        print(f"removed orphaned sibling {path}")
        changed = True
    return changed


def install_wrapper(
    paths: Paths,
    spec: WrapperSpec | str,
    *,
    token: str = "",
    model_override: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm: Callable[[Path], bool] | None = None,
) -> bool:
    """Write ``spec``'s script. Return True iff it wrote (or would, in dry-run).

    Idempotent: a byte-identical re-install is a no-op.

    The shape selects the file plan — one file for most shapes
    (:func:`_wrapper_plan`), three for ``OPENAI_TOML``
    (:func:`_openai_toml_plan`: bash wrapper + TOML profile + model catalog) —
    and one orchestrator (:func:`_install_plan`) writes whichever plan it was
    handed, so the ownership guard, dry-run, force, and confirm behaviour are
    identical per file regardless of how many files a shape writes.

    Args:
        paths: Resolved :class:`Paths`.
        spec: A :class:`WrapperSpec`, or a preset name (resolved via
            :func:`get_spec`) for backwards compatibility.
        token: Auth token to embed. Resolved by the CALLER — ``auth_value`` for
            literal providers, ``resolve_token`` for secret ones.
        model_override: Only meaningful with a preset name; ignored when a
            fully-resolved spec is passed (its model is already decided).
        dry_run: Print what would happen, write nothing, ask nothing.
        force: Overwrite a foreign file without asking.
        confirm: Asked before overwriting a foreign file. ``None`` means "no
            way to ask" and is treated as refusal — this is what guarantees a
            non-interactive run FAILS FAST instead of blocking on stdin.

    Raises:
        CodeHelperError: a foreign file is in the way and was not confirmed.
    """
    resolved: WrapperSpec = (
        spec_from_preset(get_preset(spec), model_override=model_override)
        if isinstance(spec, str)
        else spec
    )
    spec = resolved

    plan = (
        _openai_toml_plan(paths, spec, token)
        if spec.shape is ConfigShape.OPENAI_TOML
        else _wrapper_plan(paths, spec, token)
    )
    wrote = _install_plan(
        paths, spec, plan, dry_run=dry_run, force=force, confirm=confirm
    )

    # A non-OPENAI_TOML install under an alias that previously held an
    # OPENAI_TOML install leaves the ``~/.codex/<alias>.*`` siblings orphaned
    # — the new wrapper no longer dispatches ``codex --profile <alias>``. Runs
    # regardless of whether the wrapper slot itself changed (a SKIP re-install
    # under an alias that still carries stale siblings must still clean them
    # up — the siblings are the whole point, not a side effect of the wrapper
    # write), but only for OUR siblings; a refusal above already raised before
    # reaching here, so nothing here is racing an unresolved guard decision.
    # Its own return value is folded into ``wrote`` so a run whose only effect
    # was deleting orphaned siblings is not reported as "no changes".
    if spec.shape is not ConfigShape.OPENAI_TOML:
        cleaned = _cleanup_openai_toml_siblings(paths, spec.alias, dry_run=dry_run)
        wrote = wrote or cleaned

    return wrote


def describe_wrapper(
    spec: WrapperSpec,
    *,
    installed: bool,
    installed_word: str,
    not_installed_word: str,
    profile_name: str | None = None,
    default: bool = False,
) -> str:
    """Format one ``name / install-state / description`` row.

    The single source of truth for this row's column widths (``:12``/``:13``)
    and field order — every caller that lists wrappers with their install
    state (:func:`list_wrappers`, the TUI's wrapper picker, ``edit-token``'s
    picker) renders through here instead of re-deriving the format string, so
    the three menus can't silently drift apart on layout. ``installed_word``/
    ``not_installed_word`` are caller-supplied because callers render in
    different languages (``list_wrappers`` in English, the TUI in Russian) —
    only the format itself is shared.

    ``default=True`` prefixes the row with ``● `` — the marker the TUI's main
    screen (issue #29) draws on the one wrapper that is the
    ``default_wrapper`` for its agent, so the eye lands on it without scanning
    the column. It is a PREFIX to ``spec.name`` (not a new column), so the
    ``:12``/``:13`` alignment every other caller depends on is unchanged.
    ``list_wrappers`` passes ``default=False`` (the default) — the CLI list is
    a flat registry dump and the marker belongs only to the interactive
    screen that can CHANGE which wrapper is default.
    """
    state = installed_word if installed else not_installed_word
    profile = f" [profile: {profile_name}]" if profile_name else ""
    mark = "● " if default else ""
    return f"{mark}{spec.name:12} {state:13} {spec.description}{profile}"


def describe_all(
    paths: Paths,
    specs: Sequence[WrapperSpec],
    *,
    installed_word: str,
    not_installed_word: str,
    defaults: dict[str, str | None] | None = None,
) -> list[tuple[str, str]]:
    """``(name, describe_wrapper(...))`` pairs for a menu built over ``specs``.

    Both ``edit-token``'s picker (over the ``secret``-auth subset) and the
    TUI's ``add`` wrapper picker (over the full registry) build this exact
    shape — the same ``is_installed`` lookup per spec, wrapped by
    :func:`describe_wrapper` — differing only in which specs they iterate and
    which language's install-state words they pass. Sharing the loop here
    means that wiring (not just the row's column widths) can't drift between
    the two menus.

    ``defaults`` is an optional ``{agent_name: alias}`` map (the TUI's main
    screen passes the live ``valid_default_wrapper`` result per agent). When
    given, the row whose ``spec`` matches ``defaults[spec.agent.name]`` is
    rendered with ``default=True`` (the ``●`` marker). ``None`` (the default)
    renders no marker at all — ``list_wrappers`` and ``edit-token``'s picker
    are flat dumps that have no notion of "the default for this agent", so
    they pass nothing and stay marker-free. The map is the CALLER's
    responsibility (it must already have resolved staleness via
    :func:`valid_default_wrapper`) so this function stays a pure function of
    its arguments and does not read ``state.json``.
    """
    return [
        (
            spec.name,
            describe_wrapper(
                spec,
                installed=is_installed(paths, spec.name),
                installed_word=installed_word,
                not_installed_word=not_installed_word,
                profile_name=profile_from_installed(paths, spec.name),
                default=(
                    defaults is not None and defaults.get(spec.agent.name) == spec.name
                ),
            ),
        )
        for spec in specs
    ]


def list_wrappers(paths: Paths, *, print_fn=print) -> None:
    """Print the wrapper registry + whether each is present on disk.

    Read-only (no write). Goes through :func:`describe_all` like the two
    menus do, rather than re-deriving ``script_for(...).exists()`` inline —
    that helper exists to share the ``Paths``/:func:`is_installed` wiring,
    not only the row format.
    """
    for _name, label in describe_all(
        paths,
        WRAPPERS,
        installed_word="installed",
        not_installed_word="not installed",
    ):
        print_fn(label)

    ad_hoc = discover_managed(paths)
    if ad_hoc:
        print_fn("")
        print_fn("ad-hoc wrappers:")
        for name in ad_hoc:
            profile = profile_from_installed(paths, name)
            suffix = f" [profile: {profile}]" if profile else ""
            print_fn(f"{name:12} {'installed':13}{suffix}")


def discover_managed(paths: Paths) -> list[str]:
    """Names of managed wrappers on disk that are NOT presets.

    Without this, a wrapper built from the axes would be invisible to ``list``
    — the preset registry cannot know about it, and there is no state file.
    The marker in the script body is the only record that it is ours.

    A marked file whose name is not a usable alias (a reserved name like
    ``claude``, or anything ``validate_alias`` rejects) is skipped: no command
    in the tool can act on it, so listing it advertises a wrapper the user
    cannot then edit or reinstall.
    """
    if not paths.bin_dir.is_dir():
        return []
    known = set(preset_names())
    found = [
        entry.name
        for entry in paths.bin_dir.iterdir()
        if entry.is_file()
        and entry.name not in known
        and is_managed(paths, entry.name)
        and _is_usable_alias(entry.name)
    ]
    return sorted(found)


def valid_default_wrapper(paths: Paths, agent_name: str) -> str | None:
    """The saved default-wrapper alias for ``agent_name`` if it still exists.

    A saved alias can go stale (uninstalled/renamed), so a reader must fall
    back to ``None`` on a miss rather than highlight a ghost wrapper. Lives
    here (not in ``state.py``) because it needs the live wrapper set, and
    ``state.py`` must not depend on this module — the same one-way edge
    ``secrets.valid_active_profile`` relies on. Checks the single alias
    directly (``is_installed``/``is_managed``) rather than scanning the whole
    ``bin_dir``, since this runs on the TUI's hot render path.

    The alias must also belong to ``agent_name`` — a per-agent consumer must
    never be handed a wrapper that launches a different agent. And the read
    never raises: a malformed alias (path separator, ``.``/``..``) degrades to
    ``None`` before any path arithmetic.
    """
    from code_helper.services.state import default_wrapper

    alias = default_wrapper(paths, agent_name)
    if alias is None:
        return None
    if not _is_usable_alias(alias):
        return None
    # A managed wrapper on disk takes precedence over a same-named preset: the
    # on-disk wrapper's agent is authoritative, not the preset's. Checking the
    # installed wrapper first means a codex wrapper named "glm" (colliding with
    # the claude preset) is correctly handed to codex and withheld from claude.
    if is_installed(paths, alias) and is_managed(paths, alias):
        spec = spec_from_installed(paths, alias)
        if spec is not None and spec.agent.name == agent_name:
            return alias
        return None
    if alias in preset_names():
        if get_preset(alias).agent == agent_name:
            return alias
        return None
    return None


def _is_usable_alias(name: str) -> bool:
    """True iff ``name`` is one the rest of the tool can still act on."""
    try:
        validate_alias(name)
    except CodeHelperError:
        return False
    return True
