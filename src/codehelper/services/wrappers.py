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
import stat
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote, unquote

from codehelper.backends._atomic import atomic_write, read_text_or_none, remove_file
from codehelper.errors import CodeHelperError
from codehelper.services.agents import (
    all_agents,
    load_user_agents_strict,
)
from codehelper.services.agents import (
    get_agent as get_any_agent,
)
from codehelper.services.model import (
    Agent,
    BaseUrlPolicy,
    ConfigShape,
    Provider,
    get_provider_for_legacy_read,
    provider_storage_names,
    with_auth,
    with_base_url,
)
from codehelper.services.naming import is_valid_alias_shape, validate_alias
from codehelper.services.paths import Paths
from codehelper.services.profiles import NewProfileOutcome, validate_new_profile_name
from codehelper.services.render import (
    MARKER_PREFIX,
    render_legacy_script,
    render_script,
)
from codehelper.services.spec import (
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
    "remove_wrapper",
    "rename_wrapper",
    "rename_provider_profile",
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
    "removal_discards_only_secret",
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


def _ownership_marker_at(path: Path) -> bool:
    """True iff ``path`` carries our marker on its first or second line.

    Reads only the first couple of lines. Anything unreadable (a binary, a
    permission error, a dangling symlink) counts as *not* ours — the safe
    answer, since it makes the guard refuse rather than clobber. The single
    marker-sniff behind :func:`is_managed` (which resolves the path from a
    wrapper name) and :func:`_ownership_marker_only` (which takes a raw path for
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
    return _ownership_marker_at(paths.script_for(name))


def removal_discards_only_secret(paths: Paths, name: str) -> bool:
    """True iff removing ``name`` would destroy the only durable copy of a token.

    The removal-flavored counterpart of :func:`_discards_only_secret`, which
    guards the install/overwrite path: that path already refuses to discard
    "the only copy of its token", and ``disable`` deletes with a strictly
    wider blast radius, so it asks the same question before unlinking. A
    wrapper is at risk when its OWN marker names a secret-auth provider (the
    profile-independent check — a missing/corrupt ``OPENAI_TOML`` sibling
    must not lower the guard) AND its token is not durably stored in
    ``credentials.json`` under its provider: an env-sourced token is never
    cached, and an ``add`` that saw a disagreeing cache entry invalidated it,
    so the file can genuinely be the last copy. Literal wrappers carry no
    credential and a cached token survives ``disable`` by contract — False.
    Anything ambiguous (unreadable body, unresolvable provider) counts as
    at-risk, the same fail-closed direction the overwrite guard takes.
    """
    if not is_managed(paths, name):
        return False
    if not _ownership_marker_provider_is_secret(paths, name):
        return False
    provider_name = _installed_provider_name(paths, name)
    token = (
        token_from_installed(paths, name, provider_name)
        if provider_name is not None
        else None
    )
    if not token:
        return True
    from codehelper.services.secrets import credential_for, profile_names

    return token not in (
        credential_for(paths, provider_name, profile)
        for profile in profile_names(paths, provider_name)
    )


def _marker_fields_from(body: str) -> dict[str, str]:
    """The ``key=value`` fields of the ownership marker inside ``body``.

    The ONE parser for the marker's field syntax — the first-two-line scan
    and the ``key=value`` regex — behind :func:`_marker_fields`,
    :func:`_ownership_marker_provider_is_secret`,
    :func:`_installed_provider_name`, and :func:`spec_from_installed`. The
    regex has grown fields twice (``auth=`` #81, ``ctx=`` #83); the next one
    happens here or nowhere. Empty dict when ``body`` carries no marker on
    its first two lines.
    """
    marker = next(
        (ln for ln in body.split("\n")[:2] if ln.startswith(MARKER_PREFIX)), None
    )
    if marker is None:
        return {}
    return dict(re.findall(r"(\w+)=([^,()\s]+)", marker[len(MARKER_PREFIX) :]))


def _marker_fields(paths: Paths, name: str) -> dict[str, str]:
    """The marker fields of the INSTALLED wrapper ``name`` (file-reading form
    of :func:`_marker_fields_from`). Empty dict for a missing/unreadable
    file."""
    body = read_text_or_none(paths.script_for(name))
    if body is None:
        return {}
    return _marker_fields_from(body)


def _installed_provider_name(paths: Paths, name: str) -> str | None:
    """The provider name recorded by the INSTALLED wrapper ``name``, or ``None``.

    Reads ONLY the marker's ``provider=`` field — every caller arrives
    marker-gated (``is_managed``/``discover_managed``), and the marker is the
    same source :func:`spec_from_installed` would resolve the provider from,
    so a full spec reconstruction (body parse, possible TOML profile read,
    ``build_spec`` validation) would buy nothing but a second file read per
    candidate. Survives a missing/corrupt ``OPENAI_TOML`` profile sibling the
    way :func:`_ownership_marker_provider_is_secret` does; ``None`` for a
    wrapper whose provider cannot even be named — not this function's
    problem to guess.
    """
    provider_name = _marker_fields(paths, name).get("provider")
    if not provider_name:
        return None
    try:
        return get_provider_for_legacy_read(provider_name).name
    except CodeHelperError:
        return None


def _ownership_marker_provider_is_secret(paths: Paths, name: str) -> bool:
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
    fields = _marker_fields(paths, name)
    # The RECORDED auth first (issue #81): a wrapper installed with
    # `--auth secret` on an OVERRIDABLE provider (ollama-direct) is secret
    # even though the registry answers "literal" — the registry check below
    # is the fallback for markers predating the auth field, which is all it
    # was ever able to answer for anyway.
    if fields.get("auth") == "secret":
        return True
    provider_name = fields.get("provider")
    if not provider_name:
        return False
    try:
        return get_provider_for_legacy_read(provider_name).auth == "secret"
    except CodeHelperError:
        return False


def _recover_base_url(
    provider_obj: Provider,
    shape_field: str | None,
    toml_profile: dict | None,
    body: str,
    provider_name: str,
) -> Provider | None:
    """Recover the installed wrapper's ``base_url`` onto ``provider_obj``.

    Round-trip rule: the FILE wins for anything but a FIXED base_url. This is
    a direct consequence of edit-token's "rotate a credential WITHOUT
    re-expanding a preset from scratch" contract applied to ``base_url``: an
    installed wrapper is a complete record of itself, and a credential
    rotation is not licence to change an unrelated setting. For FIXED the
    registry wins instead — there is no user-supplied value on that axis, so
    the file could not have recorded a legitimate override, only a hand-edit;
    siding with the registry is what lets a genuine address change (e.g. z.ai
    moving domains) reach already-installed wrappers on the next edit-token.

    Returns the (possibly substituted) provider, or ``None`` when recovery is
    impossible — a malformed recovered URL or a REQUIRED provider with
    nothing to recover. ``None`` is the same "unrecoverable → spec_from_installed
    returns None" outcome as a missing model, keeping that function's
    never-raises contract intact.
    """
    if provider_obj.base_url_policy is BaseUrlPolicy.FIXED:
        return provider_obj

    if shape_field == ConfigShape.OPENAI_TOML.value:
        recovered_url = _base_url_from_toml_data(toml_profile, provider_name)
    else:
        recovered_url = _env_value(body, "ANTHROPIC_BASE_URL")

    if recovered_url:
        try:
            return with_base_url(provider_obj, recovered_url)
        except CodeHelperError:
            # The recovered value came from a hand-edited or truncated file,
            # not our own renderer — validate_base_url can reject it (bad
            # scheme, control chars). Same "unrecoverable" outcome as a
            # missing value, not an exception for the caller to catch.
            return None

    if provider_obj.base_url_policy is BaseUrlPolicy.REQUIRED:
        # No registry fallback exists for REQUIRED, and the file didn't carry
        # one either — returning a spec with an empty base_url would let
        # edit-token silently reinstall pointed at nothing.
        return None

    # OVERRIDABLE with nothing recovered — the registry default on
    # provider_obj (untouched by with_base_url) stands.
    return provider_obj


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
    body = read_text_or_none(paths.script_for(name))
    if body is None:
        return None

    fields = _marker_fields_from(body)
    if not fields:
        # A markerless file predates the marker, so its axes are not recorded
        # anywhere — but if a preset of this name renders to the same body
        # (any model), that preset supplies them and the FILE supplies the
        # models. Without this, rotating a legacy install customized with
        # ``--model`` reverted it to the preset defaults: the very bug the
        # installed-spec lookup exists to prevent, just one release older.
        return _spec_from_legacy_body(name, body)

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
        # The RECORDED auth, honoured the way the recorded shape is (issue
        # #81): `with_auth` re-substitutes the runtime override the wrapper
        # was installed with, so every consumer of this reconstruction —
        # `switch --from-wrapper`, the TUI chipset, `edit-token`, the
        # ownership checks — gets the same answer the apply path would.
        # A marker WITHOUT the field predates the recording (or names a
        # literal wrapper): want_secret=False leaves the registry default
        # standing, which is exactly the pre-#81 behaviour — the migration
        # fallback. `auth=secret` on a provider the registry no longer lets
        # override (a hand-edited marker, a registry downgrade) raises and
        # fails closed to None, like any other unrecognised marker value.
        provider_obj = with_auth(
            get_provider_for_legacy_read(fields["provider"]),
            want_secret=fields.get("auth") == "secret",
        )
    except CodeHelperError:
        return None

    # Round-trip rule: the FILE wins for anything but a FIXED base_url —
    # see :func:`_recover_base_url` for the full rationale (edit-token's
    # "rotate WITHOUT re-expanding a preset" contract applied to base_url).
    provider_obj = _recover_base_url(
        provider_obj, shape_field, toml_profile, body, fields["provider"]
    )
    if provider_obj is None:
        return None

    # Resolve the agent through the MERGED registry (built-ins + user-defined
    # agents), not model.get_agent's built-in-only lookup: a wrapper created
    # for a user-defined agent records that agent's name in its marker, and
    # build_spec would otherwise reject it as unknown, hiding the wrapper from
    # the TUI's list/remove/token actions. Pass the resolved Agent object so
    # build_spec skips its own string resolution.
    try:
        agent_obj = get_any_agent(paths, fields["agent"])
    except CodeHelperError:
        return None

    try:
        return build_spec(
            agent=agent_obj,
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
            # The RECORDED window, honoured the way the recorded shape/auth
            # are (issue #83): a garbage value (``ctx=abc`` → ValueError,
            # ``ctx=-5``/oversized → build_spec's range check) fails closed
            # to None below, like any other unrecognised marker value. A
            # marker WITHOUT the field predates the recording — the catalog
            # derivation stands, the pre-#83 migration fallback.
            context_window=int(fields["ctx"]) if fields.get("ctx") else None,
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

    body = read_text_or_none(paths.script_for(name))
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
    profile = read_text_or_none(paths.codex_config_for(alias))
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


def _ownership_full_match(paths: Paths, spec: WrapperSpec, token: str) -> bool:
    """True iff the file at ``spec.alias`` is one we may replace unasked.

    Two ways to qualify:

    - it carries the marker (:func:`is_managed`) — the normal case; or
    - it is byte-identical to what the PREVIOUS, markerless release would
      have written for this same spec.

    The second clause is the migration path. The marker did not exist before
    this branch, so every already-installed wrapper lacks it, and without this
    the guard would classify the tool's own prior output as a third-party file
    — telling users it "was not created by codehelper" about a file it did
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
    existing = read_text_or_none(paths.script_for(spec.alias))
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
    return _ownership_marker_provider_is_secret(paths, script.name)


def _respec_from_body(spec: WrapperSpec, body: str) -> WrapperSpec | None:
    """``spec`` with the models the FILE actually carries, or None.

    Only the models are taken from the body — agent, provider, and shape stay
    the caller's, so this can widen *which model* counts as ours but never
    which agent or provider does.

    The explicit context window is deliberately NOT taken from anywhere: this
    feeds only the markerless legacy byte-compare in ``_ownership_full_match``,
    and a pre-marker body can never carry an explicit ``ctx=`` decision — its
    window line is catalog-derived and re-derives identically. Do not "fix"
    that reflexively.
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
    """:attr:`body` with the embedded auth token blanked, for structural comparison.

    Both places the renderer puts it: the ``export ANTHROPIC_AUTH_TOKEN=``
    line, and — since wrappers forward their env via ``--settings`` — the
    JSON payload in the exec line, where it sits under
    ``"ANTHROPIC_AUTH_TOKEN"`` as a JSON string. That half is escape-aware
    (``(?:[^"\\\\]|\\\\.)*``) so a token containing a quote or backslash is
    blanked just as completely; without it a token rotation under a legacy
    (markerless) wrapper would compare old-token bytes against new-token
    bytes and refuse a file this tool wrote.
    """
    body = re.sub(
        r"^export ANTHROPIC_AUTH_TOKEN=.*$",
        "export ANTHROPIC_AUTH_TOKEN=",
        body,
        flags=re.MULTILINE,
    )
    return re.sub(
        r'("ANTHROPIC_AUTH_TOKEN":")((?:[^"\\]|\\.)*)(")',
        r"\1\3",
        body,
    )


def _ownership_marker_only(path: Path) -> bool:
    """True iff ``path`` carries our marker on its first or second line.

    The OPENAI_TOML TOML profile carries the same marker comment the bash
    wrapper does, which is what lets the ownership guard recognise it as ours.
    That file has NO pre-marker legacy form (the shape is new), so unlike
    :func:`_ownership_full_match` there is no byte-identical-to-legacy migration clause —
    the marker alone is the proof of authorship. Unreadable (binary, perms, a
    dangling symlink) counts as not ours, so the guard refuses rather than
    clobbers — the same safe answer :func:`is_managed` gives.
    """
    return _ownership_marker_at(path)


def _ownership_catalog_marker(path: Path) -> bool:
    """True iff ``path`` is JSON carrying our own ``managed_by`` field.

    No renderer writes a catalog any more (Codex requires a non-optional
    ``base_instructions`` per entry, and an empty synthesized one silently
    replaces Codex's real system prompt — see ``render.openai_toml_body``).
    This check survives ONLY to let :func:`_warn_stale_catalog` recognise a
    catalog a PREVIOUS version of this tool wrote (and :func:`remove_wrapper`
    to delete one as part of an explicit removal) — never to gate a write,
    since there is no longer a catalog write to gate. Unreadable/non-JSON/
    missing key → False, the same safe-refuse answer every other ownership
    check in this module gives (an unrecognised file is left alone, not swept
    up as ours).
    """
    body = read_text_or_none(path)
    if body is None:
        return False
    try:
        import json

        data = json.loads(body)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("managed_by") == "codehelper"


class _FilePlan(NamedTuple):
    """One file a wrapper install writes, and how to treat an existing copy.

    The plan is the shape-driven answer to "which files does this install
    write?" — one entry for a single-shape wrapper, two for ``OPENAI_TOML``
    (wrapper + TOML profile — no catalog, see ``_openai_toml_plan``). Pairing
    each file with its own ``managed_check`` and ``is_wrapper`` flag as data
    is what lets one orchestrator (:func:`_install_plan`) handle every shape:
    the per-file ownership predicate varies (marker+legacy for the wrapper,
    marker-only for the TOML profile) without the orchestrator knowing which
    shape it is looking at.
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
        "{path} exists and was not created by codehelper — "
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
    if read_text_or_none(f.path) == f.body:
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
            lambda _p: _ownership_full_match(paths, spec, token),
            True,
        )
    ]


def _openai_toml_plan(paths: Paths, spec: WrapperSpec, token: str) -> list[_FilePlan]:
    """The two-file plan: TOML profile + bash wrapper.

    No model catalog is written any more (see ``render.openai_toml_body`` for
    why: Codex's ``ModelInfo`` requires a per-entry ``base_instructions``
    string, and an empty synthesized one silently replaces Codex's real
    system prompt) — a stale catalog from a PREVIOUS version of this tool is
    only warned about, never written or deleted here (see
    :func:`_warn_stale_catalog` in :func:`install_wrapper`).

    Ordered so the on-PATH executable lands LAST: if the profile write fails
    (disk full, permission), no invocable wrapper is left pointing at a
    missing profile — only a harmless orphaned wrapper-less state the next
    idempotent install repairs. The wrapper-first order this replaced could
    leave a broken executable on ``PATH`` during the failure window.

    Each slot carries its own ownership check — :func:`_ownership_marker_only`
    for the wrapper AND the TOML profile (both carry the marker comment;
    OPENAI_TOML is a new shape with no pre-marker legacy form, so the
    legacy byte-match clause :func:`_ownership_full_match` uses for the other shapes would
    only ever match a third-party hand-written ``exec codex --profile`` script
    and adopt it as ours). Modes: the wrapper follows :func:`_mode_for` (``0o700``
    for a secret, else ``0o755``); the profile is ``0o600`` owner-only.
    """
    from codehelper.services.render import openai_toml_body

    config_path = paths.codex_config_for(spec.alias)
    return [
        _FilePlan(
            config_path,
            openai_toml_body(spec),
            0o600,
            _ownership_marker_only,
            False,
        ),
        _FilePlan(
            paths.script_for(spec.alias),
            render_script(spec, token),
            _mode_for(spec),
            _ownership_marker_only,
            True,
        ),
    ]


def _remove_owned_paths(paths_to_check: list[Path], *, dry_run: bool) -> bool:
    """Remove each path in ``paths_to_check`` that exists. Best-effort.

    Callers pre-filter to paths already proven ours — this function only
    performs the removal and reports it, so a failure here (permissions, a
    concurrent delete) is reported on stderr and skipped, never raised: it
    runs AFTER the wrapper install already succeeded, so it must never make a
    successful install look failed.

    Returns:
        True iff anything was removed (or, in dry-run, would be).
    """
    import sys

    changed = False
    for path in paths_to_check:
        if not path.exists():
            continue
        if dry_run:
            print(f"would remove orphaned sibling {path}")
            changed = True
            continue
        try:
            remove_file(path)
        except OSError as exc:
            print(
                f"warning: failed to remove orphaned sibling {path}: {exc}",
                file=sys.stderr,
            )
            continue
        print(f"removed orphaned sibling {path}")
        changed = True
    return changed


def _warn_stale_catalog(paths: Paths, alias: str) -> None:
    """Warn about a ``<alias>.model.json`` an OLDER version wrote; never delete.

    No renderer writes a catalog any more (see ``render.openai_toml_body``:
    Codex's ``ModelInfo`` requires a per-entry ``base_instructions`` string,
    and an empty synthesized one silently replaces Codex's real system prompt
    for every wrapper reading it). A leftover catalog is now INERT: no
    ``set-default`` writes ``model_catalog_json`` any more, and the set-default
    patch scrubs a stale ``model_catalog_json`` line when it rewrites the
    profile. Deleting it on every install would therefore be convenience, not
    hygiene — and it would be UNSAFE convenience: :func:`_ownership_catalog_marker`
    proves the file was at some point written by this tool, but a user can
    hand-edit a marker-bearing catalog afterwards, and a fresh install must
    never destroy data it cannot prove is byte-identical to what it wrote
    (the old renderer that produced the canonical bytes no longer exists to
    compare against). Warn and leave the file alone; ``remove_wrapper`` still
    removes a marker-owned catalog, because there the removal IS the explicit,
    confirmed action.
    """
    import sys

    catalog_path = paths.codex_catalog_for(alias)
    if catalog_path.exists() and _ownership_catalog_marker(catalog_path):
        print(
            f"warning: stale codehelper-managed model catalog {catalog_path} "
            f"left in place (nothing reads it any more; delete manually if "
            f"you want it gone)",
            file=sys.stderr,
        )


def _cleanup_openai_toml_siblings(paths: Paths, alias: str, *, dry_run: bool) -> bool:
    """Remove the ``~/.codex/<alias>.config.toml`` profile we wrote.

    Only meaningful when a PREVIOUS install under ``alias`` was OPENAI_TOML and
    the new one is not: the old profile no longer matches anything the new
    wrapper dispatches to, so leaving it is silent clutter. A stale catalog
    from an older version of this tool is NOT swept here — it is only warned
    about via :func:`_warn_stale_catalog`, for the same reason an install
    never deletes one: ``managed_by=codehelper`` alone does not prove the file
    was not hand-edited since.

    The profile is gated by ITS OWN ownership proof — :func:`_ownership_marker_only`.
    This is deliberately NOT "the profile's marker decides the sibling's fate":
    an earlier version inferred the catalog's fate from the profile alone,
    which meant a hand-curated catalog sitting next to OUR profile was deleted
    with no ``--force`` and no prompt — the exact thing the install-time
    ownership guard exists to prevent, just reached through a different door.
    A foreign profile is left untouched, matching what the install guard
    would have refused to overwrite.

    Best-effort by design: called AFTER the wrapper install already succeeded
    (see ``install_wrapper``), so a failure here must never make a successful
    install look failed.

    Returns:
        True iff the profile was removed (or, in dry-run, would be) — folded
        into ``install_wrapper``'s own return value so a run whose only effect
        was deleting an orphaned sibling is not reported as "no changes".
    """
    config_path = paths.codex_config_for(alias)
    _warn_stale_catalog(paths, alias)
    return _remove_owned_paths(
        [config_path] if _ownership_marker_only(config_path) else [], dry_run=dry_run
    )


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
    (:func:`_wrapper_plan`), two for ``OPENAI_TOML``
    (:func:`_openai_toml_plan`: bash wrapper + TOML profile — no catalog, see
    its docstring) — and one orchestrator (:func:`_install_plan`) writes
    whichever plan it was handed, so the ownership guard, dry-run, force, and
    confirm behaviour are identical per file regardless of how many files a
    shape writes.

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

    if spec.shape is ConfigShape.OPENAI_TOML:
        # An earlier version of this tool wrote a ``<alias>.model.json``
        # catalog next to the profile; no renderer writes one any more (see
        # ``render.openai_toml_body``). Warn about a leftover from THIS
        # alias's own prior install, never delete: the ``managed_by=
        # codehelper`` marker alone does not prove the file was not
        # hand-edited since (see ``_warn_stale_catalog``).
        _warn_stale_catalog(paths, spec.alias)
    else:
        # A non-OPENAI_TOML install under an alias that previously held an
        # OPENAI_TOML install leaves the ``~/.codex/<alias>.*`` siblings
        # orphaned — the new wrapper no longer dispatches ``codex --profile
        # <alias>``. Runs regardless of whether the wrapper slot itself
        # changed (a SKIP re-install under an alias that still carries stale
        # siblings must still clean them up — the siblings are the whole
        # point, not a side effect of the wrapper write), but only for OUR
        # siblings; a refusal above already raised before reaching here, so
        # nothing here is racing an unresolved guard decision. Its own
        # return value is folded into ``wrote`` so a run whose only effect
        # was deleting orphaned siblings is not reported as "no changes".
        cleaned = _cleanup_openai_toml_siblings(paths, spec.alias, dry_run=dry_run)
        wrote = wrote or cleaned

    return wrote


def remove_wrapper(
    paths: Paths,
    name: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    confirm: Callable[[list[Path]], bool] | None = None,
) -> bool:
    """Remove a wrapper and its managed OPENAI_TOML siblings safely.

    Refuses to delete an unmanaged executable unless the caller explicitly
    passes ``force``. Only siblings carrying their own ownership proof are
    removed, so a wrapper removal cannot silently delete user configuration.

    ``confirm`` is asked once, with the full list of files about to be
    unlinked, before anything is touched — mirroring ``install_wrapper``'s
    per-file ``confirm`` gate on an overwrite. ``force`` bypasses it (it
    already means "I know what I'm removing"); dry-run never calls it, since
    dry-run never mutates anything to confirm.
    """
    if not _is_usable_alias(name):
        raise CodeHelperError(f"invalid wrapper name: {name}")
    wrapper = paths.script_for(name)
    if not wrapper.exists():
        raise CodeHelperError(f"wrapper not found: {wrapper}")
    if not is_managed(paths, name) and not force:
        raise CodeHelperError(
            f"refusing to remove unmanaged file {wrapper} (use --force)"
        )

    siblings = [
        path
        for path, ours in (
            (
                paths.codex_config_for(name),
                _ownership_marker_only(paths.codex_config_for(name)),
            ),
            (
                paths.codex_catalog_for(name),
                _ownership_catalog_marker(paths.codex_catalog_for(name)),
            ),
        )
        if path.exists() and ours
    ]
    # Siblings first, the wrapper executable last: `is_managed`/`is_installed`
    # both key off the executable's presence, so keeping it around until
    # every sibling is confirmed gone means a failed unlink partway through
    # (permissions, a transient filesystem error) leaves a wrapper that is
    # still recognized and retryable — not a dangling alias a retry can no
    # longer find (`wrapper not found`) with orphaned siblings behind it.
    targets = [*siblings, wrapper]
    if dry_run:
        for path in targets:
            print(f"would remove {path}")
        return True

    if not force and confirm is not None and not confirm(targets):
        raise CodeHelperError(f"removal of {wrapper} was not confirmed")

    for path in targets:
        try:
            remove_file(path)
        except OSError as exc:
            # Every file is untouched or gone at this point — the default
            # pointer has NOT been cleared yet, so a still-installed wrapper
            # never loses its default out from under a failed removal.
            raise CodeHelperError(f"failed to remove {path}: {exc}") from exc
        print(f"removed {path}")

    # Only clear the default pointer once every unlink above has actually
    # succeeded — clearing it any earlier would strip a still-installed
    # wrapper's default the moment a LATER sibling unlink fails. A failure
    # in this write itself is reported, not swallowed as success, but does
    # NOT retroactively undo the deletion that already happened: the files
    # are gone regardless, so raising here only signals "the pointer may
    # still be stale", which the message says explicitly.
    from codehelper.services.state import clear_default_wrapper

    try:
        clear_default_wrapper(paths, name)
    except OSError as exc:
        raise CodeHelperError(
            f"removed {wrapper} but failed to clear its default pointer: {exc}"
        ) from exc
    return True


def _marker_with_repointed_profile(
    body: str, old_quoted: str, new_quoted: str
) -> str | None:
    """The body with its marker's ``profile=`` field re-pointed, or ``None``.

    Only the ownership-marker region (the first two lines — the same region
    :func:`_ownership_marker_only` trusts) is searched, and the field is
    anchored between a comma and the closing delimiter so a profile value can
    never be confused with a sibling field. ``None`` means the body carries no
    matching ``profile=`` — a corrupt or already-re-pointed marker; callers
    decide whether that is fatal.
    """
    lines = body.split("\n")
    pattern = re.compile(rf"(?<=, )profile={re.escape(old_quoted)}(?=[,)])")
    changed = False
    for i, line in enumerate(lines[:2]):
        new_line, count = pattern.subn(f"profile={new_quoted}", line)
        if count:
            lines[i] = new_line
            changed = True
    if not changed:
        return None
    return "\n".join(lines)


def rename_wrapper(
    paths: Paths,
    old_alias: str,
    new_alias: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Move an installed managed wrapper to a new alias (issue #95).

    The marker records no alias — for every shape except ``OPENAI_TOML`` the
    body is byte-identical under any name, so the move is a byte-preserving
    copy of the file: the token, model and profile binding travel untouched.
    ``OPENAI_TOML`` is the one alias-referencing shape (the exec line launches
    ``codex --profile '<alias>'`` and the companion TOML is alias-keyed), so
    there the wrapper is re-rendered under the new spec with the token
    recovered from the old file — a literal-auth wrapper keeps ``token=""``,
    which is exactly what its body had.

    The new name goes through the full creation gate —
    :func:`naming.validate_alias` including reserved names, plus user-agent
    binaries, the same boundary every wrapper creation flows through — because
    a rename must not reach a name ``add`` could not have taken. The
    default-wrapper pointers follow the move.

    Returns True iff the rename happened (or was previewed).
    """
    if not _is_usable_alias(old_alias):
        raise CodeHelperError(f"invalid wrapper name: {old_alias}")
    wrapper = paths.script_for(old_alias)
    if not wrapper.exists():
        raise CodeHelperError(f"wrapper not found: {wrapper}")
    if not is_managed(paths, old_alias):
        raise CodeHelperError(f"refusing to rename unmanaged file {wrapper}")

    validate_alias(new_alias)
    user_binaries = {a.binary for a in load_user_agents_strict(paths)}
    if new_alias in user_binaries:
        raise CodeHelperError(
            f"{new_alias!r} is a reserved name — a wrapper named after a "
            f"user-defined agent's binary would shadow the real one on PATH"
        )
    new_path = paths.script_for(new_alias)
    if new_path.exists():
        raise CodeHelperError(f"wrapper already exists: {new_path}")
    if paths.codex_config_for(new_alias).exists():
        raise CodeHelperError(
            f"codex profile already exists: {paths.codex_config_for(new_alias)}"
        )

    spec = spec_from_installed(paths, old_alias)
    if spec is None:
        raise CodeHelperError(
            f"cannot rename {wrapper}: its marker is not recognisable"
        )

    from codehelper.services.state import default_wrapper

    agents_to_repoint = [
        agent.name
        for agent in all_agents(paths)
        if default_wrapper(paths, agent.name) == old_alias
    ]

    new_spec = replace(spec, alias=new_alias)
    if spec.shape is ConfigShape.OPENAI_TOML:
        token = ""
        if spec.auth == "secret":
            token = token_from_installed(paths, old_alias, spec.provider.name) or ""
            if not token:
                raise CodeHelperError(
                    f"cannot rename {wrapper}: its token could not be recovered"
                )
        install_wrapper(paths, new_spec, token=token, dry_run=dry_run)
    else:
        mode = stat.S_IMODE(wrapper.stat().st_mode)
        if dry_run:
            print(f"would move {wrapper} -> {new_path}")
        else:
            atomic_write(new_path, wrapper.read_text(encoding="utf-8"), mode=mode)
            print(f"renamed wrapper {old_alias} -> {new_alias}")

    from codehelper.services.state import set_default_wrapper

    for agent_name in agents_to_repoint:
        set_default_wrapper(paths, agent_name, new_alias)

    try:
        remove_wrapper(paths, old_alias, dry_run=dry_run)
    except CodeHelperError as exc:
        raise CodeHelperError(
            f"renamed to {new_alias} but the old file could not be removed: {exc}"
        ) from exc
    return True


def rename_provider_profile(
    paths: Paths,
    provider_name: str,
    old_name: str,
    new_name: str,
    *,
    dry_run: bool = False,
) -> int:
    """Rename a provider's token profile AND follow the wrappers (issue #95).

    A cache-only rename strands every installed wrapper whose marker records
    the old profile: ``list`` keeps showing it and an ``edit-token --profile
    <old>``, aimed at rotation, would silently create a fresh empty profile.
    So the verb owns the whole move — the credentials key via
    :func:`secrets.rename_profile`, the ``profile=`` field in every installed
    wrapper marker naming it (wrapper body AND the ``OPENAI_TOML`` companion,
    which carries the same marker line), and the stored active-selection
    pointer, compared through the provider's storage names (a pre-rename
    ``state.json`` may hold a retired spelling).

    Returns the number of installed wrappers re-pointed.
    """
    provider = get_provider_for_legacy_read(provider_name)

    from codehelper.services.secrets import profile_names, rename_profile

    if old_name == new_name:
        raise CodeHelperError("old and new profile names are identical")
    existing = profile_names(paths, provider.name)
    if old_name not in existing:
        raise CodeHelperError(
            f"unknown profile: {old_name!r} for provider {provider.name!r}"
        )
    outcome = validate_new_profile_name(
        new_name, [n for n in existing if n != old_name]
    )
    if outcome in (NewProfileOutcome.EMPTY, NewProfileOutcome.COLLISION_NEW):
        raise CodeHelperError(f"invalid new profile name: {new_name!r}")

    targets = [
        name
        for name in wrappers_for_provider(paths, provider)
        if profile_from_installed(paths, name) == old_name
    ]

    old_quoted = quote(old_name, safe="._-")
    new_quoted = quote(new_name, safe="._-")
    rewrites: list[tuple[Path, str]] = []
    for name in targets:
        files = [paths.script_for(name)]
        spec = spec_from_installed(paths, name)
        if spec is not None and spec.shape is ConfigShape.OPENAI_TOML:
            companion = paths.codex_config_for(name)
            if companion.exists() and _ownership_marker_only(companion):
                files.append(companion)
        for path in files:
            body = path.read_text(encoding="utf-8")
            new_body = _marker_with_repointed_profile(body, old_quoted, new_quoted)
            if new_body is None:
                raise CodeHelperError(
                    f"cannot re-point {path}: no profile={old_quoted} in its marker"
                )
            rewrites.append((path, new_body))

    if dry_run:
        for path, _ in rewrites:
            print(f"would update profile marker in {path}")
        print(f"would rename profile {old_name} -> {new_name}")
        return len(targets)

    for path, new_body in rewrites:
        atomic_write(path, new_body)
        print(f"re-pointed {path}")

    rename_profile(paths, provider.name, old_name, new_name)

    from codehelper.services.state import active_selection, set_active_selection

    selection = active_selection(paths)
    if (
        selection is not None
        and selection[0] in provider_storage_names(provider.name)
        and selection[1] == old_name
    ):
        set_active_selection(paths, provider.name, new_name)

    print(f"renamed profile {old_name} -> {new_name}")
    return len(targets)


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
    name field shrinks by the marker's width and the ``:12``/``:13`` column
    alignment every other caller depends on is unchanged.
    ``list_wrappers`` passes ``default=False`` (the default) — the CLI list is
    a flat registry dump and the marker belongs only to the interactive
    screen that can CHANGE which wrapper is default.
    """
    state = installed_word if installed else not_installed_word
    profile = f" [profile: {profile_name}]" if profile_name else ""
    mark = "● " if default else ""
    return f"{mark}{spec.name:{12 - len(mark)}} {state:13} {spec.description}{profile}"


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


def describe_wrapper_columns(
    spec: WrapperSpec, *, installed: bool, default: bool = False
) -> tuple[str, str, str]:
    """TUI-only name/provider/model columns, independent of prose descriptions."""
    name = f"● {spec.name}" if default else f"  {spec.name}"
    model = spec.model + (" (not installed)" if not installed else "")
    return name, spec.provider.name, model


def describe_all_columns(
    paths: Paths, specs: Sequence[WrapperSpec], *, defaults: dict[str, str | None]
) -> list[tuple[str, tuple[str, str, str]]]:
    """Return aligned semantic columns for the interactive main screen."""
    raw = [
        (
            spec.name,
            describe_wrapper_columns(
                spec,
                installed=is_installed(paths, spec.name),
                default=defaults.get(spec.agent.name) == spec.name,
            ),
        )
        for spec in specs
    ]
    widths = [max((len(columns[i]) for _, columns in raw), default=0) for i in range(2)]
    return [
        (name, (columns[0].ljust(widths[0]), columns[1].ljust(widths[1]), columns[2]))
        for name, columns in raw
    ]


def column_header(rows: Sequence[tuple[str, tuple[str, str, str]]]) -> str:
    """``WRAPPER  PROVIDER  MODEL``, aligned to ``describe_all_columns``.

    Takes the already-padded rows rather than the specs: every row's column 0
    and column 1 are ``ljust``-ed to the same width by
    :func:`describe_all_columns`, so the first row's widths are every row's
    widths — recomputing a max here would be a second source of truth that
    silently drifts the moment :func:`describe_wrapper_columns` changes its
    padding.

    Column 0 carries the two-character ``"● "``/``"  "`` default marker, so
    the header is indented by the same two spaces; without that the word
    ``WRAPPER`` would sit two columns left of every name under it.
    """
    if not rows:
        return ""
    name_col, provider_col, _ = rows[0][1]
    return (
        f"{'  WRAPPER'.ljust(len(name_col))}  "
        f"{'PROVIDER'.ljust(len(provider_col))}  MODEL"
    )


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

    A marked file whose name is not structurally a usable alias
    (:func:`is_valid_alias_shape`) is skipped: no command in the tool can act
    on it, so listing it advertises a wrapper the user cannot then edit or
    reinstall. A RESERVED name (e.g. ``opencode``) is still listed — it is a
    real, removable wrapper, and hiding it would strand it on PATH with no way
    to clean it up.
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


def wrappers_for_provider(paths: Paths, provider: Provider) -> list[str]:
    """Installed managed wrappers whose resolved provider is ``provider``.

    The blast-radius enumeration behind ``disable <provider>`` (issue #89):
    every alias ``disable`` must physically delete so the provider vanishes
    from PATH. Covers BOTH kinds — presets installed on this provider (their
    marker names it, exactly like an ad-hoc wrapper) and ``discover_managed``
    entries — resolved through :func:`_installed_provider_name`, so a marker
    predating a provider rename (legacy ``ollama``) still matches via
    :func:`get_provider_for_legacy_read`.

    Sorted for a deterministic removal/preview order. Not installed → empty
    list; never raises — a wrapper whose provider cannot even be named
    (corrupt marker) is skipped rather than guessed at, consistent with
    :func:`_installed_provider_name`'s contract.
    """
    names: set[str] = set()
    for spec in WRAPPERS:
        if is_installed(paths, spec.name) and is_managed(paths, spec.name):
            names.add(spec.name)
    names.update(discover_managed(paths))
    return sorted(
        name for name in names if _installed_provider_name(paths, name) == provider.name
    )


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
    from codehelper.services.state import default_wrapper

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
    return None


def _is_usable_alias(name: str) -> bool:
    """True iff ``name`` is one the rest of the tool can still act on.

    Structural validity only — NOT the reserved-name check. A managed wrapper
    whose name became reserved (``opencode`` after the agent registry grew)
    must stay discoverable and removable so the user can clean it up; only
    NEW creation under a reserved name is blocked, by ``validate_alias`` at
    the install boundary.
    """
    return is_valid_alias_shape(name)
