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
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.model import Agent, ConfigShape, Provider
from code_helper.services.naming import validate_alias
from code_helper.services.paths import Paths
from code_helper.services.render import (
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
    "list_wrappers",
    "describe_wrapper",
    "describe_all",
    "discover_managed",
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
    # shapes carry the model in the rendered script itself.
    shape_field = fields.get("shape")
    if shape_field == ConfigShape.OPENAI_TOML.value:
        model = _model_from_toml_profile(paths, name)
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
        return build_spec(
            agent=fields["agent"],
            provider=fields["provider"],
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
        )
    except (CodeHelperError, ValueError):
        # A marker naming an agent/provider/shape this build no longer knows,
        # or a pairing that is no longer valid (ValueError comes from the
        # ConfigShape lookup). Not our problem to resolve here.
        return None


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
    :func:`_model_from_toml_profile`.
    """
    sonnet = _env_value(body, "ANTHROPIC_DEFAULT_SONNET_MODEL")
    if sonnet is not None:
        return sonnet
    found = re.search(r"--model '(.*?)' --", body)
    return found.group(1).replace("'\"'\"'", "'") if found else None


def _toml_unescape(value: str) -> str:
    """Reverse :func:`render._toml_string` for a TOML basic-string body.

    Only the escapes ``_toml_string`` emits are handled (``\\\\``, ``\\"``,
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


def _model_from_toml_profile(paths: Paths, alias: str) -> str | None:
    """The ``model`` value from the sibling ``<alias>.config.toml`` profile.

    The OPENAI_TOML wrapper body carries no model (it only dispatches
    ``codex --profile <alias>``), so ``edit-token``/``spec_from_installed``
    cannot recover the model from the wrapper the way the other shapes do —
    the model is in the TOML profile written alongside it. ``tomllib`` (3.11+)
    parses it correctly; on 3.10 (the project's minimum) a line regex reads
    the ``model = "..."`` key this renderer is the only writer of. Unreadable
    or absent → None, so the caller falls back to the preset path rather than
    raising.
    """
    profile = _read_text_or_none(paths.codex_config_for(alias))
    if profile is None:
        return None
    try:
        import tomllib  # py3.11+

        data = tomllib.loads(profile)
        model = data.get("model")
        return model if isinstance(model, str) and model else None
    except ModuleNotFoundError:
        # 3.10 fallback: this renderer is the only writer of the profile, so a
        # plain ``^model = "..."`` line match is sufficient — tomllib's
        # validation is not needed for a file we authored.
        found = re.search(r'^model = "(.*)"$', profile, re.MULTILINE)
        return _toml_unescape(found.group(1)) if found else None


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
    return installed is not None and installed.auth == "secret"


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


def _is_our_catalog(paths: Paths, alias: str) -> bool:
    """True iff the catalog for ``alias`` is one we wrote.

    The catalog is JSON, and JSON has no comments — so the marker the wrapper
    and TOML profile carry cannot ride along. A purely structural test
    (``{"version": 1, "models": [...]}``) is too permissive: that is a generic
    shape a hand-curated or third-party catalog plausibly uses, and treating it
    as proof of authorship clobbers a researched ``context_window`` with the
    :data:`_DEFAULT_CONTEXT_WINDOW` floor — silent data loss without
    ``--force``. Instead the catalog is proven ours by its SIBLING: the
    ``<alias>.config.toml`` profile is written alongside it and carries our
    marker, so the catalog is ours iff the profile is. A foreign catalog with
    no (or a foreign) profile routes to ``OVERWRITE_FOREIGN`` like the other
    two slots. The byte-identical idempotence case is handled earlier in
    :func:`_decide` (SKIP), so this only gates non-identical existing catalogs.
    """
    return _marker_at(paths.codex_config_for(alias))


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


class _Action(str, Enum):
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


def _cleanup_openai_toml_siblings(paths: Paths, alias: str, *, dry_run: bool) -> None:
    """Remove the ``~/.codex/<alias>.config.toml`` + ``<alias>.model.json`` we wrote.

    Only run when a PREVIOUS install under ``alias`` was OPENAI_TOML and the new
    one is not: the old profile/catalog no longer match anything the new wrapper
    dispatches to, so leaving them is silent clutter (and a stale catalog could
    mislead a later ``codex --profile <alias>`` if the alias is ever reused for
    OPENAI_TOML again). Only OUR siblings are removed — proven the same way the
    install guard proves them: the profile by its marker, the catalog by its
    sibling profile's marker. A foreign profile/catalog under the alias is left
    untouched, exactly as the install guard would refuse to overwrite it.
    """
    config_path = paths.codex_config_for(alias)
    catalog_path = paths.codex_catalog_for(alias)

    # The profile marker is the single proof both siblings share (the catalog
    # is JSON — no comment marker; see ``_is_our_catalog``). Decide once, BEFORE
    # removing anything: unlinking the profile first would make the re-check
    # false and strand the catalog.
    profile_is_ours = _marker_at(config_path)
    if not profile_is_ours:
        return

    if config_path.exists():
        if dry_run:
            print(f"would remove orphaned sibling {config_path}")
        else:
            try:
                config_path.unlink()
            except OSError as exc:
                raise CodeHelperError(
                    f"failed to remove orphaned sibling {config_path}: {exc}"
                ) from exc
            print(f"removed orphaned sibling {config_path}")

    if catalog_path.exists():
        if dry_run:
            print(f"would remove orphaned sibling {catalog_path}")
        else:
            try:
                catalog_path.unlink()
            except OSError as exc:
                raise CodeHelperError(
                    f"failed to remove orphaned sibling {catalog_path}: {exc}"
                ) from exc
            print(f"removed orphaned sibling {catalog_path}")


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
    # — the new wrapper no longer dispatches ``codex --profile <alias>``. Only
    # clean up AFTER a successful (non-dry-run) write or a dry-run that would
    # have written, and only for OUR siblings; a refusal above already raised
    # before reaching here.
    if spec.shape is not ConfigShape.OPENAI_TOML:
        _cleanup_openai_toml_siblings(paths, spec.alias, dry_run=dry_run)

    return wrote


def describe_wrapper(
    spec: WrapperSpec, *, installed: bool, installed_word: str, not_installed_word: str
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
    """
    state = installed_word if installed else not_installed_word
    return f"{spec.name:12} {state:13} {spec.description}"


def describe_all(
    paths: Paths,
    specs: Sequence[WrapperSpec],
    *,
    installed_word: str,
    not_installed_word: str,
) -> list[tuple[str, str]]:
    """``(name, describe_wrapper(...))`` pairs for a menu built over ``specs``.

    Both ``edit-token``'s picker (over the ``secret``-auth subset) and the
    TUI's ``add`` wrapper picker (over the full registry) build this exact
    shape — the same ``is_installed`` lookup per spec, wrapped by
    :func:`describe_wrapper` — differing only in which specs they iterate and
    which language's install-state words they pass. Sharing the loop here
    means that wiring (not just the row's column widths) can't drift between
    the two menus.
    """
    return [
        (
            spec.name,
            describe_wrapper(
                spec,
                installed=is_installed(paths, spec.name),
                installed_word=installed_word,
                not_installed_word=not_installed_word,
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
            print_fn(f"{name:12} {'installed':13}")


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


def _is_usable_alias(name: str) -> bool:
    """True iff ``name`` is one the rest of the tool can still act on."""
    try:
        validate_alias(name)
    except CodeHelperError:
        return False
    return True
