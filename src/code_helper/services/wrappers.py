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
from pathlib import Path

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


def is_managed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` exists AND carries our marker.

    Reads only the first couple of lines. Anything unreadable (a binary, a
    permission error, a dangling symlink) counts as *not* ours — the safe
    answer, since it makes the guard refuse rather than clobber.
    """
    script = paths.script_for(name)
    try:
        with script.open("r", encoding="utf-8") as fh:
            for _ in range(2):
                line = fh.readline()
                if not line:
                    break
                if line.startswith(MARKER_PREFIX):
                    return True
    except (OSError, UnicodeDecodeError):
        return False
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
    """
    sonnet = _env_value(body, "ANTHROPIC_DEFAULT_SONNET_MODEL")
    if sonnet is not None:
        return sonnet
    found = re.search(r"--model '(.*?)' --", body)
    return found.group(1).replace("'\"'\"'", "'") if found else None


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

    body = render_script(spec, token)
    script = paths.script_for(spec.alias)

    if script.exists() and _read_text_or_none(script) == body:
        return False  # identical script already installed

    # Ownership check runs only for a file we did not write. Order matters:
    # the idempotence check above means an unchanged reinstall never prompts.
    if script.exists() and not _is_ours(paths, spec, token):
        if dry_run:
            print(f"would overwrite UNMANAGED file {script}")
            return True
        if not force:
            if confirm is None or not confirm(script):
                raise CodeHelperError(
                    f"{script} exists and was not created by code-helper — "
                    f"refusing to overwrite (use --force)"
                )
        print(f"overwriting unmanaged file {script}")
    elif script.exists() and _discards_only_secret(paths, spec, script):
        # Ours, but a DIFFERENT wrapper whose token exists nowhere else.
        if dry_run:
            print(f"would discard the only copy of {script}'s token")
            return True
        if not force:
            if confirm is None or not confirm(script):
                raise CodeHelperError(
                    f"{script} holds a wrapper whose token exists nowhere else, "
                    f"and the replacement does not use one — refusing to discard "
                    f"the only copy of its token (use --force)"
                )
        print(f"discarding the only copy of {script}'s token")

    if dry_run:
        print(f"would write {script}")
        return True
    atomic_write(script, body, mode=_mode_for(spec))
    print(f"wrote {script}")
    return True


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
