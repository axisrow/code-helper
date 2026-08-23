"""``switch`` — patch the ``env`` block of Claude Code's OWN ``~/.claude/settings.json``, in place.

Every other module in this project bends over backwards to never touch an
agent's own configuration (see ``services/paths.py``'s ``codex_dir``/
``claude_dir`` docstrings, ``services/render.py``'s ``openai_toml_body``).
That invariant is deliberately broken here — the SAME way ``services/
codex_default.py`` deliberately breaks it for Codex's ``config.toml`` — for a
different reason: a generated wrapper script fixes its backend at PROCESS
START (``export ANTHROPIC_*; claude "$@"``), so switching backends normally
means quitting and relaunching. Claude Code re-reads ``settings.json``,
including its ``env`` block, BETWEEN PROMPTS — so patching that file changes
what an ALREADY RUNNING ``claude`` does on its next prompt, no restart
needed. See ``ConfigShape.ANTHROPIC_SETTINGS`` for why this is a distinct
mechanism from ``ANTHROPIC_ENV``, not a variant of it.

Why this is simpler than ``codex_default.py``
-----------------------------------------------
``config.toml`` is TOML with no round-trip-preserving stdlib parser, so that
module patches by regex/string manipulation and uses ``tomllib`` only as a
verification net. ``settings.json`` is JSON: ``json.loads``/``json.dumps``
round-trip a well-formed file structurally (key order survives, because
Python dicts preserve insertion order — only whitespace style may change).
So the patch here is an ordinary dict operation, not text surgery.

What is NOT simpler: this file may hold a real credential
------------------------------------------------------------
A generated wrapper script carries its token at ``0o700``, on the theory
that a script only ever runs for the user who owns it. ``settings.json`` is
typically ``0o644`` and routinely dotfile-synced or screenshotted — writing
``ANTHROPIC_AUTH_TOKEN`` into it is a real widening of the blast radius, not
a cosmetic difference. This module writes the file ``0o600`` and redacts the
token in every diff/preview it produces, but does not eliminate the exposure
— see the ``switch`` command's own docs for the caveat spelled out to users.

Ownership marker: a closed key list, not a comment
-----------------------------------------------------
Every other file this project writes carries a ``# codehelper: managed``
comment as its ownership proof (``render.MARKER_PREFIX``). JSON has no
comments. Ownership of *individual keys inside someone else's file* is
instead defined by :data:`MANAGED_ENV_KEYS` — a fixed, closed tuple. Every
operation here touches ONLY those keys inside ``env``; every other key, in
``env`` or at the top level (``permissions``, ``hooks``, ``HTTPS_PROXY``,
...), is carried through byte-for-byte. :func:`_verify_patch_applied` proves
this before any write, rather than merely hoping it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from codehelper.backends._atomic import (
    atomic_write,
    file_lock,
    read_text_or_none,
    rotate_backups,
)
from codehelper.errors import CodeHelperError
from codehelper.services.model import ConfigShape, Provider
from codehelper.services.paths import Paths
from codehelper.services.render import anthropic_base_url, uniform_context_window
from codehelper.services.spec import TierModels

__all__ = [
    "MANAGED_ENV_KEYS",
    "SettingsPatch",
    "resolve_switch_patch",
    "read_settings",
    "patch_settings",
    "diff_preview",
    "dump_settings",
    "credential_values",
    "redact_credential",
    "settings_backup_slots",
    "without_env_keys",
    "read_env",
    "current_switch",
    "active_switch_env",
    "live_axes_for_spec",
    "matches_switch_spec",
    "apply_switch",
    "restore_settings",
]

#: The env keys this module owns inside settings.json's ``env`` block. This
#: tuple IS the ownership marker: JSON admits no comment, so "mine to
#: remove" is defined by an explicit, closed list rather than a marker line.
#: Every name here is one ``render._render_anthropic_env`` also writes into a
#: generated wrapper script — the two mechanisms are kept in lockstep by
#: ``tests/test_claude_settings.py::test_managed_keys_match_renderer``.
#: Anything else in ``env`` (``HTTPS_PROXY``, ``NO_PROXY``, ``IS_DEMO``,
#: ``CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS``, ...) is the user's and is
#: preserved byte-for-byte through every operation this module performs.
MANAGED_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
)


@dataclass(frozen=True)
class SettingsPatch:
    """The exact ``env`` values a ``switch`` writes into ``settings.json``.

    ``env`` is the COMPLETE set of managed keys to SET; every managed key
    NOT present in it is REMOVED. A reset patch (``provider.env_reset``)
    explicitly sets every managed key to ``""``. Claude Code's settings
    watcher applies updates to the running process, but removing an env key
    from the file does not unset its already-applied process value. Empty
    values both reset that live state and remain false-y on a fresh launch,
    which restores Claude's native OAuth/model defaults.
    """

    provider_name: str
    env: dict[str, str]

    @property
    def is_reset(self) -> bool:
        return bool(self.env) and all(value == "" for value in self.env.values())


def resolve_switch_patch(
    provider: Provider,
    *,
    tier_models: TierModels | None,
    token: str,
    subagent_model: str | None = None,
    current_env: dict[str, str] | None = None,
) -> SettingsPatch:
    """Resolve the patch from the axes. Pure, no IO.

    The single place ``provider.env_reset`` is consulted. For a reset
    provider the result explicitly blanks every MANAGED_ENV_KEYS key that is
    already present in ``current_env`` — never a key that was never set —
    and ``tier_models``/``token`` are ignored entirely — the CLI layer is
    expected to collect neither for ``switch native`` (see
    ``cli/parser.py``'s ``_handle_switch``), but even if it did, nothing here
    would leak them into the patch. ``current_env=None`` (the default, used
    by callers with no live snapshot to hand, e.g. a bare CLI invocation with
    no existing settings.json) blanks every managed key, matching the prior
    unconditional behaviour. Blanking a key that was never set would turn
    ``switch native`` on an already-native file into a write that ADDS keys
    nobody set — the opposite of "native means clear the override" — since a
    key with no prior value has no stale already-applied process value to
    reset.

    ``ANTHROPIC_BASE_URL`` is derived via :func:`render.anthropic_base_url` —
    the SAME function the wrapper renderer uses — so a ``switch`` to a
    provider and a generated wrapper for that same provider always agree on
    the resulting URL. ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` follows the same
    shared-derivation rule via :func:`render.uniform_context_window`: both
    mechanisms declare a third-party model's real context window (Claude
    Code would otherwise assume its 200k fallback and auto-compact there) or
    neither does.

    Raises:
        CodeHelperError: ``provider`` does not declare
            :attr:`~codehelper.services.model.ConfigShape.ANTHROPIC_SETTINGS`
            (the error lists the providers that do); or a non-reset provider
            with no ``tier_models``; or a ``BaseUrlPolicy.REQUIRED`` provider
            whose ``base_url`` was never resolved via ``model.with_base_url``
            — the same invariant :func:`~codehelper.services.spec.build_spec`
            enforces, mirrored here for the same reason.
    """
    from codehelper.services.model import BaseUrlPolicy, switchable_providers

    if ConfigShape.ANTHROPIC_SETTINGS not in provider.shapes:
        known = ", ".join(p.name for p in switchable_providers())
        raise CodeHelperError(
            f"provider {provider.name!r} cannot be switched to live: it does "
            f"not present the anthropic-settings mechanism (switchable: {known})"
        )

    if provider.env_reset:
        keys = MANAGED_ENV_KEYS if current_env is None else current_env.keys()
        return SettingsPatch(
            provider_name=provider.name,
            env={key: "" for key in MANAGED_ENV_KEYS if key in keys},
        )

    if provider.base_url_policy is BaseUrlPolicy.REQUIRED and not provider.base_url:
        raise CodeHelperError(
            f"provider {provider.name!r} requires a base URL — supply one "
            f"via model.with_base_url(provider, url) before resolve_switch_patch"
        )

    if tier_models is None:
        raise CodeHelperError(f"a model is required for claude + {provider.name}")

    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": anthropic_base_url(provider.base_url),
        "ANTHROPIC_AUTH_TOKEN": token,
        # Always emptied — same rationale as render._render_anthropic_env: an
        # inherited real Anthropic key would otherwise outrank the token.
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": tier_models.haiku,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": tier_models.sonnet,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": tier_models.opus,
    }
    if subagent_model is not None:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent_model
    # Same derivation the wrapper renderer uses — a switch and a wrapper for
    # the same model must agree on whether the window is declared at all.
    # Conditional for the same reason it is there: no declaration beats a
    # guessed window (an oversized claim overflows the real one mid-session).
    models = [tier_models.haiku, tier_models.sonnet, tier_models.opus]
    if subagent_model is not None:
        models.append(subagent_model)
    if (window := uniform_context_window(models)) is not None:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
    return SettingsPatch(provider_name=provider.name, env=env)


def read_settings(paths: Paths) -> tuple[str, dict]:
    """Read ``settings.json`` as ``(raw_text, parsed_object)``.

    Returns ``("", {})`` for a MISSING file — a fresh install (no
    ``~/.claude/settings.json`` yet) is not an error, matching
    ``codex_default``'s own ``read_text_or_none(...) or ""`` convention.

    Raises:
        CodeHelperError: the file exists but is unreadable (permission
            denied, undecodable as UTF-8), does not parse as JSON, or
            parses to something other than a JSON object. REFUSING here
            (rather than silently treating a broken file as empty) is the
            whole safety story: starting from ``{}`` would replace a
            corrupt-but-recoverable settings.json — with the user's proxy
            config and hooks in it — with a two-key file. This tool never
            repairs the file; it tells the user to fix or move it.
    """
    settings_path = paths.claude_settings()
    # `read_text_or_none` folds "missing" and "exists but unreadable"
    # (permission denied, binary/non-UTF-8) into the same `None` — by
    # design, for callers that treat both as "not ours, leave it alone".
    # THIS caller may not: a MISSING file is a fresh install (safe to start
    # from `{}`), but an EXISTING, unreadable file is exactly the
    # corrupt-but-recoverable case the docstring above promises to refuse,
    # not silently overwrite. `Path.exists()` disambiguates the two.
    if not settings_path.exists():
        return "", {}
    raw = read_text_or_none(settings_path)
    if raw is None:
        raise CodeHelperError(
            f"{settings_path} exists but could not be read (permission denied "
            f"or not valid UTF-8) — refusing to touch it; fix permissions or "
            f"move the file, then retry"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodeHelperError(
            f"{paths.claude_settings()} is not valid JSON ({exc}) — refusing "
            f"to touch it; fix or move the file, then retry"
        ) from exc
    if not isinstance(parsed, dict):
        raise CodeHelperError(
            f"{paths.claude_settings()} does not contain a JSON object at its "
            f"top level — refusing to touch it"
        )
    return raw, parsed


def without_env_keys(data: dict, keys: tuple[str, ...]) -> dict:
    """``data`` minus ``keys`` inside ``env``; everything else intact.

    The comparison basis for "this patch touched nothing outside the key set
    its module owns" — used by :func:`_verify_patch_applied` here and by
    ``services/proxy.py``'s own verification, each passing its OWN key tuple.
    Never used to build real output (that is :func:`patch_settings`'s job).

    Normalises the same way ``patch_settings`` does: an ``env`` block holding
    nothing but ``keys`` collapses to "no env key at all", so an ``original``
    with no ``env`` and a ``patched`` whose ``env`` was entirely owned keys
    compare equal — both mean "the user had no foreign env entries".
    """
    result = dict(data)
    env = result.get("env")
    if isinstance(env, dict):
        foreign = {k: v for k, v in env.items() if k not in keys}
        if foreign:
            result["env"] = foreign
        else:
            result.pop("env", None)
    return result


def _without_managed_env(data: dict) -> dict:
    """``data`` minus :data:`MANAGED_ENV_KEYS` — this module's own key set."""
    return without_env_keys(data, MANAGED_ENV_KEYS)


def patch_settings(original: dict, patch: SettingsPatch) -> dict:
    """Apply ``patch`` to a parsed settings object. PURE — returns a NEW object.

    Never mutates ``original``: the caller diffs old against new, and both
    must stay independently readable/printable.

    Within ``env`` ONLY: every key in :data:`MANAGED_ENV_KEYS` is removed,
    then ``patch.env`` is inserted. Every other key of ``env`` (a user's
    ``HTTPS_PROXY``, ``IS_DEMO``, ...) and every top-level key
    (``permissions``, ``hooks``, ``statusLine``, ``model``, ...) is carried
    through unchanged. Native deliberately keeps managed keys with empty
    values: deleting them would leave their prior values alive in an already
    running Claude Code process. For non-native patches, if ``env`` becomes
    empty after the removal, the ``env`` key itself is dropped.
    """
    result = dict(original)
    env = dict(result.get("env", {}))
    for key in MANAGED_ENV_KEYS:
        env.pop(key, None)
    env.update(patch.env)
    if env:
        result["env"] = env
    else:
        result.pop("env", None)
    return result


def _verify_patch_applied(original: dict, patched: dict, patch: SettingsPatch) -> None:
    """Assert, BEFORE any write, that the patch did exactly what it claims.

    Two checks: (1) every key in ``patch.env`` carries its intended value in
    ``patched["env"]``, and no OTHER managed key survives there; (2)
    everything outside the managed keys — every foreign ``env`` entry, every
    top-level key — is structurally identical between ``original`` and
    ``patched`` (via :func:`_without_managed_env`). The second check is the
    mechanical guarantee, not a hope, that ``HTTPS_PROXY``/``NO_PROXY``/
    ``hooks``/``permissions`` cannot be lost by a bug here — the direct JSON
    analogue of ``codex_default._verify_patch_applied``'s managed-region diff.

    Also re-parses ``json.dumps(patched)`` to prove the text about to be
    written round-trips — the JSON analogue of that module's ``tomllib``
    re-parse.

    Raises:
        CodeHelperError: any of the above fails. Always an internal-error
            message ("this is a codehelper bug") — a well-formed
            ``original``/``patch`` pair can never legitimately fail this.
    """
    patched_env = patched.get("env", {})
    if not isinstance(patched_env, dict):
        raise CodeHelperError(
            "internal error: patch_settings produced a non-dict `env` — "
            "this is a codehelper bug, please report it"
        )
    for key, value in patch.env.items():
        if patched_env.get(key) != value:
            raise CodeHelperError(
                "internal error: patch_settings failed to apply "
                f"{key}={value!r} — this is a codehelper bug, please report it"
            )
    survivors = set(patched_env) & set(MANAGED_ENV_KEYS) - set(patch.env)
    if survivors:
        raise CodeHelperError(
            f"internal error: patch_settings left stale managed key(s) "
            f"{sorted(survivors)} — this is a codehelper bug, please report it"
        )
    if _without_managed_env(original) != _without_managed_env(patched):
        raise CodeHelperError(
            "internal error: patching settings.json appears to have altered "
            "content outside the managed env keys — refusing to write; this "
            "is a codehelper bug, please report it"
        )
    try:
        if json.loads(json.dumps(patched, ensure_ascii=False)) != patched:
            raise CodeHelperError(
                "internal error: patched settings.json does not round-trip "
                "through JSON — this is a codehelper bug, please report it"
            )
    except (TypeError, ValueError) as exc:
        raise CodeHelperError(
            f"internal error: patched settings.json is not JSON-serialisable "
            f"({exc}) — this is a codehelper bug, please report it"
        ) from exc


def dump_settings(data: dict) -> str:
    """Render a patched settings object back to text.

    ``indent=2`` + ``ensure_ascii=False`` — a reasonable, stable default.
    Note this normalises whitespace: a hand-indented settings.json is
    reformatted to this style. Key ORDER survives (Python dicts preserve
    insertion order through the read-modify-write), only the indentation
    style may change — visible in the diff shown at confirm time.
    """
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def diff_preview(original: str, patched: str, *, label: str = "settings.json") -> str:
    """Unified diff of ``original`` -> ``patched``, stdlib only.

    Empty string when the two are identical (the no-op case). Mirrors
    ``codex_default.diff_preview``'s exact shape and calling convention.
    """
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"{label} (current)",
            tofile=f"{label} (new)",
        )
    )


def redact_credential(value: str) -> str:
    """``sk-nJyifvWJ1lscrXOR0oLrEg`` -> ``sk-n...rEg`` for a confirm preview.

    Short enough to still prove "yes, a token is here", too short to be
    useful if the preview is ever screenshotted or pasted somewhere. Values
    of 8 chars or fewer are fully masked rather than partially shown.
    """
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-3:]}"


#: Managed keys whose VALUE may itself be a credential, as opposed to a URL
#: or a model name (``ANTHROPIC_BASE_URL``, the ``*_MODEL`` keys). These are
#: the values ``_redacted_preview`` must mask — not just the newly supplied
#: ``token`` — because a PREVIOUS switch may have already left one of these
#: in ``original``, and a preview is printed to stdout / a confirm prompt.
_CREDENTIAL_ENV_KEYS: tuple[str, ...] = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")


def credential_values(parsed: dict) -> set[str]:
    """Every non-empty value of :data:`_CREDENTIAL_ENV_KEYS` in ``parsed["env"]``."""
    env = parsed.get("env")
    if not isinstance(env, dict):
        return set()
    return {
        value
        for key in _CREDENTIAL_ENV_KEYS
        if isinstance(value := env.get(key), str) and value
    }


def _redacted_preview(
    original: str,
    patched: str,
    token: str,
    *,
    original_parsed: dict,
    patched_parsed: dict,
) -> str:
    """:func:`diff_preview` with every credential value replaced everywhere.

    Redacts the newly supplied ``token`` AND every credential already
    present in ``original_parsed``/``patched_parsed`` (e.g.
    ``ANTHROPIC_AUTH_TOKEN`` left by a PREVIOUS switch) — not just the one
    this call is writing. Switching AWAY from a provider whose token is
    still sitting in ``env`` must not print that old value verbatim.
    """
    preview = diff_preview(original, patched)
    values = credential_values(original_parsed) | credential_values(patched_parsed)
    if token:
        values.add(token)
    for value in values:
        preview = preview.replace(value, redact_credential(value))
    return preview


def current_switch(paths: Paths) -> str | None:
    """Which provider ``settings.json`` is currently pointed at, or ``None``.

    Read-only, NEVER raises — a missing or corrupt file reads as ``None``,
    mirroring ``state.load_state``'s posture. Resolves by matching the live
    ``ANTHROPIC_BASE_URL`` against ``anthropic_base_url(p.base_url)`` for
    each switchable, non-reset provider. A managed but unrecognised endpoint
    reports ``"custom"``; only the absence of managed keys is ``None``.

    Deliberately derives this from the FILE ITSELF, not from ``state.json``:
    a second source of truth for "what's active" would go stale the moment
    the user hand-edits settings.json, the exact failure mode
    ``wrappers.valid_default_wrapper`` had to grow a staleness check for.
    The trade-off: a ``litellm`` switch (user-supplied ``base_url``) and any
    other provider that happens to share that same address are
    indistinguishable here — acceptable, and it resolves to whichever
    switchable provider matches first.
    """
    from codehelper.services.model import switchable_providers

    env = read_env(paths)
    if env is None:
        return None
    live_base_url = env.get("ANTHROPIC_BASE_URL")
    if not live_base_url:
        return None
    for provider in switchable_providers():
        if provider.env_reset or not provider.base_url:
            continue
        if anthropic_base_url(provider.base_url) == live_base_url:
            return provider.name
    return "custom"


def read_env(paths: Paths) -> dict | None:
    """The ``env`` block of settings.json, or None when unreadable/absent.

    The single place the read-only "never raises" posture toward a missing or
    corrupt settings.json is defined — shared by ``current_switch``/
    ``active_switch_env`` here and by ``services/proxy.py``'s ``proxy_status``,
    so a change to that posture is made once.
    """
    try:
        _, parsed = read_settings(paths)
    except CodeHelperError:
        return None
    env = parsed.get("env")
    if not isinstance(env, dict):
        return None
    return env


def active_switch_env(paths: Paths) -> dict[str, str] | None:
    """Snapshot managed live settings, or None when no override is present."""
    env = read_env(paths)
    if env is None:
        return None
    managed = {key: value for key, value in env.items() if key in MANAGED_ENV_KEYS}
    return managed or None


def live_axes_for_spec(spec) -> tuple[Provider, TierModels, str | None]:
    """Translate a Claude wrapper/preset into live-settings axes."""
    if ConfigShape.ANTHROPIC_SETTINGS not in spec.provider.shapes:
        raise CodeHelperError(
            f"wrapper {spec.name!r} cannot switch a live Claude session"
        )
    tiers = spec.tier_models or TierModels.uniform(spec.model)
    subagent = (
        spec.model if spec.shape is ConfigShape.OLLAMA_LAUNCH else spec.subagent_model
    )
    return spec.provider, tiers, subagent


def matches_switch_spec(active_env: dict[str, str] | None, spec) -> bool:
    """True iff a managed snapshot exactly represents ``spec``'s target."""
    if active_env is None:
        return False
    try:
        provider, tiers, subagent = live_axes_for_spec(spec)
        patch = resolve_switch_patch(
            provider,
            tier_models=tiers,
            token=active_env.get("ANTHROPIC_AUTH_TOKEN", ""),
            subagent_model=subagent,
        )
    except CodeHelperError:
        return False
    return active_env == patch.env


def apply_switch(
    paths: Paths,
    *,
    provider: Provider,
    tier_models: TierModels | None = None,
    token: str = "",
    subagent_model: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Patch ``~/.claude/settings.json``'s ``env`` block to point Claude Code
    at ``provider`` — or, for an ``env_reset`` provider, to clear the override.

    Flow: resolve the patch -> read + parse (refuse outright on a corrupt
    file) -> patch (pure) -> verify (before any write) -> gate (force/confirm,
    raise on refusal) -> rotate backups -> ``atomic_write`` at ``0o600``
    (this file may now carry a token). ``--dry-run`` prints the redacted diff
    and returns without touching anything — never prompts, mirrors every
    other command's ``--dry-run`` contract in this project.

    Args:
        confirm: ``(path: Path, preview: str) -> bool`` — called ONLY when a
            real, visible write is about to happen and neither ``force`` nor
            an existing no-op short-circuit applies. ``preview`` is a unified
            diff with any token value redacted. ``None`` behaves like "always
            refuse".

    Returns:
        True if anything was written (or would be, under ``--dry-run``);
        False on a true no-op (the file already matches the resolved patch)
        — a no-op consumes no backup slot (see :func:`~codehelper.backends
        ._atomic.rotate_backups`'s ring semantics).

    Raises:
        CodeHelperError: an incompatible/unresolvable provider (from
            :func:`resolve_switch_patch`), an unparseable existing
            settings.json, a patch that fails its own post-write
            verification, or a refused overwrite (no ``--force``/confirmation).
    """
    settings_path = paths.claude_settings()
    original_text, original = read_settings(paths)

    original_env = original.get("env")
    patch = resolve_switch_patch(
        provider,
        tier_models=tier_models,
        token=token,
        subagent_model=subagent_model,
        current_env=original_env if isinstance(original_env, dict) else {},
    )

    patched = patch_settings(original, patch)
    _verify_patch_applied(original, patched, patch)

    patched_text = dump_settings(patched)
    # Compare the PARSED objects, not the raw text: a byte-for-byte text
    # comparison would treat re-formatting alone (e.g. a hand-indented
    # existing file normalised to dump_settings' 2-space style) as a real change,
    # rotating a backup and writing for something that resolves to the exact
    # same env either way.
    if patched == original:
        return False

    preview = _redacted_preview(
        original_text,
        patched_text,
        token,
        original_parsed=original,
        patched_parsed=patched,
    )

    if dry_run:
        print(preview or "(no textual change)")
        print(f"would write {settings_path}")
        return True

    if not force and not (confirm and confirm(settings_path, preview)):
        raise CodeHelperError(
            f"about to patch {settings_path} — refusing without confirmation "
            f"(use --force, or re-run interactively)"
        )

    # Re-read immediately before writing: `original_text` was captured back
    # at function entry, and an interactive `confirm` prompt (or simply the
    # gap between read and write) gives a concurrent `switch`/hand-edit a
    # window to change the file in between. Without this recheck that
    # concurrent write is silently lost — this refuses instead of clobbering
    # it, mirroring codex_default's own stale-file guard.
    # The lock must cover the final check, backup rotation, and replacement.
    # Locking only the check still permits two cooperating writers to both
    # pass it and then rotate/write stale snapshots.
    with file_lock(settings_path):
        current_text = read_text_or_none(settings_path) or ""
        if current_text != original_text:
            raise CodeHelperError(
                f"{settings_path} changed since it was read — refusing to write "
                f"a patch computed off a stale snapshot (a concurrent switch or "
                f"hand-edit may have run; re-run to patch the current file)"
            )

        if original_text:
            rotate_backups(settings_backup_slots(paths), current=original_text)
        atomic_write(settings_path, patched_text, mode=0o600)
    action = "reset to native" if patch.is_reset else "wrote"
    print(f"{action} {settings_path} (backup: {paths.claude_settings_backup(1)})")
    return True


def settings_backup_slots(paths: Paths) -> tuple[Path, Path, Path]:
    return (
        paths.claude_settings_backup(3),
        paths.claude_settings_backup(2),
        paths.claude_settings_backup(1),
    )


def restore_settings(
    paths: Paths,
    *,
    slot: int = 1,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Restore ``~/.claude/settings.json`` from backup slot ``slot``.

    Does NOT itself create a new backup slot — restoring is "put it back",
    not a fresh edit to archive; a subsequent ``switch`` backs up the
    just-restored state normally. Mirrors ``codex_default.restore_default``'s
    shape for the single-file case (no paired catalog here).

    Raises:
        CodeHelperError: no backup exists at ``slot``, or the overwrite is
            refused (no ``--force``/confirmation) — restoring can still
            discard a DIFFERENT current settings.json if it was hand-edited
            (or changed by Claude Code itself) since the backup was taken.
    """
    backup_path = paths.claude_settings_backup(slot)
    backup_body = read_text_or_none(backup_path)
    if backup_body is None:
        raise CodeHelperError(f"no backup found at {backup_path}; nothing to restore")

    # A backup is never written by anything but rotate_backups (this
    # module's own prior writes), but it CAN be hand-corrupted or truncated
    # on disk afterward. Validate it the same way read_settings validates a
    # live settings.json — restoring a malformed file would leave Claude
    # Code unable to load its config, the exact failure this command exists
    # to recover FROM, not cause.
    try:
        backup_parsed = json.loads(backup_body)
    except json.JSONDecodeError as exc:
        raise CodeHelperError(
            f"{backup_path} is not valid JSON ({exc}) — refusing to restore "
            f"a corrupted backup"
        ) from exc
    if not isinstance(backup_parsed, dict):
        raise CodeHelperError(
            f"{backup_path} does not contain a JSON object at its top level "
            f"— refusing to restore it"
        )

    settings_path = paths.claude_settings()
    current = read_text_or_none(settings_path) or ""

    if current == backup_body:
        return False

    # Same credential-aware redaction as apply_switch's _redacted_preview:
    # both `current` and `backup_body` may legitimately hold a live
    # ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY, and this preview goes to
    # stdout / an interactive confirm prompt exactly like a switch's does.
    # `backup_parsed` is already validated above; `current` may still be
    # unparseable (it is NOT validated — a corrupt live file is
    # read_settings/apply_switch's problem, restore's job is only to not
    # WRITE a corrupt one), so its parse is still guarded.
    preview = diff_preview(current, backup_body)
    try:
        secrets = credential_values(json.loads(current) if current else {})
    except json.JSONDecodeError:
        secrets = set()  # unparseable current text can't be key-scanned
    secrets |= credential_values(backup_parsed)
    for value in secrets:
        preview = preview.replace(value, redact_credential(value))

    if dry_run:
        print(preview or "(no textual change)")
        print(f"would restore {settings_path} from {backup_path}")
        return True

    if not force and not (confirm and confirm(settings_path, preview)):
        raise CodeHelperError(
            f"about to restore {settings_path} from {backup_path} — refusing "
            f"without confirmation (use --force)"
        )

    # Re-read immediately before writing — same stale-snapshot guard as
    # apply_switch: a concurrent switch/restore or hand-edit during the
    # confirm prompt must not be silently clobbered.
    with file_lock(settings_path):
        current_now = read_text_or_none(settings_path) or ""
        if current_now != current:
            raise CodeHelperError(
                f"{settings_path} changed since it was read — refusing to "
                f"restore over a concurrent change (a concurrent switch or "
                f"hand-edit may have run; re-run to restore over the current file)"
            )

        atomic_write(settings_path, backup_body, mode=0o600)
    print(f"restored {settings_path} from {backup_path}")
    return True
