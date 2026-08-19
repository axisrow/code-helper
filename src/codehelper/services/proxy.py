"""``proxy`` — toggle the proxy variables in Claude Code's OWN ``~/.claude/settings.json``.

The second module allowed to patch that file, alongside
``services/claude_settings.py``. The two share the file and share nothing
else: ``claude_settings`` owns the ``ANTHROPIC_*`` backend keys
(:data:`~codehelper.services.claude_settings.MANAGED_ENV_KEYS`), this module
owns the proxy keys below. Neither may touch the other's set, and
:func:`_verify_proxy_patch` proves that mechanically before every write —
the same posture ``claude_settings._verify_patch_applied`` takes toward the
proxy keys from the other side.

Why a separate module rather than more keys in ``MANAGED_ENV_KEYS``
--------------------------------------------------------------------
Adding the proxy keys there would pull them under ``switch native``, whose
whole job is to blank every managed key — so changing model backends would
silently drop the user's proxy. ``test_managed_keys_match_renderer`` also
pins that tuple against ``render._render_anthropic_env``, which has no
business emitting proxy variables into a wrapper script. Two closed key
sets, two owners.

Why all four proxy keys move together
----------------------------------------
Claude Code reads "the first one that's set in the order ``https_proxy``,
``HTTPS_PROXY``, ``http_proxy``, ``HTTP_PROXY``" (Enterprise network
configuration). Blanking only the uppercase pair would leave a lowercase
``https_proxy`` to win, and the toggle would report "off" while traffic
still went through the proxy. So :data:`PROXY_ENV_KEYS` is the whole
precedence chain and every write sets all of it.

Why ``NO_PROXY`` is NOT part of the toggle
---------------------------------------------
It is a bypass list, inert while no proxy is set, so blanking it on "off"
would destroy a hand-maintained list for no benefit. It is edited
separately, via ``no_proxy=`` on :func:`apply_proxy`.

Both case spellings of it are kept in sync deliberately.
``NO_PROXY``/``no_proxy`` in one file looks like a duplicate, and for Claude
Code itself it arguably is. But the ``env`` block is inherited by every child
process — hooks, MCP servers, ``npx`` — and those disagree: Python's
``urllib`` honours ONLY lowercase ``no_proxy``, curl takes either. Dropping
one spelling would tidy the file and break the bypass for some consumer, so
both are written, always with the same value: two spellings holding
DIFFERENT lists is the genuinely harmful case this avoids.

Off means empty, not absent — for keys that exist
----------------------------------------------------
Same reasoning as ``claude_settings``' ``env_reset``: Claude Code applies
settings updates to the running process, but removing a key from the file
does not unset an already-applied process value. An empty string both
resets the live session and stays false-y on a fresh launch. That argument
only holds for a key that HAS a live value, so ``off`` blanks only keys
already present — writing ``""`` into a file with no proxy at all would add
entries nobody set and turn a no-op into a real write.

The address is not lost: it is banked in ``state.json``
(``state.set_saved_proxy``), which is what lets ``proxy on`` put it back.
That write happens AFTER ``settings.json`` was successfully patched, never
before — see :func:`set_proxy_state` for why the ordering is load-bearing
against a refused confirmation and against a concurrent writer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from codehelper.backends._atomic import (
    atomic_write,
    file_lock,
    read_text_or_none,
    rotate_backups,
)
from codehelper.errors import CodeHelperError
from codehelper.services.claude_settings import (
    credential_values,
    diff_preview,
    dump_settings,
    read_env,
    read_settings,
    redact_credential,
    settings_backup_slots,
    without_env_keys,
)
from codehelper.services.paths import Paths

__all__ = [
    "PROXY_ENV_KEYS",
    "NO_PROXY_ENV_KEYS",
    "ProxyStatus",
    "validate_proxy_url",
    "resolve_proxy_patch",
    "patch_proxy_settings",
    "proxy_status",
    "apply_proxy",
    "set_proxy_state",
]

#: The proxy env keys this module owns, in Claude Code's documented
#: precedence order (``https_proxy`` wins over ``HTTPS_PROXY`` wins over
#: ``http_proxy`` wins over ``HTTP_PROXY``). This tuple IS the ownership
#: marker — JSON admits no comment — exactly as ``MANAGED_ENV_KEYS`` is for
#: ``claude_settings``. Every write sets the WHOLE chain: leaving any one of
#: them behind lets it outrank the others and makes the toggle lie.
PROXY_ENV_KEYS: tuple[str, ...] = (
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
)

#: The bypass-list keys. Owned by this module (so the toggle's verification
#: allows editing them) but deliberately NOT touched by on/off — see the
#: module docstring. Both spellings are always written with the same value.
NO_PROXY_ENV_KEYS: tuple[str, ...] = ("NO_PROXY", "no_proxy")

#: Every key this module may write. Anything outside it — the ``ANTHROPIC_*``
#: keys ``claude_settings`` owns, ``IS_DEMO``, ``hooks``, ``permissions`` —
#: must survive byte-for-byte, which :func:`_verify_proxy_patch` proves.
_OWNED_ENV_KEYS: tuple[str, ...] = (*PROXY_ENV_KEYS, *NO_PROXY_ENV_KEYS)

#: Schemes Claude Code can actually use. It "does not support SOCKS
#: proxies", and rejecting one here — with that reason — beats letting the
#: agent fail later at a layer that cannot explain why.
_ALLOWED_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class ProxyStatus:
    """What the settings file and ``state.json`` currently say about the proxy.

    ``url`` is the live address read from the file (the first non-empty key
    in :data:`PROXY_ENV_KEYS` order, i.e. the one Claude Code would itself
    use); ``saved_url`` is the address remembered for the next ``proxy on``.
    They differ exactly when the proxy is off but an address is remembered.
    """

    url: str | None
    saved_url: str | None
    no_proxy: str | None

    @property
    def enabled(self) -> bool:
        """Whether a proxy is live. Derived, so it cannot disagree with ``url``."""
        return self.url is not None

    @property
    def restorable_url(self) -> str | None:
        """The address ``proxy on`` would use — live first, then remembered."""
        return self.url or self.saved_url

    # The display forms exist so the SAFE spelling is also the obvious one:
    # a proxy URL may carry basic-auth credentials, and every UI that shows
    # an address would otherwise have to remember to call `redact_proxy_url`
    # itself. A raw `.url` read is now a visible deviation rather than the
    # default. The unredacted fields stay available for the code that needs
    # the real value (applying it, comparing it).
    @property
    def display_url(self) -> str:
        """The live address, password-masked, for printing."""
        return redact_proxy_url(self.url or "")

    @property
    def display_saved_url(self) -> str:
        """The remembered address, password-masked, for printing."""
        return redact_proxy_url(self.saved_url or "")

    @property
    def display_restorable_url(self) -> str:
        """:attr:`restorable_url`, password-masked, for printing."""
        return redact_proxy_url(self.restorable_url or "")


def validate_proxy_url(url: str) -> str:
    """Return ``url`` unchanged if Claude Code can use it, else raise.

    Claude Code parses the proxy URL AT STARTUP and refuses to launch when
    it cannot — "when it can't parse the value, such as one missing the
    ``http://`` scheme, Claude Code stops launch with an error naming the
    variable to fix". A toggle that can write an unparseable value is a
    toggle that can leave the agent unable to start, so this runs before any
    write rather than leaving the user to discover it on the next launch.

    Raises:
        CodeHelperError: empty, missing/unsupported scheme (SOCKS is called
            out by name — Claude Code does not support it), or no host.
    """
    candidate = url.strip()
    if not candidate:
        raise CodeHelperError(
            "proxy URL is empty — give one like http://127.0.0.1:8118"
        )

    parsed = urlparse(candidate)
    if parsed.scheme in ("socks", "socks4", "socks5", "socks5h"):
        raise CodeHelperError(
            f"{candidate!r} is a SOCKS proxy — Claude Code does not support "
            f"SOCKS; use an http:// or https:// proxy"
        )
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise CodeHelperError(
            f"{candidate!r} has no usable scheme — Claude Code refuses to "
            f"start on a proxy URL it cannot parse; write it as "
            f"http://host:port (supported: {', '.join(_ALLOWED_SCHEMES)})"
        )
    if not parsed.hostname:
        raise CodeHelperError(
            f"{candidate!r} names no host — write it as http://host:port"
        )
    return candidate


def redact_proxy_url(url: str) -> str:
    """``http://u:pw@host:8118`` -> ``http://u:***@host:8118``.

    Claude Code's docs explicitly allow basic-auth credentials inside the
    proxy URL, so a diff printed to stdout or an interactive confirm prompt
    can carry a password. Mirrors ``claude_settings.redact_credential``'s purpose: keep
    enough to recognise the value, not enough to reuse it. A URL with no
    password is returned unchanged.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = f"{parsed.username or ''}:***@{host}"
    return urlunparse(parsed._replace(netloc=netloc))


def _redacted_preview(
    original_text: str,
    patched_text: str,
    *,
    original: dict,
    patched: dict,
) -> str:
    """:func:`~codehelper.services.claude_settings.diff_preview`, fully masked.

    Masks TWO different kinds of secret, because this file has two owners and
    a unified diff carries CONTEXT lines: the proxy passwords this module
    writes, AND the ``ANTHROPIC_AUTH_TOKEN``/``ANTHROPIC_API_KEY`` that
    ``claude_settings`` may have left in the same ``env`` block. Redacting
    only its own values would print someone else's credential verbatim —
    ownership of a KEY does not limit what a diff of the FILE can reveal.

    Takes the parsed objects the caller already holds rather than re-parsing
    the text: ``apply_proxy`` has both in scope, and ``patched`` was just
    serialised to produce ``patched_text``.
    """
    preview = diff_preview(original_text, patched_text)
    if not preview:
        return preview
    values = _proxy_values_in(original) | _proxy_values_in(patched)
    for raw in values:
        redacted = redact_proxy_url(raw)
        if redacted != raw:
            preview = preview.replace(raw, redacted)
    # The other owner's credentials, masked with that owner's own redactor.
    for raw in credential_values(original) | credential_values(patched):
        preview = preview.replace(raw, redact_credential(raw))
    return preview


def _proxy_values_in(data: dict) -> set[str]:
    """Every non-empty proxy value in a parsed settings object's ``env``."""
    env = data.get("env")
    if not isinstance(env, dict):
        return set()
    return {
        value
        for key in PROXY_ENV_KEYS
        if isinstance(value := env.get(key), str) and value
    }


def resolve_proxy_patch(
    *,
    url: str | None,
    no_proxy: str | None = None,
    current_env: dict | None = None,
) -> dict[str, str]:
    """The exact ``env`` entries to write. PURE — no IO.

    ``url=""`` blanks the proxy chain (off); a non-empty ``url`` sets every
    key in it (on); ``url=None`` leaves the proxy chain alone, which is what
    a ``NO_PROXY``-only edit needs.

    Blanking touches only keys ALREADY PRESENT in ``current_env``, plus the
    two canonical spellings ``HTTPS_PROXY``/``HTTP_PROXY``. Blanking a key
    nobody ever set would add entries on an off->off no-op — the same
    reasoning behind ``claude_settings.resolve_switch_patch``'s ``current_env``
    handling. Turning the proxy ON writes the whole chain regardless, because
    a lowercase key left unset would be fine but a lowercase key left with a
    STALE value would silently outrank the new one.

    ``no_proxy`` is written to BOTH spellings with the same value (see the
    module docstring on why both exist and why they must never diverge).
    """
    present = set(current_env or {})
    patch: dict[str, str] = {}

    if url is not None:
        if url:
            patch.update({key: url for key in PROXY_ENV_KEYS})
        else:
            # Blank only what is actually there. A key that was never set has
            # no already-applied process value to reset, so writing "" to it
            # adds an entry nobody asked for and — on a file with no proxy at
            # all — turns `off` into a real write that burns a backup slot
            # instead of the no-op it should be. Same reasoning as
            # ``claude_settings.resolve_switch_patch``'s ``current_env``
            # handling for ``switch native``.
            patch.update({key: "" for key in PROXY_ENV_KEYS if key in present})

    if no_proxy is not None:
        # An empty bypass list is written only where a key already exists, for
        # the same reason `off` blanks only present keys: "no bypass list" is
        # the ABSENCE of the key, so writing "" into a file that never had one
        # adds entries that `proxy_status` then reports as absent anyway —
        # display and file disagreeing over a value neither needs.
        patch.update(
            {key: no_proxy for key in NO_PROXY_ENV_KEYS if no_proxy or key in present}
        )

    return patch


def patch_proxy_settings(original: dict, patch: dict[str, str]) -> dict:
    """Apply ``patch`` to a parsed settings object. PURE — returns a NEW object.

    Never mutates ``original``: the caller diffs one against the other. Only
    keys named in ``patch`` are written; every other ``env`` entry and every
    top-level key is carried through untouched. Unlike
    ``claude_settings.patch_settings`` this does NOT pre-remove its whole key
    set — off writes empty values rather than deleting keys, and a
    ``NO_PROXY``-only edit must leave the proxy chain exactly as it found it.
    """
    result = dict(original)
    env = dict(result.get("env", {}))
    env.update(patch)
    if env:
        result["env"] = env
    else:
        result.pop("env", None)
    return result


def _verify_proxy_patch(original: dict, patched: dict, patch: dict[str, str]) -> None:
    """Assert, BEFORE any write, that the patch did exactly what it claims.

    Two checks: every key in ``patch`` carries its intended value, and
    NOTHING outside :data:`_OWNED_ENV_KEYS` differs between ``original`` and
    ``patched``. The second is the mechanical guarantee that this module
    cannot damage ``claude_settings``' ``ANTHROPIC_*`` keys, the user's
    ``hooks``/``permissions``/``autoMode``, or anything else in the file —
    the mirror image of ``claude_settings._verify_patch_applied``'s promise
    about the proxy keys.

    Raises:
        CodeHelperError: always an internal-error message; a well-formed
            ``original``/``patch`` pair cannot legitimately fail this.
    """
    patched_env = patched.get("env", {})
    if not isinstance(patched_env, dict):
        raise CodeHelperError(
            "internal error: patch_proxy_settings produced a non-dict `env` — "
            "this is a codehelper bug, please report it"
        )
    for key, value in patch.items():
        if patched_env.get(key) != value:
            raise CodeHelperError(
                f"internal error: patch_proxy_settings failed to apply "
                f"{key}={value!r} — this is a codehelper bug, please report it"
            )
    if without_env_keys(original, _OWNED_ENV_KEYS) != without_env_keys(
        patched, _OWNED_ENV_KEYS
    ):
        raise CodeHelperError(
            "internal error: patching settings.json appears to have altered "
            "content outside the proxy keys — refusing to write; this is a "
            "codehelper bug, please report it"
        )


def proxy_status(paths: Paths) -> ProxyStatus:
    """What the proxy configuration currently is. Read-only, NEVER raises.

    A missing, unreadable, or corrupt settings.json reads as "off, nothing
    configured" — the same never-fails-on-its-way-out posture
    ``claude_settings.current_switch`` and ``state.load_state`` take, because
    this feeds a status line and a TUI label that must render on any machine.

    ``enabled`` is true when any key in :data:`PROXY_ENV_KEYS` holds a
    non-empty value, and ``url`` reports the first such key in precedence
    order — i.e. the address Claude Code would actually use, not merely the
    one most recently written.
    """
    from codehelper.services.state import saved_proxy

    env = read_env(paths) or {}
    url = next(
        (
            value
            for key in PROXY_ENV_KEYS
            if isinstance(value := env.get(key), str) and value
        ),
        None,
    )
    # `saved_url` has a second source deliberately. `off` blanks the live
    # values and banks the address in state.json, so that bank is normally
    # the only copy — but it is a SEPARATE file, and a write to it can fail
    # (disk full, permissions) or be lost after the settings write already
    # landed. The address is still in the backup `off` just rotated, so fall
    # back to it rather than reporting "no address configured" while the
    # value sits one file over. This is why the two writes need no
    # cross-file transaction: the settings backup already IS the redundancy.
    saved = saved_proxy(paths) or _url_from_backup(paths)
    no_proxy = next(
        (
            value
            for key in NO_PROXY_ENV_KEYS
            if isinstance(value := env.get(key), str) and value
        ),
        None,
    )
    return ProxyStatus(
        url=url,
        saved_url=saved,
        no_proxy=no_proxy,
    )


def _url_from_backup(paths: Paths) -> str | None:
    """The proxy address in the most recent settings.json backup, if any.

    Read-only and never raises — an absent or unparseable backup is simply
    "no fallback", the same posture :func:`proxy_status` takes toward the
    live file. Only slot 1 is consulted: it is the state `off` rotated away
    moments earlier, so it is the copy that matters; an older slot could hold
    an address the user has since deliberately changed.
    """
    import json

    body = read_text_or_none(paths.claude_settings_backup(1))
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    env = parsed.get("env")
    if not isinstance(env, dict):
        return None
    return next(
        (
            value
            for key in PROXY_ENV_KEYS
            if isinstance(value := env.get(key), str) and value
        ),
        None,
    )


def apply_proxy(
    paths: Paths,
    *,
    url: str | None,
    no_proxy: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Patch the proxy keys in ``~/.claude/settings.json``'s ``env`` block.

    Flow, deliberately identical to ``claude_settings.apply_switch``: resolve
    the patch -> read + parse (refuse outright on a corrupt file) -> patch
    (pure) -> verify (before any write) -> gate (force/confirm) -> lock ->
    re-check for a concurrent change -> rotate backups -> ``atomic_write`` at
    ``0o600``. ``--dry-run`` prints the redacted diff and touches nothing.

    ``0o600`` matches what ``apply_switch`` leaves the file at, and is
    warranted on its own terms: a proxy URL may embed basic-auth credentials.

    Args:
        url: the proxy address to set, ``""`` to turn the proxy off, or
            ``None`` to leave the proxy chain untouched (a ``no_proxy``-only
            edit). A non-empty value is validated first.
        no_proxy: new bypass list, written to both spellings. ``None`` leaves
            it alone.
        confirm: ``(path, preview) -> bool``, called only when a real write
            is about to happen and ``force`` is not set. ``None`` behaves as
            "always refuse".

    Returns:
        True if anything was written (or would be, under ``--dry-run``);
        False on a true no-op, which consumes no backup slot.

    Raises:
        CodeHelperError: an invalid proxy URL, an unparseable settings.json,
            a patch that fails its own verification, or a refused overwrite.
    """
    if url:
        url = validate_proxy_url(url)

    settings_path = paths.claude_settings()
    original_text, original = read_settings(paths)

    original_env = original.get("env")
    patch = resolve_proxy_patch(
        url=url,
        no_proxy=no_proxy,
        current_env=original_env if isinstance(original_env, dict) else {},
    )
    if not patch:
        return False

    patched = patch_proxy_settings(original, patch)
    _verify_proxy_patch(original, patched, patch)

    patched_text = dump_settings(patched)
    # Compare PARSED objects, not raw text: reformatting alone (a
    # hand-indented file normalised to dump_settings' style) is not a real change
    # and must not rotate a backup.
    if patched == original:
        return False

    preview = _redacted_preview(
        original_text, patched_text, original=original, patched=patched
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

    # Re-read immediately before writing. `original_text` was captured at
    # entry, and an interactive confirm prompt gives a concurrent `switch`,
    # `proxy`, or hand-edit a window to change the file. The lock must cover
    # the check, the rotation and the replacement together — locking only the
    # check would let two writers both pass it and then write stale snapshots.
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

    action = "disabled proxy in" if url == "" else "wrote"
    print(f"{action} {settings_path} (backup: {paths.claude_settings_backup(1)})")
    return True


def set_proxy_state(
    paths: Paths,
    *,
    action: str | None = None,
    url: str | None = None,
    no_proxy: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm=None,
) -> bool:
    """Resolve a proxy verb into a write, saving the address first.

    The whole two-file state machine in one place: resolve ``toggle`` against
    what is actually on disk, work out which address ``on`` should restore,
    persist that address to ``state.json`` BEFORE ``settings.json`` is
    blanked, then delegate to :func:`apply_proxy`.

    The ordering is the point. ``off`` writes empty values, so once it has
    run the address is gone from the file and ``on`` would have nothing to
    restore — it has to be captured first. Keeping that here rather than in
    the CLI handler means every entry point (the ``proxy`` command and the
    TUI's chip row and settings screen) gets it from one place, and
    ``dry_run`` is honoured in one place instead of on each branch.

    Args:
        action: ``"on"``, ``"off"``, ``"toggle"``, or ``None`` when only
            ``url``/``no_proxy`` is being set.
        url: a new address to set, which implies turning the proxy on.
        no_proxy: a new bypass list, applied independently of ``action``.

    Returns:
        Whatever :func:`apply_proxy` returns — False on a true no-op.

    Raises:
        CodeHelperError: ``on`` with no address known anywhere, or an
            invalid ``url`` (rejected before anything is written).
    """
    from codehelper.services.state import set_saved_proxy

    status = proxy_status(paths)

    if action == "toggle":
        action = "off" if status.enabled else "on"

    target: str | None = None
    if url is not None:
        target = validate_proxy_url(url)
    elif action == "on":
        target = status.restorable_url
        if not target:
            raise CodeHelperError(
                "no proxy address configured — set one first: "
                "codehelper proxy --url http://host:port"
            )
    elif action == "off":
        target = ""

    to_save = status.url if target == "" else target

    # Write settings.json FIRST, bank the address only once that write really
    # landed. Ordering matters twice over:
    #
    #   * a refused confirmation raises out of `apply_proxy`, so state.json is
    #     never left pointing at an address the user declined to apply;
    #   * `apply_proxy` re-reads under its lock and refuses a patch computed
    #     off a stale snapshot, so a concurrent writer that switches the live
    #     proxy between `proxy_status` above and the write cannot get its new
    #     endpoint disabled while THIS command banks the old one for a later
    #     `proxy on` to restore.
    #
    # The earlier "save before blanking" ordering was aimed at a real problem —
    # `off` erases the address, so it must be captured beforehand — but
    # capturing it in `to_save` above is what solves that; persisting it early
    # only widened the window in which the two files disagree.
    wrote = apply_proxy(
        paths,
        url=target,
        no_proxy=no_proxy,
        dry_run=dry_run,
        force=force,
        confirm=confirm,
    )

    if wrote and to_save and to_save != status.saved_url and not dry_run:
        try:
            set_saved_proxy(paths, to_save)
        except CodeHelperError as exc:
            # The settings write already landed, so raising here would report
            # failure for an operation that DID happen. Warn instead: the
            # address is still in the backup this write just rotated, which
            # `proxy_status` falls back to — so `proxy on` keeps working.
            print(f"warning: {exc}", file=sys.stderr)

    return wrote
