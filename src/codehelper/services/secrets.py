"""Resolving a wrapper's secret token.

The resolution order is **env var → selected profile → hidden prompt**, and a
token resolved from a prompt is cached back so the next ``add``/``--list-models``
does not ask again. The cache is the ONLY persistent copy of a token this project
keeps outside a generated wrapper script — and it is a CACHE, not a session:

- A wrapper carries its token baked into the script itself (``0o700``). Deleting
  ``credentials.json`` does NOT break an installed wrapper; it only means the
  next install/discovery prompts again.
- There is no "logged in" state, no provider round-trip to validate a token, no
  command that "connects" codehelper to a service. ``codehelper`` configures
  agents and aliases; it does not authenticate with anything.
- The file is written ``0o600`` (owner-only) and read by no one but this module.

The profile shape is ``{provider_name: {profile_name: token}}`` — profiles are
scoped to providers, never wrappers, because one provider backs many wrappers.
The first key uses the internal ``default`` profile. When a second key is
added, the UI names both profiles. The old flat ``{provider_name: token}``
shape is read as ``default`` for backward compatibility.

Provider-specific key FORMAT validation (the archived project's strict
``<32-hex>.<16-alnum>`` Z.ai regex) is deliberately NOT reproduced here — this
project has no reason to know what a valid Z.ai (or any other provider's) key
looks like. Instead, :func:`codehelper.services.wrappers.render_script`
neutralizes shell metacharacters in EVERY interpolated value via POSIX
single-quoting, so an arbitrary (even adversarial) token string can never
break out of the generated script — see the injection tests in
``tests/test_wrappers.py``.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from codehelper.backends._atomic import atomic_write, best_effort_lock, read_json_object
from codehelper.errors import CodeHelperError
from codehelper.services.model import RETIRED_PROVIDER_NAMES, provider_storage_names
from codehelper.services.paths import Paths


def _storage_names(provider_name: str) -> tuple[str, ...]:
    """``provider_name``, plus its retired predecessor name if it has one.

    The read-path consumer of ``model.provider_storage_names`` — the ONE
    decision point for "which spellings may this provider be stored under"
    now lives THERE, next to :data:`model.RETIRED_PROVIDER_NAMES`. Every
    read path below (:func:`profile_names`, :func:`valid_active_profile`,
    :func:`credential_for`) consults this wrapper instead of re-deriving
    the current-name/legacy-name fallback separately.
    """
    return provider_storage_names(provider_name)


__all__ = [
    "ResolvedToken",
    "DEFAULT_PROFILE",
    "resolve_token",
    "env_cache_conflict",
    "resolve_with_conflict_check",
    "load_credentials",
    "mask_token",
    "profile_rows",
    "render_token",
    "profile_names",
    "seed_default_profile",
    "rename_profile",
    "credential_for",
    "valid_active_profile",
    "save_credential",
    "invalidate_cached_credential",
    "cache_freshly_typed_token",
    "token_for_discovery",
]

#: Where a resolved token came from — ``resolve_token`` returns a
#: :class:`ResolvedToken` so a caller can decide whether to cache it (only a
#: value the user just typed is worth caching; an env value already outlives
#: this process, and a cached value is already in the file).
SOURCE_ENV = "env"
SOURCE_CACHE = "file"
SOURCE_PROMPT = "prompt"
DEFAULT_PROFILE = "default"


def mask_token(value: str) -> str:
    """A display-safe form of a stored token: ``fe42****6309``.

    The viewer's default mask (``codehelper tokens``, the TUI Tokens screen):
    the head and tail stay readable so a user can tell WHICH key a store
    holds — the whole reason the viewer exists — while everything between is
    unrecoverable. Distinct from ``claude_settings.redact_credential`` (the
    ``4...3`` diff-preview mask), which is contracted by tests and used by
    both settings.json owners; that one answers "is this secret safe to
    print", this one answers "is this the key I think it is".

    A token of 8 characters or fewer is ALL asterisks: head+tail of a short
    value would leave nothing hidden.

    "Display-safe" is enforced, not assumed: nothing constrains what a
    cached or environment-provided value may contain, and a control
    character surviving into the readable head/tail could spoof the
    terminal (ANSI colouring, cursor moves, fake rows) — so every
    non-printable renders as a visible ``\\xNN`` escape.
    """
    if len(value) <= 8:
        return "*" * len(value)

    def printable(part: str) -> str:
        return "".join(ch if ch.isprintable() else f"\\x{ord(ch):02x}" for ch in part)

    return printable(value[:4]) + "****" + printable(value[-4:])


def render_token(value: str, reveal: bool) -> str:
    """The viewer's rendering of one token — the ONE place mask meets reveal.

    Shared by the CLI's ``tokens`` command and the TUI Tokens screen so their
    two renderings of the same value cannot drift (e.g. one gaining a
    display-safety fix the other misses). ``reveal=True`` is the documented
    explicit raw path — see :func:`mask_token` for the masked default.
    """
    return value if reveal else mask_token(value)


def profile_rows(paths: Paths) -> list[tuple[str, str, str, bool]]:
    """Canonical viewer rows for the cached credentials.

    ``(provider, profile, token, is_active)`` — derived through the SAME
    resolvers the runtime uses: :func:`credential_for` for the token and
    :func:`valid_active_profile` for the marker. This is the ROOT invariant
    the token viewer's review cycles converged on: the display must show
    what an install would actually resolve, never its own re-derivation.

    Storage keys are canonicalized to current provider names (a pre-rename
    key stays reachable), and a legacy/current duplicate pair collapses to
    the single entry :func:`credential_for` resolves — the canonical key
    when present — so a rename can never yield two rows, or two active
    markers, for one ``(provider, profile)``. Order: provider alphabetical,
    ``default`` profile first.
    """
    creds = load_credentials(paths)
    groups: dict[tuple[str, str], None] = {}
    for key, profiles in creds.items():
        name = RETIRED_PROVIDER_NAMES.get(key, key)
        for profile in profiles:
            groups.setdefault((name, profile), None)
    rows: list[tuple[str, str, str, bool]] = []
    for name, profile in sorted(
        groups, key=lambda g: (g[0], g[1] != DEFAULT_PROFILE, g[1])
    ):
        token = credential_for(paths, name, profile)
        rows.append(
            (name, profile, token, valid_active_profile(paths, name) == profile)
        )
    return rows


@dataclass(frozen=True)
class ResolvedToken:
    """A resolved token plus where it came from.

    ``source`` is one of :data:`SOURCE_ENV`, :data:`SOURCE_CACHE`,
    :data:`SOURCE_PROMPT`. Only :data:`SOURCE_PROMPT` should be written back to
    ``credentials.json`` — the other two are already durable.
    """

    value: str
    source: str


def load_credentials(paths: Paths) -> dict[str, dict[str, str]]:
    """Read ``credentials.json`` as ``{provider_name: {profile: token}}``.

    **Never raises.** A missing, unreadable, malformed, or oddly-shaped file is
    equivalent to "no cached credentials" — the same never-fails-on-its-way-out
    contract :func:`codehelper.services.models_api.list_models` follows,
    because this is read on the optional discovery path where the absence of a
    token is a normal state, not an error. Non-string values are skipped (not
    fatal): a stray entry must not cost the user the rest of the cache.
    """
    data = read_json_object(paths.credentials_file())
    if data is None:
        return {}
    credentials: dict[str, dict[str, str]] = {}
    for provider_name, value in data.items():
        if not isinstance(provider_name, str) or not provider_name:
            continue
        if isinstance(value, str):
            # Pre-profile cache: preserve the token as the unnamed profile.
            if value:
                credentials[provider_name] = {DEFAULT_PROFILE: value}
            continue
        if not isinstance(value, dict):
            continue
        profiles = {
            profile_name: token
            for profile_name, token in value.items()
            if isinstance(profile_name, str)
            and profile_name
            and isinstance(token, str)
            and token
        }
        if profiles:
            credentials[provider_name] = profiles
    return credentials


def profile_names(paths: Paths, provider_name: str) -> tuple[str, ...]:
    """Return the provider's profiles in stable display order.

    Also includes any profile still stored under a RETIRED provider name
    (see :func:`_storage_names`) — a rename must not orphan profiles a user
    already saved under the old key.
    """
    creds = load_credentials(paths)
    names: set[str] = set()
    for name in _storage_names(provider_name):
        names |= set(creds.get(name, {}))
    return tuple(sorted(names, key=lambda name: (name != DEFAULT_PROFILE, name)))


def valid_active_profile(paths: Paths, provider_name: str) -> str | None:
    """The stored active profile for ``provider_name`` if it still exists.

    Cross-checks ``state.active_selection`` against the live
    ``profile_names``: a saved profile can go stale (renamed via
    :func:`rename_profile` or dropped via
    :func:`invalidate_cached_credential`), and every reader must fall back to
    ``None`` on a miss rather than let a pre-selection install a wrapper under
    a nonexistent profile. Also returns ``None`` when the active selection's
    provider is not ``provider_name`` — the store now holds a single
    (provider, profile) pointer (issue #23 follow-up), not one slot per
    provider, so a profile only "belongs" to the provider it was last
    selected for. Lives here (not in ``state.py``) because it needs
    ``profile_names``, and ``state.py`` must not depend on this module.
    """
    from codehelper.services.state import active_selection

    selection = active_selection(paths)
    if selection is None:
        return None
    active_provider_name, name = selection
    # A pre-existing selection may still name a RETIRED provider (e.g.
    # "ollama" before it was renamed to "ollama-direct") — match it against
    # every name this provider may be stored under (see _storage_names).
    if active_provider_name not in _storage_names(provider_name):
        return None
    return name if name in profile_names(paths, provider_name) else None


def seed_default_profile(paths: Paths, provider_name: str, token: str) -> bool:
    """Cache an existing token as ``default`` when no profile exists yet.

    This is the migration bridge for wrappers installed before the profile
    cache was introduced. It is deliberately conservative: an existing
    provider profile is never overwritten, and a malformed credentials file is
    left untouched rather than replaced with a partial reconstruction.

    Returns ``True`` only when the cache was written successfully. Filesystem
    failures are reported as warnings because the installed wrapper remains a
    valid source of the token and the caller can continue without a cache.

    The existence check and the write happen inside ONE
    :func:`_locked_update` window (issue #17 follow-up) rather than as two
    separate operations — checking ``profile_names`` before acquiring the
    lock let two concurrent migrations both observe "no profile yet," both
    pass the guard, and then both write, with the second silently clobbering
    the first despite this function's own "never overwritten" contract. This
    can't reuse :func:`save_credential` (it acquires its own lock, and
    ``flock`` on a second file descriptor for the same file blocks even
    within one process) — the read-modify-write is inlined here instead.
    """
    if not token:
        return False

    path = paths.credentials_file()
    with _locked_update(paths):
        if profile_names(paths, provider_name):
            return False

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                print(
                    f"warning: could not recover the {provider_name} token "
                    f"profile because {path} is not valid JSON",
                    file=sys.stderr,
                )
                return False
            if not isinstance(data, dict):
                print(
                    f"warning: could not recover the {provider_name} token "
                    f"profile because {path} does not contain a JSON object",
                    file=sys.stderr,
                )
                return False

        try:
            data = load_credentials(paths)
            data.setdefault(provider_name, {})[DEFAULT_PROFILE] = token
            atomic_write(
                paths.credentials_file(),
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                mode=0o600,
            )
        except OSError as e:
            print(
                f"warning: could not cache the existing {provider_name} token "
                f"({e}) — the installed wrapper remains unchanged",
                file=sys.stderr,
            )
            return False
    return True


def credential_for(
    paths: Paths, provider_name: str, profile_name: str = DEFAULT_PROFILE
) -> str:
    """The cached token for a profile, or ``""`` if none — never raises.

    Falls back to a RETIRED provider name (see :func:`_storage_names`, e.g.
    ``"ollama"`` before it was renamed to ``"ollama-direct"``) when the
    current name has no entry — a token cached under the old key before a
    rename must stay reachable under the new one. Read-time fallback only:
    this never rewrites ``credentials.json`` to migrate the key, so the old
    entry is left in place (harmless — merely unreachable under its own
    name going forward) until the next :func:`save_credential` for this
    provider naturally lands it under the current name.
    """
    creds = load_credentials(paths)
    for name in _storage_names(provider_name):
        hit = creds.get(name, {}).get(profile_name, "")
        if hit:
            return hit
    return ""


@contextlib.contextmanager
def _locked_update(paths: Paths):
    """Serialize one read-modify-write cycle against ``credentials.json``.

    Every writer in this module (:func:`save_credential`,
    :func:`rename_profile`, :func:`invalidate_cached_credential`) goes
    through this ONE context manager rather than each independently pairing
    :func:`load_credentials` with ``atomic_write`` — the same "one decision
    point" pattern this project already uses for
    :func:`cache_freshly_typed_token` and ``render.openai_base_url``, so the
    three writers cannot drift on how they serialize.

    The lock itself is :func:`backends._atomic.best_effort_lock` — an
    ``fcntl.flock`` held for the read-modify-write window on a sibling
    ``credentials.json.lock`` file (a separate file, since ``atomic_write``
    replaces the destination inode), opened at ``0o600`` to match the
    credentials file's own permissions. Its acquisition degrades to an
    unlocked yield on ANY ``OSError``, mirroring every other function in
    this module's never-raises contract (:func:`load_credentials`,
    :func:`invalidate_cached_credential`): an unwritable ``config_dir``
    degrades to NO locking rather than an uncaught exception, which is
    strictly no worse than this project's pre-#17 behavior.

    Empirically (see ``tests/test_credentials_concurrency.py``), two
    concurrent ``codehelper`` invocations racing this window reliably lose
    one side's update — not a rare, hard-to-hit interleaving, but one that
    reproduced on ordinary GIL-scheduled threads with no forced delay. The
    lock closes that window: a second holder blocks until the first releases
    it (via the ``with`` block's exit, which always runs, success or
    exception), so the two read-modify-write cycles serialize instead of
    interleaving.
    """
    with best_effort_lock(paths.credentials_file()):
        yield


def save_credential(
    paths: Paths,
    provider_name: str,
    token: str,
    profile_name: str = DEFAULT_PROFILE,
) -> None:
    """Cache ``token`` for a provider profile in ``credentials.json`` (``0o600``).

    Read-modify-write so one provider's credential never clobbers another's —
    serialized against other writers in this module via :func:`_locked_update`
    (issue #17). Written through ``atomic_write`` with ``mode=0o600`` — the
    same crash-safe primitive that writes wrapper scripts, owner-only because
    the file holds secrets in plain text. ``atomic_write`` creates
    ``config_dir`` if absent.
    """
    if not token:
        return  # never write an empty credential (would only delete later reads)
    with _locked_update(paths):
        data = load_credentials(paths)
        data.setdefault(provider_name, {})[profile_name] = token
        atomic_write(
            paths.credentials_file(),
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            mode=0o600,
        )


def rename_profile(
    paths: Paths, provider_name: str, old_name: str, new_name: str
) -> None:
    """Rename one provider profile without exposing or losing its token.

    ``old_name`` is looked up across every name in :func:`_storage_names`
    (``provider_name`` plus its retired predecessor, if any) — not just
    ``provider_name`` itself — because :func:`profile_names` (which is what a
    caller lists ``old_name`` from in the first place) already includes
    profiles still parked under a RETIRED provider name. Without this, a
    profile visible only because of that legacy fallback would silently fail
    to rename: the write always lands under the CURRENT name regardless of
    which name the old entry was found under, so a stale legacy entry gets
    migrated forward by the rename itself rather than left behind.

    Serialized against other writers in this module via :func:`_locked_update`
    (issue #17).
    """
    if not new_name or old_name == new_name:
        return
    with _locked_update(paths):
        data = load_credentials(paths)
        source_name = next(
            (
                name
                for name in _storage_names(provider_name)
                if old_name in data.get(name, {})
            ),
            None,
        )
        if source_name is None:
            return
        profiles = data[source_name]
        target_profiles = data.setdefault(provider_name, {})
        if new_name in target_profiles or (
            source_name != provider_name and new_name in profiles
        ):
            raise CodeHelperError(
                f"profile {new_name!r} already exists for provider {provider_name}"
            )
        target_profiles[new_name] = profiles.pop(old_name)
        if source_name != provider_name and not profiles:
            del data[source_name]
        try:
            atomic_write(
                paths.credentials_file(),
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                mode=0o600,
            )
        except OSError as e:
            print(
                f"warning: could not rename the token profile for {provider_name} "
                f"({e}) — the wrapper is installed, but the old profile name "
                "remains",
                file=sys.stderr,
            )


def invalidate_cached_credential(
    paths: Paths, provider_name: str, profile_name: str = DEFAULT_PROFILE
) -> None:
    """Drop a provider profile's cached credential, if any — never raises.

    A ``FIXED``-policy provider's cache can go stale relative to what is
    actually installed: an env-sourced token (never itself cached — see
    :func:`cache_freshly_typed_token`) can replace an OLDER prompt-cached
    token in the installed wrapper without touching the cache at all, and a
    later, env-free ``add`` would then resolve that now-stale cached value
    and silently reinstall the wrapper back to it — reverting a rotated or
    revoked credential with no confirmation. The caller (``_handle_add``) is
    responsible for deciding WHEN that situation applies (see its own
    comment); this is just the drop, read-modify-write like
    :func:`save_credential` so it never disturbs another provider's entry.
    Dropping (rather than overwriting with the env value) is deliberate: an
    env value is not meant to be cached at all, so simply removing the stale
    entry is enough — the next env-free run falls through to a fresh prompt.

    Also drops the same profile cached under ``provider_name``'s RETIRED
    predecessor name, if any (see :func:`_storage_names`) — :func:`credential_for`
    falls back to that legacy key when the current name has no entry, so
    dropping only the current name here would leave a stale/revoked token
    cached under the old key fully reachable again on the very next
    ``credential_for`` call, defeating the whole point of invalidating it.

    Like :func:`cache_freshly_typed_token`, this runs AFTER the wrapper
    install already succeeded, so a write failure here (full disk,
    unwritable ``config_dir``, permission error) must not surface as an
    uncaught exception — see that function's docstring for the full
    reasoning; both go through the identical best-effort try/except so the
    two cannot drift on it. Serialized against other writers in this module
    via :func:`_locked_update` (issue #17).
    """
    with _locked_update(paths):
        for name in _storage_names(provider_name):
            _invalidate_locked(paths, name, profile_name)


def _invalidate_locked(paths: Paths, provider_name: str, profile_name: str) -> None:
    data = load_credentials(paths)
    profiles = data.get(provider_name)
    if not profiles or profile_name not in profiles:
        return
    del profiles[profile_name]
    if not profiles:
        del data[provider_name]
    try:
        atomic_write(
            paths.credentials_file(),
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            mode=0o600,
        )
    except OSError as e:
        print(
            f"warning: could not invalidate the stale cached token for "
            f"{provider_name} ({e}) — a future run without the environment "
            f"variable set may resolve it again",
            file=sys.stderr,
        )


def cache_freshly_typed_token(
    paths: Paths,
    provider_name: str,
    token: str,
    *,
    profile_name: str = DEFAULT_PROFILE,
    source: str,
    dry_run: bool,
) -> None:
    """Cache ``token`` iff it was just typed at a prompt and this is a real run.

    The ONE decision point behind both callers that persist a resolved token —
    ``cli/parser.py``'s ``_handle_add`` (gated on ``resolve_token``'s
    ``source``) and ``_handle_edit_token`` (which always prompts, so it always
    caches after a successful write). Before this helper existed the two call
    sites each re-implemented "cache unless dry-run", which is exactly the kind
    of two-writers-of-one-fact split this project's ``openai_env_key`` avoids
    for the same reason (see its docstring) — one place decides, so the two
    cannot drift on the guard.

    Never caches ``SOURCE_ENV``/``SOURCE_CACHE`` (already durable) or anything
    under ``--dry-run`` (the project-wide rule that ``--dry-run`` writes
    nothing holds for credentials too).

    Both callers run this AFTER their wrapper install already succeeded (see
    each call site's own comment on why caching is ordered last) — so a write
    failure here (a full disk, an unwritable ``config_dir``, a permission
    error) must not surface as an uncaught exception. ``save_credential``
    goes through ``atomic_write``, which raises a plain ``OSError`` on
    failure — not a :class:`CodeHelperError`, so ``__main__.main()``'s
    ``except CodeHelperError`` would not catch it, and the command would
    crash with a raw traceback for what the user just watched succeed. The
    cache is optional convenience, not the source of truth (the token is
    already baked into the installed script); treat a failure to persist it
    as best-effort and warn instead of crashing.
    """
    if source == SOURCE_PROMPT and not dry_run:
        try:
            save_credential(paths, provider_name, token, profile_name)
        except OSError as e:
            print(
                f"warning: could not cache the token for {provider_name} "
                f"({e}) — the wrapper is installed, but the next add/"
                f"--list-models will prompt again",
                file=sys.stderr,
            )


def resolve_token(
    *,
    env_var: str,
    prompt: str,
    paths: Paths,
    provider_name: str,
    profile_name: str | None = None,
    base_url_policy: str = "fixed",
    environ: Mapping[str, str] = os.environ,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    retries: int = 3,
) -> ResolvedToken:
    """Return a non-empty token from a selected or default profile.

    Args:
        env_var: Environment variable name checked first (headless/scripting
            path — e.g. ``"ZAI_API_KEY"``).
        prompt: The ``getpass`` prompt shown when neither env nor cache has it.
        paths: Resolved paths — locates ``credentials.json``.
        provider_name: Key into ``credentials.json`` (a provider name, not a
            wrapper name).
        base_url_policy: The resolved provider's
            :class:`~codehelper.services.model.BaseUrlPolicy` value (as a
            plain string — this module stays free of the ``model`` import,
            same reasoning as :func:`token_for_discovery`). Defaults to
            ``"fixed"`` so every pre-existing caller keeps the original
            cache-using behaviour unless it opts in by passing the real
            value. See :func:`token_for_discovery`'s docstring for why the
            cache is skipped for anything else.
        environ: Injected environment (default ``os.environ``).
        getpass_fn: Hidden-input source (default ``getpass.getpass``, NEVER
            echoed to the terminal).
        retries: How many empty-input attempts are tolerated before giving up.

    Returns:
        A :class:`ResolvedToken` — a non-empty ``value`` and the ``source`` it
        came from. Never logged or printed by this function.

    Raises:
        CodeHelperError: if none of env/cache/prompt yields a non-empty token.

    An explicitly selected profile's own cached value wins over the
    environment, so a user can deliberately choose a named profile even when
    ``ZAI_API_KEY`` is set. But a brand-new profile with nothing cached yet
    still falls back to the environment before prompting — otherwise a
    first-time ``--profile`` use in a headless/CI run would hang on a
    prompt nobody is there to answer, purely because the profile happened
    to be new. Without an explicit profile, the environment wins outright,
    followed by the default profile. A prompt-resolved value is the only
    one a caller should cache — see :func:`save_credential`.

    ``base_url_policy`` gates only the UNNAMED/default cache: reusing it
    across a changed ``--base-url`` would be a silent, accidental leak,
    since nothing chose it for the current invocation. A NAMED profile is
    the opposite — the user typed ``--profile <name>`` on purpose, the same
    deliberate choice that already lets a profile win over the environment,
    so its cache is trusted for any ``base_url_policy`` (this is what makes
    profiles usable at all for a REQUIRED-policy provider like ``litellm``,
    which the profile feature explicitly supports — see the README).
    """
    if profile_name:
        cached = credential_for(paths, provider_name, profile_name)
        if cached:
            return ResolvedToken(cached, SOURCE_CACHE)
        env_value = environ.get(env_var)
        if env_value:
            return ResolvedToken(env_value, SOURCE_ENV)
    else:
        env_value = environ.get(env_var)
        if env_value:
            return ResolvedToken(env_value, SOURCE_ENV)

        if base_url_policy == "fixed":
            cached = credential_for(paths, provider_name)
            if cached:
                return ResolvedToken(cached, SOURCE_CACHE)

    for _ in range(retries):
        value = getpass_fn(prompt)
        if value:
            return ResolvedToken(value, SOURCE_PROMPT)

    raise CodeHelperError(
        f"no token provided — set {env_var} or enter a non-empty value when prompted"
    )


def env_cache_conflict(
    resolved: ResolvedToken,
    *,
    env_var: str,
    paths: Paths,
    provider_name: str,
    profile_name: str | None = None,
    base_url_policy: str = "fixed",
) -> str | None:
    """A redacted warning line when an env-resolved token silently beat a
    DIFFERENT cached one — ``None`` when there is nothing to warn about.

    :func:`resolve_token` resolves env → cache → prompt, so a token exported
    in the shell (``ZAI_API_KEY`` in ``~/.zshrc``) wins over the cached
    profile even when the export is stale or revoked and the cache holds the
    working key. The value written then 401s in the session with no hint
    which credential was used or where it came from — and the same command
    from a clean environment succeeds, which looks nondeterministic (issue
    #71: an already-open terminal kept a superseded export after the rc file
    was fixed; happened twice in one day). This helper names the
    disagreement, redacting both sides the way a settings diff preview does.

    Fires only where a cache was actually BEATEN — mirroring
    :func:`resolve_token`'s own candidate rules:

    - only a :data:`SOURCE_ENV` resolution can have beaten anything — a
      cache win is the documented precedence, not a trap, and a prompt means
      neither source had a value;
    - an explicit ``profile_name`` caches ABOVE the environment, so an env
      win means that profile was empty — nothing disagrees;
    - a non-``fixed`` ``base_url_policy`` never consults the default cache
      (the cross-address rule in :func:`token_for_discovery`'s docstring),
      so it was never a candidate.

    Pure: returns the line, never prints. The CLI layer owns presentation
    (stderr), which also lets the TUI's chip hot-apply — its silent mode
    captures stdout only — surface the warning. The redaction helper is
    imported lazily from ``claude_settings`` so this module's import-time
    dependency graph (deliberately free of the model/render chain) is
    unchanged.
    """
    if resolved.source != SOURCE_ENV or profile_name:
        return None
    if base_url_policy != "fixed":
        return None
    cached = credential_for(paths, provider_name)
    if not cached or cached == resolved.value:
        return None
    from codehelper.services.claude_settings import redact_credential

    return (
        f"warning: {env_var} from the environment differs from the cached "
        f"profile {DEFAULT_PROFILE!r} ({redact_credential(resolved.value)} vs "
        f"{redact_credential(cached)}) — using the environment value; unset "
        f"{env_var} to use the profile"
    )


def resolve_with_conflict_check(
    *,
    env_var: str,
    prompt: str,
    paths: Paths,
    provider_name: str,
    profile_name: str | None = None,
    base_url_policy: str = "fixed",
    getpass_fn: Callable[[str], str] | None = None,
) -> tuple[ResolvedToken, str | None]:
    """Resolve a token AND its #71 env-vs-cache disagreement in one call.

    Exactly :func:`resolve_token` + :func:`env_cache_conflict` with the same
    arguments — the ONE combined entry point for every caller that resolves a
    token for a write (``add``, ``switch``, ``edit``), so the resolution and
    its conflict check cannot drift the way call sites each re-pairing the
    two could. Returns ``(resolved, conflict)`` where ``conflict`` is the
    redacted warning line from :func:`env_cache_conflict`, or ``None``.

    Pure with respect to presentation: NOTHING is printed here — the caller
    owns the terminal. A CLI caller prints a non-``None`` ``conflict`` to
    stderr, so a TUI silent capture (stdout only) still surfaces it;
    ``edit``'s service layer does the same (see
    :func:`codehelper.services.wrappers.edit_wrapper`).

    ``getpass_fn`` keeps :func:`resolve_token`'s seam: ``None`` leaves
    :func:`resolve_token`'s own default hidden prompt in place (the default
    lives in one signature, not re-stated here), and a caller with no
    interactive terminal passes a raising closure instead (``switch``'s
    chip hot-apply).
    """
    passthrough = {}
    if getpass_fn is not None:
        passthrough["getpass_fn"] = getpass_fn
    resolved = resolve_token(
        env_var=env_var,
        prompt=prompt,
        paths=paths,
        provider_name=provider_name,
        profile_name=profile_name,
        base_url_policy=base_url_policy,
        **passthrough,
    )
    conflict = env_cache_conflict(
        resolved,
        env_var=env_var,
        paths=paths,
        provider_name=provider_name,
        profile_name=profile_name,
        base_url_policy=base_url_policy,
    )
    return resolved, conflict


def token_for_discovery(
    paths: Paths,
    provider,
    *,
    profile_name: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> str:  # type: ignore[no-untyped-def]
    """A token for an OPTIONAL listing request: explicit profile → env → default.

    Mirrors :func:`codehelper.services.models_api.list_models`'s contract: it
    never prompts and never raises. An empty return is legitimate — the listing
    just goes out unauthenticated (as it did before caching existed) and the
    caller falls back to manual model entry. Returns ``""`` for any non-secret
    provider, so a caller need not branch on ``provider.auth`` itself.

    ``environ`` is the same injectable seam :func:`resolve_token` declares
    (default ``os.environ``), so a test can prove "no env var" without
    monkeypatching the real process environment.

    ``provider`` is a :class:`codehelper.services.model.Provider`; typed loose
    to avoid importing the model module here (``secrets`` → ``model`` would be a
    fine edge, but this helper reads only three attributes and staying free of
    the import keeps the dependency arrow one-directional at call sites).

    The UNNAMED/default cache is consulted **only for a**
    ``BaseUrlPolicy.FIXED`` **provider.** Its cache key is the provider
    *name*, not an address — safe as long as a provider has exactly one
    true address (``FIXED``), but a ``REQUIRED``/``OVERRIDABLE`` provider's
    ``base_url`` can be a different, caller-supplied host on every
    invocation (that is the whole point of ``--base-url``). Handing a
    cached secret to whatever host the caller names next would silently
    send it to an address it was never cached for.

    A NAMED profile is different: the caller chose it deliberately for
    *this* invocation (mirroring :func:`resolve_token`'s same distinction),
    so its cache is trusted regardless of ``base_url_policy`` — otherwise
    profiles would be unusable for a REQUIRED-policy provider like
    ``litellm``, which the profile feature explicitly supports. An explicit
    env var is honoured either way: the caller set it for *this*
    invocation, so it carries no cross-invocation ambiguity.
    """
    if provider.auth != "secret":
        return ""
    if profile_name:
        cached = credential_for(paths, provider.name, profile_name)
        if cached:
            return cached
        return environ.get(provider.token_env_var, "")
    env_value = environ.get(provider.token_env_var, "")
    if env_value:
        return env_value
    if provider.base_url_policy != "fixed":
        return ""
    return credential_for(paths, provider.name)
