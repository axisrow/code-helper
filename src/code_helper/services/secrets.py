"""Resolving a wrapper's secret token.

The resolution order is **env var → cached credential → hidden prompt**, and a
token resolved from a prompt is cached back so the next ``add``/``--list-models``
does not ask again. The cache is the ONLY persistent copy of a token this project
keeps outside a generated wrapper script — and it is a CACHE, not a session:

- A wrapper carries its token baked into the script itself (``0o700``). Deleting
  ``credentials.json`` does NOT break an installed wrapper; it only means the
  next install/discovery prompts again.
- There is no "logged in" state, no provider round-trip to validate a token, no
  command that "connects" code-helper to a service. ``code-helper`` configures
  agents and aliases; it does not authenticate with anything.
- The file is written ``0o600`` (owner-only) and read by no one but this module.

The flat shape is ``{provider_name: token}`` — keys are provider names, never
wrapper names, because one provider backs many wrappers and the credential
belongs to the provider.

Provider-specific key FORMAT validation (the archived project's strict
``<32-hex>.<16-alnum>`` Z.ai regex) is deliberately NOT reproduced here — this
project has no reason to know what a valid Z.ai (or any other provider's) key
looks like. Instead, :func:`code_helper.services.wrappers.render_script`
neutralizes shell metacharacters in EVERY interpolated value via POSIX
single-quoting, so an arbitrary (even adversarial) token string can never
break out of the generated script — see the injection tests in
``tests/test_wrappers.py``.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.paths import Paths

__all__ = [
    "ResolvedToken",
    "resolve_token",
    "load_credentials",
    "credential_for",
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


@dataclass(frozen=True)
class ResolvedToken:
    """A resolved token plus where it came from.

    ``source`` is one of :data:`SOURCE_ENV`, :data:`SOURCE_CACHE`,
    :data:`SOURCE_PROMPT`. Only :data:`SOURCE_PROMPT` should be written back to
    ``credentials.json`` — the other two are already durable.
    """

    value: str
    source: str


def load_credentials(paths: Paths) -> dict[str, str]:
    """Read ``credentials.json`` as ``{provider_name: token}``.

    **Never raises.** A missing, unreadable, malformed, or oddly-shaped file is
    equivalent to "no cached credentials" — the same never-fails-on-its-way-out
    contract :func:`code_helper.services.models_api.list_models` follows,
    because this is read on the optional discovery path where the absence of a
    token is a normal state, not an error. Non-string values are skipped (not
    fatal): a stray entry must not cost the user the rest of the cache.
    """
    path = paths.credentials_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(name): str(value)
        for name, value in data.items()
        if isinstance(name, str)
        and name  # a provider name is never empty
        and isinstance(value, str)
        and value
    }


def credential_for(paths: Paths, provider_name: str) -> str:
    """The cached token for ``provider_name``, or ``""`` if none — never raises."""
    return load_credentials(paths).get(provider_name, "")


def save_credential(paths: Paths, provider_name: str, token: str) -> None:
    """Cache ``token`` for ``provider_name`` in ``credentials.json`` (``0o600``).

    Read-modify-write so one provider's credential never clobbers another's.
    Written through ``atomic_write`` with ``mode=0o600`` — the same crash-safe
    primitive that writes wrapper scripts, owner-only because the file holds
    secrets in plain text. ``atomic_write`` creates ``config_dir`` if absent.
    """
    if not token:
        return  # never write an empty credential (would only delete later reads)
    data = load_credentials(paths)
    data[provider_name] = token
    atomic_write(
        paths.credentials_file(),
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )


def invalidate_cached_credential(paths: Paths, provider_name: str) -> None:
    """Drop ``provider_name``'s cached credential, if any — never raises.

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

    Like :func:`cache_freshly_typed_token`, this runs AFTER the wrapper
    install already succeeded, so a write failure here (full disk,
    unwritable ``config_dir``, permission error) must not surface as an
    uncaught exception — see that function's docstring for the full
    reasoning; both go through the identical best-effort try/except so the
    two cannot drift on it.
    """
    data = load_credentials(paths)
    if provider_name not in data:
        return
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
    paths: Paths, provider_name: str, token: str, *, source: str, dry_run: bool
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
            save_credential(paths, provider_name, token)
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
    base_url_policy: str = "fixed",
    environ: Mapping[str, str] = os.environ,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    retries: int = 3,
) -> ResolvedToken:
    """Return a non-empty token: env var → cached credential → hidden prompt.

    Args:
        env_var: Environment variable name checked first (headless/scripting
            path — e.g. ``"ZAI_API_KEY"``).
        prompt: The ``getpass`` prompt shown when neither env nor cache has it.
        paths: Resolved paths — locates ``credentials.json``.
        provider_name: Key into ``credentials.json`` (a provider name, not a
            wrapper name).
        base_url_policy: The resolved provider's
            :class:`~code_helper.services.model.BaseUrlPolicy` value (as a
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

    The env var wins first on purpose: a headless/CI run must be able to
    override a stale cached value. A prompt-resolved value is the only one a
    caller should cache — see :func:`save_credential`.
    """
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


def token_for_discovery(
    paths: Paths, provider, *, environ: Mapping[str, str] = os.environ
) -> str:  # type: ignore[no-untyped-def]
    """A token for an OPTIONAL listing request: env var → cache → ``""``.

    Mirrors :func:`code_helper.services.models_api.list_models`'s contract: it
    never prompts and never raises. An empty return is legitimate — the listing
    just goes out unauthenticated (as it did before caching existed) and the
    caller falls back to manual model entry. Returns ``""`` for any non-secret
    provider, so a caller need not branch on ``provider.auth`` itself.

    ``environ`` is the same injectable seam :func:`resolve_token` declares
    (default ``os.environ``), so a test can prove "no env var" without
    monkeypatching the real process environment.

    ``provider`` is a :class:`code_helper.services.model.Provider`; typed loose
    to avoid importing the model module here (``secrets`` → ``model`` would be a
    fine edge, but this helper reads only three attributes and staying free of
    the import keeps the dependency arrow one-directional at call sites).

    The cache is consulted **only for a** ``BaseUrlPolicy.FIXED`` **provider.**
    The cache key is the provider *name*, not an address — safe as long as a
    provider has exactly one true address (``FIXED``), but a ``REQUIRED``/
    ``OVERRIDABLE`` provider's ``base_url`` can be a different, caller-supplied
    host on every invocation (that is the whole point of ``--base-url``). Handing
    a cached secret to whatever host the caller names next would silently send
    it to an address it was never cached for. An explicit env var is still
    honoured either way: the caller set it for *this* invocation, so it carries
    no such cross-invocation ambiguity.
    """
    if provider.auth != "secret":
        return ""
    env_value = environ.get(provider.token_env_var, "")
    if env_value:
        return env_value
    if provider.base_url_policy != "fixed":
        return ""
    return credential_for(paths, provider.name)
