"""Read-only health report for the live codex / claude wiring.

``doctor`` answers the question a bare ``401 Invalid API key`` hides: which
provider does ``~/.codex/config.toml`` actually name, does its table carry the
``env_key`` a secret provider needs, is that env variable visible to codex
(process env, or ``launchctl`` for GUI-launched apps), does the env token
disagree with the cached profile (the #71 class), is the claude settings'
``env`` block coherent, does the backend actually answer a discovery probe,
and — for each installed OPENAI_TOML wrapper — which proxy-env names the
current environment supplies it (issue #76). Each answer is one
:class:`CheckRow`; the exit code is 1 only when a row FAILs.

Read-only and never-raise, the same posture :mod:`models_api` documents: a
missing file, a garbled config, or a dead endpoint is an answer, not an
exception. Tokens appear only through :func:`secrets.mask_token`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from codehelper.errors import CodeHelperError
from codehelper.services import models_api, secrets
from codehelper.services.claude_settings import CREDENTIAL_ENV_KEYS, active_switch_env
from codehelper.services.codex_default import read_default_config
from codehelper.services.model import (
    ConfigShape,
    Provider,
    get_provider_for_legacy_read,
    with_base_url,
)
from codehelper.services.paths import Paths
from codehelper.services.proxy import redact_proxy_url
from codehelper.services.render import openai_env_key
from codehelper.services.wrappers import (
    discover_managed,
    is_managed,
    preset_names,
    spec_from_installed,
)

__all__ = ["CheckRow", "DoctorReport", "OK", "WARN", "FAIL", "proxy_env_row", "run"]

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class CheckRow:
    """One check's outcome. ``hint`` names the fix, empty when self-evident."""

    status: str  #: :data:`OK` / :data:`WARN` / :data:`FAIL`
    name: str
    detail: str
    hint: str = ""


@dataclass(frozen=True)
class DoctorReport:
    rows: tuple[CheckRow, ...]

    @property
    def has_failure(self) -> bool:
        return any(row.status == FAIL for row in self.rows)


#: ``(env_var) -> value`` — the launchd environment a GUI-launched codex sees.
LaunchctlFn = Callable[[str], str]


def _launchctl_getenv(env_var: str) -> str:
    """``launchctl getenv``'s stdout, or ``""`` — never raises.

    Covers the GUI path (the Codex desktop app inherits launchd's env, not a
    shell's): a var that exists only there is invisible to a NEW terminal.
    No ``launchctl`` (non-macOS) or a failed call reads as "not set" — an
    answer, not an error, like every other read in this module.
    """
    if not shutil.which("launchctl"):
        return ""
    try:
        proc = subprocess.run(
            ["launchctl", "getenv", env_var],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def _token_source_rows(
    env_var: str,
    paths: Paths,
    provider_name: str,
    environ: Mapping[str, str],
    launchctl_fn: LaunchctlFn,
    base_url_policy: str = "fixed",
) -> list[CheckRow]:
    """Where can codex actually read ``env_var`` — env, launchctl, nowhere?

    ``base_url_policy`` threads the provider's own gate into
    ``env_cache_conflict``: a non-fixed policy never consults the default
    cache, so the "unset the env var to use the profile" remedy the warning
    names would be FALSE there. Custom tables keep the default — no cache
    entry exists for a name outside the registry, so the check is a no-op
    either way.
    """
    value = environ.get(env_var, "")
    if value:
        rows = [
            CheckRow(
                OK, env_var, f"set in the environment ({secrets.mask_token(value)})"
            )
        ]
        conflict = secrets.env_cache_conflict(
            secrets.ResolvedToken(value, secrets.SOURCE_ENV),
            env_var=env_var,
            paths=paths,
            provider_name=provider_name,
            base_url_policy=base_url_policy,
        )
        if conflict:
            rows.append(CheckRow(WARN, env_var, conflict))
        return rows
    if launchctl_fn(env_var):
        return [
            CheckRow(
                WARN,
                env_var,
                "visible only via launchctl — GUI apps launched after `setenv` "
                "see it, but a NEW terminal does not",
                hint=f"add `export {env_var}=…` to ~/.zshrc (value: `codehelper tokens --reveal`)",
            )
        ]
    return [
        CheckRow(
            FAIL,
            env_var,
            "not set in the environment or launchctl — codex fails with "
            "`Missing environment variable`",
            hint=(
                f"`export {env_var}=…` in ~/.zshrc, or `launchctl setenv "
                f"{env_var} …` for GUI apps (value: `codehelper tokens --reveal`)"
            ),
        )
    ]


def _codex_rows(
    paths: Paths,
    environ: Mapping[str, str],
    launchctl_fn: LaunchctlFn,
) -> tuple[list[CheckRow], Provider | None]:
    """The config.toml side: which provider is live, can it authenticate?

    Returns the rows plus the resolved registry provider (``None`` for native
    or custom) so the caller knows whether a backend probe applies. The
    provider comes back with the table's own base_url substituted in
    (``with_base_url``) — the probe must test the address codex will
    actually hit, not the registry placeholder.
    """
    text, data, parse_error = read_default_config(paths)
    if text is None:
        return (
            [
                CheckRow(
                    OK,
                    "codex default",
                    "config.toml not found — codex on stock defaults",
                )
            ],
            None,
        )
    if data is None:
        return (
            [
                CheckRow(
                    WARN,
                    "codex default",
                    f"config.toml does not parse as TOML: {parse_error}",
                )
            ],
            None,
        )

    name = data.get("model_provider")
    if not isinstance(name, str) or not name:
        return (
            [
                CheckRow(
                    OK,
                    "codex default",
                    "model_provider not set — codex on its native provider",
                )
            ],
            None,
        )

    providers = data.get("model_providers")
    table = providers.get(name) if isinstance(providers, dict) else None
    table = table if isinstance(table, dict) else {}

    try:
        provider = get_provider_for_legacy_read(name)
    except CodeHelperError:
        provider = None

    # The probe must test the endpoint codex will actually hit: for a
    # non-fixed base_url_policy the registry address is a placeholder
    # (REQUIRED = empty) and the real one lives in the table this function
    # just read. with_base_url is THE substitution point and its own policy
    # branch gates what may be replaced — a CodeHelperError (a FIXED
    # provider, or a hand-mangled URL) keeps the registry entry, the same
    # never-raise posture as every other read here.
    if provider is not None:
        configured = table.get("base_url")
        if isinstance(configured, str) and configured:
            try:
                provider = with_base_url(provider, configured)
            except CodeHelperError:
                pass

    if provider is None:
        # A hand-written table this tool does not know. Its env_key (when it
        # declares one) is still checkable — the variable is the variable.
        rows = [
            CheckRow(
                OK,
                "codex default",
                f"custom model_provider {name!r} — outside codehelper's registry",
            )
        ]
        env_var = table.get("env_key")
        if isinstance(env_var, str) and env_var:
            rows.extend(_token_source_rows(env_var, paths, name, environ, launchctl_fn))
        return rows, None

    # openai_env_key IS the secret→token_env_var rule — the same helper both
    # codex-table writers use — so this diagnosis cannot drift from what
    # set-default actually writes.
    expected = openai_env_key(provider)
    if not expected:
        return (
            [
                CheckRow(
                    OK,
                    "codex default",
                    f"{provider.name}: no token needed (auth={provider.auth!r})",
                )
            ],
            provider,
        )
    actual = table.get("env_key")
    if actual is None:
        return (
            [
                CheckRow(
                    FAIL,
                    "codex default",
                    f"{provider.name} requires a token ({expected}), but "
                    f"[model_providers.{name}] has no env_key — codex sends NO "
                    f"key and the backend answers 401 Invalid API key",
                    hint=(
                        f"re-run `codehelper set-default --agent codex "
                        f"--provider {provider.name}` (it writes env_key now), "
                        f'or add env_key = "{expected}" to the table'
                    ),
                )
            ],
            provider,
        )
    if actual != expected:
        return (
            [
                CheckRow(
                    FAIL,
                    "codex default",
                    f"env_key in config.toml is {actual!r}, the registry says "
                    f"{expected!r}",
                    hint=(
                        f"re-run `codehelper set-default --agent codex "
                        f"--provider {provider.name}`"
                    ),
                )
            ],
            provider,
        )

    rows = [
        CheckRow(OK, "codex default", f"{provider.name}: env_key={expected} in place")
    ]
    rows.extend(
        _token_source_rows(
            expected,
            paths,
            provider.name,
            environ,
            launchctl_fn,
            base_url_policy=provider.base_url_policy,
        )
    )
    return rows, provider


def _claude_row(paths: Paths) -> CheckRow:
    """The ~/.claude/settings.json managed block: a base URL without a
    credential is the claude-side twin of the missing-env_key failure.

    Reads through the owner's own surface — :func:`active_switch_env` answers
    "is there a managed override at all" and :data:`CREDENTIAL_ENV_KEYS` is
    the owner's credential vocabulary — so this check cannot drift from what
    ``switch`` actually writes.
    """
    managed = active_switch_env(paths)
    if not managed:
        return CheckRow(OK, "claude settings", "no managed env override — native mode")
    base = managed.get("ANTHROPIC_BASE_URL")
    if base and not any(managed.get(key) for key in CREDENTIAL_ENV_KEYS):
        return CheckRow(
            FAIL,
            "claude settings",
            f"ANTHROPIC_BASE_URL is set ({base}) but none of the credential "
            f"keys are — Claude Code goes out with no key",
            hint="re-run `codehelper switch <provider>`, or restore the token",
        )
    if base:
        return CheckRow(OK, "claude settings", f"switched to {base}")
    return CheckRow(
        OK,
        "claude settings",
        "managed override present, no base URL — native endpoint",
    )


def _probe_row(
    provider: Provider,
    paths: Paths,
    fetch: models_api.Fetcher,
    environ: Mapping[str, str],
) -> CheckRow:
    """Live discovery probe — reuses models_api's never-raises contract."""
    token = secrets.token_for_discovery(paths, provider, environ=environ)
    result = models_api.list_models(provider, fetch=fetch, token=token)
    if result.ok:
        return CheckRow(
            OK,
            f"{provider.name}: probe",
            f"endpoint answered, discovery listed {len(result.models)} models",
        )
    return CheckRow(
        WARN, f"{provider.name}: probe", f"discovery unavailable: {result.error}"
    )


#: The C2 proxy-env universe of the issue #76 contract — every name a wrapper
#: must never write. This report only READS them (a wrapper's proxy comes from
#: the caller's environment; the wrapper itself is env-transparent), and the
#: net is deliberately wider than ``proxy.PROXY_ENV_KEYS`` (``ALL_PROXY``
#: included): the report mirrors the contract's test universe, not one
#: consumer's habit.
PROXY_ENV_NAMES: tuple[str, ...] = (
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "ALL_PROXY",
    "all_proxy",
)

#: The bypass-list names, the other half of C2's universe.
BYPASS_ENV_NAMES: tuple[str, ...] = ("NO_PROXY", "no_proxy")

#: Claude Code's documented precedence (``proxy.PROXY_ENV_KEYS``): the
#: "effective address" below is the first NON-EMPTY of these. One consumer's
#: reading order, worded as such — codex, curl, and urllib each read the same
#: names in their own order, which is why the row says so instead of claiming
#: a universal answer.
_EFFECTIVE_ORDER: tuple[str, ...] = (
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
)

#: Case spellings of the SAME variable. Two different values behind two
#: spellings is the genuinely harmful case (``proxy.py``'s docstring):
#: consumers disagree on which spelling wins, so some traffic can miss the
#: proxy — or the bypass.
_DIVERGENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("https_proxy", "HTTPS_PROXY"),
    ("http_proxy", "HTTP_PROXY"),
    ("ALL_PROXY", "all_proxy"),
    ("NO_PROXY", "no_proxy"),
)


def _endpoint_host(base_url: str) -> str | None:
    """The host a wrapper's endpoint URL names, or ``None`` — never raises.

    A hand-mangled ``base_url`` is an answer here (other rows already judge
    the URL itself); this function only needs a host to check the bypass list
    against, and "unparseable" means the check is skipped.
    """
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return None
    return host or None


def _host_is_bypassed(host: str, bypass_value: str) -> bool:
    """Whether a NO_PROXY-style comma list names ``host`` — exact or suffix.

    Deliberately ONLY those two forms: no CIDR, no port, no ``*`` magic. This
    is a diagnostic wording one predicate, not a network engine — when in
    doubt, the proxy's own log is the ground truth (contract §4, step 4).
    """
    host = host.lower().rstrip(".")
    for entry in bypass_value.split(","):
        entry = entry.strip().lower().lstrip(".").rstrip(".")
        if entry and (host == entry or host.endswith("." + entry)):
            return True
    return False


def proxy_env_row(
    wrapper: str,
    endpoint_host: str | None,
    environ: Mapping[str, str],
) -> CheckRow:
    """One OPENAI_TOML wrapper's proxy picture in THIS environment.

    The issue #76 contract's diagnostic half (its §4, steps 1–2): wrappers
    are env-transparent (C2), so what the wrapper's session sees is exactly
    what this process sees — run doctor from the same shell that launches the
    wrapper. Pure, so the unit tests own it; WARN/INFO at most and NEVER a
    FAIL (a proxy-less environment is legitimate — the wrapper runs direct —
    and doctor's exit code stays "1 only on FAIL"). Hostile environments
    degrade to an answer: a non-string value, or a mapping that raises on
    ``get``, reads as "not set" — the same never-raise posture as every other
    read in this module.

    ``endpoint_host`` is the wrapper's endpoint host (``None`` when the
    profile carries no readable ``base_url``): the bypass check compares it
    against ``NO_PROXY``/``no_proxy`` — exact host or domain suffix only, and
    only when a forward-proxy address is actually configured, since a bypass
    list without a proxy is inert.
    """
    values: dict[str, str] = {}
    for name in (*PROXY_ENV_NAMES, *BYPASS_ENV_NAMES):
        try:
            value = environ.get(name, "")
        except Exception:
            value = ""
        if isinstance(value, str) and value:
            values[name] = value

    if not values:
        return CheckRow(
            OK,
            f"{wrapper}: proxy",
            "no proxy env names set — the wrapper inherits this environment "
            "verbatim and runs direct, which is correct behaviour",
            hint=(
                "to proxy it, export the standard names (e.g. https_proxy) in "
                "the shell or launcher that runs the wrapper, or prefix one "
                "invocation — codehelper proxy configures Claude Code only"
            ),
        )

    host = endpoint_host if isinstance(endpoint_host, str) and endpoint_host else None
    parts = [
        "proxy env set: "
        + ", ".join(n for n in (*PROXY_ENV_NAMES, *BYPASS_ENV_NAMES) if n in values)
    ]
    effective = next(
        (values[name] for name in _EFFECTIVE_ORDER if name in values),
        "",
    )
    if effective:
        # A proxy URL may embed basic-auth credentials — same masking rule as
        # every other value this command prints.
        parts.append(
            f"effective address {redact_proxy_url(effective)} (first non-empty "
            "of https_proxy → HTTPS_PROXY → http_proxy → HTTP_PROXY — each "
            "agent reads these names in its own order)"
        )

    status = OK
    diverged = [
        f"{a} vs {b}"
        for a, b in _DIVERGENCE_PAIRS
        if a in values and b in values and values[a] != values[b]
    ]
    if diverged:
        status = WARN
        parts.append(
            f"{' and '.join(diverged)} case spellings hold different values — "
            "consumers disagree on which spelling wins, so traffic may split"
        )

    # The bypass verdict needs a proxy to bypass: with no forward-proxy
    # address configured, nothing is being proxied and "traffic bypasses the
    # proxy" would be false — so the check itself is gated, not just worded.
    proxy_configured = any(name in values for name in PROXY_ENV_NAMES)
    bypassed = [
        name
        for name in BYPASS_ENV_NAMES
        if proxy_configured
        and name in values
        and host
        and _host_is_bypassed(host, values[name])
    ]
    if bypassed:
        status = WARN
        parts.append(
            f"endpoint {host} is in {', '.join(bypassed)} — API traffic to it "
            "bypasses the proxy"
        )
    elif not proxy_configured:
        # A bypass list without a proxy is the normal aftermath of
        # `proxy off` (proxy.py deliberately leaves NO_PROXY standing), so
        # this is an answer, not a warning: the list is inert.
        parts.append("no proxy address set — the NO_PROXY list is inert")
    elif not host:
        parts.append("wrapper endpoint unknown — bypass check skipped")

    hint = ""
    if bypassed:
        hint = f"if the bypass is unintended, remove {host} from the NO_PROXY list"
    elif diverged:
        hint = "align the case spellings — export both with the same value"
    return CheckRow(status, f"{wrapper}: proxy", "; ".join(parts), hint)


def _proxy_rows(paths: Paths, environ: Mapping[str, str]) -> list[CheckRow]:
    """One row per installed OPENAI_TOML wrapper: its proxy picture right here.

    The issue #76 contract scopes the report to OPENAI_TOML wrappers — the
    shape whose endpoint lives in a per-alias profile, and the one the issue's
    failure mode ("valid base_url, unintended network path") is about.
    ``discover_managed`` deliberately lists only ad-hoc wrappers, so installed
    PRESETS on the shape are unioned in: a future codex preset must not lose
    its row. An unreconstructable wrapper (corrupt profile) reads as no row —
    its health is the other rows' job; this report only describes proxies.
    """
    aliases = set(discover_managed(paths))
    aliases.update(name for name in preset_names() if is_managed(paths, name))
    rows: list[CheckRow] = []
    for alias in sorted(aliases):
        spec = spec_from_installed(paths, alias)
        if spec is None or spec.shape is not ConfigShape.OPENAI_TOML:
            continue
        rows.append(
            proxy_env_row(alias, _endpoint_host(spec.provider.base_url), environ)
        )
    return rows


def run(
    paths: Paths,
    *,
    environ: Mapping[str, str] = os.environ,
    launchctl_fn: LaunchctlFn = _launchctl_getenv,
    fetch: models_api.Fetcher = models_api.urlopen_fetch,
) -> DoctorReport:
    """Run every check. Never raises; writes nothing; masks every token.

    ``fetch`` defaults to models_api's real transport (not a ``None``
    sentinel) so the probe has exactly one call site, the way every other
    ``list_models`` caller just omits the seam.
    """
    rows, provider = _codex_rows(paths, environ, launchctl_fn)
    rows.append(_claude_row(paths))
    if provider is not None:
        rows.append(_probe_row(provider, paths, fetch, environ))
    rows.extend(_proxy_rows(paths, environ))
    return DoctorReport(tuple(rows))
