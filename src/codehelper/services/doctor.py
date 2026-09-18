"""Read-only health report for the live codex / claude wiring.

``doctor`` answers the question a bare ``401 Invalid API key`` hides: which
provider does ``~/.codex/config.toml`` actually name, does its table carry the
``env_key`` a secret provider needs, is that env variable visible to codex
(process env, or ``launchctl`` for GUI-launched apps), does the env token
disagree with the cached profile (the #71 class), is the claude settings'
``env`` block coherent, and does the backend actually answer a discovery
probe. Each answer is one :class:`CheckRow`; the exit code is 1 only when a
row FAILs.

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

from codehelper.errors import CodeHelperError
from codehelper.services import models_api, secrets
from codehelper.services.claude_settings import CREDENTIAL_ENV_KEYS, active_switch_env
from codehelper.services.codex_default import read_default_config
from codehelper.services.model import Provider, get_provider_for_legacy_read
from codehelper.services.paths import Paths
from codehelper.services.render import openai_env_key

__all__ = ["CheckRow", "DoctorReport", "OK", "WARN", "FAIL", "run"]

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
) -> list[CheckRow]:
    """Where can codex actually read ``env_var`` — env, launchctl, nowhere?"""
    value = environ.get(env_var, "")
    if value:
        rows = [
            CheckRow(OK, env_var, f"set in the environment ({secrets.mask_token(value)})")
        ]
        conflict = secrets.env_cache_conflict(
            secrets.ResolvedToken(value, secrets.SOURCE_ENV),
            env_var=env_var,
            paths=paths,
            provider_name=provider_name,
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
    or custom) so the caller knows whether a backend probe applies.
    """
    text, data, parse_error = read_default_config(paths)
    if text is None:
        return (
            [CheckRow(OK, "codex default", "config.toml not found — codex on stock defaults")],
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
            rows.extend(
                _token_source_rows(env_var, paths, name, environ, launchctl_fn)
            )
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
    rows.extend(_token_source_rows(expected, paths, provider.name, environ, launchctl_fn))
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
        return CheckRow(
            OK, "claude settings", "no managed env override — native mode"
        )
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
    return DoctorReport(tuple(rows))
