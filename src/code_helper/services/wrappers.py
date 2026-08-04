"""The wrapper registry + generated-script lifecycle (install/list).

A "wrapper" is a small bash script in ``~/.local/bin`` that exports the
``ANTHROPIC_*`` environment variables Claude Code reads for its model tiers,
then execs ``claude "$@"`` — redirecting Claude Code at a different backend
(a local Ollama daemon, Z.ai, or any other Anthropic-API-compatible endpoint)
without touching Claude Code's own config.

Wrappers are DATA, not hand-written strings — one :class:`WrapperSpec` per
provider in :data:`WRAPPERS`. This mirrors the archived project's
``services/aliases.py`` design (aliases-as-data), applied to generated scripts
instead of ``.zshrc`` fence lines.

The ``auth`` field is what makes this generic across providers with wildly
different trust models:

- ``"literal"`` — the token is a known, non-secret constant (e.g. Ollama's
  local daemon accepts the literal string ``"ollama"``). No prompt, no env
  lookup, mode ``0o755``.
- ``"secret"`` — the token is a real credential (e.g. a Z.ai API key),
  resolved via :func:`code_helper.services.secrets.resolve_token` and never
  logged. Mode ``0o700`` (owner-only — the script carries the secret in plain
  text).

``install_wrapper`` writes ``~/.local/bin/<name>`` unconditionally by name —
whatever is already there (helper-generated or not) gets overwritten. There
is no ownership marker and no "foreign file" guard.
"""

from __future__ import annotations

from dataclasses import dataclass

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.paths import Paths

__all__ = [
    "WrapperSpec",
    "WRAPPERS",
    "render_script",
    "install_wrapper",
    "is_installed",
    "list_wrappers",
    "get_spec",
]

#: Owner-only. A "secret" wrapper carries a real credential in plain text —
#: group/other must have NO bits. A "literal" wrapper (no real secret) is
#: 0o755 — a normal executable.
_MODE_SECRET = 0o700
_MODE_LITERAL = 0o755


@dataclass(frozen=True)
class WrapperSpec:
    """One managed wrapper — a generated ``~/.local/bin/<name>`` script.

    ``auth`` selects how the token is obtained and how sensitive the on-disk
    script is treated (see module docstring). ``auth_value`` is the literal
    token when ``auth == "literal"`` and is ignored otherwise.
    """

    name: str
    base_url: str
    haiku_model: str
    sonnet_model: str
    opus_model: str
    subagent_model: str | None
    auth: str  # "literal" | "secret"
    auth_value: str = ""
    token_env_var: str = ""
    description: str = ""


_DEEPSEEK_MODEL = "deepseek-v4-flash:0731-cloud"

#: The full registry — every wrapper this tool knows how to generate.
WRAPPERS: list[WrapperSpec] = [
    WrapperSpec(
        name="deepseek",
        base_url="http://127.0.0.1:11434",
        haiku_model=_DEEPSEEK_MODEL,
        sonnet_model=_DEEPSEEK_MODEL,
        opus_model=_DEEPSEEK_MODEL,
        subagent_model=_DEEPSEEK_MODEL,
        auth="literal",
        auth_value="ollama",
        description="Claude Code → deepseek-v4-flash via the local Ollama daemon",
    ),
    WrapperSpec(
        name="glm",
        base_url="https://api.z.ai/api/anthropic",
        haiku_model="glm-4.7",
        sonnet_model="glm-5-turbo",
        opus_model="glm-5.2[1m]",
        subagent_model=None,
        auth="secret",
        token_env_var="ZAI_API_KEY",
        description="Claude Code → Z.ai",
    ),
]


def get_spec(name: str) -> WrapperSpec:
    """Return the :class:`WrapperSpec` for ``name``.

    Raises:
        CodeHelperError: if ``name`` is not a known wrapper. Fails BEFORE any
            token is resolved or file is touched — a typo cannot trigger an
            interactive secret prompt for a name that will never be used.
    """
    for spec in WRAPPERS:
        if spec.name == name:
            return spec
    known = ", ".join(w.name for w in WRAPPERS)
    raise CodeHelperError(f"unknown wrapper name: {name} (known: {known})")


def _shell_single_quote(value: str) -> str:
    """POSIX-safe single-quoting: wraps ``value`` so it is always ONE shell word.

    Every value interpolated into the generated script body goes through this
    — the token, the base URL, and every model name — because ANY of them can
    now be adversarial input: the token may come from an untrusted env var,
    and the model name can come straight from the user via ``--model``. A bare
    ``f"'{value}'"`` is not safe: a value containing a single quote (``'``)
    closes the string early and lets the rest be interpreted as shell syntax
    (arbitrary command execution when the generated script later runs). The
    standard escape is to close the quote, emit an escaped literal quote, and
    reopen: ``'`` → ``'"'"'``. Applied to a value with none, it is a no-op
    except for the wrapping quotes.
    """
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_script(
    spec: WrapperSpec, token: str, *, model_override: str | None = None
) -> str:
    """Return the bash wrapper body for ``spec`` (pure, no IO).

    Shape: a subshell that exports the Anthropic endpoint + auth token + the
    tier-model envs (haiku/sonnet/opus, and the subagent model if the spec has
    one), then runs ``claude "$@"``. Every interpolated value is single-quoted
    via :func:`_shell_single_quote` — including model names, which after
    ``model_override`` can be arbitrary user input from ``--model``.

    ``ANTHROPIC_API_KEY=` is always emptied — this mirrors how Ollama's own
    ``ollama launch claude`` integration does it (``cmd/launch/claude.go``): a
    real Anthropic key inherited from the caller's environment would otherwise
    take priority over ``ANTHROPIC_AUTH_TOKEN`` and silently defeat the
    wrapper.

    Args:
        spec: The wrapper's provider description (endpoint + tier models).
        token: The resolved auth token to embed (literal or secret).
        model_override: When given, used for ALL tier models AND the subagent
            model instead of ``spec``'s defaults (the ``--model`` flag).

    Returns:
        The complete script body, including the shebang.
    """
    haiku = model_override or spec.haiku_model
    sonnet = model_override or spec.sonnet_model
    opus = model_override or spec.opus_model
    subagent = (model_override or spec.subagent_model) if spec.subagent_model else None

    lines = [
        "#!/bin/bash",
        "(",
        f"export ANTHROPIC_BASE_URL={_shell_single_quote(spec.base_url)}",
        f"export ANTHROPIC_AUTH_TOKEN={_shell_single_quote(token)}",
        "export ANTHROPIC_API_KEY=",
        f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={_shell_single_quote(haiku)}",
        f"export ANTHROPIC_DEFAULT_SONNET_MODEL={_shell_single_quote(sonnet)}",
        f"export ANTHROPIC_DEFAULT_OPUS_MODEL={_shell_single_quote(opus)}",
    ]
    if subagent is not None:
        lines.append(
            f"export CLAUDE_CODE_SUBAGENT_MODEL={_shell_single_quote(subagent)}"
        )
    lines.append('claude "$@"')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _mode_for(spec: WrapperSpec) -> int:
    return _MODE_SECRET if spec.auth == "secret" else _MODE_LITERAL


def is_installed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` currently exists."""
    get_spec(name)  # validate the name
    return paths.script_for(name).exists()


def install_wrapper(
    paths: Paths,
    name: str,
    *,
    token: str = "",
    model_override: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Generate the ``name`` wrapper script. Return True iff it wrote.

    Overwrites ``~/.local/bin/<name>`` unconditionally — whatever was there
    before (helper-generated or not) is replaced. Idempotent: a
    byte-identical re-install is a no-op (returns False).

    Args:
        paths: Resolved :class:`Paths` (``paths.bin_dir``).
        name: The wrapper name — must be a key in :data:`WRAPPERS`.
        token: The auth token to embed. The caller resolves this BEFORE
            calling — ``spec.auth_value`` for ``auth == "literal"`` specs, or
            :func:`code_helper.services.secrets.resolve_token` for
            ``auth == "secret"`` specs.
        model_override: Forwarded to :func:`render_script`.
        dry_run: When True, print the would-be path and write nothing.

    Returns:
        True iff a write happened (or would happen, under ``dry_run``).

    Raises:
        CodeHelperError: unknown name.
    """
    spec = get_spec(name)
    body = render_script(spec, token, model_override=model_override)
    script = paths.script_for(name)
    if script.exists() and script.read_text(encoding="utf-8") == body:
        return False  # idempotent: identical script already installed
    if dry_run:
        print(f"would write {script}")
        return True
    atomic_write(script, body, mode=_mode_for(spec))
    print(f"wrote {script}")
    return True


def list_wrappers(paths: Paths, *, print_fn=print) -> None:
    """Print the wrapper registry + whether each is present on disk.

    Read-only (no write).
    """
    for spec in WRAPPERS:
        state = "installed" if paths.script_for(spec.name).exists() else "not installed"
        print_fn(f"{spec.name:12} {state:13} {spec.description}")
