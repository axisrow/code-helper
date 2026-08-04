"""The wrapper registry + generated-script lifecycle (install/uninstall/list).

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

Ownership is proven by a MARKER comment embedded in the script body, not by
matching the body or the token — so re-running ``add`` after a token rotation
still recognizes (and safely updates) the same script, and a foreign,
hand-written file with the same name is never touched. This mirrors the
archived project's ``services/glm_script.py`` marker discipline exactly.
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
    "uninstall_wrapper",
    "is_installed",
    "list_wrappers",
    "get_spec",
]

#: Stable marker comment embedded in every generated script. Ownership is
#: proven by this marker, NOT by a body/token match — so detection survives a
#: token rotation and a model override, and a foreign
#: ``~/.local/bin/<name>`` (no marker) is never touched.
_MARKER = "# code-helper managed (do not edit)"

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
        The complete script body, including the shebang and marker line.
    """
    haiku = model_override or spec.haiku_model
    sonnet = model_override or spec.sonnet_model
    opus = model_override or spec.opus_model
    subagent = (model_override or spec.subagent_model) if spec.subagent_model else None

    lines = [
        "#!/bin/bash",
        _MARKER,
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


def _is_ours(script) -> bool:
    """True iff ``script`` exists and carries the helper marker (any body).

    Marker-based, so it survives a token rotation or a ``--model`` override
    (the generated body changes, the marker doesn't) and works without any
    external state. A nonexistent or unreadable file is treated as NOT ours
    (fail-safe toward "do not touch").
    """
    if not script.exists():
        return False
    try:
        return _MARKER in script.read_text(encoding="utf-8")
    except OSError:
        return False


def is_installed(paths: Paths, name: str) -> bool:
    """True iff the wrapper named ``name`` is currently installed (by marker).

    A foreign ``~/.local/bin/<name>`` (no marker) is "not installed" — the
    single predicate shared by :func:`list_wrappers` and the install/remove
    ownership checks.
    """
    get_spec(name)  # validate the name
    return _is_ours(paths.script_for(name))


def install_wrapper(
    paths: Paths,
    name: str,
    *,
    token: str = "",
    model_override: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Generate the ``name`` wrapper script. Return True iff it wrote.

    Refuses to clobber a FOREIGN ``~/.local/bin/<name>`` (one without the
    helper marker) — raises :class:`CodeHelperError` rather than destroying a
    user's hand-written script. A helper-owned script (marker present) is
    updated in place (so re-running ``add`` after a token rotation or a
    different ``--model`` refreshes it). Idempotent: a byte-identical
    re-install is a no-op (returns False).

    Order of checks is load-bearing: the caller resolves ``token`` BEFORE
    calling this (so an interactive secret prompt never fires for a name that
    turns out to collide with a foreign file only AFTER the prompt) — this
    function itself checks byte-equality (no-op) before the marker check
    (refuse foreign), and only then writes.

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
        CodeHelperError: unknown name, OR a foreign file is in the way.
    """
    spec = get_spec(name)
    body = render_script(spec, token, model_override=model_override)
    script = paths.script_for(name)
    if script.exists():
        existing = script.read_text(encoding="utf-8")
        if existing == body:
            return False  # idempotent: identical script already installed
        if _MARKER not in existing:
            raise CodeHelperError(
                f"{script} exists and is not code-helper-managed (no marker) — "
                "refusing to overwrite; remove or rename it first"
            )
    if dry_run:
        print(f"would write {script}")
        return True
    atomic_write(script, body, mode=_mode_for(spec))
    print(f"wrote {script}")
    return True


def uninstall_wrapper(paths: Paths, name: str, *, dry_run: bool = False) -> bool:
    """Remove the ``name`` wrapper script. Return True iff it was removed.

    Marker-based: removes the script only when it carries the marker. A
    foreign ``~/.local/bin/<name>`` (no marker) is left intact — this tool
    never deletes files it did not create. Idempotent: returns False when the
    script is absent or not ours (a no-op performs NO write at all — no
    touched mtime, no mode change).
    """
    get_spec(name)  # validate the name even if nothing is installed
    script = paths.script_for(name)
    if not _is_ours(script):
        return False
    if dry_run:
        print(f"would remove {script}")
        return True
    script.unlink()
    print(f"removed {script}")
    return True


def list_wrappers(paths: Paths, *, print_fn=print) -> None:
    """Print the wrapper registry + whether each is present on disk.

    Read-only (no write). A script that exists but lacks the marker is
    reported as ``foreign`` rather than ``not installed`` — distinguishing
    "never created" from "something else is already there".
    """
    for spec in WRAPPERS:
        script = paths.script_for(spec.name)
        if _is_ours(script):
            state = "installed"
        elif script.exists():
            state = "foreign"
        else:
            state = "not installed"
        print_fn(f"{spec.name:12} {state:13} {spec.description}")
