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

from collections.abc import Sequence
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
    "describe_wrapper",
    "describe_all",
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

    Two shapes, discriminated by ``launch_command``:

    - **env-var shape** (``launch_command is None``, the original form): the
      script exports ``ANTHROPIC_*`` env vars and runs ``claude "$@"``.
      ``base_url``/``haiku_model``/``sonnet_model``/``opus_model``/
      ``subagent_model`` describe the endpoint + tier models.
    - **command shape** (``launch_command`` set): the script ``exec``s a
      provider's own launcher (e.g. ``ollama launch claude --model {model}``),
      letting that launcher set up ``ANTHROPIC_*`` itself. ``launch_model`` is
      the default model substituted into the ``{model}`` placeholder; the
      env-var fields are unused. ``--model`` overrides ``launch_model``.

    ``auth`` selects how the token is obtained and how sensitive the on-disk
    script is treated (see module docstring). ``auth_value`` is the literal
    token when ``auth == "literal"`` and is ignored otherwise. The env-var
    fields carry defaults so a command-shape spec only names what it uses.
    """

    name: str
    base_url: str = ""
    haiku_model: str = ""
    sonnet_model: str = ""
    opus_model: str = ""
    subagent_model: str | None = None
    auth: str = "literal"  # "literal" | "secret"
    auth_value: str = ""
    token_env_var: str = ""
    launch_command: str | None = None
    launch_model: str = ""
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
    WrapperSpec(
        name="glm-ollama",
        auth="literal",
        launch_command="ollama launch claude --model {model}",
        launch_model="glm-5.2:cloud",
        description="Claude Code → glm-5.2:cloud via `ollama launch claude`",
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

    Two shapes, discriminated by ``spec.launch_command``:

    - **command shape** (``launch_command`` set): ``exec`` the provider's own
      launcher (e.g. ``ollama launch claude --model {model} -- "$@"``), which
      sets up ``ANTHROPIC_*`` itself. ``launch_model`` is the default model
      substituted into the ``{model}`` placeholder; ``model_override``
      replaces it. The model is the only user-controlled interpolation and is
      single-quoted via :func:`_shell_single_quote` BEFORE substitution
      (``str.replace``, not ``str.format``, so a model can never hijack the
      template); ``launch_command`` itself is a trusted registry constant.
      The ``--`` before ``"$@"`` is required: without it, the launcher parses
      forwarded Claude flags (e.g. ``-p``) as its own and rejects them.
    - **env-var shape** (``launch_command is None``): a subshell that exports
      the Anthropic endpoint + auth token + the tier-model envs
      (haiku/sonnet/opus, and the subagent model if the spec has one), then runs
      ``claude "$@"``. Every interpolated value is single-quoted via
      :func:`_shell_single_quote` — including model names, which after
      ``model_override`` can be arbitrary user input from ``--model``.

    ``ANTHROPIC_API_KEY=` is always emptied in the env-var shape — this mirrors
    how Ollama's own ``ollama launch claude`` integration does it
    (``cmd/launch/claude.go``): a real Anthropic key inherited from the
    caller's environment would otherwise take priority over
    ``ANTHROPIC_AUTH_TOKEN`` and silently defeat the wrapper.

    Args:
        spec: The wrapper's provider description (endpoint + tier models, or a
            launch command).
        token: The resolved auth token to embed (literal or secret). Ignored by
            the command shape.
        model_override: When given, used for ALL tier models AND the subagent
            model (env-var shape) or for the single ``{model}`` placeholder
            (command shape), instead of ``spec``'s defaults (the ``--model``
            flag).

    Returns:
        The complete script body, including the shebang.
    """
    if spec.launch_command is not None:
        model = model_override or spec.launch_model
        quoted_model = _shell_single_quote(model)
        cmd = spec.launch_command.replace("{model}", quoted_model)
        lines = ["#!/bin/bash", f'exec {cmd} -- "$@"']
        return "\n".join(lines) + "\n"

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

    Read-only (no write).
    """
    for spec in WRAPPERS:
        installed = paths.script_for(spec.name).exists()
        print_fn(
            describe_wrapper(
                spec,
                installed=installed,
                installed_word="installed",
                not_installed_word="not installed",
            )
        )
