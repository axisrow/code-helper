"""Turn a resolved wrapper into a bash script body — pure, no IO.

One renderer per :class:`ConfigShape`, looked up in :data:`_RENDERERS`. The
dispatch key is the *same* shape :func:`resolve_shape` produced, which is what
keeps "this combination is allowed" and "this is how it's written" from ever
drifting apart.

Why a registry keyed by shape rather than by ``(agent, provider)``: the pair
table grows as agents × providers, while shapes stay a handful — and a new
provider that reuses an existing shape needs no renderer at all. It is also not
a method on ``Provider``, because that would turn the provider registry from
plain data into behaviour and block "a provider is one record you add".

Quoting rule, unchanged from the original implementation: every value that can
originate outside the registry — the token, the base URL, every model name —
goes through :func:`_shell_single_quote`. Model names in particular arrive
straight from ``--model``. The one deliberate exception is
``agent.binary``, which is interpolated bare so the generated line reads
``ollama launch claude`` rather than ``ollama launch 'claude'``; that is safe
because ``_validate_registries`` constrains it to ``[a-z][a-z0-9_-]*`` at
import time. Structural validation where the input is a constant, quoting where
it is not.
"""

from __future__ import annotations

from collections.abc import Callable

from code_helper.errors import CodeHelperError
from code_helper.services.model import ConfigShape
from code_helper.services.spec import WrapperSpec

__all__ = [
    "render_script",
    "render_legacy_script",
    "openai_toml_body",
    "openai_catalog_body",
    "MARKER_PREFIX",
]

#: Second line of every generated script. Presence of this prefix is how
#: ``is_managed`` tells a file this tool wrote from a file it merely found —
#: see ``wrappers.py``. Kept as a comment so it rides along if the script is
#: copied, and so it costs no runtime behaviour.
MARKER_PREFIX = "# code-helper: managed wrapper"


def _shell_single_quote(value: str) -> str:
    """POSIX-safe single-quoting: wraps ``value`` so it is always ONE shell word.

    A bare ``f"'{value}'"`` is not safe: a value containing a single quote
    closes the string early and lets the rest be read as shell syntax —
    arbitrary command execution when the generated script later runs. The
    standard escape closes the quote, emits an escaped literal quote, and
    reopens: ``'`` → ``'"'"'``.
    """
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _marker(spec: WrapperSpec) -> str:
    return (
        f"{MARKER_PREFIX} (agent={spec.agent.name}, "
        f"provider={spec.provider.name}, shape={spec.shape.value})"
    )


def _render_anthropic_env(spec: WrapperSpec, token: str) -> str:
    """Subshell exporting ``ANTHROPIC_*``, then ``claude "$@"``.

    ``ANTHROPIC_API_KEY=`` is always emptied — mirroring Ollama's own
    ``cmd/launch/claude.go``. A real Anthropic key inherited from the caller's
    environment would otherwise outrank ``ANTHROPIC_AUTH_TOKEN`` and silently
    defeat the wrapper.

    The subagent line is emitted only when the spec carries one: Claude Code
    treats an unset ``CLAUDE_CODE_SUBAGENT_MODEL`` differently from an empty
    one, and the ``glm`` preset deliberately omits it.
    """
    q = _shell_single_quote
    tiers = spec.tier_models
    lines = [
        "#!/bin/bash",
        _marker(spec),
        "(",
        f"export ANTHROPIC_BASE_URL={q(spec.provider.base_url)}",
        f"export ANTHROPIC_AUTH_TOKEN={q(token)}",
        "export ANTHROPIC_API_KEY=",
        f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={q(tiers.haiku)}",
        f"export ANTHROPIC_DEFAULT_SONNET_MODEL={q(tiers.sonnet)}",
        f"export ANTHROPIC_DEFAULT_OPUS_MODEL={q(tiers.opus)}",
    ]
    if spec.subagent_model is not None:
        lines.append(f"export CLAUDE_CODE_SUBAGENT_MODEL={q(spec.subagent_model)}")
    lines.append(f'{spec.agent.binary} "$@"')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _render_ollama_launch(spec: WrapperSpec, token: str) -> str:
    """``exec ollama launch <agent> --model <model> -- "$@"``.

    ``ollama launch`` configures the agent itself (env vars for Claude Code, a
    TOML profile plus ``-m`` for Codex), so the wrapper stays one line and
    embeds no token — hence ``token`` is unused here.

    The ``--`` before ``"$@"`` is load-bearing for BOTH agents: without it the
    launcher's own flag parser consumes forwarded agent flags. Verified — a
    bare ``-p`` yields ``unknown shorthand flag: 'p'``.
    """
    q = _shell_single_quote
    return (
        "#!/bin/bash\n"
        f"{_marker(spec)}\n"
        f'exec ollama launch {spec.agent.binary} --model {q(spec.model)} -- "$@"\n'
    )


def _render_openai_toml(spec: WrapperSpec, token: str) -> str:
    """``exec codex --profile <alias> "$@"`` — the OPENAI_TOML wrapper.

    The model, base URL, and wire protocol live in a sibling TOML profile
    (``openai_toml_body``) plus a model catalog (``openai_catalog_body``),
    both written alongside this script by ``install_wrapper``. The wrapper
    itself is just the dispatch line, so ``token`` is unused: ollama is
    ``literal``-auth and the daemon takes care of authentication itself.

    ``agent.binary`` is interpolated bare (registry constant, constrained at
    import time); ``spec.alias`` is user-chosen, so it is single-quoted — the
    same structural-vs-quoted split as the launch renderer's ``--model``.
    """
    q = _shell_single_quote
    return (
        "#!/bin/bash\n"
        f"{_marker(spec)}\n"
        f'exec {spec.agent.binary} --profile {q(spec.alias)} "$@"\n'
    )


#: ``model_catalog_json`` floors for models the catalog describes. No axis
#: carries a context window, so these are pragmatic defaults a future data
#: source can override in one place.
_DEFAULT_CONTEXT_WINDOW = 128000
_DEFAULT_MAX_OUTPUT = 32768


def _openai_base_url(provider_base_url: str) -> str:
    """Derive the OpenAI-compatible ``base_url`` for the TOML profile.

    A dual-shape provider like ollama serves both the Anthropic protocol and
    the OpenAI ``/v1`` endpoint off the same root, so its ``base_url`` carries
    no version segment and the profile needs ``/v1/`` appended. An OpenAI-only
    provider follows the ``/v1``-suffix convention instead, so appending again
    would double it (``.../v1/v1/``) and every request 404s. Detect the suffix
    and append only when it is absent — one renderer serving both shapes.
    """
    root = provider_base_url.rstrip("/")
    if root.endswith("/v1"):
        return root + "/"
    return root + "/v1/"


def openai_toml_body(spec: WrapperSpec, catalog_path: str) -> str:
    """The ``~/.codex/<alias>.config.toml`` profile body (pure, no IO).

    Codex ≥ 0.146.0 resolves profiles from per-file config: the basename
    ``<alias>.config.toml`` is addressed as ``--profile <alias>``, and legacy
    ``[profiles.X]`` tables inside ``config.toml`` are rejected — so this is a
    whole file, not a fragment. ``config.toml`` is never read or modified.

    Every provider-specific value derives from ``spec.provider`` so the shape
    is a real extension point — wiring a second OpenAI-compatible provider is
    only a ``PROVIDERS`` entry, not a renderer edit:

    - the ``[model_providers.<name>]`` table key and the matching
      ``model_provider`` value are ``spec.provider.name`` (a registry constant
      constrained to a bare-key-safe shape at import time by
      ``model._validate_registries``, so it is interpolated bare like
      ``agent.binary`` — the one unquoted interpolation);
    - the display ``name`` is ``spec.provider.description`` (falling back to
      the provider name);
    - ``base_url`` is :func:`_openai_base_url` (handles both ``/v1``-suffixed
      and bare roots);
    - ``wire_api`` is ``spec.provider.wire_api``.

    ``catalog_path`` is the already-resolved ``<alias>.model.json`` location,
    passed in (rather than computed) because this function is pure and has no
    ``Paths`` — the install path owns the resolution.

    The marker on line 1 is the same comment the bash wrapper carries, which is
    what lets the ownership guard recognise this as ours. Every quoted value
    goes through :func:`_toml_string`; the table key is the only bare
    interpolation, safe by the import-time check.
    """
    table = spec.provider.name
    display_name = spec.provider.description or spec.provider.name
    base_url = _openai_base_url(spec.provider.base_url)
    return (
        f"{_marker(spec)}\n"
        f'model = "{_toml_string(spec.model)}"\n'
        f'model_provider = "{_toml_string(table)}"\n'
        f'model_catalog_json = "{_toml_string(catalog_path)}"\n'
        "\n"
        f"[model_providers.{table}]\n"
        f'name = "{_toml_string(display_name)}"\n'
        f'base_url = "{_toml_string(base_url)}"\n'
        f'wire_api = "{_toml_string(spec.provider.wire_api)}"\n'
    )


def openai_catalog_body(spec: WrapperSpec) -> str:
    """The ``~/.codex/<alias>.model.json`` catalog body (pure, no IO).

    Without a catalog, Codex does not learn the context window of a model it
    does not ship knowledge of (``glm-5.2:cloud`` etc.). This is a minimal
    entry so the model is usable; the real window is unknown to this tool, so
    :data:`_DEFAULT_CONTEXT_WINDOW` is a floor, not a measurement.
    """
    import json

    entry = {
        "id": spec.model,
        "name": spec.model,
        "context_window": _DEFAULT_CONTEXT_WINDOW,
        "max_output_tokens": _DEFAULT_MAX_OUTPUT,
    }
    return json.dumps({"version": 1, "models": [entry]}, indent=2) + "\n"


def _toml_string(value: str) -> str:
    """Quote ``value`` for a TOML basic string.

    TOML basic strings are ``"..."`` with ``\\`` and ``"`` escaped, and they
    forbid literal control characters (U+0000–U+001F except tab) — a ``--model``
    containing a newline would otherwise produce a profile Codex rejects, while
    the shell sibling of the same install handles it fine via
    :func:`_shell_single_quote`. Control characters are emitted as their TOML
    escapes (``\\uXXXX`` is always valid). This is the per-target equivalent of
    the shell single-quote escape, applied to every value that originates
    outside the registry (models, base URLs, ``wire_api``).
    """
    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


#: Shape -> renderer. Each renderer returns the bash WRAPPER body only; the
#: OPENAI_TOML shape additionally writes a TOML profile and catalog via the
#: ``openai_toml_body`` / ``openai_catalog_body`` helpers (orchestrated in
#: ``install_wrapper``, not here). Adding a provider that reuses an existing
#: shape still needs no new renderer — only a shape with no entry surfaces as
#: "not implemented yet".
_RENDERERS: dict[ConfigShape, Callable[[WrapperSpec, str], str]] = {
    ConfigShape.ANTHROPIC_ENV: _render_anthropic_env,
    ConfigShape.OLLAMA_LAUNCH: _render_ollama_launch,
    ConfigShape.OPENAI_TOML: _render_openai_toml,
}


def render_legacy_script(spec: WrapperSpec, token: str = "") -> str:
    """The body the PREVIOUS, markerless release would have written for ``spec``.

    Used by the install guard to recognise the tool's own prior output, which
    predates :data:`MARKER_PREFIX` and would otherwise be classified as a
    third-party file (see ``wrappers._is_ours``). Every renderer here differs
    from its pre-marker ancestor by exactly the marker line — verified against
    the base commit — so this drops that line rather than duplicating the
    bodies, which would let the two copies drift apart silently.

    Not part of the write path: nothing renders a legacy body to install it.
    """
    body = render_script(spec, token)
    lines = body.split("\n")
    return "\n".join(line for line in lines if not line.startswith(MARKER_PREFIX))


def render_script(spec: WrapperSpec, token: str = "") -> str:
    """Return the complete bash body for ``spec`` (pure, no IO).

    Note there is no ``model_override`` parameter: the model is already
    resolved on the spec by ``build_spec``. Keeping override logic out of the
    renderer is what lets each renderer be a straight-line function of its
    input.

    Raises:
        CodeHelperError: the shape is recognised but has no renderer yet.
            Distinct from ``resolve_shape``'s "incompatible" error — this one
            means "possible, not built", and it is what a caller hits for
            ``codex × an OpenAI-only provider`` today.
    """
    renderer = _RENDERERS.get(spec.shape)
    if renderer is None:
        raise CodeHelperError(
            f"configuration shape {spec.shape.value!r} is not implemented yet "
            f"({spec.agent.name} + {spec.provider.name})"
        )
    return renderer(spec, token)
