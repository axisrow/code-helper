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

#: The ``[model_providers.X]`` table name written into the profile. Stable so
#: the profile the wrapper points at (via ``--profile <alias>``) and the table
#: name inside it cannot drift apart.
_PROVIDER_TABLE = "ollama-launch"


def openai_toml_body(spec: WrapperSpec, catalog_path: str) -> str:
    """The ``~/.codex/<alias>.config.toml`` profile body (pure, no IO).

    Codex ≥ 0.146.0 resolves profiles from per-file config: the basename
    ``<alias>.config.toml`` is addressed as ``--profile <alias>``, and legacy
    ``[profiles.X]`` tables inside ``config.toml`` are rejected — so this is a
    whole file, not a fragment. ``config.toml`` is never read or modified.

    ``base_url`` gets ``/v1/`` appended because the provider's ``base_url`` is
    the Anthropic-protocol root (no version segment), which is what
    ``ANTHROPIC_ENV`` wants; the OpenAI-compatible endpoint this profile
    targets lives under ``/v1``. Derived here rather than stored on the
    provider, so the one field serves both shapes without a conflict.

    ``catalog_path`` is the already-resolved ``<alias>.model.json`` location,
    passed in (rather than computed) because this function is pure and has no
    ``Paths`` — the install path owns the resolution.

    The marker on line 1 is the same comment the bash wrapper carries, which is
    what lets the ownership guard recognise this as ours. The profile name
    inside (``model_provider``) is a fixed registry-style identifier, not user
    input, so it needs no quoting.
    """
    base_url = spec.provider.base_url.rstrip("/") + "/v1/"
    return (
        f"{_marker(spec)}\n"
        f'model = "{_toml_string(spec.model)}"\n'
        f'model_provider = "{_PROVIDER_TABLE}"\n'
        f'model_catalog_json = "{_toml_string(catalog_path)}"\n'
        "\n"
        f"[model_providers.{_PROVIDER_TABLE}]\n"
        f'name = "Ollama"\n'
        f'base_url = "{_toml_string(base_url)}"\n'
        f'wire_api = "{spec.provider.wire_api}"\n'
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

    TOML basic strings are ``"..."`` with ``\\`` and ``"`` escaped; a bare
    model like ``glm-5.2:cloud`` (``:``/``.``) MUST be quoted or Codex rejects
    the file. This is the TOML equivalent of the shell ``_shell_single_quote``
    — a per-target escape for values that originate outside the registry.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


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
