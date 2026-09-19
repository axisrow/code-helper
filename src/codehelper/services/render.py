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

import json
from collections.abc import Callable, Iterable
from urllib.parse import quote, urlsplit, urlunsplit

from codehelper.errors import CodeHelperError
from codehelper.services.model import ConfigShape, Provider
from codehelper.services.spec import WrapperSpec

__all__ = [
    "render_script",
    "render_legacy_script",
    "openai_toml_body",
    "openai_base_url",
    "anthropic_base_url",
    "openai_env_key",
    "openai_provider_table",
    "toml_string",
    "MARKER_PREFIX",
    "MODEL_CONTEXT_WINDOWS",
    "uniform_context_window",
]

#: Second line of every generated script. Presence of this prefix is how
#: ``is_managed`` tells a file this tool wrote from a file it merely found —
#: see ``wrappers.py``. Kept as a comment so it rides along if the script is
#: copied, and so it costs no runtime behaviour.
MARKER_PREFIX = "# codehelper: managed wrapper"


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
    """The ownership line, and the record ``spec_from_installed`` rebuilds from.

    ``auth`` is written ONLY for a secret wrapper (issue #81): it is the one
    axis reconstruction cannot re-derive — a marker naming an OVERRIDABLE
    provider (ollama-direct) reads back the registry's literal default, and
    every consumer of the reconstruction (``switch --from-wrapper``, the TUI
    chipset, ``edit-token``) then applies or edits the literal credential
    instead of the account token embedded in the wrapper's own body. For a
    literal wrapper the field is omitted, so the marker stays byte-identical
    to the pre-#81 form and every already-installed literal wrapper keeps
    matching ``_decide``'s SKIP path; for a registry-secret provider (zai)
    the field is redundant with the registry — and honoured as a no-op — but
    writing it keeps the marker self-sufficient should that registry default
    ever move.
    """
    auth = ", auth=secret" if spec.auth == "secret" else ""
    profile = (
        f", profile={quote(spec.profile_name, safe='._-')}" if spec.profile_name else ""
    )
    # Recorded LAST (after profile) so every existing marker's bytes — and
    # every prefix-matching consumer — are untouched; the generic fields
    # regex reads it back. Written ONLY for an explicit answer (issue #83):
    # a catalog-derived window is re-derivable, and recording it would churn
    # every already-installed known-model wrapper's marker (the #82 rule).
    ctx = f", ctx={spec.context_window}" if spec.context_window is not None else ""
    # effort continues the same append-only rule (issue #100): conditional,
    # after ctx, so a pre-#100 marker's bytes never move. Only ever set for
    # the OPENAI_TOML shape — build_spec refuses it everywhere else.
    effort = f", effort={spec.effort}" if spec.effort else ""
    return (
        f"{MARKER_PREFIX} (agent={spec.agent.name}, "
        f"provider={spec.provider.name}, shape={spec.shape.value}{auth}{profile}{ctx}{effort})"
    )


def _settings_flag(env: dict[str, str]) -> str:
    """``--settings '<json>'`` carrying ``env`` at the command-line level.

    Why the exports alone are not enough: since Claude Code 2.0.1 every
    ``env`` entry in ``settings.json`` is written into the process
    environment at startup, REPLACING the value inherited from the shell —
    an empty string included (treated as unset for provider selection). So
    ``~/.claude/settings.json`` holding ``ANTHROPIC_BASE_URL: ""`` (what
    ``switch native`` leaves behind) silently defeats every wrapper export
    and launches native. ``--settings`` is one level ABOVE the user file in
    Claude Code's precedence, merges ``env`` per key, and lasts a single
    session — exactly the override a per-invocation wrapper needs. The
    exports stay in the script for subprocesses and for
    ``wrappers.spec_from_installed`` recovery; ``--settings`` is what
    guarantees they win.
    """
    payload = json.dumps({"env": env}, separators=(",", ":"))
    return f"--settings {_shell_single_quote(payload)}"


#: Real context windows of third-party models this tool knows, in tokens.
#: Claude Code cannot resolve a non-``claude-`` model ID, so it assumes its
#: 200k fallback and proactively auto-compacts there — for a model whose real
#: window is 1M that abandons 80% of the context every session. Declaring the
#: window via ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` makes compaction continue at
#: the declared window instead (Claude Code docs, "Correct the window for a
#: gateway or custom model ID": for an ID that is not ``claude-*``, carries no
#: ``[1m]``, and resolves to no Claude model, the variable applies directly).
#: Data, not code: a model missing here simply gets NO declaration — never
#: guess a window, an oversized claim overflows the real one mid-session.
#: Deliberately absent: ``glm-5-turbo`` / ``glm-4.7`` (200k real — the
#: fallback assumption already matches them, a declaration would add nothing
#: but a lie's risk).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "glm-5.3": 1_000_000,
    "glm-5.2": 1_000_000,
    "glm-5.2:cloud": 1_000_000,
    "deepseek-v4-flash:0731-cloud": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-flash-vision-exp": 1_000_000,
    # Kept for ALREADY-INSTALLED wrappers (issue #112 review): gemini-3.7-flash
    # was the removed gemini-litellm preset's model — the preset and the
    # gemini provider are dead history, but the MODEL is not, and its window
    # is still 1M (ai.google.dev model docs). Its wrappers' markers carry no
    # `ctx=` (the #82 rule: catalog-known models record nothing), so dropping
    # this entry would make the next re-render (edit-token/edit) silently
    # lose the 1M CLAUDE_CODE_MAX_CONTEXT_TOKENS declaration those wrappers
    # still carry. Catalog entries are per-model data, not provider story.
    "gemini-3.7-flash": 1_000_000,
    # B.AI's documented window for gpt-6-astra (docs.b.ai; 1,050,000 input /
    # 128,000 output) — the catalog's first non-1M value: windows are
    # per-model data, never a shared assumption.
    "gpt-6-astra": 1_050_000,
}


def uniform_context_window(models: Iterable[str]) -> int | None:
    """The one context window shared by EVERY model in ``models``, or None.

    None unless every name resolves in :data:`MODEL_CONTEXT_WINDOWS` AND all
    resolve to the same value: ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` declares
    ONE window for the whole session, so tiers that genuinely differ (or any
    unknown model) must yield no declaration rather than a guess. The
    subagent model participates for the same reason — it shares the variable.

    Public (not ``_``-prefixed) for the same reason ``anthropic_base_url``
    is: ``claude_settings.resolve_switch_patch`` derives the same value for
    the switch mechanism, so wrapper and switch cannot disagree on whether a
    model's window is declared.
    """
    windows = {MODEL_CONTEXT_WINDOWS.get(model) for model in models}
    if len(windows) == 1 and None not in windows:
        (window,) = windows
        return window
    return None


def _declared_window(spec: WrapperSpec) -> int | None:
    """The window this spec declares, honouring the explicit axis (issue #83).

    The covered model set comes from ``spec.window_models`` — the same
    source the interactive question asks about, so ask and declare can
    never drift (three review rounds, PR #84).

    An explicit ``spec.context_window`` wins over everything: ``> 0`` is the
    window the user (or the recorded answer behind a marker's ``ctx=``)
    decided, ``0`` is an explicit "no declaration" that suppresses even a
    catalog hit — the user was asked and answered, so the mixed-tier
    uniformity rule does not get a vote. Without an explicit value the
    catalog derivation stands, unchanged.
    """
    if spec.context_window is not None:
        return spec.context_window or None
    return uniform_context_window(spec.window_models)


def _render_anthropic_env(spec: WrapperSpec, token: str) -> str:
    """Subshell exporting ``ANTHROPIC_*``, then ``claude --settings … "$@"``.

    ``ANTHROPIC_API_KEY=`` is always emptied — mirroring Ollama's own
    ``cmd/launch/claude.go``. A real Anthropic key inherited from the caller's
    environment would otherwise outrank ``ANTHROPIC_AUTH_TOKEN`` and silently
    defeat the wrapper.

    The subagent line is emitted only when the spec carries one: Claude Code
    treats an unset ``CLAUDE_CODE_SUBAGENT_MODEL`` differently from an empty
    one, and the ``glm`` preset deliberately omits it.

    ``ANTHROPIC_BASE_URL`` goes through :func:`anthropic_base_url`, not raw —
    see that function for why.

    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` rides along when every tier (plus the
    subagent, when set) resolves to one known window via
    :func:`uniform_context_window` — see :data:`MODEL_CONTEXT_WINDOWS` for
    why it is conditional and never a guess.

    One dict feeds BOTH the exports and the ``--settings`` payload (see
    :func:`_settings_flag`), so the two can never drift; a value the JSON
    carries but an export lost (or vice versa) would resurface as the very
    settings.json-overrides-the-wrapper bug this flag exists to fix.
    """
    q = _shell_single_quote
    tiers = spec.tier_models
    if tiers is None:
        # Only the ANTHROPIC_ENV renderer dispatches here, and build_spec
        # materializes uniform tiers for that shape (spec.build_spec) — the
        # same invariant resolve_switch_patch guards explicitly. Spelled out
        # so a future shape/registry change fails as a domain error here
        # instead of an AttributeError mid-render.
        raise CodeHelperError(
            f"wrapper {spec.name!r} has no tier models — the anthropic-env "
            "shape always carries them; build_spec should have materialized "
            "uniform tiers"
        )
    env = {
        "ANTHROPIC_BASE_URL": anthropic_base_url(spec.provider.base_url),
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": tiers.haiku,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": tiers.sonnet,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": tiers.opus,
    }
    if spec.subagent_model is not None:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = spec.subagent_model
    if (window := _declared_window(spec)) is not None:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
    lines = ["#!/bin/bash", _marker(spec), "("]
    lines += [f"export {key}={q(value)}" for key, value in env.items()]
    lines.append(f'{spec.agent.binary} {_settings_flag(env)} "$@"')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _render_ollama_launch(spec: WrapperSpec, token: str) -> str:
    """``exec ollama launch <agent> --model <model> -- [--settings …] "$@"``.

    ``ollama launch`` configures the agent itself (env vars for Claude Code, a
    TOML profile plus ``-m`` for Codex), so the wrapper embeds no token of its
    own — ``token``, when non-empty, is only echoed into the ``--settings``
    payload below.

    The ``--`` before the forwarded flags is load-bearing for BOTH agents:
    without it the launcher's own flag parser consumes them. Verified — a
    bare ``-p`` yields ``unknown shorthand flag: 'p'``.

    For an agent that speaks ``ANTHROPIC_*`` env (claude — i.e. any agent
    declaring the ANTHROPIC_ENV shape, never codex or the launch-only CLI
    agents), the wrapper also forwards :func:`_settings_flag`: the env vars
    ``ollama launch`` injects into the agent's process are, from Claude
    Code's side, indistinguishable from shell exports — and get REPLACED by
    ``~/.claude/settings.json``'s ``env`` block the same way (an
    ``ANTHROPIC_BASE_URL: ""`` left by ``switch native`` redirects the
    session to native). The payload mirrors what ``cmd/launch/claude.go``
    injects for a local daemon: the provider's base URL, the literal token
    (or the resolved secret, for an ``--auth secret`` install), and the
    model in every tier slot. For every other agent the line is unchanged —
    ``--settings`` is a Claude Code flag and would be rejected by their
    argument parsers.
    """
    q = _shell_single_quote
    launch = f"exec ollama launch {spec.agent.binary} --model {q(spec.model)}"
    # The separator comes FIRST and unconditionally: everything after it is
    # forwarded to the agent, never parsed by `ollama launch` itself.
    launch += " --"
    if ConfigShape.ANTHROPIC_ENV in spec.agent.shapes:
        env = {
            "ANTHROPIC_BASE_URL": anthropic_base_url(spec.provider.base_url),
            "ANTHROPIC_AUTH_TOKEN": token or spec.provider.auth_value,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": spec.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": spec.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": spec.model,
            "CLAUDE_CODE_SUBAGENT_MODEL": spec.model,
        }
        if (window := _declared_window(spec)) is not None:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
        launch += f" {_settings_flag(env)}"
    launch += ' "$@"'
    return f"#!/bin/bash\n{_marker(spec)}\n{launch}\n"


def _render_openai_toml(spec: WrapperSpec, token: str) -> str:
    """``exec codex --profile <alias> "$@"`` — the OPENAI_TOML wrapper.

    The model, base URL, and wire protocol live in a sibling TOML profile
    (``openai_toml_body``) plus a model catalog (``openai_catalog_body``),
    both written alongside this script by ``install_wrapper``.

    For a non-secret provider (``ollama``, ``literal``-auth) ``token`` is
    unused and the wrapper is just the dispatch line — the daemon
    authenticates itself. For a SECRET provider (:func:`openai_env_key`
    non-empty), one ``export`` line precedes ``exec``: Codex has no
    ``OPENAI_BASE_URL``-style env config, but a TOML profile CAN name an
    ``env_key`` to read its token from, and this is the writer of that
    variable — the matching ``env_key`` line lives in ``openai_toml_body``.
    ``export`` + ``exec`` rather than ``exec env {key}=... codex ...``: the
    latter would replace the process image with ``env`` (breaking ``$0``) and
    add a dependency on ``env`` being on ``PATH`` for no benefit — no
    subshell is needed either, since ``exec`` is the last line and nothing
    after it could see the leaked variable.

    Conditional on purpose: for a non-secret provider this must render
    BYTE-IDENTICAL to before the env_key/export mechanism existed, or every
    already-installed ``codex × ollama`` wrapper stops matching the ownership
    guard's byte comparison and gets rewritten on the next ``add``.

    ``agent.binary`` is interpolated bare (registry constant, constrained at
    import time); ``spec.alias`` is user-chosen, so it is single-quoted — the
    same structural-vs-quoted split as the launch renderer's ``--model``. The
    exported token goes through the same single-quoting.

    Note the exported variable is visible to ``codex`` and every child
    process it spawns (including MCP servers it launches) — that is the
    mechanism by which ``codex`` receives the credential, not an oversight.
    """
    q = _shell_single_quote
    lines = ["#!/bin/bash", _marker(spec)]
    env_key = openai_env_key(spec.provider)
    if env_key:
        lines.append(f"export {env_key}={q(token)}")
    lines.append(f'exec {spec.agent.binary} --profile {q(spec.alias)} "$@"')
    return "\n".join(lines) + "\n"


def _render_agent_native(spec: WrapperSpec, token: str) -> str:
    """``exec <agent> --model <model> "$@"`` — the AGENT_NATIVE wrapper.

    The honest minimal form: the agent's own native auth applies (for ``agy``,
    Google OAuth against its ``~/.gemini`` config home), so there is NO env
    block, NO token, NO config file — this tool writes nothing the agent would
    have to be pointed with. ``token`` is therefore unused by contract: this
    renderer is only ever dispatched for ``auth="none"`` providers, where
    ``build_spec``/``_add_resolve_token`` have already skipped resolution —
    a fake env var would be a lie, and an assertion-here would be dead code
    for a state the shape system cannot produce.

    ``spec.model`` goes through :func:`_shell_single_quote` like every value
    that can originate outside the registry; ``agent.binary`` is interpolated
    bare (registry constant, constrained at import time — same rule as the
    other renderers). Only ``--model`` is threaded: it is the one documented
    flag the wrapper exists to pin (agy's model ids encode the effort tier,
    so a separate ``--effort`` would be redundant). Everything the user adds
    rides ``"$@"`` untouched.
    """
    del token  # native auth: there is no credential to place anywhere
    return (
        f"#!/bin/bash\n"
        f"{_marker(spec)}\n"
        f'exec {spec.agent.binary} --model {_shell_single_quote(spec.model)} "$@"\n'
    )


def openai_base_url(provider_base_url: str) -> str:
    """Derive the OpenAI-compatible ``base_url`` for the TOML profile.

    A dual-shape provider like ollama serves both the Anthropic protocol and
    the OpenAI ``/v1`` endpoint off the same root, so its ``base_url`` carries
    no version segment and the profile needs ``/v1/`` appended. An OpenAI-only
    provider follows the ``/v1``-suffix convention instead, so appending again
    would double it (``.../v1/v1/``) and every request 404s. Detect the suffix
    and append only when it is absent — one renderer serving both shapes.

    Public (not ``_``-prefixed): ``services/codex_default.py`` reuses this
    exact derivation for ``set-default``, so the two OPENAI_TOML writers (a
    per-alias profile here, Codex's own default config there) cannot disagree
    on how a provider's ``base_url`` becomes a ``/v1/`` endpoint.
    """
    root = provider_base_url.rstrip("/")
    if root.endswith("/v1"):
        return root + "/"
    return root + "/v1/"


def anthropic_base_url(provider_base_url: str) -> str:
    """Derive the ``ANTHROPIC_BASE_URL`` for the anthropic-env shape.

    :func:`openai_base_url`'s mirror image, and for the same reason a dual-shape
    provider like litellm needs both: Claude Code itself appends
    ``/v1/messages`` to ``ANTHROPIC_BASE_URL``, so a stored value that already
    carries a ``/v1`` suffix (typed by the user, or round-tripped through
    ``wrappers._recover_base_url`` off an already-``/v1/``-suffixed OPENAI_TOML
    profile) would double it to ``/v1/v1/messages`` and every request 404s.
    Strip the suffix; leave everything else — including a bare root, and a
    ``/v1`` that is not a whole trailing path segment — untouched. Idempotent,
    like its sibling, so a recover-then-reinstall round trip is a no-op.

    Public (not ``_``-prefixed) for the same reason ``openai_base_url`` is:
    both the renderer and any future reader of ``provider.base_url`` for this
    shape must derive the endpoint the same way.

    Uses :func:`urllib.parse.urlsplit` rather than a bare string suffix check
    so a host that merely *ends in* ``v1`` (``http://v1``) or carries ``/v1``
    as a non-trailing path segment (``http://h/v1/proxy``) is never touched —
    only a trailing whole ``/v1`` *path* segment qualifies.
    """
    root = provider_base_url.rstrip("/")
    parts = urlsplit(root)
    path = parts.path
    if path.endswith("/v1"):
        stripped = path[: -len("/v1")]
        return urlunsplit((parts.scheme, parts.netloc, stripped, "", ""))
    return root


def openai_env_key(provider: Provider) -> str:
    """The env var name a secret ``provider`` carries its token under, or ``""``.

    ``""`` for anything but ``auth == "secret"`` — the two writers below
    (:func:`openai_toml_body`'s ``env_key`` line, ``_render_openai_toml``'s
    ``export``) both branch on this return value being non-empty, so a
    non-secret provider (``ollama``, ``literal``-auth) gets neither: the
    OPENAI_TOML wrapper's output for it must stay byte-identical to before
    this function existed, or every previously-installed ``codex × ollama``
    wrapper stops matching the ownership guard's byte comparison.

    One function so both writers agree on the name — the same reasoning that
    makes :func:`openai_base_url`/:func:`toml_string` public and shared with
    ``codex_default.py`` rather than each writer deriving its own value.

    ``provider.token_env_var`` is validated at import time
    (``model._validate_provider``) to match ``[A-Z][A-Z0-9_]*`` precisely
    because this value is interpolated into the wrapper UNQUOTED, left of
    ``=`` in ``export {key}=...`` — shell syntax has no quoting form there.
    """
    if provider.auth != "secret":
        return ""
    return provider.token_env_var


def openai_provider_table(
    table: str,
    display_name: str,
    base_url: str,
    wire_api: str,
    env_key: str,
) -> str:
    """The ``[model_providers.<table>]`` block BOTH codex-table writers emit.

    ONE home for the field set, the line order, and the conditional-emission
    rule (``env_key`` only for a secret provider, LAST) — the rule whose
    duplication between this module's per-alias profile and
    ``codex_default``'s ``config.toml`` patcher is exactly what once left
    bare ``codex`` authenticating with no key at all (401) while the wrapper
    on the same endpoint worked. Both writers call this; neither spells the
    block by hand any more. Public like :func:`toml_string` and
    :func:`openai_env_key` for the same reason: the two writers must not
    drift. Empty ``env_key`` emits no line, byte-identical to the pre-env_key
    output — required by the ownership guard's byte comparison.
    """
    body = (
        f"[model_providers.{table}]\n"
        f'name = "{toml_string(display_name)}"\n'
        f'base_url = "{toml_string(base_url)}"\n'
        f'wire_api = "{toml_string(wire_api)}"\n'
    )
    if env_key:
        body += f'env_key = "{toml_string(env_key)}"\n'
    return body


def openai_toml_body(spec: WrapperSpec) -> str:
    """The ``~/.codex/<alias>.config.toml`` profile body (pure, no IO).

    Codex ≥ 0.146.0 resolves profiles from per-file config: the basename
    ``<alias>.config.toml`` is addressed as ``--profile <alias>``, and legacy
    ``[profiles.X]`` tables inside ``config.toml`` are rejected — so this is a
    whole file, not a fragment. This renderer never reads or modifies Codex's
    own ``config.toml`` — that file has a separate writer entirely
    (``services/codex_default.py``'s ``set-default`` patcher), which never
    touches this per-alias profile in turn. The two files are siblings, not
    layers: neither is a fragment of the other.

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
    - ``base_url`` is :func:`openai_base_url` (handles both ``/v1``-suffixed
      and bare roots);
    - ``wire_api`` is ``spec.provider.wire_api``.

    No model catalog is written any more (see the removed
    ``openai_catalog_body``): Codex's ``ModelInfo`` requires a ``base_instructions``
    string per catalog entry, and this tool has no real system prompt to put
    there — an empty one is not "no override", it IS the session's
    instructions (``ModelInfo::get_model_instructions`` returns
    ``base_instructions`` verbatim when no ``model_messages`` template is set),
    silently replacing Codex's real coding-agent prompt for every wrapper on a
    matched catalog entry. Without a catalog, an unrecognised slug falls
    through to Codex's own ``model_info_from_slug``, which carries the real
    bundled prompt — worse context-window guessing, but a working agent.
    ``model_context_window`` (a plain top-level ``config.toml`` key, applied
    unconditionally by Codex's ``with_config_overrides``) covers the one thing
    the catalog existed for, and only rides along when the model resolves to a
    known window via :func:`uniform_context_window` — never a guessed floor,
    the same conditional-emission rule :func:`_render_anthropic_env` uses for
    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS``.

    The marker on line 1 is the same comment the bash wrapper carries, which is
    what lets the ownership guard recognise this as ours. Every quoted value
    goes through :func:`toml_string`; the table key is the only bare
    interpolation, safe by the import-time check.

    A secret provider (:func:`openai_env_key` non-empty) gets one more line,
    ``env_key``, LAST in the table — Codex reads the named env var for its
    Authorization header, and the matching ``export`` is written by
    ``_render_openai_toml`` in the sibling wrapper script. Conditional and
    appended last so a non-secret provider's output (``ollama``) is byte-
    identical to before this field existed — required for the ownership
    guard's byte comparison on already-installed wrappers to keep matching.
    """
    table = spec.provider.name
    display_name = spec.provider.description or spec.provider.name
    base_url = openai_base_url(spec.provider.base_url)
    lines = [
        f"{_marker(spec)}\n",
        f'model = "{toml_string(spec.model)}"\n',
        f'model_provider = "{toml_string(table)}"\n',
    ]
    if (window := _declared_window(spec)) is not None:
        lines.append(f"model_context_window = {window}\n")
    # Conditional, emitted with the other optional declarations (issue #100):
    # an effort-less profile must stay byte-identical to the pre-#100 output,
    # so the ownership guard's byte comparison keeps matching installed
    # wrappers that never carried the key.
    if spec.effort is not None:
        lines.append(f'model_reasoning_effort = "{toml_string(spec.effort)}"\n')
    lines += [
        "\n",
        openai_provider_table(
            table,
            display_name,
            base_url,
            spec.provider.wire_api,
            openai_env_key(spec.provider),
        ),
    ]
    return "".join(lines)


def toml_string(value: str) -> str:
    """Quote ``value`` for a TOML basic string.

    TOML basic strings are ``"..."`` with ``\\`` and ``"`` escaped, and they
    forbid literal control characters (U+0000–U+001F except tab) — a ``--model``
    containing a newline would otherwise produce a profile Codex rejects, while
    the shell sibling of the same install handles it fine via
    :func:`_shell_single_quote`. Control characters are emitted as their TOML
    escapes (``\\uXXXX`` is always valid). This is the per-target equivalent of
    the shell single-quote escape, applied to every value that originates
    outside the registry (models, base URLs, ``wire_api``).

    Public (not ``_``-prefixed): ``services/codex_default.py`` reuses this
    exact encoder for the values it patches into Codex's own ``config.toml``,
    so the two writers of an OPENAI_TOML-shaped value cannot drift on escaping.
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
    ConfigShape.AGENT_NATIVE: _render_agent_native,
}


def render_legacy_script(spec: WrapperSpec, token: str = "") -> str:
    """The body the PREVIOUS, markerless release would have written for ``spec``.

    Used by the install guard to recognise the tool's own prior output, which
    predates :data:`MARKER_PREFIX` and would otherwise be classified as a
    third-party file (see ``wrappers._ownership_full_match``). Every renderer here differs
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
