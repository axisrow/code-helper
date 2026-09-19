"""Renderer-level pins for the issue #76 proxy contract (its §5: T1 + T2).

C1 — ``base_url`` is an endpoint, never a proxy. C2 — a generated wrapper is
env-transparent: it inherits the caller's environment and never assigns,
exports, or unsets a proxy variable. The eight names in
:data:`C2_PROXY_ENV_NAMES` are the contract's C2 test universe — deliberately
wider than the four keys ``proxy.py`` manages (``ALL_PROXY`` included), so a
renderer that emits ANY proxy variable fails here whatever consumer would
have read it — and they are owned as one tuple in this file.

Everything here is pure: ``render_script`` / ``openai_toml_body`` do no IO.
The byte-stability and ownership-guard tests in ``test_wrappers.py`` prove
the contract cost the templates nothing.
"""

from __future__ import annotations

import re
import tomllib

import pytest

from codehelper.services.model import ConfigShape, ModelListAPI, Provider
from codehelper.services.render import openai_toml_body, render_script
from codehelper.services.spec import WrapperSpec, build_spec
from codehelper.services.wrappers import get_spec

#: Every proxy-env name, all case spellings (issue #76, C2). THE test
#: universe for "a wrapper never writes proxy env".
C2_PROXY_ENV_NAMES: tuple[str, ...] = (
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _writes(name: str, line: str) -> bool:
    """Whether one script line writes ``name`` into the environment.

    Covers the shell write forms — assignment (plain or as a command
    prefix), ``export`` (with or without ``=``), ``declare``/``typeset``/
    ``local`` (with their flags), ``unset``, and ``env NAME=...`` — the three
    verbs the contract names (assign, export, unset) plus their aliases. A
    comment merely mentioning the name is not a write and must not fire.
    """
    escaped = re.escape(name)
    return (
        re.match(
            rf"^\s*(?:export\s+|declare\s+(?:-[a-zA-Z]+\s+)*|typeset\s+"
            rf"(?:-[a-zA-Z]+\s+)*|local\s+|env\s+)?{escaped}\s*=",
            line,
        )
        is not None
        or re.match(rf"^\s*export\s+(?:-[a-zA-Z]+\s+)*{escaped}\s*$", line) is not None
        or re.match(rf"^\s*unset\s+(?:-[a-zA-Z]+\s+)*{escaped}\s*$", line) is not None
    )


def _secret_toml_provider() -> Provider:
    """An out-of-registry OPENAI_TOML secret provider — the same technique as
    ``test_wrappers``' fixtures: the shape is the extension point, the
    registry is not needed to exercise the renderer."""
    return Provider(
        name="secret-openai",
        shapes=frozenset({ConfigShape.OPENAI_TOML}),
        base_url="https://api.secret.invalid/v1",
        auth="secret",
        token_env_var="SECRET_API_KEY",
        model_list_api=ModelListAPI.OPENAI_V1,
        wire_api="responses",
    )


def _openai_toml_specs() -> list[tuple[str, WrapperSpec]]:
    """Both OPENAI_TOML variants: literal (no export, no ``env_key``) and
    secret (the one export a wrapper is allowed)."""
    return [
        (
            "literal",
            build_spec(
                agent="codex",
                provider="ollama-direct",
                model="glm-5.2:cloud",
                alias="glm-5-codex",
            ),
        ),
        (
            "secret",
            build_spec(
                agent="codex",
                provider=_secret_toml_provider(),
                model="m",
                alias="x",
            ),
        ),
    ]


def _contract_bodies() -> list[tuple[str, str]]:
    """One representative render per shape; OPENAI_TOML in both variants."""
    bodies = [
        ("anthropic-env-literal", render_script(get_spec("deepseek-ollama"), "ollama")),
        ("anthropic-env-secret", render_script(get_spec("glm"), "sk-tok")),
        ("ollama-launch", render_script(get_spec("glm-ollama"), "")),
        ("agent-native", render_script(get_spec("agy-native"), "")),
    ]
    bodies.extend(
        (
            f"openai-toml-{label}",
            render_script(spec, "" if label == "literal" else "sk-tok"),
        )
        for label, spec in _openai_toml_specs()
    )
    return bodies


# --------------------------------------------------------------------------- #
# T1 — renderers never emit proxy env
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "body", [pytest.param(body, id=label) for label, body in _contract_bodies()]
)
def test_no_renderer_line_writes_a_proxy_env_name(body):
    """T1 (issue #76 §5): for every shape, no line of ``render_script``
    output assigns, exports, or unsets any of the eight C2 names. A wrapper
    is env-transparent: it inherits the invoking process's environment and
    writes nothing proxy-shaped."""
    for line in body.splitlines():
        for name in C2_PROXY_ENV_NAMES:
            assert not _writes(name, line), (
                f"line {line!r} writes proxy env {name!r} — a wrapper is "
                "env-transparent (issue #76, C2)"
            )


# --------------------------------------------------------------------------- #
# T2 — profile keys are whitelisted
# --------------------------------------------------------------------------- #

#: T2's whitelists (issue #76 §5). ``model_providers`` is the table container
#: itself; its inner tables are checked against PROVIDER_TABLE_KEYS below.
#: No proxy key is in either vocabulary, so one cannot appear without
#: failing here (C1: ``base_url`` is an endpoint, never a proxy).
TOP_LEVEL_KEYS = {
    "model",
    "model_provider",
    "model_context_window",
    "model_reasoning_effort",
    "model_providers",
}
PROVIDER_TABLE_KEYS = {"name", "base_url", "wire_api", "env_key"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec",
    [pytest.param(spec, id=label) for label, spec in _openai_toml_specs()],
)
def test_openai_toml_body_keys_are_whitelisted(spec):
    """T2 (issue #76 §5): ``openai_toml_body`` parses with ``tomllib`` and
    its top-level keys and provider-table keys are whitelisted — there is no
    proxy key in the vocabulary for a future renderer edit to emit."""
    data = tomllib.loads(openai_toml_body(spec))
    assert set(data) <= TOP_LEVEL_KEYS
    tables = data.get("model_providers", {})
    assert tables, "no provider table — test lost its subject"
    for table in tables.values():
        assert set(table) <= PROVIDER_TABLE_KEYS
