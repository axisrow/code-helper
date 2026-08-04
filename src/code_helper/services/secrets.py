"""Resolving a wrapper's secret token — env first, hidden prompt otherwise.

Unlike the archived project's ``services/api_key.py``, there is no persistent
config file here (no ``moonbridge-zai.yml``): the token lives ONLY inside the
generated wrapper script. This module's only job is producing the token value
to embed, never storing it anywhere else and never printing/logging it.

Provider-specific key FORMAT validation (the old project's strict
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
import os
from collections.abc import Callable

from code_helper.errors import CodeHelperError

__all__ = ["resolve_token"]


def resolve_token(
    *,
    env_var: str,
    prompt: str,
    environ: os._Environ[str] = os.environ,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    retries: int = 3,
) -> str:
    """Return a non-empty secret token: ``env_var`` wins, else a hidden prompt.

    Args:
        env_var: Environment variable name checked first (headless/scripting
            path — e.g. ``"ZAI_API_KEY"``).
        prompt: The ``getpass`` prompt shown when the env var is unset.
        environ: Injected environment (default ``os.environ``).
        getpass_fn: Hidden-input source (default ``getpass.getpass``, NEVER
            echoed to the terminal).
        retries: How many empty-input attempts are tolerated before giving up.

    Returns:
        The resolved, non-empty token string. Never logged or printed by this
        function.

    Raises:
        CodeHelperError: if the env var is unset AND every interactive
            attempt yields an empty string.
    """
    env_value = environ.get(env_var)
    if env_value:
        return env_value

    for _ in range(retries):
        value = getpass_fn(prompt)
        if value:
            return value

    raise CodeHelperError(
        f"no token provided — set {env_var} or enter a non-empty value when prompted"
    )
