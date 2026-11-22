"""Launch ChatGPT.app with a process-scoped HTTP proxy on macOS."""

from __future__ import annotations

import subprocess
import sys
from urllib.parse import urlsplit

from codehelper.errors import CodeHelperError
from codehelper.services.paths import Paths
from codehelper.services.proxy import (
    ProxyStatus,
    proxy_status,
    resolve_proxy_patch,
    validate_proxy_url,
)

_APP_ID = "com.openai.codex"
_LOOPBACK = "localhost,127.0.0.1,::1"


def chatgpt_proxy_url(
    paths: Paths,
    *,
    state: dict[str, object] | None = None,
    app_proxy: ProxyStatus | None = None,
) -> str | None:
    if app_proxy is None:
        app_proxy = proxy_status(paths, state=state)
    return app_proxy.restorable_url


def _is_running() -> bool:
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", f'application id "{_APP_ID}" is running'],
        capture_output=True,
        check=False,
        timeout=5,
        text=True,
    )
    status = result.stdout.strip()
    if result.returncode != 0 or status not in ("true", "false"):
        raise CodeHelperError("could not check whether ChatGPT.app is running")
    return status == "true"


def launch_chatgpt(paths: Paths, *, dry_run: bool = False) -> None:
    address = chatgpt_proxy_url(paths)
    if not address:
        raise CodeHelperError(
            "set the shared proxy with `codehelper proxy --url` first"
        )
    address = validate_proxy_url(address)
    parsed = urlsplit(address)
    if parsed.username is not None or parsed.password is not None:
        raise CodeHelperError(
            "ChatGPT.app needs an HTTP proxy URL without embedded credentials"
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise CodeHelperError(
            "ChatGPT.app needs an HTTP proxy URL in the form http://host:port"
        )
    if dry_run:
        print(f"Would launch ChatGPT.app via {address} with the existing Codex history")
        return
    if sys.platform != "darwin":
        raise CodeHelperError("ChatGPT.app launch requires macOS")

    # The embedded Codex runtime uses proxy env; Electron handles its own
    # network stack through Chromium switches.
    command = ["/usr/bin/open", "-b", _APP_ID]
    for key, value in resolve_proxy_patch(url=address, no_proxy=_LOOPBACK).items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(
        [
            "--args",
            f"--proxy-server={address}",
            "--proxy-bypass-list=localhost;127.0.0.1;[::1]",
        ]
    )
    try:
        if _is_running():
            raise CodeHelperError(
                "ChatGPT.app is running — quit it, then run codehelper chatgpt from Terminal"
            )
        subprocess.run(command, capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeHelperError(
            f"ChatGPT.app launch failed: {type(exc).__name__}"
        ) from None
    print(f"Opened ChatGPT.app with proxy {address}")
