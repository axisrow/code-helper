"""Switch Claude Desktop between native and gateway profiles."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from codehelper.backends._atomic import (
    atomic_write,
    file_lock,
    guarded_replace,
    read_json_object,
    read_text_or_none,
)
from codehelper.errors import CodeHelperError
from codehelper.services.claude_settings import dump_settings, read_settings_file
from codehelper.services.paths import Paths
from codehelper.services.proxy import NATIVE_API_HOST

OLLAMA_PROFILE_ID = "00000000-0000-4000-8000-000000000114"
GATEWAY_PROFILE_ID = "7a6116c2-b392-48f4-973a-02c2f5c3e495"
_GATEWAY_PROFILE_NAME = "Codehelper Gateway"
_APP_ID = "com.anthropic.claudefordesktop"
_CONFIG_NAME = "claude_desktop_config.json"
_OLLAMA_ROUTING_KEYS = (
    "inferenceProvider",
    "inferenceGatewayBaseUrl",
    "inferenceGatewayAuthScheme",
    "deploymentDisplayName",
    "inferenceModels",
    "coworkEgressAllowedHosts",
    "autoModeEnabled",
    "disableEssentialTelemetry",
    "disableNonessentialTelemetry",
)


@dataclass(frozen=True)
class DesktopEdit:
    path: Path
    old_text: str
    old: dict
    new: dict

    def summary(self) -> str:
        changed = [
            key
            for key in self.new.keys() | self.old.keys()
            if key != "env" and self.new.get(key) != self.old.get(key)
        ]
        if self.old.get("env") != self.new.get("env"):
            old_env, new_env = self.old.get("env", {}), self.new.get("env", {})
            changed.extend(
                f"env.{key}"
                for key in new_env.keys() | old_env.keys()
                if new_env.get(key) != old_env.get(key)
            )
        changed.sort()
        return f"{self.path} — {', '.join(changed)}"

    def write(self) -> None:
        guarded_replace(
            self.path,
            expected=self.old_text,
            new_text=dump_settings(self.new),
            stale_message=f"{self.path} changed since it was read — retry",
            backup_slots=(
                self.path.with_suffix(".json.bak3"),
                self.path.with_suffix(".json.bak2"),
                self.path.with_suffix(".json.bak1"),
            ),
            mode=0o600,
        )

    def rollback(self) -> None:
        with file_lock(self.path):
            if read_text_or_none(self.path) != dump_settings(self.new):
                raise CodeHelperError(f"{self.path} changed during rollback")
            if self.old_text:
                atomic_write(self.path, self.old_text, mode=0o600)
            else:
                self.path.unlink()


def _deployment_edits(paths: Paths, mode: str) -> list[DesktopEdit]:
    edits = []
    for third_party in (False, True):
        path = paths.claude_desktop_app_dir(third_party=third_party) / _CONFIG_NAME
        text, config = read_settings_file(path)
        edits.append(DesktopEdit(path, text, config, dict(config, deploymentMode=mode)))
    return edits


def _profile_library(paths: Paths) -> Path:
    return paths.claude_desktop_app_dir(third_party=True) / "configLibrary"


def _restore_edits(paths: Paths) -> list[DesktopEdit]:
    library = _profile_library(paths)
    meta_path = library / "_meta.json"
    meta_text, meta = read_settings_file(meta_path)
    configs = _deployment_edits(paths, "1p")
    active = meta.get("appliedId")

    if active not in (OLLAMA_PROFILE_ID, GATEWAY_PROFILE_ID):
        if any(edit.old.get("deploymentMode") == "3p" for edit in configs):
            raise CodeHelperError(
                "restore the active Desktop profile from its provider's settings"
            )
        return []

    if active == GATEWAY_PROFILE_ID:
        # Keep the saved gateway profile available for the next activation.
        next_meta = dict(meta)
        next_meta.pop("appliedId")
        configs.append(DesktopEdit(meta_path, meta_text, meta, next_meta))
        return [edit for edit in configs if edit.old != edit.new]

    entries = meta.get("entries", [])
    if not isinstance(entries, list):
        raise CodeHelperError(f"{meta_path} must contain a list of profile entries")
    next_meta = dict(meta)
    next_meta.pop("appliedId")
    if "entries" in meta:
        next_meta["entries"] = [
            entry
            for entry in entries
            if not isinstance(entry, dict) or entry.get("id") != OLLAMA_PROFILE_ID
        ]

    profile_path = library / f"{OLLAMA_PROFILE_ID}.json"
    profile_text, profile = read_settings_file(profile_path)
    if profile:
        next_profile = {
            key: value
            for key, value in profile.items()
            if key not in _OLLAMA_ROUTING_KEYS
        }
        next_profile["disableDeploymentModeChooser"] = False
        configs.append(DesktopEdit(profile_path, profile_text, profile, next_profile))
    # Clear the applied ID last so a failed restore can be retried.
    configs.append(DesktopEdit(meta_path, meta_text, meta, next_meta))
    return [edit for edit in configs if edit.old != edit.new]


def _gateway_edits(paths: Paths, url: str, token: str | None) -> list[DesktopEdit]:
    library = _profile_library(paths)
    meta_path = library / "_meta.json"
    meta_text, meta = read_settings_file(meta_path)
    entries = meta.get("entries", [])
    if not isinstance(entries, list):
        raise CodeHelperError(f"{meta_path} must contain a list of profile entries")
    next_entries = list(entries)
    if not any(
        isinstance(entry, dict) and entry.get("id") == GATEWAY_PROFILE_ID
        for entry in entries
    ):
        next_entries.append({"id": GATEWAY_PROFILE_ID, "name": _GATEWAY_PROFILE_NAME})
    next_meta = dict(meta, appliedId=GATEWAY_PROFILE_ID, entries=next_entries)

    profile_path = library / f"{GATEWAY_PROFILE_ID}.json"
    profile_text, profile = read_settings_file(profile_path)
    # A saved credential belongs only to the same gateway URL.
    existing = (
        profile.get("inferenceGatewayApiKey")
        if profile.get("inferenceGatewayBaseUrl") == url
        else None
    )
    key = token or (existing if isinstance(existing, str) else None) or "unused"
    next_profile = dict(
        profile,
        inferenceProvider="gateway",
        inferenceGatewayBaseUrl=url,
        inferenceGatewayApiKey=key,
        inferenceGatewayAuthScheme="bearer",
        deploymentDisplayName=_GATEWAY_PROFILE_NAME,
        modelDiscoveryEnabled=True,
    )
    next_profile.pop("inferenceModels", None)
    edits = [
        DesktopEdit(profile_path, profile_text, profile, next_profile),
        DesktopEdit(meta_path, meta_text, meta, next_meta),
        *reversed(_deployment_edits(paths, "3p")),
    ]
    return [edit for edit in edits if edit.old != edit.new]


def desktop_api_host(paths: Paths) -> str | None:
    native_config = paths.claude_desktop_app_dir() / _CONFIG_NAME
    config = read_json_object(native_config)
    if config is None and native_config.exists():
        return None
    if (config or {}).get("deploymentMode") != "3p":
        return NATIVE_API_HOST
    library = _profile_library(paths)
    active = (read_json_object(library / "_meta.json") or {}).get("appliedId")
    if active not in (OLLAMA_PROFILE_ID, GATEWAY_PROFILE_ID):
        return None
    profile = read_json_object(library / f"{active}.json") or {}
    url = profile.get("inferenceGatewayBaseUrl")
    try:
        return urlparse(url).hostname if isinstance(url, str) else None
    except ValueError:
        return None


def _desktop_is_running() -> bool:
    result = subprocess.run(
        ["/usr/bin/pgrep", "-f", r"/Claude\.app/Contents/MacOS/Claude([[:space:]]|$)"],
        capture_output=True,
        timeout=5,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise CodeHelperError("could not check whether Claude Desktop is running")
    return result.returncode == 0


def _quit_desktop() -> None:
    subprocess.run(
        ["/usr/bin/osascript", "-e", f'tell application id "{_APP_ID}" to quit'],
        capture_output=True,
        timeout=20,
        check=True,
    )
    deadline = time.monotonic() + 20
    while _desktop_is_running():
        if time.monotonic() >= deadline:
            raise CodeHelperError("Claude Desktop is still running — quit it and retry")
        time.sleep(0.2)


def _open_desktop(paths: Paths) -> None:
    env = os.environ.copy()
    if env.get("CLAUDE_CONFIG_DIR") == str(paths.legacy_claude_app_settings().parent):
        env.pop("CLAUDE_CONFIG_DIR")
    subprocess.run(
        ["/usr/bin/open", "-b", _APP_ID],
        env=env,
        capture_output=True,
        timeout=10,
        check=True,
    )


def _write_desktop_edits(edits: list[DesktopEdit]) -> None:
    applied = []
    try:
        for edit in edits:
            edit.write()
            applied.append(edit)
    except (CodeHelperError, OSError):
        failures = []
        for edit in reversed(applied):
            try:
                edit.rollback()
            except (CodeHelperError, OSError):
                failures.append(str(edit.path))
        if failures:
            raise CodeHelperError(
                "Desktop profile rollback failed for "
                + ", ".join(failures)
                + " — review the backups before opening Claude Desktop"
            ) from None
        raise


def _apply_desktop_profile(
    paths: Paths,
    collect: Callable[[], list[DesktopEdit]],
    *,
    dry_run: bool,
    force: bool,
    confirm,
    launch: bool = False,
) -> bool:
    edits = collect()
    if not edits and not launch:
        return False
    preview = (
        "\n".join(edit.summary() for edit in edits)
        or "Desktop profile already configured."
    )
    if dry_run:
        print(preview)
        return True
    if sys.platform != "darwin":
        raise CodeHelperError("Desktop profile changes require macOS")
    try:
        if (
            edits
            and not force
            and not (
                confirm
                and confirm(
                    edits[0].path,
                    preview + "\nA running Claude Desktop will be restarted.",
                )
            )
        ):
            raise CodeHelperError(
                "Desktop profile change requires confirmation — use `--force`"
            )
        with file_lock(paths.config_dir / "claude-desktop-profile"):
            if collect() != edits:
                raise CodeHelperError(
                    "Desktop profile changed since it was read — retry"
                )
            running = _desktop_is_running()
            if running and edits:
                _quit_desktop()
            # Desktop saves preferences during shutdown, so retain its final write.
            _write_desktop_edits(collect())
            if launch or running:
                _open_desktop(paths)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeHelperError(
            f"Claude Desktop profile change failed: {type(exc).__name__}"
        ) from None
    return True


def restore_desktop(paths: Paths, *, dry_run: bool, force: bool, confirm) -> None:
    """Return to first-party mode after the Desktop app saves its settings."""
    if not _apply_desktop_profile(
        paths,
        lambda: _restore_edits(paths),
        dry_run=dry_run,
        force=force,
        confirm=confirm,
    ):
        print("Desktop already uses Native Anthropic")
    elif not dry_run:
        print("Desktop provider: Native Anthropic (1p)")


def launch_desktop_gateway(
    paths: Paths, url: str, *, token: str | None, dry_run: bool, force: bool, confirm
) -> None:
    url = url.strip().rstrip("/")
    try:
        parsed = urlparse(url)
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and parsed.port != 0
            and parsed.username is None
            and parsed.password is None
            and not parsed.params
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 for char in url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise CodeHelperError(
            "gateway URL must be HTTP(S) without credentials, query or fragment"
        )

    _apply_desktop_profile(
        paths,
        lambda: _gateway_edits(paths, url, token),
        dry_run=dry_run,
        force=force,
        confirm=confirm,
        launch=True,
    )
    if not dry_run:
        print("Desktop provider: Codehelper Gateway (3p)")
