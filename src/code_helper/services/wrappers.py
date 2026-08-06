"""Generated-script lifecycle: render → install → describe.

A "wrapper" is a small bash script in ``~/.local/bin`` that points a coding
agent at a model backend. What a wrapper *is* now lives in
``services/spec.py`` (the resolved agent × provider × model combination) and
``services/render.py`` (how that becomes a script body); this module owns only
the lifecycle around it — writing the file, reporting what is installed, and
formatting listings.

The ``auth`` mode (carried by the provider) decides how sensitive the on-disk
script is:

- ``"literal"`` / ``"none"`` — no real credential in the file, mode ``0o755``.
- ``"secret"`` — a real key is embedded in plain text, mode ``0o700``
  (owner-only).

**Ownership.** Every generated script carries a marker comment
(``render.MARKER_PREFIX``) on its second line, and :func:`is_managed` reads it
back. This reverses the project's earlier "no ownership guard" position, and
the reason is that the position's precondition disappeared: overwriting by name
was safe while the set of names was closed and curated (``get_spec`` admitted
exactly three), but a user-chosen ``--alias`` can name anything on ``PATH`` —
including ``~/.local/bin/claude``, a real working symlink on a normal install.
Presets still overwrite freely; only a *foreign* file triggers the guard.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from code_helper.backends._atomic import atomic_write
from code_helper.errors import CodeHelperError
from code_helper.services.model import Agent, Provider
from code_helper.services.paths import Paths
from code_helper.services.render import MARKER_PREFIX, render_script
from code_helper.services.spec import (
    PRESETS,
    Preset,
    TierModels,
    WrapperSpec,
    build_spec,
    get_preset,
    preset_names,
    spec_from_preset,
    suggest_alias,
)

__all__ = [
    # re-exported so existing imports keep working
    "WrapperSpec",
    "TierModels",
    "Preset",
    "PRESETS",
    "Agent",
    "Provider",
    "build_spec",
    "get_preset",
    "preset_names",
    "spec_from_preset",
    "suggest_alias",
    "render_script",
    # lifecycle
    "install_wrapper",
    "is_installed",
    "is_managed",
    "list_wrappers",
    "describe_wrapper",
    "describe_all",
    "discover_managed",
    "get_spec",
    "WRAPPERS",
]

#: Owner-only. A "secret" wrapper carries a real credential in plain text —
#: group/other must have NO bits. Anything else is 0o755, a normal executable.
_MODE_SECRET = 0o700
_MODE_LITERAL = 0o755


def get_spec(name: str) -> WrapperSpec:
    """Return the resolved spec for the preset called ``name``.

    Kept as the preset lookup so callers written against the old flat registry
    keep working.

    Raises:
        CodeHelperError: unknown name. Fails BEFORE any token is resolved or
            file touched — a typo must not trigger a secret prompt.
    """
    return spec_from_preset(get_preset(name))


#: Backwards-compatible view of the preset registry as resolved specs.
WRAPPERS: list[WrapperSpec] = [spec_from_preset(p) for p in PRESETS]


def _mode_for(spec: WrapperSpec) -> int:
    return _MODE_SECRET if spec.auth == "secret" else _MODE_LITERAL


def is_installed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` currently exists.

    Pure existence — says nothing about who wrote it. Use :func:`is_managed`
    for that. The name is no longer validated against a registry here: an
    ad-hoc alias has no preset, and ``script_for`` already refuses anything
    that is not a single path component.
    """
    return paths.script_for(name).exists()


def is_managed(paths: Paths, name: str) -> bool:
    """True iff ``~/.local/bin/<name>`` exists AND carries our marker.

    Reads only the first couple of lines. Anything unreadable (a binary, a
    permission error, a dangling symlink) counts as *not* ours — the safe
    answer, since it makes the guard refuse rather than clobber.
    """
    script = paths.script_for(name)
    try:
        with script.open("r", encoding="utf-8") as fh:
            for _ in range(2):
                line = fh.readline()
                if not line:
                    break
                if line.startswith(MARKER_PREFIX):
                    return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def install_wrapper(
    paths: Paths,
    spec: WrapperSpec | str,
    *,
    token: str = "",
    model_override: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    confirm: Callable[[Path], bool] | None = None,
) -> bool:
    """Write ``spec``'s script. Return True iff it wrote (or would, in dry-run).

    Idempotent: a byte-identical re-install is a no-op.

    Args:
        paths: Resolved :class:`Paths`.
        spec: A :class:`WrapperSpec`, or a preset name (resolved via
            :func:`get_spec`) for backwards compatibility.
        token: Auth token to embed. Resolved by the CALLER — ``auth_value`` for
            literal providers, ``resolve_token`` for secret ones.
        model_override: Only meaningful with a preset name; ignored when a
            fully-resolved spec is passed (its model is already decided).
        dry_run: Print what would happen, write nothing, ask nothing.
        force: Overwrite a foreign file without asking.
        confirm: Asked before overwriting a foreign file. ``None`` means "no
            way to ask" and is treated as refusal — this is what guarantees a
            non-interactive run FAILS FAST instead of blocking on stdin.

    Raises:
        CodeHelperError: a foreign file is in the way and was not confirmed.
    """
    resolved: WrapperSpec = (
        spec_from_preset(get_preset(spec), model_override=model_override)
        if isinstance(spec, str)
        else spec
    )
    spec = resolved

    body = render_script(spec, token)
    script = paths.script_for(spec.alias)

    if script.exists() and script.read_text(encoding="utf-8") == body:
        return False  # identical script already installed

    # Ownership check runs only for a file we did not write. Order matters:
    # the idempotence check above means an unchanged reinstall never prompts.
    if script.exists() and not is_managed(paths, spec.alias):
        if dry_run:
            print(f"would overwrite UNMANAGED file {script}")
            return True
        if not force:
            if confirm is None or not confirm(script):
                raise CodeHelperError(
                    f"{script} exists and was not created by code-helper — "
                    f"refusing to overwrite (use --force)"
                )
        print(f"overwriting unmanaged file {script}")

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

    Read-only (no write). Goes through :func:`describe_all` like the two
    menus do, rather than re-deriving ``script_for(...).exists()`` inline —
    that helper exists to share the ``Paths``/:func:`is_installed` wiring,
    not only the row format.
    """
    for _name, label in describe_all(
        paths,
        WRAPPERS,
        installed_word="installed",
        not_installed_word="not installed",
    ):
        print_fn(label)

    ad_hoc = discover_managed(paths)
    if ad_hoc:
        print_fn("")
        print_fn("ad-hoc wrappers:")
        for name in ad_hoc:
            print_fn(f"{name:12} {'installed':13}")


def discover_managed(paths: Paths) -> list[str]:
    """Names of managed wrappers on disk that are NOT presets.

    Without this, a wrapper built from the axes would be invisible to ``list``
    — the preset registry cannot know about it, and there is no state file.
    The marker in the script body is the only record that it is ours.
    """
    if not paths.bin_dir.is_dir():
        return []
    known = set(preset_names())
    found = [
        entry.name
        for entry in paths.bin_dir.iterdir()
        if entry.is_file() and entry.name not in known and is_managed(paths, entry.name)
    ]
    return sorted(found)
