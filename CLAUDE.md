# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`code-helper` — pip-installable Python CLI that generates and manages small bash wrapper scripts in `~/.local/bin`. Each wrapper (`deepseek`, `glm`, ...) points Claude Code at a different model backend by exporting `ANTHROPIC_*` env vars and execing `claude "$@"`. Not affiliated with Z.ai, Moon Bridge, or Codex — successor to the archived `zai-codex-helper` project, stripped down to just this one feature.

## Commands

```bash
pip install -e ".[dev]"        # editable install + dev tools
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format (the project's only linter)
```

## Architecture

Three-layer split, much smaller than the archived predecessor:

- **`cli/parser.py`** — argparse tree (`list` / `add <name> [--model]` / `remove <name>`). Handlers are thin shells: resolve `Paths.default()`, delegate to a service, let `CodeHelperError` propagate to `__main__.main()`.
- **`services/wrappers.py`** — the core. `WrapperSpec` (frozen dataclass) + the `WRAPPERS` registry (`deepseek`, `glm`). `render_script` (pure), `install_wrapper` / `uninstall_wrapper` / `is_installed` / `list_wrappers`. Ownership of a generated script is proven by a marker comment (`# code-helper managed`), never by matching the body or the token — so a token rotation or `--model` override still recognizes the same script, and a foreign hand-written file with the same name is never touched.
- **`services/secrets.py`** — resolves a wrapper's token: env var first, hidden `getpass` prompt otherwise. No persistent config file — the token lives only inside the generated script.
- **`services/paths.py`** — frozen `Paths` dataclass, one field (`bin_dir = ~/.local/bin`). `Paths.from_home(home)` is pure path arithmetic; `Paths.default()` wraps `Path.home()`.
- **`backends/_atomic.py`** — the only code that touches the filesystem. `atomic_write(path, data, mode)`: temp-sibling + fsync + `os.replace`, crash-safe.
- **`errors.py`** — `CodeHelperError` lives here (not in `__main__`) to avoid a class-identity split under `python -m`.

## Key Constraints

- **Every wrapper is DATA** (a `WrapperSpec` in `WRAPPERS`), never a hand-edited string — adding a new backend is a registry entry, not new logic.
- **The `auth` field** (`"literal"` vs `"secret"`) is what keeps this generic: a literal token (e.g. Ollama's local daemon accepts `"ollama"`) needs no prompt and gets mode `0o755`; a secret token is resolved interactively/from env and gets mode `0o700` (owner-only — the script carries it in plain text).
- **Shell-injection defense**: every value interpolated into a generated script — token, base URL, model names — goes through `_shell_single_quote` (POSIX `'` → `'"'"'`). This matters because `--model` makes the model name user-controlled input, not just a registry constant.
- **Never clobber a foreign file**: `install_wrapper` raises `CodeHelperError` if `~/.local/bin/<name>` exists without the marker. `uninstall_wrapper` on a foreign/absent file is a silent no-op (`False`), never a write.
- **No Rich/Typer** — plain text, argparse (stdlib) only.
- **`--dry-run`** never writes any file — prints what would happen instead.

## Testing

- **Markers**: `@pytest.mark.unit` / `.integration`. `--strict-markers` in `addopts` turns a typo'd marker into a hard collection error.
- **HOME isolation**: `conftest.py`'s autouse `_isolate_home` fixture sets `HOME=tmp_path` for every test. `Paths.from_home(tmp_path)` is the test seam; production code only ever calls `Paths.default()`.
- Every install/uninstall/render behavior in `services/wrappers.py` is pinned in `tests/test_wrappers.py`, including two shell-injection regression tests (token and `--model`).
