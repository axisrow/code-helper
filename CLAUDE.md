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

- **`cli/parser.py`** — argparse tree (`list` / `add <name> [--model]` / `edit-token [<name>]`). Handlers are thin shells: resolve `Paths.default()`, delegate to a service, let `CodeHelperError` propagate to `__main__.main()`.
- **`cli/menu.py`** — a ~90-line hand-rolled arrow-key terminal menu (`select_from_menu`, raw `termios`/`tty` reading, Unix-only). Used by `edit-token` when no wrapper name is given. Deliberately stdlib-only, no new dependency — consistent with "No Rich/Typer" below. The raw-keypress reader is a separate function passed in as an injectable `read_key` callable, so tests never touch a real TTY.
- **`services/wrappers.py`** — the core. `WrapperSpec` (frozen dataclass) + the `WRAPPERS` registry (`deepseek`, `glm`, `glm-ollama`). `render_script` (pure), `install_wrapper` / `is_installed` / `list_wrappers`. `install_wrapper` writes `~/.local/bin/<name>` unconditionally by name — no ownership marker, no "foreign file" guard; whatever was there before (helper-generated or hand-written) is replaced.
- **`services/secrets.py`** — resolves a wrapper's token: env var first, hidden `getpass` prompt otherwise. No persistent config file — the token lives only inside the generated script. **Not used by `edit-token`**: that command always prompts via `getpass` directly, because `resolve_token`'s env-var-wins behavior would otherwise silently ignore whatever the user types when the env var (e.g. `ZAI_API_KEY`) happens to already be set.
- **`services/paths.py`** — frozen `Paths` dataclass, one field (`bin_dir = ~/.local/bin`). `Paths.from_home(home)` is pure path arithmetic; `Paths.default()` wraps `Path.home()`.
- **`backends/_atomic.py`** — the only code that touches the filesystem. `atomic_write(path, data, mode)`: temp-sibling + fsync + `os.replace`, crash-safe.
- **`errors.py`** — `CodeHelperError` lives here (not in `__main__`) to avoid a class-identity split under `python -m`.

## Key Constraints

- **Every wrapper is DATA** (a `WrapperSpec` in `WRAPPERS`), never a hand-edited string — adding a new backend is a registry entry, not new logic.
- **Two wrapper shapes**, discriminated by `launch_command`: the **env-var shape** (`launch_command is None`, the original) exports `ANTHROPIC_*` and runs `claude "$@"`; the **command shape** (`launch_command` set, with a `{model}` placeholder) `exec`s a provider's own launcher (`ollama launch claude --model {model} -- "$@"`) and skips the `ANTHROPIC_*` exports — that launcher sets them itself. The `--` before `"$@"` is required: without it, the launcher's own flag parser consumes forwarded Claude flags (e.g. `-p`) instead of passing them through. `render_script` branches on `launch_command is not None`; the env-var path is unchanged.
- **The `auth` field** (`"literal"` vs `"secret"`) is what keeps this generic: a literal token (e.g. Ollama's local daemon accepts `"ollama"`) needs no prompt and gets mode `0o755`; a secret token is resolved interactively/from env and gets mode `0o700` (owner-only — the script carries it in plain text).
- **Shell-injection defense**: every value interpolated into a generated script — token, base URL, model names — goes through `_shell_single_quote` (POSIX `'` → `'"'"'`). This matters because `--model` makes the model name user-controlled input, not just a registry constant.
- **No ownership guard**: `install_wrapper` overwrites `~/.local/bin/<name>` unconditionally, by name only — there is no "remove" command and no protection against clobbering a pre-existing, non-code-helper-managed file. Validation is limited to the wrapper `name` itself (`get_spec` rejects unknown names before any prompt or write).
- **No Rich/Typer** — plain text, argparse (stdlib) only.
- **`--dry-run`** never writes any file — prints what would happen instead.

## Testing

- **Markers**: `@pytest.mark.unit` / `.integration`. `--strict-markers` in `addopts` turns a typo'd marker into a hard collection error.
- **HOME isolation**: `conftest.py`'s autouse `_isolate_home` fixture sets `HOME=tmp_path` for every test. `Paths.from_home(tmp_path)` is the test seam; production code only ever calls `Paths.default()`.
- Every install/render behavior in `services/wrappers.py` is pinned in `tests/test_wrappers.py`, including two shell-injection regression tests (token and `--model`).
