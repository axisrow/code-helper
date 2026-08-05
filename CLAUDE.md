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

- **`cli/parser.py`** — argparse tree (`list` / `add <name> [--model]` / `edit-token [<name>]` / `tui`, plus bare `code-helper` defaulting to `tui`). Handlers are thin shells: resolve `Paths.default()`, delegate to a service, let `CodeHelperError` propagate to `__main__.main()`.
- **`cli/menu.py`** — a ~90-line hand-rolled arrow-key terminal menu (`select_from_menu`, raw `termios`/`tty` reading, Unix-only). Used by `edit-token` when no wrapper name is given, and by `cli/tui.py`. Deliberately stdlib-only, no new dependency — consistent with "No Rich/Typer" below. The raw-keypress reader is a separate function passed in as an injectable `read_key` callable, so tests never touch a real TTY. `press_any_key(prompt)` is the public "block for one keypress" primitive (TTY-gated, no-op off a TTY) used by the TUI's post-command pause. On a real TTY the menu redraws *in place* each keypress (cursor-up + clear-to-end ANSI escapes), so navigation doesn't scroll an endless stack of menus; on a non-TTY (pipe/tests with injected `print_fn`) it falls back to reprinting each frame.
- **`cli/tui.py`** — a looping arrow-key menu wrapping the CLI 1-to-1 (issue #3's contract: CLI primary, TUI secondary, no new behavior). `run_tui` mutates the shared `argparse.Namespace` (`name`/`model`/`dry_run`/`debug`) and calls the *same* `_handle_list` / `_handle_add` / `_handle_edit_token` from `cli/parser.py` — it never touches `services/` directly, so TUI and CLI cannot diverge in behavior. The menu loops: `list`/`add`/`edit-token` run then return to the menu; only `quit` (and a top-level Ctrl-C/`q`) exits, so `run_tui` always returns 0. A `CodeHelperError` from `add`/`edit-token` is printed to stderr and the menu reappears (re-raised under `--debug`, mirroring `__main__.main`); cancelling a sub-menu returns to the main menu. `--model` is a plain `input()` prompt; `--dry-run`/`--debug` are toggle menu items that redraw instead of running a command. After a command runs, the screen clears and the menu re-renders at the top — but only after a keypress (`[press any key to return to the menu]`), so the command's output stays readable instead of being wiped instantly; on a non-TTY the pause is a no-op so tests never block. Both `select_from_menu` and `input` are looked up at call time (not bound as function defaults) so tests can `monkeypatch.setattr` them — a `= input` default parameter would bind the builtin at def-time and silently ignore the patch.
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
- **TUI is a mirror, not a second implementation**: every TUI menu item must dispatch into an existing `_handle_*` with the same arguments the CLI subcommand would build — no service calls, no duplicated validation (e.g. `edit-token`'s secret-wrapper filter lives once, in `_handle_edit_token`).

## Testing

- **Markers**: `@pytest.mark.unit` / `.integration`. `--strict-markers` in `addopts` turns a typo'd marker into a hard collection error.
- **HOME isolation**: `conftest.py`'s autouse `_isolate_home` fixture sets `HOME=tmp_path` for every test. `Paths.from_home(tmp_path)` is the test seam; production code only ever calls `Paths.default()`.
- Every install/render behavior in `services/wrappers.py` is pinned in `tests/test_wrappers.py`, including two shell-injection regression tests (token and `--model`).
- **`tests/test_tui.py`** — the TUI path is exercised through `main(["tui"])` / `main([])`, patching `code_helper.cli.menu.select_from_menu` and `builtins.input` (both looked up at call time inside `cli/tui.py`, so the patch reaches them). Because the menu loops, every command test's menu sequence ends in `"quit"` (the loop would otherwise re-prompt once the fake is exhausted). One test (`test_tui_add_installs_same_as_cli`) installs via TUI and via CLI into two separate `HOME`s and asserts the generated script bodies are byte-identical — the actual proof of the 1-to-1 contract; `test_tui_loops_after_command` runs two commands before `quit` to prove the loop, and `test_tui_error_returns_to_menu` patches `resolve_token` to raise and asserts the error is printed and the menu reappears rather than exiting.
