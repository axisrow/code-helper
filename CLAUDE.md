# CLAUDE.md

Guidance for Claude Code when working in this repository.

`code-helper` — pip-installable Python CLI that generates bash wrapper scripts in `~/.local/bin`, each pointing a coding agent (Claude Code, Codex) at a model backend (Ollama, Z.ai, LiteLLM, …) under a chosen name. A wrapper is a resolved point in **agent × provider × model**.

## Package Manager

- **pip** (editable install): `pip install -e ".[dev]"`

## Commands

| Task | Command |
|------|---------|
| Install (editable) + dev tools | `pip install -e ".[dev]"` |
| Run tests | `pytest -q` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| CLI entry points | `code-helper add [<preset>] \| add --agent A --provider P --model M \| list [wrappers\|agents\|providers\|matrix] \| edit-token [<name>] \| set-default [...] \| tui` |

## Architecture

| Layer | File | Responsibility |
|-------|------|-----------------|
| CLI | `cli/parser.py` | argparse tree; thin handlers delegating to `services/` |
| CLI | `cli/menu.py` | stdlib-only arrow-key terminal menu (`termios`/`tty`/`select`) |
| CLI | `cli/tui.py` | looping menu wrapping the CLI 1:1, no new behavior |
| Service | `services/model.py` | `Agent`/`Provider`/`ConfigShape` axes; compatibility = shape-set intersection |
| Service | `services/spec.py` | `WrapperSpec` construction (`build_spec`, `spec_from_preset`) |
| Service | `services/render.py` | one renderer per `ConfigShape`, dispatched on `resolve_shape`'s result |
| Service | `services/naming.py` | `validate_alias` allow-list for executable names on `PATH` |
| Service | `services/models_api.py` | `list_models(provider)` — never raises, degrades to `error` |
| Service | `services/wrappers.py` | generated-script lifecycle: install/list/describe/ownership guard |
| Service | `services/codex_default.py` | `set-default` — patches `~/.codex/config.toml` in place |
| Service | `services/secrets.py` | token resolution: env → cached profile → prompt |
| Service | `services/paths.py` | frozen `Paths` dataclass, pure path arithmetic off `home` |
| Service | `services/state.py` | active token-profile pointer (`state.json`) |
| Backend | `backends/_atomic.py` | only code that touches the filesystem (atomic write) |
| Root | `errors.py` | `CodeHelperError` |
| Root | `__main__.py` | dispatch + `CodeHelperError`/`MenuCancelled` → exit-code contract |

## External References

| Need | File |
|------|------|
| Test suite map | `tests/` (one file per service, named `test_<module>.py`) |

## Key Conventions

- Agents and providers are **data** (registry entries) — never `if agent == ... and provider == ...` branches; compatibility is computed from `ConfigShape` set intersection.
- Runtime `base_url` and `auth` overrides are also data, via `BaseUrlPolicy`/`AuthPolicy` — substituted once (`with_base_url`, `with_auth`) before `build_spec` runs, never branched on provider name.
- One renderer per `ConfigShape` in `_RENDERERS`; "incompatible" (`resolve_shape`) and "not implemented yet" (`render_script`) are distinct errors — never collapse them.
- Every value interpolated into a generated script goes through `_shell_single_quote`/`toml_string`; the only bare interpolation is `agent.binary`/`provider.token_env_var`, both guarded by an import-time regex.
- `build_spec` validates compatibility and alias **before** any token prompt — a bad combination must never trigger interactive auth.
- Every generated script carries an ownership marker (`# code-helper: managed wrapper …`); overwriting a foreign file requires `--force` or interactive confirm; `--dry-run` never prompts or writes.
- `--dry-run` never writes any file.
- TUI is a mirror of the CLI, not a second implementation — every menu item dispatches into the same `_handle_*` functions; no duplicated validation or state-changing service calls.
- `--provider` always selects constructor mode; `add <name>` is always a preset. Never guess between them.
- `set-default` is the only command allowed to touch an agent's own config file (`~/.codex/config.toml`), and only via patch (never replace), with backups and `tomllib` verification.
- `OPENAI_TOML` is the extension point for new OpenAI-compatible providers — adding one is a `PROVIDERS` entry, no code changes.

## Testing

- Markers: `@pytest.mark.unit` / `.integration`, `--strict-markers` enforced.
- `HOME` isolation via `conftest.py`'s autouse `_isolate_home` fixture; production code only calls `Paths.default()`, tests use `Paths.from_home(tmp_path)`.
- One test file per service module (`tests/test_<module>.py`).
- Manual PTY verification (`pexpect`) is required for ANSI redraw/hang behavior — the injected-`read_key` test suite cannot see real terminal-driver bugs.
