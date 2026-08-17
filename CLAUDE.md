# CLAUDE.md

Guidance for Claude Code when working in this repository.

`codehelper` — pip-installable Python CLI that generates bash wrapper scripts in `~/.local/bin`, each pointing a coding agent (Claude Code, Codex, and every other CLI integration `ollama launch` supports — OpenCode, Copilot CLI, Droid, Cline, …) at a model backend (Ollama, Z.ai, LiteLLM, …) under a chosen name. A wrapper is a resolved point in **agent × provider × model**.

## Package Manager

- **pip** (editable install): `pip install -e ".[dev]"`

## Commands

| Task | Command |
|------|---------|
| Install (editable) + dev tools | `pip install -e ".[dev]"` |
| Run tests | `pytest -q` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| CLI entry points | `codehelper add [<preset>] \| add --agent A --provider P --model M \| list [wrappers\|agents\|providers\|matrix] \| edit-token [<name>] \| set-default [...] \| switch [<provider>\|--from-wrapper NAME] \| tui` |

## Architecture

| Layer | File | Responsibility |
|-------|------|-----------------|
| CLI | `cli/parser.py` | argparse tree; thin handlers delegating to `services/` |
| CLI | `cli/menu.py` | stdlib-only arrow-key terminal menu (`termios`/`tty`/`select`) |
| CLI | `cli/tui.py` | looping menu wrapping the CLI 1:1, no new behavior |
| Service | `services/model.py` | `Agent`/`Provider`/`ConfigShape` axes; compatibility = shape-set intersection |
| Service | `services/agents.py` | user-defined agents (`agents.json`, merged with `AGENTS`); `all_agents`/`get_agent` are the merge-aware lookups every CLI/TUI call site must use for a name that could be user-defined |
| Service | `services/spec.py` | `WrapperSpec` construction (`build_spec`, `spec_from_preset`) |
| Service | `services/render.py` | one renderer per `ConfigShape`, dispatched on `resolve_shape`'s result |
| Service | `services/naming.py` | `validate_alias` allow-list for executable names on `PATH` |
| Service | `services/models_api.py` | `list_models(provider)` — never raises, degrades to `error` |
| Service | `services/wrappers.py` | generated-script lifecycle: install/list/describe/ownership guard |
| Service | `services/codex_default.py` | `set-default` — patches `~/.codex/config.toml` in place; `current_default` reads back what is applied, `clear_default` removes the managed region (codex's "native") |
| Service | `services/claude_settings.py` | `switch` — live-patches `~/.claude/settings.json`'s `env` block so an already-running `claude` picks up a new backend on its next prompt, no restart |
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
- Every generated script carries an ownership marker (`# codehelper: managed wrapper …`); overwriting a foreign file requires `--force` or interactive confirm; `--dry-run` never prompts or writes.
- `--dry-run` never writes any file.
- TUI is a mirror of the CLI, not a second implementation — every menu item dispatches into the same `_handle_*` functions; no duplicated validation or state-changing service calls.
- The TUI main screen is a **chipset**: one selectable row per agent, each a horizontal strip of `native` + that agent's installed wrappers. Up/Down moves rows, Left/Right (and Shift+Tab) moves the chip cursor with NO I/O, Enter applies. Three chip states are distinct: `[applied]`, `<highlighted>`, `( plain )`.
- Per-agent differences (how "applied" is read, how a chip is applied, when it takes effect) live in `cli/tui.py`'s `_AGENT_BACKENDS` table — never as an `if agent.name == ...` branch. Its apply hooks are METHOD NAMES resolved via `getattr` at apply time, not captured functions, so patching a method actually takes effect. An agent with a real live-patchable config (`claude`, `codex`) needs one `AGENTS` entry + one table entry to get a chipset row. A **launch-only** agent (any `ollama launch`-only CLI — OpenCode, Droid, Cline, …) needs only the `AGENTS` entry: `ollama launch` writes that agent's own native config in its own format, which this project does not parse, so it correctly gets no `_AGENT_BACKENDS` entry and no chipset row — `add`/`list`/`remove` still work for it in full, only the chipset's live-toggle convenience is unavailable. This is the honest degradation the `_AgentBackend` docstring describes, not a gap to fill reflexively.
- A chip is an already-installed WRAPPER, never a bare provider: model, token and base URL were resolved when the wrapper was created, which is what lets Enter apply with no prompts. Creating a pairing is `add`; switching between them is the chipset. One exception: every row also carries a trailing `+ add` ACTION chip (`_ADD_CHIP`) — visually distinct (dim, never `✓`/bold), never "applied", Enter on it opens `add` pre-scoped to that row's agent. It exists so an agent with zero wrappers (e.g. a fresh `codex` row) still shows how to get its first one instead of reading as empty/broken. The no-prompt guarantee is about *backend* chips only.
- `add` is **agent-first**: `a` (or the `+ add` chip) on a chipset row skips straight to a provider list scoped to that row's agent (`compatible_providers(agent)`); `a` anywhere else opens a "Wrapper or Agent?" choice, then — for Wrapper — which agent, before any provider is shown. Filtering providers against a fixed agent (as the old claude-only `_add_provider_choices` did) would make a provider only some agent supports silently unreachable in the TUI; agent-first avoids that structurally rather than by convention.
- Anything a label callable reads MUST be cached and refreshed once per main-loop iteration (`_refresh_active_label`, or `_refresh_profile_label` for profile-only state) — the menu re-evaluates labels on every redraw frame, including pure cursor movement.
- Every key bound in a screen's `on_key` MUST also be in `menu._PASSTHROUGH`, or it is silently dead: `_translate_char` reports it as `OTHER` and the binding never fires. This shipped once (`w`/switch was bound, documented, and non-functional); `test_tui.py` now pins the main screen's bindings against `_translate_char`.
- `--provider` always selects constructor mode; `add <name>` is always a preset. Never guess between them.
- `set-default` and `switch` are the only commands allowed to touch an agent's own config file — `~/.codex/config.toml` and `~/.claude/settings.json` respectively — and only via patch (never replace), with rotating backups and post-write verification (`tomllib` for TOML, a re-parse + managed-region diff for JSON).
- `set-default` persists a **preference** applied at the agent's next launch; `switch` changes what an **already-running** `claude` does on its next prompt (Claude Code re-reads `settings.json` between prompts) — they patch different files for different lifecycles, never conflate the two. `_AgentBackend.lifecycle` (`live` / `next launch`) still carries that distinction as data, but the chipset row does NOT render it — deliberately dropped and pinned by `test_chipset_row_has_no_lifecycle_tail`, pending verification of how codex actually reads `config.toml` before the claim is asserted in the UI again.
- "native" means *clear the override*, and each agent clears its own way: claude's `env_reset` provider (`switch native`) writes empty values for every managed `env` key so Claude Code's live watcher resets the running session; codex removes the managed region from `config.toml` (`clear_default`). Deliberately NOT `--restore`, which rolls back to a backup snapshot and would undo unrelated hand-edits.
- `ConfigShape.ANTHROPIC_SETTINGS` (the `switch` mechanism) is declared by providers only, never by any `Agent` — that is what guarantees adding it cannot perturb `resolve_shape`/wrapper generation for the `ANTHROPIC_ENV`/`OPENAI_TOML`/`OLLAMA_LAUNCH` shapes. Compatibility for `switch` is computed by `claude_settings`/`model.switchable_providers`, a separate resolver from `resolve_shape`.
- A provider that means "clear the override, restore the agent's native behaviour" (e.g. `native`) is data too — `Provider.env_reset: bool`, never a name check. `_validate_provider` enforces that an `env_reset` provider carries no address and no credential.
- `OPENAI_TOML` is the extension point for new OpenAI-compatible providers — adding one is a `PROVIDERS` entry, no code changes.
- `AGENTS` (`model.py`) stays a frozen, built-in-only, pure registry — it is never mutated or extended at runtime. A user-defined agent (the `a` → "Agent" branch) is a genuinely separate storage layer, `services/agents.py`, persisted as JSON in the config dir and merged with `AGENTS` at read time (`all_agents`/`get_agent`), built-ins always first so a lookup can never resolve to a user entry when a built-in of the same name exists. A user agent is always `OLLAMA_LAUNCH`-only — the shape every non-`claude`/`codex` built-in agent already uses — since `ANTHROPIC_ENV`/`OPENAI_TOML` need a renderer contract this module has no way to supply. Every CLI/TUI call site that resolves an agent name that COULD be user-defined (`add`, `list agents`, `list matrix`) must use the merged lookup, not `model.get_agent`/`model.AGENTS` directly; `set-default` is the one deliberate exception (codex-only, a built-in with a real live-patchable config). A user agent's `name`/`binary` go through `model.validate_agent_binary` — the exact same regex the built-in registry itself is checked against at import time — because `agent.binary` is interpolated UNQUOTED into a generated script; skipping this is a shell-injection hole, not a cosmetic gap. A user agent may not shadow a built-in name/binary, an existing user agent, or anything in `naming.RESERVED_ALIASES` — rejected outright, never silently overridden. Same honest degradation as any other launch-only agent: full `add`/`list`/`remove` support, no chipset row (no live-patchable config this project knows how to read for it).
- `Provider.known_models` is the registry-declared FALLBACK model list, consulted only when `models_api.list_models` returns none (structurally, e.g. `zai`'s `model_list_api=NONE`, or transiently, e.g. the daemon being down) — discovery is always tried first and stays the source of truth, this is never used to override a real result. Selection happens in the TUI (`_choose_add_model`), not in `models_api` — that module's "never raises, returns a result" contract stays untouched, mirroring how `resolve_shape` vs `render_script` keep "incompatible" and "not implemented" as distinct questions. The prompt text must say which source fed the menu ("known models — discovery unavailable" vs a plain discovery prompt), so a stale built-in entry is never mistaken for something the endpoint just confirmed.

## Testing

- Markers: `@pytest.mark.unit` / `.integration`, `--strict-markers` enforced.
- `HOME` isolation via `conftest.py`'s autouse `_isolate_home` fixture; production code only calls `Paths.default()`, tests use `Paths.from_home(tmp_path)`.
- One test file per service module (`tests/test_<module>.py`).
- Manual PTY verification (`pexpect`) is required for ANSI redraw/hang behavior — the injected-`read_key` test suite cannot see real terminal-driver bugs.
