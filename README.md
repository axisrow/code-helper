# code-helper

Generates and manages small bash wrapper scripts in `~/.local/bin` that point a
coding agent — [Claude Code](https://claude.ai/code) or
[Codex](https://developers.openai.com/codex/cli/) — at a model backend of your
choice, without touching either tool's own configuration — except the
explicit `set-default` command below, which patches Codex's own default and
always backs it up first.

A wrapper is a point in three axes: **agent × provider × model**, installed
under a name you pick. Running `glm-5-codex` instead of `codex` transparently
routes that session to the chosen model.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# curated presets
code-helper add deepseek                  # create ~/.local/bin/deepseek
code-helper add glm                       # prompts for ZAI_API_KEY (or reads it from env)
code-helper add deepseek --model X        # override the preset's model

# build your own: agent + provider + model
code-helper add --agent codex  --provider ollama --model glm-5:cloud
                                          # -> ~/.local/bin/glm-5-codex
code-helper add --agent claude --provider ollama --model qwen3.5:9b \
                --alias qwen              # -> ~/.local/bin/qwen

# discover what's available
code-helper list                          # wrappers + install state
code-helper list agents                   # agents and how each can be configured
code-helper list providers                # backends and what they speak
code-helper list matrix                   # which agent × provider pairings work
code-helper add --agent codex --provider ollama --list-models

code-helper --dry-run add deepseek        # preview, write nothing
code-helper edit-token glm                # rotate glm's token (ignores ZAI_API_KEY)
code-helper                               # arrow-key menu over all of the above

# change what a bare `codex` (no wrapper) runs by default
code-helper set-default --agent codex --provider ollama --model glm-5.2:cloud
code-helper set-default --restore         # undo the last set-default
```

The alias defaults to `<model>-<agent>` (`glm-5:cloud` + `codex` →
`glm-5-codex`); `--alias` overrides it.

## Presets

| Name | Agent | Backend | Auth |
|---|---|---|---|
| `deepseek` | Claude Code | local Ollama daemon, `deepseek-v4-flash:0731-cloud` | literal token (`ollama`) |
| `glm` | Claude Code | Z.ai (`https://api.z.ai/api/anthropic`) | secret (`ZAI_API_KEY`) |
| `glm-ollama` | Claude Code | `ollama launch claude --model glm-5.2:cloud` | none — `ollama launch` authenticates itself |

Presets exist alongside the constructor because some combinations aren't just
"an agent and a model": `glm` uses a *different model per tier*
(`glm-4.7` / `glm-5-turbo` / `glm-5.2[1m]`), which a single `--model` can't
express.

## Not every combination is possible

`code-helper list matrix` shows which pairings exist and refuses the rest with
an explanation rather than generating a wrapper that fails at runtime:

```
        ollama         zai
claude  anthropic-env  anthropic-env
codex   ollama-launch  —
```

Codex can't speak the Anthropic protocol Z.ai serves, so that cell is empty.
Compatibility is derived from what each side supports, not from a hand-written
list — a new provider can't silently introduce a broken combination.

## `set-default`

A wrapper solves "run *this session* on another model." `set-default` solves a
different problem: making a bare `codex`, with no wrapper and no `--profile`,
start on the backend you chose. It does this by patching Codex's own
`~/.codex/config.toml` in place — the top-level `model` / `model_provider` /
`model_catalog_json` keys, and the matching `[model_providers.X]` table.

```bash
code-helper set-default --agent codex --provider ollama --model glm-5.2:cloud
code-helper set-default --agent codex --provider ollama --model glm-5.2:cloud --dry-run
code-helper set-default --restore                # undo, from the newest backup
code-helper set-default --restore --slot 2        # or an older one
```

This is the **one** command that touches `~/.codex/config.toml`. It never uses
a round-trip TOML parser (this project ships no such dependency) — it patches
only the four constructs above with anchored regex, so every other table
(`[projects.*]`, MCP servers, hooks, comments, blank lines) survives
byte-for-byte. Before any real write it verifies the existing file parses as
TOML at all (Python 3.11+), and after patching it re-parses the result and
checks the intended values landed — either check failing refuses the write.

Every real write rotates a 3-slot backup ring first
(`~/.codex/config.toml.bak1` newest, `.bak2`, `.bak3` oldest) — `--restore`
reads one back. Off a TTY, `set-default` refuses without `--force`, same as
`add`.

## Safety

- **Your other executables are protected.** A wrapper carries a marker
  comment, and `code-helper` refuses to overwrite a file it didn't create
  unless you pass `--force` or confirm interactively. In a non-interactive run
  it fails immediately rather than waiting on input.
- **Wrapper names are validated.** A name must be a single, plain path
  component; `../../etc/passwd`, names with spaces, and names matching an agent
  binary (which would make the wrapper call itself forever) are rejected.
- **No shell injection.** Every value interpolated into a generated script
  (token, base URL, model names) is POSIX single-quoted, including values that
  come from user input or an untrusted environment variable.
- **Secrets stay put.** A token lives only inside the generated script, which
  is written `0o700` when it carries a real credential. There is no config
  file — except `~/.codex/config.toml`, and only through the explicit
  `set-default` command, which patches only its own managed keys there, backs
  up before every real write, and never carries a secret into it.
- `--dry-run` writes nothing. There is no `remove` command — delete a wrapper
  script yourself if you no longer want it.

## Development

```bash
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format
```
