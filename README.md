# code-helper

Generates and manages small bash wrapper scripts in `~/.local/bin` that point a
coding agent — [Claude Code](https://claude.ai/code) or
[Codex](https://developers.openai.com/codex/cli/) — at a model backend of your
choice, without touching either tool's own configuration.

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
  is written `0o700` when it carries a real credential. There is no config file.
- `--dry-run` writes nothing. There is no `remove` command — delete a wrapper
  script yourself if you no longer want it.

## Development

```bash
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format
```
