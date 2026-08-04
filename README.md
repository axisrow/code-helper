# code-helper

Generates and manages small bash wrapper scripts in `~/.local/bin` that point
[Claude Code](https://claude.ai/code) at a different model backend — a local
Ollama daemon, Z.ai, or any other Anthropic-API-compatible endpoint — without
touching Claude Code's own configuration.

Each wrapper exports the `ANTHROPIC_*` environment variables Claude Code reads
for its model tiers, then execs `claude "$@"`. Running `deepseek` instead of
`claude` transparently routes every request to that provider.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
code-helper list                     # show the wrapper registry + install state
code-helper add deepseek             # create ~/.local/bin/deepseek
code-helper add deepseek --model X   # override the default model
code-helper add glm                  # prompts for ZAI_API_KEY (or reads it from env)
code-helper remove deepseek          # remove, only if code-helper-managed
code-helper --dry-run add deepseek   # preview, write nothing
```

## Wrappers in the registry

| Name | Backend | Auth |
|---|---|---|
| `deepseek` | local Ollama daemon (`http://127.0.0.1:11434`), `deepseek-v4-flash:0731-cloud` | literal token (`ollama`) |
| `glm` | Z.ai (`https://api.z.ai/api/anthropic`) | secret (`ZAI_API_KEY`) |

## Safety

- A wrapper this tool did not create (no `# code-helper managed` marker) is
  **never** overwritten or deleted — `add`/`remove` refuse and leave it
  untouched.
- Every value interpolated into a generated script (token, base URL, model
  names) is POSIX single-quoted to prevent shell injection, even when the
  value comes from user input (`--model`) or an untrusted environment
  variable.
- `--dry-run` writes nothing.

## Development

```bash
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format
```
