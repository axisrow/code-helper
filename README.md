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

# a provider whose endpoint YOU choose (a self-hosted LiteLLM proxy) needs
# --base-url — see the "LiteLLM" section below
code-helper add --agent claude --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
code-helper add --agent codex  --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o

# discover what's available
code-helper list                          # wrappers + install state
code-helper list agents                   # agents and how each can be configured
code-helper list providers                # backends and what they speak
code-helper list matrix                   # which agent × provider pairings work
code-helper add --agent codex --provider ollama --list-models

code-helper --dry-run add deepseek        # preview, write nothing
code-helper edit-token glm                # rotate glm's token (always prompts)
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
express. There is no `litellm` preset — a preset bundles a fixed provider
address, and `litellm`'s whole point is that its address is yours, not this
project's to bundle; use the constructor with `--base-url` instead (see
below).

## Not every combination is possible

`code-helper list matrix` shows which pairings exist and refuses the rest with
an explanation rather than generating a wrapper that fails at runtime:

```
        ollama         zai            litellm
claude  anthropic-env  anthropic-env  anthropic-env
codex   openai-toml    —              openai-toml
```

Codex can't speak the Anthropic protocol Z.ai serves, so that cell is empty.
Compatibility is derived from what each side supports, not from a hand-written
list — a new provider can't silently introduce a broken combination. `litellm`
declares both mechanisms (a LiteLLM proxy serves both protocols off one host),
so neither of its cells is empty.

## LiteLLM

`litellm` is the one provider whose address is not built into `code-helper` —
it's your own [LiteLLM](https://docs.litellm.ai/) proxy, so you supply its URL
yourself:

```bash
code-helper add --agent claude --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
code-helper add --agent codex  --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
```

**The URL must include `/v1`** (`http://localhost:4000/v1`, not just
`http://localhost:4000`) — `--list-models` requests `{url}/models`, and the
same value feeds the Codex TOML profile's `base_url`, which expects an
OpenAI-style `/v1` root. One URL serves both agents: `claude` resolves to the
direct `ANTHROPIC_*` env shape (LiteLLM's Anthropic Messages passthrough),
`codex` resolves to a `[model_providers.litellm]` TOML profile — chosen
automatically, same as every other provider here.

The token comes from `LITELLM_API_KEY` (env, a cached credential, or a hidden
prompt — see [Where tokens live](#where-tokens-live)) exactly like any other
secret-auth provider. For `codex`, which has no environment-variable config of
its own, the wrapper `export`s `LITELLM_API_KEY` right before launching
`codex` — so the credential is visible to `codex` and everything it spawns
(including MCP servers it starts), which is the mechanism, not a leak.

`set-default --provider litellm` also works, but **requires `--base-url`** —
without it the command would patch `~/.codex/config.toml` with a malformed
URL and its own verification couldn't catch it (both sides of the check would
be wrong the same way). It also does not export the token anywhere: there is
no wrapper script for a bare `codex` to source a variable from, so set the
`LITELLM_API_KEY` environment variable in your own shell profile if you use
this path.

### Fallback (429, provider outages) is LiteLLM's job, not code-helper's

`code-helper` generates one wrapper for one `agent × provider × model` point,
on purpose — it does not retry, does not race multiple backends, and will not
grow that logic. A LiteLLM proxy already does this well, in its own
`config.yaml`:

```yaml
model_list:
  - model_name: primary
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: backup
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  num_retries: 3
  fallbacks: [{"primary": ["backup"]}]
  allowed_fails: 3
  cooldown_time: 30
```

Point a wrapper at `primary` — `code-helper add ... --model primary` — and a
429 (or any failure LiteLLM is configured to catch) fails over to `backup`
transparently; the agent never sees the switch. Check the
[LiteLLM reliability docs](https://docs.litellm.ai/docs/proxy/reliability) for
the current config keys, since this is LiteLLM's surface, not this project's.

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
- **Secrets stay put.** A token lives inside the generated script, which is
  written `0o700` when it carries a real credential — that copy is what an
  installed wrapper actually runs on, and it is never affected by anything
  below. A token typed at a prompt is also cached in
  `~/.config/code-helper/credentials.json` (`0o600`) so the next `add` or
  `--list-models` doesn't ask again — see
  [Where tokens live](#where-tokens-live). The only agent config file this
  tool ever touches is `~/.codex/config.toml`, and only through the explicit
  `set-default` command, which patches only its own managed keys there, backs
  up before every real write, and never carries a secret into it.
- **A secret-auth Codex wrapper (e.g. `codex × litellm`) exports its token to
  the process it launches.** Codex has no equivalent of `ANTHROPIC_BASE_URL`,
  so the only way to hand it a credential is an environment variable named in
  its own TOML profile (`env_key`) — the wrapper `export`s that variable
  immediately before `exec`ing `codex`. That variable is visible to `codex`
  and every child process it spawns, including MCP servers it starts; this is
  how the credential reaches Codex at all, not an oversight.
- `--dry-run` writes nothing. There is no `remove` command — delete a wrapper
  script yourself if you no longer want it.

## Where tokens live

For a secret-auth provider (`zai`, `litellm`), a token is resolved in this
order every time one is needed:

1. **Environment variable** (`ZAI_API_KEY`, `LITELLM_API_KEY`, ...) — wins over
   everything, so a headless/CI run can always override.
2. **Cached credential** — `~/.config/code-helper/credentials.json` (`0o600`,
   owner-only). A token typed at an `add` prompt is written here so the *next*
   `add` or `--list-models` doesn't ask again.
3. **Hidden prompt** — asked only when neither of the above has it.

This file is a **cache of a value you typed**, not a session with the
provider: `code-helper` configures agents and aliases, it does not log in
anywhere. There is no "logged in" state and no command that connects to a
provider to validate a token. Deleting the file does not break any installed
wrapper — each wrapper carries its own token baked into the script itself
(`0o700`); the cache only means the next install/discovery prompts again.

`edit-token` never reads the cache — it always prompts via a hidden input,
because rotating a token should never silently return the value you're trying
to replace. The newly typed value is cached after a successful rotation, so
the cache stays in step.

`--list-models` discovery follows the same env → cache lookup (never the
prompt — an optional listing must never block a script waiting on stdin); with
neither available it falls back to an unauthenticated request, same as before
this cache existed.

## Development

```bash
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format
```
