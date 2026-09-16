# codehelper

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
codehelper add deepseek-ollama            # create ~/.local/bin/deepseek-ollama
codehelper add glm                       # prompts for ZAI_API_KEY (or reads it from env)
codehelper add deepseek-ollama --model X  # override the preset's model

# cloud DeepSeek API: claude via its Anthropic-compatible surface, codex via
# its OpenAI-compatible surface — both need DEEPSEEK_API_KEY
codehelper add --agent claude --provider deepseek        --model deepseek-v4-flash
codehelper add --agent codex  --provider deepseek-openai --model deepseek-v4-flash

# B.AI (b.ai): one host serves both agents — Anthropic-compatible /v1/messages
# for claude, OpenAI /v1/responses for codex — see the "B.AI" section below
codehelper add bai                                          # preset: claude -> qwen3.8-flash
codehelper add --agent codex  --provider bai --model gpt-5.5

# local FreeLLMAPI proxy: same one-host-both-protocols deal at a fixed
# loopback address; model ids come from its own /v1/models (the list rotates)
codehelper add --agent claude --provider freellmapi --model qwen3.8-flash
codehelper add --agent codex  --provider freellmapi --model auto

# build your own: agent + provider + model
codehelper add --agent codex  --provider ollama-direct --model glm-5:cloud
                                          # -> ~/.local/bin/glm-5-codex
codehelper add --agent claude --provider ollama-direct --model qwen3.5:9b \
                --alias qwen              # -> ~/.local/bin/qwen

# a provider whose endpoint YOU choose (a self-hosted LiteLLM proxy) needs
# --base-url — see the "LiteLLM" section below
codehelper add --agent claude --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
codehelper add --agent codex  --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o

# discover what's available
codehelper list                          # wrappers + install state
codehelper list agents                   # agents and how each can be configured
codehelper list providers                # backends and what they speak
codehelper list matrix                   # which agent × provider pairings work
codehelper add --agent codex --provider ollama-direct --list-models

codehelper --dry-run add deepseek-ollama  # preview, write nothing
codehelper add glm --profile work        # use a named token profile
codehelper edit-token glm --profile work # rotate the selected profile
codehelper edit-token glm                # choose a profile and rotate it
codehelper tokens                        # which keys are stored (masked head+tail)
codehelper tokens --reveal               # full values (careful: not for pasting)
codehelper                               # arrow-key menu over all of the above

# change what a bare `codex` (no wrapper) runs by default
codehelper set-default --agent codex --provider ollama-direct --model glm-5.2:cloud
codehelper set-default --restore         # undo the last set-default

# retire a provider for a while (subscription ended, daemon uninstalled):
# delete its wrappers and hide it everywhere — list/add/switch/TUI chips/tokens
codehelper disable ollama --yes          # legacy spellings accepted ('ollama' = ollama-direct)
codehelper enable ollama                 # lift it; wrappers are NOT restored — `add` recreates them
```

The alias defaults to `<model>-<agent>` (`glm-5:cloud` + `codex` →
`glm-5-codex`); `--alias` overrides it.

## Interactive menu

Running `codehelper` opens an English arrow-key menu with `List`, `Add`,
`Settings`, and `Quit`. `Add` follows one path:

`provider → profile/token → model → compatible agent → command name`

Providers with literal authentication skip the profile step; LiteLLM asks for
its base URL before it selects a profile and discovers models. `List` is a
second-level wrapper browser with `Back`; choosing a secret-backed wrapper
rotates one of its token profiles. The menu never stops on a separate
"press any key" screen. Scriptable commands, including `edit-token`, remain
unchanged.

For a secret profile, the suggested command name includes both the profile
and the agent: choosing `axisrow` for model `glm` on `claude` suggests
`glm-axisrow-claude`, so two agents sharing the same model and profile never
collide on one wrapper name. The installed wrapper records the selected
profile and `codehelper list` shows it, so each alias has an explicit
token-profile association.

## Presets

| Name | Agent | Backend | Auth |
|---|---|---|---|
| `deepseek-ollama` | Claude Code | local Ollama daemon, `deepseek-v4-flash:0731-cloud` | literal token (`ollama`) |
| `glm` | Claude Code | Z.ai (`https://api.z.ai/api/anthropic`), `glm-5.3` | secret (`ZAI_API_KEY`) |
| `glm-ollama` | Claude Code | `ollama launch claude --model glm-5.2:cloud` | none — `ollama launch` authenticates itself |
| `gemini-litellm` | Claude Code | LiteLLM proxy (see [Google Gemini via LiteLLM](#google-gemini-via-litellm)), `gemini-3.7-flash` | secret (`LITELLM_API_KEY`) |
| `bai` | Claude Code | B.AI (`https://api.b.ai`), `qwen3.8-flash` (see [B.AI](#bai)) | secret (`BAI_API_KEY`) |

There is no preset for the cloud DeepSeek API (`deepseek`/`deepseek-openai`
providers) — use the constructor, as shown above.

Presets exist alongside the constructor because they pin a curated model
choice, not just "an agent and a provider". There is no general `litellm`
preset — a preset bundles a fixed provider address, and `litellm`'s whole
point is that its address is yours, not this project's to bundle; use the
constructor with `--base-url` instead (see below). `gemini-litellm` is the
one exception: it exists precisely to bundle one concrete LiteLLM instance's
address, and `--base-url` retargets it to yours. `bai` sits on the opposite
pole: its address is fixed registry data, so the preset bundles no URL and
`--base-url` is refused.

## Google Gemini via LiteLLM

`gemini-litellm` gets Google's models into Claude Code and Codex — through a
translating proxy, because **Google serves neither protocol the agents
speak**: its OpenAI-compatible endpoint is chat-completions only (no
`/responses`, which is all current Codex accepts), and there is no
Anthropic-compatible endpoint at all (which is what Claude Code needs). The
conversion is [LiteLLM](https://docs.litellm.ai/)'s job — codehelper only
does the addressing, and never translates formats itself:

```
Claude Code ──(ANTHROPIC_BASE_URL=<litellm>, /v1/messages)──► LiteLLM ──► Google chat/completions
Codex       ──(base_url=<litellm>/v1,       responses)  ──► LiteLLM ──► Google chat/completions
```

```bash
# claude: the preset carries its curated LiteLLM address and model
codehelper add gemini-litellm                # prompts for LITELLM_API_KEY

# point it at a different LiteLLM instance
codehelper add gemini-litellm --base-url https://my-proxy.example.com

# codex: the constructor form over the same proxy
codehelper add --agent codex --provider litellm \
                --base-url https://my-proxy.example.com --model gemini-3.7-flash
```

The proxy's `config.yaml` needs one entry per model, e.g.:

```yaml
model_list:
  - model_name: gemini-3.7-flash
    litellm_params:
      model: gemini/gemini-3.7-flash
      api_key: os.environ/GEMINI_API_KEY
```

**Trust boundary, stated plainly:** the preset embeds a concrete proxy address,
so `add gemini-litellm` resolves `LITELLM_API_KEY` and bakes it into a wrapper
that sends every prompt to that address — press Enter on its chip and an
already-running session is retargeted there too. If you serve several LiteLLM
instances, note that token profiles are keyed by provider name (`litellm`),
not by host: a profile cached for one instance will be offered for the other.
Point the wrapper at a different instance with `--base-url`, and check `list`
or the script itself for the `ANTHROPIC_BASE_URL` line whenever in doubt.

Direct `codex × gemini` (and `claude × gemini` without a proxy) is
impossible, not merely unbuilt: the `gemini` provider entry is **suspended**
— `codehelper list providers` marks it, and `add` refuses the pairing with
the normal "no common configuration mechanism" error instead of installing a
wrapper that cannot work. If Google ever ships `/v1beta/openai/responses`,
unsuspending is a one-line registry change.

## Parallel sessions: native `claude` + wrappers at the same time

A bare `claude` (no wrapper) talks to Anthropic natively; every wrapper routes
its own session to its backend. They can run **simultaneously** — that is the
point of wrappers — but this needs one precaution the wrappers now carry
automatically:

Since Claude Code 2.0.1, every `env` entry in `~/.claude/settings.json` is
applied into the process environment at startup, **replacing** the value
inherited from the shell — an empty string included. So a leftover
`"ANTHROPIC_BASE_URL": ""` (which `codehelper switch native` deliberately
writes to reset a live session) would silently defeat every wrapper's exports
and launch native instead. Generated wrappers therefore pass their env to
Claude Code a second time via `--settings '<json>'`, a per-invocation settings
level that sits *above* the user file and overrides only its own keys — the
wrapper wins no matter what the global `settings.json` currently holds.

Two consequences worth knowing:

- Wrappers installed by older versions don't have the `--settings` payload —
  re-run `codehelper add <name>` once to pick it up. Hand-written wrappers can
  add the same flag to their `claude "$@"` line.
- `codehelper switch <provider>` still hot-applies to sessions launched
  *without* a wrapper (it patches the one global `settings.json`) — it no
  longer captures wrapper sessions.

### Real context windows for third-party models

Claude Code cannot resolve a non-Anthropic model ID, so it assumes a 200k
window and auto-compacts there — even for models that really hold 1M.
Generated wrappers and `switch` patches declare the real window via
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` whenever the model is in the built-in
catalog (`glm-5.3`, `glm-5.2`, `glm-5.2:cloud`,
`deepseek-v4-flash:0731-cloud`, `deepseek-v4-pro`, `deepseek-v4-flash`,
`deepseek-v4-flash-vision-exp` — all 1M — plus B.AI's `gpt-6-astra` at its
documented 1,050,000). Models outside the catalog are
left undeclared on purpose: a guessed window that exceeds the real one
overflows the session mid-flight, so no data means no claim. (Claude model
IDs need no entry — Claude Code resolves those natively.) If a model of
yours is missing, add its real number to `MODEL_CONTEXT_WINDOWS` in
`services/render.py`.

## Not every combination is possible

`codehelper list matrix` shows which pairings exist and refuses the rest with
an explanation rather than generating a wrapper that fails at runtime:

```
        ollama-direct  zai            litellm
claude  anthropic-env  anthropic-env  anthropic-env
codex   openai-toml    —              openai-toml
```

Codex can't speak the Anthropic protocol Z.ai serves, so that cell is empty.
Compatibility is derived from what each side supports, not from a hand-written
list — a new provider can't silently introduce a broken combination. `litellm`
declares both mechanisms (a LiteLLM proxy serves both protocols off one host),
so neither of its cells is empty.

## LiteLLM

`litellm` is the one provider whose address is not built into `codehelper` —
it's your own [LiteLLM](https://docs.litellm.ai/) proxy, so you supply its URL
yourself:

```bash
codehelper add --agent claude --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
codehelper add --agent codex  --provider litellm \
                --base-url http://localhost:4000/v1 --model gpt-4o
```

Either form works — `http://localhost:4000` or `http://localhost:4000/v1` —
`codehelper` derives the right endpoint for each agent from whichever you
give it. One URL serves both agents: `claude` resolves to the direct
`ANTHROPIC_*` env shape (LiteLLM's Anthropic Messages passthrough, which
never takes a `/v1` suffix — `codehelper` strips one if present), `codex`
resolves to a `[model_providers.litellm]` TOML profile whose `base_url` does
need `/v1` (`codehelper` appends one if absent) — chosen automatically, same
as every other provider here.

The token comes from `LITELLM_API_KEY`, a selected token profile, or a hidden
prompt — see [Where tokens live](#where-tokens-live) — exactly like any other
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

### Fallback (429, provider outages) is LiteLLM's job, not codehelper's

`codehelper` generates one wrapper for one `agent × provider × model` point,
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

Point a wrapper at `primary` — `codehelper add ... --model primary` — and a
429 (or any failure LiteLLM is configured to catch) fails over to `backup`
transparently; the agent never sees the switch. Check the
[LiteLLM reliability docs](https://docs.litellm.ai/docs/proxy/reliability) for
the current config keys, since this is LiteLLM's surface, not this project's.

## B.AI

[B.AI](https://docs.b.ai/llmservice/introduction/) serves both protocols off
one documented host, so both pairings work with no `--base-url`:

```bash
codehelper add bai                                          # preset: claude -> qwen3.8-flash
codehelper add --agent claude --provider bai --model claude-opus-4-8
codehelper add --agent codex  --provider bai --model gpt-5.5
```

`claude` resolves to the direct `ANTHROPIC_*` env shape (`/v1/messages`),
`codex` to a `[model_providers.bai]` TOML profile whose `base_url` gets the
`/v1` suffix appended — chosen automatically, same as every provider here.
The address is fixed registry data, so `--base-url` is refused. The token
comes from `BAI_API_KEY`, a selected token profile, or a hidden prompt —
see [Where tokens live](#where-tokens-live); model discovery uses B.AI's own
`GET /v1/models`.

Two caveats from B.AI's own docs: models marked "Premium" may answer
`403 access_denied` until the account is recharged, and `gpt-6-astra` speaks
only the Responses API — a codex-only model. Prices and the rotating
"Limited-Time Free" promotions are visible only in B.AI's dashboard; the API
model list carries no pricing, so there is nothing to filter on here.

## `set-default`

A wrapper solves "run *this session* on another model." `set-default` solves a
different problem: making a bare `codex`, with no wrapper and no `--profile`,
start on the backend you chose. It does this by patching Codex's own
`~/.codex/config.toml` in place — the top-level `model` / `model_provider` /
`model_catalog_json` keys, and the matching `[model_providers.X]` table.

```bash
codehelper set-default --agent codex --provider ollama-direct --model glm-5.2:cloud
codehelper set-default --agent codex --provider ollama-direct --model glm-5.2:cloud --dry-run
codehelper set-default --restore                # undo, from the newest backup
codehelper set-default --restore --slot 2        # or an older one
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

## `disable` / `enable`

Retire a provider at runtime without editing anything by hand — for example
when its subscription ends:

```bash
codehelper disable ollama --yes   # 'ollama' resolves to the registry name ollama-direct
codehelper enable ollama
```

`disable` **physically deletes** every managed wrapper of that provider
(installed presets included, ownership guard and Codex siblings as with
`remove`), then records the disable in `state.json` (`disabled_providers`).
From then on the provider disappears from every choice surface — `add` menus,
`switch`, the TUI chipset, the tokens and profile screens — and `list
providers` keeps showing it tagged `(disabled)`. Tokens in
`credentials.json` are kept, so `enable` reuses them — and a wrapper whose
token was never cached (it exists nowhere but the file) blocks the disable
unless you pass `--force`. A live `switch` or `set-default` config that still
names the provider is left alone (only `switch`/`set-default` touch agent
config files) — `disable` prints a note with the clearing command instead.
`enable` only lifts the mark: it does **not** restore the deleted wrappers —
recreate them with `codehelper add`. `native` (the agent's own
backend-clearing entry) cannot be disabled. If any wrapper cannot be removed,
the whole `disable` is refused with `state.json` untouched — retry after
fixing the cause (`--force` for files `codehelper` did not create).

## Safety

- **Your other executables are protected.** A wrapper carries a marker
  comment, and `codehelper` refuses to overwrite a file it didn't create
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
  below. A token typed at a prompt is also cached in named provider profiles
  in `~/.config/codehelper/credentials.json` (`0o600`) so the next `add` or
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

For a secret-auth provider (`zai`, `litellm`, `deepseek`, `deepseek-openai`,
`bai`, `freellmapi`), profiles are stored per provider:

```json
{
  "zai": {
    "work": "key-1",
    "personal": "key-2"
  }
}
```

The first key is kept under the internal `default` profile. When a second key
is added through the TUI, it asks for names for both the old and new profiles.
Each installed wrapper still contains its own copy of the selected key.
If the profile cache is missing, the TUI recovers the key from the selected
managed wrapper into `default` without rewriting that wrapper.

Without an explicit profile, a token is resolved in this order:

1. **Environment variable** (`ZAI_API_KEY`, `LITELLM_API_KEY`, ...) — wins over
   the default profile, so a headless/CI run can override it.
2. **Default profile** — a token typed at an `add` prompt is written there so
   the next `add` or `--list-models` doesn't ask again.
3. **Hidden prompt** — asked only when neither of the above has it.

With `--profile` or an explicit TUI selection, that profile is used first and
the environment variable is not substituted for it. A custom LiteLLM URL does
not receive a profile implicitly; selecting a profile is an explicit choice.

Because the environment wins silently, `add` and `switch` print a warning when
the env var and the default profile hold **different** tokens (both sides
redacted, e.g. `sk-e...ken vs sk-c...key`) — a stale export in an
already-open terminal otherwise writes a dead key that only surfaces as 401
retries later. The env value is still what gets used; unset the variable to
fall back to the cached profile.

To provision the cache from a script (CI, dotfiles bootstrap), pipe the token
on stdin: `printf %s "$BAI_API_KEY" | codehelper edit-token bai --profile
default --token-stdin` (the same flag exists on `add`). It reads exactly one
line, refuses on a terminal — the interactive path stays the hidden prompt —
and the value never appears in argv, so `ps` and shell history never see it
(the `gh auth login --with-token` / `docker login --password-stdin` pattern;
`printf %s` rather than `echo` keeps the value out of the piping command's
own argv).

A named profile is **portable across a provider's addresses.** For a provider
whose URL you supply yourself (`litellm`), the same `--profile work` reuses
its cached key against whatever `--base-url` you give — the key is stored per
provider and profile, never per URL, so there is no warning if you point it at
a different host later. Without `--profile`, a cached key is *not* reused for
such a provider: nothing selected it for that run, so `codehelper` prompts
instead. Naming a profile is what tells `codehelper` you mean that key for
this address; if a key is really scoped to one host, give it its own profile.

This file is a **cache, not a session with the provider**: `codehelper`
configures agents and aliases, it does not log in anywhere. There is no
"logged in" state and no command that connects to a provider to validate a
token. Deleting the file does not break any installed wrapper — each wrapper
carries its own token baked into the script itself (`0o700`); the TUI can
recreate `default` from the selected managed wrapper, while the cache only
controls whether the next install/discovery prompts again.

`edit-token` always prompts for the replacement key. With `--profile` it updates
that profile; without it, the CLI/TUI lets you choose one when several exist
(with only one profile, that profile is used). The newly typed value is cached
after a successful rotation, so the cache stays in step.

`--list-models` discovery follows the same env → cache lookup (never the
prompt — an optional listing must never block a script waiting on stdin); with
neither available it falls back to an unauthenticated request, same as before
this cache existed.

## Development

```bash
pytest -q                       # run tests
ruff check . && ruff format .   # lint + format
```
