# Configuration

Create a `.env` file in the gru folder:

```bash
cp .env.example .env
```

## Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key from [console.anthropic.com](https://console.anthropic.com) |

Plus at least one bot interface (Telegram, Discord, or Slack).

## Telegram

| Variable | Description |
|----------|-------------|
| `GRU_TELEGRAM_TOKEN` | Bot token from @BotFather |
| `GRU_ADMIN_IDS` | Your Telegram user ID(s), comma-separated |

## Discord

| Variable | Description |
|----------|-------------|
| `GRU_DISCORD_TOKEN` | Bot token from Developer Portal |
| `GRU_DISCORD_ADMIN_IDS` | Your Discord user ID(s), comma-separated |
| `GRU_DISCORD_GUILD_ID` | Optional: restrict to one server |

## Slack

| Variable | Description |
|----------|-------------|
| `GRU_SLACK_BOT_TOKEN` | Bot User OAuth Token (`xoxb-...`) |
| `GRU_SLACK_APP_TOKEN` | App-Level Token (`xapp-...`) |
| `GRU_SLACK_ADMIN_IDS` | Your Slack member ID(s), comma-separated |

## Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `GRU_MASTER_PASSWORD` | - | Password for encrypting stored secrets |
| `GRU_GITHUB_TOKEN` | - | GitHub PAT for private repos |
| `GRU_DATA_DIR` | `~/.gru` | Database location |
| `GRU_WORKDIR` | `~/gru-workspace` | Default agent working directory |
| `GRU_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `GRU_MAX_TOKENS` | `8192` | Max tokens per response |
| `GRU_DEFAULT_TIMEOUT` | `300` | Agent timeout (seconds) |
| `GRU_MAX_AGENTS` | `10` | Max concurrent agents |
| `GRU_PROGRESS_REPORT_INTERVAL` | `0` | Minutes between progress reports (0 = disabled) |

## Ollama (Local Models)

Run models locally for cost savings and unrestricted capabilities.

| Variable | Default | Description |
|----------|---------|-------------|
| `GRU_OLLAMA_URL` | - | Ollama server URL (e.g., `http://localhost:11434`) |
| `GRU_OLLAMA_MODEL` | `llama3:8b` | Default Ollama model |
| `GRU_DEFAULT_PROVIDER` | `anthropic` | Default provider (`anthropic` or `ollama`) |

When Ollama is configured, Gru automatically routes tasks:
- **Simple cleanup** (formatting, linting) → Ollama
- **Web search** → Ollama
- **Complex reasoning** → Claude Opus
- **Code generation** → Claude Sonnet
- **Vision tasks** → Claude (required)

## Webhooks (Vercel)

| Variable | Default | Description |
|----------|---------|-------------|
| `GRU_WEBHOOK_ENABLED` | `false` | Enable webhook server |
| `GRU_WEBHOOK_HOST` | `0.0.0.0` | Host to bind |
| `GRU_WEBHOOK_PORT` | `8080` | Port to listen on |
| `GRU_WEBHOOK_SECRET` | - | Vercel webhook signing secret |

## Examples

**Telegram only:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxx
GRU_TELEGRAM_TOKEN=1234567890:ABCdef
GRU_ADMIN_IDS=123456789
```

**All platforms:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxx
GRU_TELEGRAM_TOKEN=1234567890:ABCdef
GRU_ADMIN_IDS=123456789
GRU_DISCORD_TOKEN=your-discord-token
GRU_DISCORD_ADMIN_IDS=123456789012345678
GRU_SLACK_BOT_TOKEN=xoxb-xxx
GRU_SLACK_APP_TOKEN=xapp-xxx
GRU_SLACK_ADMIN_IDS=U01ABC123DE
```

[Back to README](../README.md)
