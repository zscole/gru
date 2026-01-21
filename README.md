# Gru

[![CI](https://github.com/zscole/gru/actions/workflows/ci.yml/badge.svg)](https://github.com/zscole/gru/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

AI agents you can control from your phone. Message a bot, get a coding assistant.

## Quick Start

**1. Get your keys** (5 min)

| Key | Where |
|-----|-------|
| Anthropic API Key | [console.anthropic.com](https://console.anthropic.com) |
| Telegram Bot Token | Message [@BotFather](https://t.me/BotFather) > `/newbot` |
| Your Telegram ID | Message [@userinfobot](https://t.me/userinfobot) |

**2. Deploy** (1 min)

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/PGw7ud?referralCode=O4yxwb)

Set these env vars:
- `ANTHROPIC_API_KEY`
- `GRU_TELEGRAM_TOKEN`
- `GRU_ADMIN_IDS`

**3. Chat**

Open Telegram, message your bot:
```
Build me a landing page for a coffee shop
```

That's it.

## Features

- **Natural language** - Just describe what you want
- **Screenshots** - Send a UI photo, Gru builds it
- **Voice notes** - Speak your request
- **Templates** - `/gru create react-app`
- **Live deploy** - `/gru deploy vercel`
- **Health check** - `/gru doctor`
- **Ralph loops** - Iterative AI development cycles

## Contents

- [Platform Setup](#platform-setup)
- [Self-Hosted Installation](#self-hosted-installation)
- [Documentation](#documentation)
- [Architecture](#architecture)

## Platform Setup

<details>
<summary><strong>Telegram</strong> (recommended)</summary>

1. Message [@BotFather](https://t.me/BotFather) > `/newbot`
2. Copy the token
3. Get your user ID from [@userinfobot](https://t.me/userinfobot)

```bash
GRU_TELEGRAM_TOKEN=your-token
GRU_ADMIN_IDS=your-user-id
```

[Full guide](docs/telegram.md)

</details>

<details>
<summary><strong>Discord</strong></summary>

1. Create app at [discord.com/developers](https://discord.com/developers/applications)
2. Get bot token, enable Message Content Intent
3. Invite with `bot` + `applications.commands` scopes

```bash
GRU_DISCORD_TOKEN=your-token
GRU_DISCORD_ADMIN_IDS=your-user-id
```

[Full guide](docs/discord.md)

</details>

<details>
<summary><strong>Slack</strong></summary>

1. Create app at [api.slack.com/apps](https://api.slack.com/apps)
2. Enable Socket Mode, get both tokens
3. Add `/gru` slash command

```bash
GRU_SLACK_BOT_TOKEN=xoxb-xxx
GRU_SLACK_APP_TOKEN=xapp-xxx
GRU_SLACK_ADMIN_IDS=U01ABC123DE
```

[Full guide](docs/slack.md)

</details>

## Self-Hosted Installation

<details>
<summary><strong>Standard</strong></summary>

```bash
git clone https://github.com/zscole/gru.git
cd gru
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your keys
PYTHONPATH=src python -m gru.main
```

</details>

<details>
<summary><strong>Docker</strong></summary>

```bash
git clone https://github.com/zscole/gru.git
cd gru
cp .env.example .env  # Edit with your keys
docker-compose up -d
```

</details>

## Documentation

| Doc | Description |
|-----|-------------|
| [Commands](docs/commands.md) | Full command reference |
| [Ralph Loops](docs/ralph.md) | Iterative AI development cycles |
| [Configuration](docs/configuration.md) | All environment variables |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [MCP Plugins](docs/mcp-plugins.md) | Extend with MCP servers |
| [Webhooks](docs/webhooks.md) | Vercel deployment notifications |

## Architecture

```
Telegram --+
Discord  --+--> Orchestrator <-> Claude API
Slack    --+         |
                     +-> Scheduler
                     +-> Database (SQLite)
                     +-> MCP Client
                     +-> Secret Store
```

## Security

- Only users in admin list can use the bot
- Supervised mode (default) requires approval for commands/file writes
- Agents run with your permissions
- Use `GRU_WORKDIR` to isolate agent workspace

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
ruff check src/ tests/
mypy src/
```

## Funding & Sponsors

This project is funded by the [$GRU token](https://gruonsol.com/) community on Solana.

**Contract Address:** `HXU8HiXKeMBTZ9QCSxWrdGySBhbHeJLhGbY6b4z6BAGS`

**Buy $GRU:** [Jupiter Exchange](https://jup.ag/swap?sell=So11111111111111111111111111111111111111112&buy=HXU8HiXKeMBTZ9QCSxWrdGySBhbHeJLhGbY6b4z6BAGS)

Special thanks to our sponsors and community members who support open-source AI development. Your contributions enable continuous improvement and new features like Ralph loops.

## License

MIT
