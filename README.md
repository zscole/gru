# Gru

[![CI](https://github.com/zscole/gru/actions/workflows/ci.yml/badge.svg)](https://github.com/zscole/gru/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

AI agents you can control from your phone. Message a bot, get a coding assistant.

Just tell it what to build. Send screenshots of UI designs. Use voice notes. Get live preview URLs.

## Quick Start

**Step 1:** Get your API keys (5 min)

| Key | Where to get it |
|-----|-----------------|
| Anthropic API Key | [console.anthropic.com](https://console.anthropic.com) → API Keys → Create Key |
| Telegram Bot Token | Open Telegram → message [@BotFather](https://t.me/BotFather) → send `/newbot` → copy the token |
| Your Telegram User ID | Message [@userinfobot](https://t.me/userinfobot) → copy the number it sends back |

**Step 2:** Deploy (1 min)

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/PGw7ud?referralCode=O4yxwb)

Click the button above, then paste your 3 keys:
- `ANTHROPIC_API_KEY` → your Anthropic key
- `GRU_TELEGRAM_TOKEN` → your bot token from BotFather
- `GRU_ADMIN_IDS` → your user ID from userinfobot

**Step 3:** Start chatting

Open Telegram, find your bot, and just tell it what you want:
```
Build me a landing page for a coffee shop
```

Or use quick-start templates:
```
/gru create landing-page
```

That's it. You now have AI agents on your phone.

*Need help creating a Telegram bot? See the [detailed Telegram setup guide](#telegram-setup-recommended).*

---

**Want Discord or Slack instead?** See [Discord Setup](#discord-setup) or [Slack Setup](#slack-setup).

**Prefer to self-host?** See [Installation](#installation-self-hosted).

---

## What is Gru?

Gru lets you spawn and control Claude-powered AI agents from your phone. Agents can run commands, read/write files, and work on coding tasks autonomously. Think of it as a coding assistant you can message from anywhere.

## Features

**Talk naturally** - Just describe what you want. No commands needed.
```
Build me a REST API with user authentication
```

**Send screenshots** - Send a photo of any UI design, Gru builds it.

**Voice notes** - Send a voice message describing your task.

**Project templates** - Quick-start common projects:
```
/gru create react-app
/gru create landing-page
/gru create express-api
```

**Live preview** - Deploy to Vercel, Netlify, or Surge:
```
/gru deploy vercel
```

**Health check** - Diagnose setup issues:
```
/gru doctor
```

**Progress tracking** - Visual progress bar shows agent status:
```
[=====---------] ~35%
Turn 12 | 3m | 15,234 tokens
Recent: bash, write_file, read_file
```

---

## Telegram Setup (Recommended)

Telegram is the easiest way to get started. You already did this if you followed Quick Start above.

### Create a Telegram Bot

1. Open Telegram on your phone or desktop
2. Search for **@BotFather** (the official Telegram bot for creating bots)
3. Start a chat and send: `/newbot`
4. BotFather will ask for a **name** for your bot. This is the display name (e.g., "My Gru Bot")
5. BotFather will ask for a **username**. This must end in `bot` (e.g., "my_gru_bot")
6. BotFather will give you a **token** that looks like this:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
7. **Save this token.** You'll need it for your `.env` file.

**Optional but recommended:** Send `/setprivacy` to @BotFather, select your bot, and choose "Disable". This lets your bot read all messages in groups (if you want to use it in groups later).

---

### Get Your Telegram User ID

Gru only responds to authorized users. You need your Telegram user ID (a number, not your username).

1. Open Telegram
2. Search for **@userinfobot**
3. Start a chat and send any message
4. The bot will reply with your user ID:
   ```
   Id: 123456789
   ```
5. **Save this number.** You'll need it for your `.env` file.

**Want to add multiple admins?** You can add multiple user IDs separated by commas (e.g., `123456789,987654321`).

---

## Discord Setup

### Create a Discord Bot

You need to create a bot in the Discord Developer Portal. This is free.

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Click **New Application**
3. Give it a name (e.g., "Gru") and click **Create**
4. Go to the **Bot** tab in the left sidebar
5. Click **Reset Token** to generate a bot token
6. Copy the token. It looks like:
   ```
   your-bot-token-here
   ```
7. **Save this token.** You'll need it for your `.env` file.

**Important Bot Settings:**
- Under **Privileged Gateway Intents**, enable **Message Content Intent**
- This allows the bot to read message content for natural language commands

**Invite the bot to your server:**

1. Go to the **OAuth2** tab, then **URL Generator**
2. Under **Scopes**, select `bot` and `applications.commands`
3. Under **Bot Permissions**, select:
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Read Message History
4. Copy the generated URL and open it in your browser
5. Select your server and click **Authorize**

---

### Get Your Discord User ID

Gru only responds to authorized users. You need your Discord user ID (a number).

1. Open Discord
2. Go to **User Settings** (gear icon)
3. Go to **Advanced** and enable **Developer Mode**
4. Close settings
5. Right-click on your username anywhere in Discord
6. Click **Copy User ID**
7. **Save this number.** You'll need it for your `.env` file.

**Want to add multiple admins?** You can add multiple user IDs separated by commas (e.g., `123456789012345678,987654321098765432`).

---

## Slack Setup

### Create a Slack App

You need to create an app in the Slack API dashboard. This is free.

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App**
3. Choose **From scratch**
4. Give it a name (e.g., "Gru") and select your workspace
5. Click **Create App**

**Enable Socket Mode (required):**

1. Go to **Socket Mode** in the left sidebar
2. Toggle **Enable Socket Mode** to ON
3. You'll be prompted to create an App-Level Token
4. Name it (e.g., "gru-socket") and add the `connections:write` scope
5. Click **Generate**
6. Copy the token. It starts with `xapp-`:
   ```
   xapp-1-xxxxxxxxxxxx
   ```
7. **Save this token.** This is your `GRU_SLACK_APP_TOKEN`.

**Add Bot Permissions:**

1. Go to **OAuth & Permissions** in the left sidebar
2. Scroll to **Scopes** > **Bot Token Scopes**
3. Add these scopes:
   - `chat:write` - Send messages
   - `commands` - Handle slash commands
   - `im:history` - Read DM history
   - `im:write` - Send DMs

**Create the Slash Command:**

1. Go to **Slash Commands** in the left sidebar
2. Click **Create New Command**
3. Set Command to `/gru`
4. Set Description to "Gru agent orchestrator"
5. Click **Save**

**Install to Workspace:**

1. Go to **Install App** in the left sidebar
2. Click **Install to Workspace**
3. Authorize the app
4. Copy the **Bot User OAuth Token**. It starts with `xoxb-`:
   ```
   xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
   ```
5. **Save this token.** This is your `GRU_SLACK_BOT_TOKEN`.

---

### Get Your Slack User ID

Gru only responds to authorized users. You need your Slack user ID.

1. Open Slack
2. Click on your profile picture (bottom left)
3. Click **Profile**
4. Click the **three dots** menu
5. Click **Copy member ID**
6. **Save this ID.** It looks like `U01ABC123DE`.

**Want to add multiple admins?** You can add multiple user IDs separated by commas (e.g., `U01ABC123DE,U02XYZ456FG`).

---

## API Key

### Get an Anthropic API Key

Gru uses Claude (made by Anthropic) as its AI brain. You need an API key.

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Go to **API Keys** in the left sidebar
4. Click **Create Key**
5. Give it a name (e.g., "Gru")
6. Copy the key. It looks like:
   ```
   sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
7. **Save this key.** You'll need it for your `.env` file.

**Note:** Anthropic API usage costs money. Check their [pricing page](https://www.anthropic.com/pricing). Claude Sonnet is recommended for a balance of capability and cost.

---

## Installation (Self-Hosted)

If you prefer to run Gru on your own machine or server instead of Railway:

### Option A: Standard Installation

**Requires Python 3.10+.** Check with `python3 --version`. Install from [python.org](https://www.python.org/downloads/) if needed.

Open your terminal and run these commands one at a time:

```bash
# 1. Clone the repository
git clone https://github.com/zscole/gru.git

# 2. Enter the directory
cd gru

# 3. Create a virtual environment (keeps dependencies isolated)
python3 -m venv .venv

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

**Windows users:** Instead of `source .venv/bin/activate`, use:
```bash
.venv\Scripts\activate
```

### Option B: Docker Installation

If you have Docker installed, this is even easier:

```bash
# 1. Clone the repository
git clone https://github.com/zscole/gru.git
cd gru

# 2. Create your .env file (see next section)

# 3. Run with Docker Compose
docker-compose up -d
```

That's it. The bot will start automatically.

---

## Configuration

Create a file called `.env` in the `gru` folder. This file stores your secret keys.

**Easy way:** Copy the example file and edit it:

```bash
cp .env.example .env
```

Then open `.env` in any text editor and fill in your values.

**Manual way:** Create a new file called `.env` with this content:

```bash
# Required - Anthropic API key
ANTHROPIC_API_KEY=paste_your_anthropic_key_here

# Telegram (optional)
GRU_TELEGRAM_TOKEN=paste_your_telegram_bot_token_here
GRU_ADMIN_IDS=paste_your_telegram_user_id_here

# Discord (optional)
GRU_DISCORD_TOKEN=paste_your_discord_bot_token_here
GRU_DISCORD_ADMIN_IDS=paste_your_discord_user_id_here
GRU_DISCORD_GUILD_ID=optional_server_id_to_restrict_to

# Slack (optional)
GRU_SLACK_BOT_TOKEN=xoxb-your-bot-token
GRU_SLACK_APP_TOKEN=xapp-your-app-token
GRU_SLACK_ADMIN_IDS=U01ABC123DE

# Optional - defaults work fine for most users
GRU_MASTER_PASSWORD=pick_any_password_for_encrypting_secrets
GRU_DATA_DIR=~/.gru
GRU_WORKDIR=~/gru-workspace
GRU_DEFAULT_MODEL=claude-sonnet-4-20250514
GRU_MAX_TOKENS=8192
GRU_DEFAULT_TIMEOUT=300
GRU_MAX_AGENTS=10

# GitHub access for private repos (optional)
GRU_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Example with Telegram only:**

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRU_TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
GRU_ADMIN_IDS=123456789
```

**Example with Discord only:**

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRU_DISCORD_TOKEN=your-discord-bot-token
GRU_DISCORD_ADMIN_IDS=123456789012345678
```

**Example with Slack only:**

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRU_SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
GRU_SLACK_APP_TOKEN=xapp-1-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
GRU_SLACK_ADMIN_IDS=U01ABC123DE
```

**Example with multiple platforms:**

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRU_TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
GRU_ADMIN_IDS=123456789
GRU_DISCORD_TOKEN=your-discord-bot-token
GRU_DISCORD_ADMIN_IDS=123456789012345678
GRU_SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
GRU_SLACK_APP_TOKEN=xapp-1-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
GRU_SLACK_ADMIN_IDS=U01ABC123DE
```

### Configuration Options Explained

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `GRU_TELEGRAM_TOKEN` | If using Telegram | Your bot token from @BotFather |
| `GRU_ADMIN_IDS` | If using Telegram | Your Telegram user ID(s), comma-separated |
| `GRU_DISCORD_TOKEN` | If using Discord | Your bot token from Discord Developer Portal |
| `GRU_DISCORD_ADMIN_IDS` | If using Discord | Your Discord user ID(s), comma-separated |
| `GRU_DISCORD_GUILD_ID` | No | Restrict bot to a specific Discord server |
| `GRU_SLACK_BOT_TOKEN` | If using Slack | Your Bot User OAuth Token (`xoxb-...`) |
| `GRU_SLACK_APP_TOKEN` | If using Slack | Your App-Level Token for Socket Mode (`xapp-...`) |
| `GRU_SLACK_ADMIN_IDS` | If using Slack | Your Slack user ID(s), comma-separated |
| `GRU_MASTER_PASSWORD` | No | Password for encrypting stored secrets |
| `GRU_GITHUB_TOKEN` | No | GitHub personal access token for private repos |
| `GRU_DATA_DIR` | No | Where Gru stores its database (default: `~/.gru`) |
| `GRU_WORKDIR` | No | Default directory for agents to work in (default: `~/gru-workspace`) |
| `GRU_DEFAULT_MODEL` | No | Claude model to use (default: `claude-sonnet-4-20250514`) |
| `GRU_MAX_TOKENS` | No | Max tokens per response (default: `8192`) |
| `GRU_DEFAULT_TIMEOUT` | No | Agent timeout in seconds (default: `300`) |
| `GRU_MAX_AGENTS` | No | Max concurrent agents (default: `10`) |
| `GRU_WEBHOOK_ENABLED` | No | Enable webhook server (default: `false`) |
| `GRU_WEBHOOK_PORT` | No | Webhook server port (default: `8080`) |
| `GRU_WEBHOOK_SECRET` | No | Vercel webhook secret for signature verification |
| `GRU_PROGRESS_REPORT_INTERVAL` | No | Minutes between progress reports (default: `0` = disabled) |

**Note:** You must configure at least one bot interface (Telegram, Discord, or Slack). You can use multiple simultaneously.

---

## Running Gru

### Standard Installation

Every time you want to run Gru:

```bash
# 1. Open terminal and go to the gru folder
cd gru

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Start Gru
PYTHONPATH=src python -m gru.main
```

**Windows users:** Use this instead:
```bash
set PYTHONPATH=src
python -m gru.main
```

### Docker Installation

```bash
docker-compose up -d
```

To stop: `docker-compose down`

To view logs: `docker-compose logs -f`

### What Success Looks Like

When Gru starts correctly, you'll see:

```
Gru server started
Data directory: /home/you/.gru
Telegram admin IDs: [123456789]
Discord admin IDs: [123456789012345678]
Slack admin IDs: ['U01ABC123DE']
```

If you have MCP servers configured (optional), you'll also see:

```
MCP server 'filesystem' started with 14 tools
Started 1 MCP server(s)
```

**Now open Telegram, Discord, or Slack and send a message to your bot!**

---

## Using Gru

### Your First Agent

Open Telegram, find your bot, and try:

```
/gru spawn write a hello world python script
```

The agent will:
1. Start working on your task
2. Ask for approval before writing files (supervised mode)
3. Send you the result when done

### Talk Naturally

You don't have to use commands. Just chat:

```
build me a simple todo app in python
```

```
what files are in my workspace?
```

```
check on my agents
```

Gru understands what you want and either spawns an agent or answers directly.

### Send Screenshots

Send a photo of any UI design (mockup, website screenshot, hand-drawn sketch) and Gru will build it. Add a caption for specific instructions:

1. Take a screenshot or photo of a UI you like
2. Send it to the bot with an optional caption like "Use React and Tailwind"
3. Gru spawns an agent that analyzes the image and recreates it in code

### Send Voice Notes

Too lazy to type? Send a voice message:

1. Hold the microphone button in Telegram
2. Describe what you want built
3. Gru transcribes it and spawns an agent

### Project Templates

Quick-start common project types:

| Template | What it creates |
|----------|-----------------|
| `react-app` | React + TypeScript + Tailwind |
| `next-app` | Next.js 14 with app router |
| `express-api` | Express + TypeScript + JWT auth |
| `fastapi` | FastAPI + SQLAlchemy + Pydantic |
| `landing-page` | Responsive landing page |
| `cli-tool` | Python CLI with Click |
| `discord-bot` | Discord.py bot |
| `chrome-extension` | Chrome extension (Manifest v3) |

Usage: `/gru create <template>` or `/gru create <template> --workdir /path`

### Commands Reference

Commands work the same across all platforms. On Telegram, use `/gru <command>`. On Discord and Slack, use the `/gru` slash command.

#### Telegram Commands

All commands start with `/gru`:

**Quick Start:**
```
/gru examples                      Show copy-paste example tasks
/gru create <template>             Create from template (react-app, landing-page, etc.)
/gru doctor                        Health check - verify setup is working
/gru setup                         Guided setup help
```

**Spawning Agents:**
```
/gru spawn <task>                  Start an agent (supervised mode)
/gru spawn <task> --unsupervised   No approval needed
/gru spawn <task> --oneshot        Fire and forget
/gru spawn <task> --workdir /path  Work in specific directory
/gru spawn <task> --priority high  Set priority (high/normal/low)
```

**Deploy:**
```
/gru deploy vercel                 Deploy to Vercel
/gru deploy netlify                Deploy to Netlify
/gru deploy surge                  Deploy to Surge.sh (free)
```

**Managing Agents:**
```
/gru status                        Show overall status
/gru status <agent_id>             Show specific agent
/gru list                          List all agents
/gru list running                  Filter by status
/gru pause <agent_id>              Pause an agent
/gru resume <agent_id>             Resume an agent
/gru terminate <agent_id>          Stop an agent
/gru nudge <agent_id> <message>    Send a message to an agent
/gru logs <agent_id>               View conversation history
```

**Approvals (supervised mode):**
```
/gru pending                       List pending approvals
/gru approve <approval_id>         Approve an action
/gru reject <approval_id>          Reject an action
```

**Secrets (encrypted storage):**
```
/gru secret set KEY value          Store a secret
/gru secret get KEY                Retrieve a secret
/gru secret list                   List all secret keys
/gru secret delete KEY             Delete a secret
```

**Templates:**
```
/gru template save <name> <task>   Save a task template
/gru template list                 List all templates
/gru template use <name>           Spawn from template
/gru template delete <name>        Delete a template
```

#### Discord Commands

Discord uses slash commands. Type `/gru` and Discord will show available subcommands:

**Spawning Agents:**
```
/gru spawn task:<task>
/gru spawn task:<task> mode:unsupervised
/gru spawn task:<task> mode:oneshot
/gru spawn task:<task> workdir:/path priority:high
```

**Managing Agents:**
```
/gru status
/gru status agent_id:<id>
/gru list
/gru list status:running
/gru pause agent_id:<id>
/gru resume agent_id:<id>
/gru terminate agent_id:<id>
/gru nudge agent_id:<id> message:<text>
/gru logs agent_id:<id>
```

**Approvals (supervised mode):**
```
/gru pending
/gru approve approval_id:<id>
/gru reject approval_id:<id>
```

**Secrets:**
```
/gru secret_set key:<KEY> value:<value>
/gru secret_get key:<KEY>
/gru secret_list
/gru secret_delete key:<KEY>
```

**Templates:**
```
/gru template_save name:<name> task:<task>
/gru template_list
/gru template_use name:<name>
/gru template_delete name:<name>
```

**Discord-specific features:**
- Approval requests appear with clickable Approve/Reject buttons
- Multi-option approvals show numbered option buttons
- Natural language works in any channel the bot can see

#### Slack Commands

Use `/gru <command>` in any channel or DM:

**Spawning Agents:**
```
/gru spawn <task>
/gru spawn <task> --unsupervised
/gru spawn <task> --oneshot
/gru spawn <task> --workdir /path --priority high
```

**Managing Agents:**
```
/gru status
/gru status <agent_id>
/gru list
/gru list running
/gru pause <agent_id>
/gru resume <agent_id>
/gru terminate <agent_id>
/gru nudge <agent_id> <message>
/gru logs <agent_id>
```

**Approvals:**
```
/gru pending
/gru approve <approval_id>
/gru reject <approval_id>
```

**Secrets:**
```
/gru secret set <key> <value>
/gru secret get <key>
/gru secret list
/gru secret delete <key>
```

**Templates:**
```
/gru template save <name> <task>
/gru template list
/gru template use <name>
/gru template delete <name>
```

**Slack-specific features:**
- Approval requests appear with clickable buttons in DMs
- Natural language works in DMs with the bot
- Uses Socket Mode (no public URL required)

### Execution Modes Explained

**Supervised (default):**
- Agent asks permission before running commands or writing files
- You get Telegram buttons to Approve or Reject
- Safest mode, recommended for beginners

**Unsupervised:**
- Agent runs without asking permission
- Use when you trust the task
- Add `--unsupervised` flag

**Oneshot:**
- Fire and forget
- Agent runs to completion without any interaction
- Results are sent when done
- Add `--oneshot` flag

---

### Monitoring Agents

**Progress Reports:**

Enable automatic progress updates during long-running tasks:

```bash
GRU_PROGRESS_REPORT_INTERVAL=5  # Report every 5 minutes
```

When enabled, agents send periodic updates showing:
- Current turn count and runtime
- Recent tool calls (e.g., `bash: git status`, `write_file: src/app.py`)

**Logs:**

View an agent's conversation history:

```
/gru logs <agent_id>
```

Logs show formatted tool calls instead of raw JSON:
- `[tool] bash(git status)` - tool invocation
- `[result] output...` - tool result
- `[error] message` - error result

You can use agent numbers or nicknames: `/gru logs 1` or `/gru logs linter`

---

## MCP Plugins (Optional)

MCP (Model Context Protocol) lets you extend Gru with additional tools. This is optional but powerful.

### Setup

1. Install Node.js if you don't have it: [nodejs.org](https://nodejs.org)

2. Create `mcp_servers.json` in the gru folder:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/your/files"],
      "env": {}
    }
  }
}
```

3. Restart Gru

### Popular MCP Servers

| Server | What it does |
|--------|--------------|
| `@modelcontextprotocol/server-filesystem` | Advanced file operations |
| `@modelcontextprotocol/server-github` | GitHub integration |
| `@modelcontextprotocol/server-postgres` | PostgreSQL database |
| `@modelcontextprotocol/server-puppeteer` | Browser automation |

See [MCP Servers](https://github.com/modelcontextprotocol/servers) for more.

---

## Webhook Setup (Vercel)

Gru can receive webhooks from Vercel to notify agents when preview deployments are ready. This is useful when agents are building web apps and push to branches.

### How It Works

1. Agent pushes code to a branch named `gru-agent-<agent_id>`
2. Vercel deploys the branch as a preview
3. Vercel sends a webhook to Gru
4. Gru notifies the agent with the preview URL

### Configuration

Add these to your `.env`:

```bash
GRU_WEBHOOK_ENABLED=true
GRU_WEBHOOK_HOST=0.0.0.0
GRU_WEBHOOK_PORT=8080
GRU_WEBHOOK_SECRET=your-vercel-webhook-secret
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GRU_WEBHOOK_ENABLED` | No | Enable webhook server (default: `false`) |
| `GRU_WEBHOOK_HOST` | No | Host to bind (default: `0.0.0.0`) |
| `GRU_WEBHOOK_PORT` | No | Port to listen on (default: `8080`) |
| `GRU_WEBHOOK_SECRET` | No | Vercel webhook secret for signature verification |

### Vercel Setup

1. Go to your Vercel project settings
2. Navigate to **Git** > **Deploy Hooks** or **Settings** > **Webhooks**
3. Add a webhook pointing to `http://your-server:8080/webhook/vercel`
4. Copy the signing secret and set it as `GRU_WEBHOOK_SECRET`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook/vercel` | POST | Receives Vercel deployment webhooks |
| `/health` | GET | Health check (returns `{"status": "ok"}`) |

### Branch Naming

For webhooks to work, agents must push to branches matching the pattern `gru-agent-<agent_id>`. The webhook handler extracts the agent ID from the branch name and notifies the correct agent.

---

## Troubleshooting

### Bot not responding

**Telegram:**
1. **Check your user ID**: Make sure `GRU_ADMIN_IDS` in `.env` matches your Telegram user ID exactly
2. **Check the token**: Make sure `GRU_TELEGRAM_TOKEN` is correct (no extra spaces)
3. **Check Gru is running**: Look at the terminal for errors
4. **Restart Gru**: Stop it (Ctrl+C) and start again

**Discord:**
1. **Check your user ID**: Make sure `GRU_DISCORD_ADMIN_IDS` matches your Discord user ID
2. **Check the token**: Make sure `GRU_DISCORD_TOKEN` is correct
3. **Check Message Content Intent**: Enable it in Discord Developer Portal > Bot settings
4. **Check bot permissions**: Bot needs Send Messages and Read Message History
5. **Check slash commands**: Type `/gru` - if no commands appear, the bot may need to be re-invited

**Slack:**
1. **Check your user ID**: Make sure `GRU_SLACK_ADMIN_IDS` matches your Slack member ID
2. **Check both tokens**: You need both `GRU_SLACK_BOT_TOKEN` (xoxb-...) AND `GRU_SLACK_APP_TOKEN` (xapp-...)
3. **Check Socket Mode**: Must be enabled in your Slack app settings
4. **Check bot scopes**: Needs `chat:write`, `commands`, `im:history`, `im:write`
5. **Check slash command**: The `/gru` command must be created in your app settings
6. **Reinstall the app**: If permissions changed, reinstall to your workspace

### "Command not found: python3"

- Make sure Python is installed (see [Step 1](#step-1-install-python))
- On Windows, try `python` instead of `python3`

### "No module named 'gru'"

Make sure you're running with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m gru.main
```

### Agent seems stuck

Try these in order:

1. Send a nudge: `/gru nudge <agent_id> hurry up`
2. Check status: `/gru status <agent_id>`
3. Terminate: `/gru terminate <agent_id>`

### Approvals not appearing

- Make sure you're in supervised mode (the default)
- Check Telegram notifications are enabled
- Try `/gru pending` to see if approvals are queued

### API errors

- **"Invalid API key"**: Check `ANTHROPIC_API_KEY` is correct
- **"Rate limited"**: You're sending too many requests, wait a bit
- **"Insufficient credits"**: Add credits at [console.anthropic.com](https://console.anthropic.com)

### Docker issues

- **Port conflict**: Another service is using the same port
- **Container won't start**: Check logs with `docker-compose logs`
- **Permission denied**: Try `sudo docker-compose up -d`

---

## Security Notes

- **Only admins can use the bot**: Users not in `GRU_ADMIN_IDS` are ignored
- **Supervised mode is default for a reason**: Always start here
- **Agents run with your permissions**: They can do anything you can do
- **Use `GRU_WORKDIR`**: Isolate agents to a specific directory
- **Review MCP servers**: They can execute arbitrary code

---

## Architecture

```
Telegram Bot --+
               |
Discord Bot  --+--> Orchestrator <-> Claude API
               |         |
Slack Bot    --+         +-> Scheduler (priority queue)
                         +-> Database (SQLite)
                         +-> MCP Client (plugin tools)
                         +-> Secret Store (encrypted)
```

All bots share the same orchestrator. You can run any combination simultaneously.

**Components:**

| File | Purpose |
|------|---------|
| `telegram_bot.py` | Telegram interface, command parsing |
| `discord_bot.py` | Discord interface, slash commands |
| `slack_bot.py` | Slack interface, Socket Mode |
| `orchestrator.py` | Agent lifecycle, tool execution |
| `claude.py` | Claude API client |
| `scheduler.py` | Priority queue scheduling |
| `webhook.py` | Vercel deployment webhooks |
| `mcp.py` | MCP server management |
| `crypto.py` | Secret encryption |
| `db.py` | SQLite database |

---

## Development

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/
```

---

## License

MIT - see [LICENSE](LICENSE)
