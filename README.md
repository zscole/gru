# Gru

[![CI](https://github.com/zscole/gru/actions/workflows/ci.yml/badge.svg)](https://github.com/zscole/gru/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Your personal army. Spin up autonomous agents that complete tasks, learn your patterns, and remember everything.

```
You: Research the top 3 competitors and build me a comparison landing page
Gru: Spinning up research agent... Found 3 competitors. Analyzing features.
     Spinning up dev agent... Building landing page.
     Done. Deployed to vercel: https://compare.yoursite.com

You: What did I discuss with Sarah last week?
Gru: 3 interactions found:
     - API redesign planning (Tuesday)
     - Q1 budget review (Wednesday)
     - Deployment timeline sync (Friday)

You: Book Nobu for Saturday 7pm, venmo @mike for the concert, and remind me Friday
Gru: Done:
     - Reserved Nobu for Saturday 7:00 PM (party of 2)
     - Sent $50 to @mike via Venmo
     - Reminder set for Friday
```

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

## What Gru Can Do

### Multi-Agent Orchestration
Gru spins up specialized agents that work in parallel to complete complex tasks.

| Say this | Gru does this |
|----------|---------------|
| "Research AI developments and build me a PoC" | Spawns research agent + dev agent |
| "Organize my inbox and summarize the important stuff" | Email agent processes, filters, summarizes |
| "Build a landing page based on this screenshot" | Vision agent analyzes, dev agent builds |
| "Check my calendar and prep me for tomorrow's meetings" | Calendar agent + research agent on attendees |

Agents learn from each task and share context, getting better over time.

### Autonomous Actions
Gru doesn't just tell you how - it does it for you.

| Say this | Gru does this |
|----------|---------------|
| "Book Nobu for Saturday 7pm" | Makes the OpenTable reservation |
| "Venmo @sarah $30 for lunch" | Sends the payment |
| "Order my usual from Chipotle" | Places the DoorDash order |
| "Schedule a call with Mike tomorrow 3pm" | Creates the calendar event |
| "Email John the project update" | Sends the email |

### Personal Knowledge Graph
Gru remembers every conversation, email, and meeting - and connects the dots.

| Ask this | Gru answers |
|----------|-------------|
| "What did I discuss with Sarah last month?" | Pulls all interactions with timestamps |
| "Who's working on the API redesign?" | Maps relationships from your conversations |
| "What topics have I been focused on?" | Analyzes patterns across all sources |
| "When did I last talk to the investors?" | Searches across email, Slack, calendar |

### Persistent Memory
Gru learns as it goes. Every interaction builds context that makes it more useful.

- **Pattern recognition** - "You usually check metrics on Fridays"
- **Preference learning** - Remembers your usual orders, common contacts, meeting preferences
- **Cross-session context** - Picks up where you left off, even weeks later
- **Proactive insights** - "You mentioned following up with Mike - no response yet"

### Proactive Intelligence
Gru anticipates what you need before you ask.

- **6am daily briefing** - Calendar, unread emails, Slack mentions
- **Smart reminders** - Surfaces uncommitted follow-ups and deadlines
- **AI research** - Monitors HN/Reddit, builds working PoCs of interesting projects
- **Anticipation** - "Traffic is heavy, leave now for your 4pm"

### Self-Healing
Gru monitors itself and fixes problems automatically.

- **Self-diagnostics** - Continuous health monitoring
- **Stuck detection** - Notices when agents aren't progressing
- **Auto-recovery** - Restarts failed services, fixes configuration
- **Bug detection** - Identifies errors and spawns fix agents
- **Performance monitoring** - Tracks slow operations and resource usage

### Development Tools
The original Gru - still the best way to code from your phone.

- **Natural language** - "Build me a landing page for a coffee shop"
- **Screenshots** - Send a UI mockup, get working code
- **Voice notes** - Speak your request while walking
- **Ralph loops** - Iterative AI development cycles
- **Live deploy** - Ship to Vercel in one message

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
| [Multi-Agent Orchestration](docs/agents.md) | Spawning and coordinating agents |
| [Self-Healing](docs/self-healing.md) | Diagnostics, auto-recovery, bug detection |
| [Autonomous Actions](docs/autonomous-actions.md) | Book reservations, send payments, order food |
| [Knowledge Graph](docs/knowledge-graph.md) | Query your relationships and history |
| [Proactive Intelligence](docs/proactive-intelligence.md) | Morning briefings, pattern learning |
| [Commands](docs/commands.md) | Full command reference |
| [Ralph Loops](docs/ralph.md) | Iterative AI development cycles |
| [Configuration](docs/configuration.md) | All environment variables |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [MCP Plugins](docs/mcp-plugins.md) | Extend with MCP servers |

## Why Gru?

**Other AI assistants** answer one question at a time, then forget.

**Gru** is your personal army. It spins up agents to handle complex tasks, learns your patterns over time, remembers every conversation, and takes real action in the world.

| Feature | ChatGPT | Siri | Gru |
|---------|---------|------|-----|
| Multi-agent orchestration | No | No | Yes |
| Autonomous task completion | No | No | Yes |
| Self-healing | No | No | Yes - auto-recovery |
| Book restaurants | No | Limited | Yes - OpenTable, Resy |
| Send payments | No | No | Yes - Venmo |
| Order food | No | No | Yes - DoorDash |
| Query past conversations | No | No | Yes - full history |
| Learn your patterns | No | No | Yes |
| Persistent memory | Limited | No | Yes - cross-session |
| Morning briefings | No | No | Yes |
| Run code | No | No | Yes |
| Deploy apps | No | No | Yes |

## Architecture

```
Telegram/Discord/Slack
         |
         v
   Orchestrator <---------> Claude API
         |
         +---> Agent Pool (spawn, coordinate, parallelize)
         |         |
         |         +---> Research Agents
         |         +---> Dev Agents
         |         +---> Task Agents
         |
         +---> Knowledge Graph (entities, relationships, history)
         +---> Memory Store (facts, preferences, learned patterns)
         +---> Proactive Engine (triggers, anticipation, briefings)
         +---> Autonomous Actions (reservations, payments, orders)
         +---> Connectors (Gmail, Calendar, Slack)
         +---> MCP Plugins (extensible tools)
```

## Privacy & Security

**Your data stays yours.** Gru runs on your infrastructure - not ours.

- All data stored locally in SQLite
- Knowledge graph lives on your machine
- No telemetry, no tracking, no cloud dependency
- Conversation history never leaves your server
- Browser sessions stored in your home directory

**Security controls:**
- Admin allowlist - only approved users can interact
- Action confirmations - payments and reservations require approval
- Supervised mode - approve commands before execution
- Workspace isolation - sandbox agent file access

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
ruff check src/ tests/
mypy src/
```

## Funding & Sponsors

This project is funded by the [$GRU token](https://gruonsol.com/) community on Solana.

**Contract Address:** `S782cLXpcS5agE3T6g7ADZ8LkmQsiDMKd7S2GzfGapp`

**Buy $GRU:** [Apps.fun](https://www.apps.fun/app/207) | [Jupiter Exchange](https://jup.ag/swap?sell=So11111111111111111111111111111111111111112&buy=S782cLXpcS5agE3T6g7ADZ8LkmQsiDMKd7S2GzfGapp)

Special thanks to our sponsors and community members who support open-source AI development.

## License

MIT
