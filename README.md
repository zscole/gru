# Gru

[![CI](https://github.com/zscole/gru/actions/workflows/ci.yml/badge.svg)](https://github.com/zscole/gru/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Self-hosted AI agent orchestration service controlled via Telegram.

Gru lets you spawn, manage, and interact with Claude-powered AI agents from your phone. Agents can execute bash commands, read/write files, and work autonomously on tasks in specified directories.

## Features

- **Telegram Control**: Spawn and manage agents from anywhere via Telegram
- **Natural Language**: Chat naturally instead of using commands
- **Multiple Execution Modes**: Supervised, unsupervised, or fully autonomous (oneshot)
- **Working Directory Isolation**: Each agent operates in its own directory
- **Tool Support**: Bash execution, file operations, glob/grep search
- **MCP Plugins**: Extend agent capabilities with Model Context Protocol servers
- **Secret Storage**: Encrypted storage for API keys and credentials
- **Task Templates**: Save and reuse common task configurations
- **Priority Scheduling**: High/normal/low priority with anti-starvation

## Requirements

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Anthropic API Key
- Your Telegram User ID (from [@userinfobot](https://t.me/userinfobot))

## Installation

```bash
git clone https://github.com/zscole/gru.git
cd gru
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```bash
# Required
GRU_TELEGRAM_TOKEN=your_bot_token_here
GRU_ADMIN_IDS=your_telegram_user_id
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional
GRU_MASTER_PASSWORD=your_encryption_password    # For secret storage
GRU_DATA_DIR=/path/to/data                      # Default: ~/.gru
GRU_WORKDIR=/path/to/workspace                  # Default: ~/gru-workspace
GRU_DEFAULT_MODEL=claude-sonnet-4-20250514      # Claude model to use
GRU_MAX_TOKENS=8192                             # Max tokens per response
GRU_DEFAULT_TIMEOUT=300                         # Agent timeout in seconds
GRU_MAX_AGENTS=10                               # Max concurrent agents
```

## Running

```bash
source .venv/bin/activate
PYTHONPATH=src python -m gru.main
```

Output on successful start:
```
MCP server 'filesystem' started with 14 tools
Started 1 MCP server(s)
Gru server started
Data directory: /path/to/data
Admin IDs: [your_id]
```

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
PYTHONPATH=src pytest tests/ -v
```

Lint:

```bash
ruff check src/ tests/
```

## Usage

### Commands

All commands use the `/gru` prefix:

```
/gru help                          Show all commands
/gru spawn <task>                  Spawn agent (supervised mode)
/gru spawn <task> --unsupervised   No approval required
/gru spawn <task> --oneshot        Fully autonomous, fire and forget
/gru spawn <task> --workdir /path  Work in specific directory
/gru spawn <task> --priority high  Set priority (high/normal/low)

/gru status                        Overall orchestrator status
/gru status <agent_id>             Specific agent status
/gru list                          List all agents
/gru list running                  Filter by status

/gru pause <agent_id>              Pause agent execution
/gru resume <agent_id>             Resume paused agent
/gru terminate <agent_id>          Stop and remove agent
/gru nudge <agent_id> <message>    Send message to running agent
/gru logs <agent_id>               View agent conversation log
```

### Execution Modes

**Supervised (default)**: Agent asks for approval before file writes and bash commands. You receive inline buttons to approve or reject each action.

**Unsupervised**: Agent executes all actions without asking. Use `--unsupervised` flag.

**Oneshot**: Fully autonomous fire-and-forget mode. Agent runs to completion without any interaction. Use `--oneshot` flag.

### Natural Language

You can chat naturally instead of using commands:

```
"build a hello world python script"
"check on my running agents"
"what's in the current directory?"
```

Gru interprets your intent and either spawns an agent or answers directly.

### Secrets

Store encrypted credentials that agents can access:

```
/gru secret set GITHUB_TOKEN ghp_xxxxx
/gru secret get GITHUB_TOKEN
/gru secret list
/gru secret delete GITHUB_TOKEN
```

Requires `GRU_MASTER_PASSWORD` to be set.

### Templates

Save and reuse task configurations:

```
/gru template save pytest-runner run pytest and fix any failures
/gru template list
/gru template use pytest-runner
/gru template delete pytest-runner
```

### Approvals

In supervised mode, agents request approval for sensitive actions:

```
/gru pending                       List pending approvals
/gru approve <approval_id>         Approve action
/gru reject <approval_id>          Reject action
```

You can also use the inline buttons sent with each approval request.

## MCP Plugins

Gru supports [Model Context Protocol](https://modelcontextprotocol.io/) servers to extend agent capabilities.

### Configuration

Create `mcp_servers.json` in the project root:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your_token"
      }
    }
  }
}
```

MCP servers start automatically when Gru launches. Tools are prefixed with the server name (e.g., `filesystem__read_file`).

### Available MCP Servers

- `@modelcontextprotocol/server-filesystem` - File system operations
- `@modelcontextprotocol/server-github` - GitHub API
- `@modelcontextprotocol/server-postgres` - PostgreSQL queries
- `@modelcontextprotocol/server-sqlite` - SQLite database
- `@modelcontextprotocol/server-puppeteer` - Browser automation

See [MCP Servers](https://github.com/modelcontextprotocol/servers) for more.

## Architecture

```
Telegram Bot <-> Orchestrator <-> Claude API
                     |
                     +-> Scheduler (priority queue)
                     +-> Database (SQLite)
                     +-> MCP Client (plugin tools)
                     +-> Secret Store (encrypted)
```

**Components**:

- `telegram_bot.py` - Telegram interface, command parsing, natural language routing
- `orchestrator.py` - Agent lifecycle, tool execution, approval flow
- `claude.py` - Claude API client, tool definitions
- `scheduler.py` - Priority queue with anti-starvation
- `mcp.py` - MCP server management, tool routing
- `crypto.py` - Encryption for secret storage
- `db.py` - SQLite persistence

## Agent Tools

Agents have access to these built-in tools:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `glob` | Find files by pattern |
| `grep` | Search file contents |
| `task_complete` | Signal task completion |

Plus any tools provided by configured MCP servers.

## Security Notes

- Only users in `GRU_ADMIN_IDS` can interact with the bot
- Supervised mode is the default for a reason
- Agents execute commands with your user permissions
- Set `GRU_WORKDIR` to isolate agent file operations
- Review `mcp_servers.json` carefully before adding servers

## Troubleshooting

**Bot not responding**: Check that your Telegram user ID is in `GRU_ADMIN_IDS`

**Agent stuck**: Use `/gru nudge <id> <message>` or `/gru terminate <id>`

**MCP server failed**: Check server command exists (e.g., `npx` requires Node.js)

**No approvals appearing**: Ensure you're in supervised mode (default)

## License

MIT
