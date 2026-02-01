# Self-Healing

Gru monitors itself, detects problems, and fixes them automatically. It can notice suboptimal performance, stuck agents, service failures, and code bugs - then take corrective action.

## How It Works

The self-heal engine continuously monitors:

1. **Health checks** - Python environment, dependencies, database, services
2. **Error patterns** - Tracks error frequency and detects spikes
3. **Performance metrics** - Agent duration, token usage, response times
4. **Agent state** - Detects stuck agents not making progress
5. **Resource usage** - Memory, disk space

When issues are detected, Gru can:
- Restart failed services
- Nudge stuck agents
- Install missing packages
- Spawn fix agents for code bugs
- Alert you to problems it can't auto-fix

## Self-Diagnostics

Run a full diagnostic check:

```
You: Run self-diagnostics

Gru: Diagnostic Results:
     Status: DEGRADED
     Checks passed: 7
     Checks failed: 1

     Issues:
     [HIGH] Unhealthy MCP server: file-search

     Recommendations:
     - Run self-heal to attempt automatic fixes
     - 1 issue can be auto-fixed
```

## Auto-Healing

Fix detected issues automatically:

```
You: Self-heal

Gru: Self-heal complete:
     Fixed: 1
     Failed: 0

     Details:
     [FIXED] Unhealthy MCP server: Recovered 1 server(s)
```

## Issue Categories

| Category | Description | Auto-fixable |
|----------|-------------|--------------|
| `performance` | Slow operations detected | No |
| `error_rate` | Error spike detected | Yes (resets counters) |
| `memory` | High memory usage | No |
| `stuck_agent` | Agent not progressing | Yes (sends nudge) |
| `service_down` | Service unhealthy | Yes (restarts) |
| `code_bug` | Code error detected | Yes (spawns fix agent) |
| `configuration` | Missing package or config | Yes (installs) |

## Stuck Agent Detection

Gru detects when agents stop making progress:

- Tracks turns without tool calls
- After threshold (default 5), marks as stuck
- Auto-fix: Sends a nudge message to prompt action
- If still stuck: Alerts you

```
You: Why is my agent stuck?

Gru: Agent agent-abc123 has gone 7 turns without using any tools.
     This usually means it's looping on explanations instead of acting.

     Running self-heal...
     [FIXED] Sent nudge message to stuck agent
```

## Error Pattern Detection

Gru tracks errors and detects spikes:

```
Error spike detected: API timeout occurred 12 times in last hour

Possible causes:
- Network issues
- Rate limiting
- Service degradation

Action taken:
- Reset error counters
- Logged for investigation
```

## Code Bug Detection

When Gru encounters an error in its own code:

```
You: analyze_error "KeyError: 'user_id'" traceback="..." file_path="src/gru/agent.py"

Gru: Issue created: abc123
     Category: code_bug
     Severity: high

     Spawning fix agent to investigate and repair...
     Fix agent agent-fix-abc123 started.
```

The fix agent will:
1. Analyze the error and traceback
2. Locate the relevant code
3. Implement a fix
4. Verify the fix works

## Health Status

Get a quick health overview:

```
You: What's my health status?

Gru: Health Status:
     Status: healthy
     Last diagnostic: 2024-01-30T10:15:00
     Open issues: 0

     Issues by severity:
       critical: 0
       high: 0
       medium: 0
       low: 0

     Total errors recorded: 3
     Fixes attempted: 1
```

## Tools

| Tool | Description |
|------|-------------|
| `run_self_diagnostic` | Run comprehensive health checks |
| `self_heal` | Attempt to fix detected issues |
| `get_health_status` | Get health summary |
| `list_issues` | List detected issues |
| `analyze_error` | Analyze an error and attempt fix |

## Configuration

Self-healing runs automatically. Thresholds can be adjusted:

| Setting | Default | Description |
|---------|---------|-------------|
| Stuck threshold | 5 turns | Turns without tool calls before stuck alert |
| Error spike threshold | 10/hour | Errors per hour before spike alert |
| Slow operation threshold | 30s | Duration before slow operation alert |
| Memory alert threshold | 1GB | Memory usage before alert |

## Privacy

- All diagnostics run locally
- No health data sent externally
- Error patterns stored in memory (not persisted)
- Fix agents work on local codebase only

## Examples

### Proactive Health Check

Gru can run periodic health checks:

```
You: Check health every hour and fix any issues

Gru: Created proactive trigger:
     - Schedule: Every hour
     - Action: Run diagnostics, auto-fix if issues found
     - Notify: On critical issues only
```

### Self-Improvement

Gru can analyze its own performance:

```
You: Analyze your performance this week

Gru: Performance Analysis:

     Operations:
     - 47 agent runs, avg 45s
     - 3 slow operations (>30s)

     Errors:
     - 12 total errors
     - Most common: timeout (5x)

     Self-heals:
     - 2 stuck agents recovered
     - 1 MCP server restarted

     Recommendations:
     - Consider increasing timeout for research tasks
     - MCP server 'web-search' has been unstable
```
