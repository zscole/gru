# Multi-Agent Orchestration

Gru is not a single assistant. It's an orchestrator that spawns specialized agents to handle complex tasks in parallel.

## How It Works

When you give Gru a complex task, it:

1. Analyzes the request and breaks it into subtasks
2. Spawns specialized agents for each subtask
3. Coordinates between agents, sharing context
4. Aggregates results and delivers a unified response

Agents share a common memory layer, so insights from one task inform future tasks.

## Agent Types

| Agent | Specialty |
|-------|-----------|
| Research Agent | Web search, data gathering, summarization |
| Dev Agent | Code generation, debugging, deployment |
| Task Agent | Calendar, email, Slack, autonomous actions |
| PoC Agent | Builds working proof-of-concepts from research |
| Analysis Agent | Data processing, pattern recognition |

## Examples

### Research + Build

```
You: Research the top 3 competitors and build me a comparison landing page

Gru: Spinning up research agent...
     - Found Competitor A: focus on enterprise, $99/mo
     - Found Competitor B: focus on SMB, freemium model
     - Found Competitor C: focus on API-first, usage-based pricing

     Spinning up dev agent...
     - Building comparison table component
     - Adding feature matrix
     - Styling and responsive layout

     Done. Deployed to: https://compare.yoursite.com
```

### Multi-Task Coordination

```
You: Prep me for tomorrow's board meeting

Gru: Spinning up calendar agent...
     - Board meeting: 10am, Conference Room A
     - Attendees: Sarah (CEO), Mike (CFO), investors

     Spinning up research agent...
     - Sarah: recently focused on Q1 targets
     - Mike: preparing budget review
     - Investor concerns: burn rate, growth metrics

     Spinning up email agent...
     - 3 relevant threads from last week
     - Outstanding questions about runway

     Summary ready. Key topics:
     1. Q1 revenue vs targets
     2. Hiring plan for Q2
     3. Runway projections
```

### Parallel Autonomous Actions

```
You: Book dinner at Nobu, venmo Mike for the tickets, and remind me Friday

Gru: Executing in parallel:
     - [Reservation] Booking Nobu for Saturday 7pm...
     - [Payment] Sending $50 to @mike...
     - [Reminder] Setting Friday reminder...

     All done:
     - Nobu confirmed for 7:00 PM
     - Venmo sent ($50 to @mike, note: "tickets")
     - Reminder set for Friday 9:00 AM
```

## Agent Communication

Agents communicate through a shared context layer:

- **Memory Store** - Facts, preferences, learned patterns
- **Knowledge Graph** - Entities, relationships, interaction history
- **Task Queue** - Pending work, dependencies, status

When one agent learns something (e.g., "user prefers window seats"), all agents have access.

## Spawning Custom Agents

Gru can spawn agents for specific workflows:

```
You: Create an agent that monitors my inbox for investor emails

Gru: Created "investor-monitor" agent:
     - Triggers: new email from @investor-domain.com
     - Actions: summarize, flag as important, notify via Telegram
     - Learning: adapts to your response patterns
```

## Parallelization

By default, independent subtasks run in parallel. Sequential dependencies are detected automatically.

```
Research competitors  ----+
                          |---> Build comparison page ---> Deploy
Analyze their pricing ----+
```

## Limits

- Max concurrent agents: 5 (configurable)
- Agent timeout: 5 minutes per subtask
- Memory sharing: real-time within session, async across sessions

## Tools

| Tool | Description |
|------|-------------|
| `spawn_agent` | Create a new specialized agent |
| `list_agents` | See running and recent agents |
| `agent_status` | Check status of specific agent |
| `stop_agent` | Cancel a running agent |
| `create_workflow` | Define a reusable multi-agent workflow |
