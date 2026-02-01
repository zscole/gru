# Knowledge Graph

Gru builds a personal knowledge graph of your relationships, conversations, and history. Ask questions like "What did I discuss with John last month?" and get instant answers.

## How It Works

Gru automatically extracts entities and relationships from:
- Conversations with Gru
- Emails (via Gmail)
- Slack messages
- Calendar events

Everything is stored locally in your database and indexed for fast retrieval.

## Querying Your Knowledge

### Natural Language Queries

```
"What did I discuss with John last month?"
"Who have I been meeting with recently?"
"What do I know about Project Alpha?"
"What topics have we talked about this week?"
```

### Tools

| Tool | Description |
|------|-------------|
| `ask_knowledge` | Answer natural language questions about your history |
| `get_person_info` | Get details about a known person |
| `get_recent_interactions` | Filter by entity, source, or time |
| `get_known_people` | List all known contacts |
| `get_recent_topics` | Topics discussed recently |
| `add_person` | Manually add someone to the graph |
| `add_relationship` | Add a relationship between entities |
| `knowledge_stats` | View graph statistics |

## Entity Types

Gru tracks these entity types:

| Type | Examples |
|------|----------|
| `person` | John Smith, Sarah, @john |
| `project` | Project Alpha, gru, website redesign |
| `company` | Acme Corp, Google, Anthropic |
| `topic` | API design, database migration, funding |
| `tool` | React, PostgreSQL, Docker |
| `place` | San Francisco, NYC office |

## Relationships

Entities are connected by relationships:

- `works_with` - Professional relationship
- `knows` - Personal acquaintance
- `works_on` - Person works on project
- `discussed_with` - Talked about topic with person
- `mentioned_with` - Co-occurred in conversation

## Time-Based Queries

All interactions are timestamped, enabling queries like:

- "last month" - 30-60 days ago
- "this week" - Past 7 days
- "recently" - Past 14 days
- "yesterday" - Past 2 days

## Data Sources

### Conversations
Every conversation with Gru is analyzed for entities and relationships.

### Email
When Gmail is connected, Gru extracts:
- Sender as person entity
- Subject and topics discussed
- Timestamp for time-based queries

### Slack
When Slack is connected, Gru extracts:
- Message senders
- Channel context
- Topics mentioned

### Calendar
Calendar events provide:
- Meeting attendees
- Event topics
- Time context

## Privacy

- All data stored locally in SQLite
- No data sent to external services (except LLM for extraction)
- Entity extraction uses Claude Haiku for efficiency
- You control what gets indexed

## Example Session

```
You: What did I discuss with Sarah last week?

Gru: Found 3 interactions:
  [2024-01-28] (conversation) Sarah: Discussion about API redesign
  [2024-01-27] (email) Sarah: Q1 planning meeting notes
  [2024-01-25] (slack) Sarah: Quick sync on deployment timeline

You: Who else is working on the API redesign?

Gru: About API redesign:
  - Sarah works_on API redesign
  - Mike works_on API redesign
  - discussed_with John (review meeting)
```

## Statistics

View your knowledge graph stats:

```
You: knowledge stats

Gru: Knowledge Graph Statistics:
  Entities by type:
    person: 47
    project: 12
    topic: 89
    company: 8
  Total relationships: 234
  Total interactions: 1,847
  Interactions (last 7 days): 42
```
