# Proactive Intelligence

Gru learns your patterns, anticipates your needs, and proactively provides useful information without being asked.

## Features

### Morning Briefings

Every morning at 6am, Gru sends you a summary:

```
Good morning. Here's your daily briefing:

CALENDAR
  9:00 AM - Standup
  2:00 PM - 1:1 with Sarah
  4:00 PM - Product review

EMAIL
  12 unread emails
  3 marked important:
    - Mike: Q1 Budget Review
    - Sarah: API docs ready
    - HR: Benefits enrollment

SLACK
  5 unread DMs
  2 mentions in #engineering
```

### Pattern Learning

Gru learns your behavior patterns:

- "You usually check email at 9am"
- "You often order lunch around noon"
- "You typically review PRs on Monday mornings"

### Anticipation

Based on learned patterns, Gru anticipates needs:

- "Your meeting with John is in 30 minutes - traffic is heavy, consider leaving now"
- "You haven't responded to Sarah's email from yesterday"
- "Based on your calendar, you might want to prep for tomorrow's board meeting"

### AI Research

Gru monitors AI news and creates proof-of-concept implementations:

- Scans HackerNews, Reddit, GitHub trending
- Identifies significant technical developments
- Automatically creates working PoCs
- Sends you a summary with GitHub repo link

## Configuration

### Triggers

Built-in triggers in the proactive engine:

| Trigger | Schedule | Description |
|---------|----------|-------------|
| `morning_briefing` | 6:00 AM | Daily summary |
| `slack_sync` | Every 10 min | Check for new messages |
| `email_check` | Every 30 min | Check for important emails |
| `calendar_reminder` | Before events | Remind about upcoming meetings |
| `ai_research` | Daily | Check for AI developments |

### Customization

Adjust proactive behavior:

```
You: Don't send morning briefings on weekends
You: Check email every hour instead of 30 minutes
You: Only alert me for emails from @important-domain.com
```

## Behavior Tracking

Gru tracks actions to learn patterns:

- Message send times
- Tool usage patterns
- Topic interests
- Response patterns

This data stays local and is used only for improving anticipation.

## Observations

Gru detects follow-up items from conversations:

- **Deadlines** - "The report is due Friday"
- **Commitments** - "I'll send you the doc tomorrow"
- **Questions** - Unanswered questions to follow up on
- **Opportunities** - Suggestions that weren't acted on

## Privacy

- All pattern data stored locally
- No behavior data sent externally
- You can view and delete learned patterns
- Proactive features can be disabled

## Tools

| Tool | Description |
|------|-------------|
| `get_morning_summary` | Get briefing on demand |
| `check_breaking_news` | Check AI news now |
| `get_patterns` | View learned patterns |
| `check_anticipations` | See what Gru anticipates |
| `get_insights` | Get proactive insights |

## Example Anticipations

```
You: What should I know about today?

Gru: Based on your patterns and calendar:

ANTICIPATIONS:
  - You have a 1:1 with Sarah at 2pm - she usually asks about sprint progress
  - You typically review the weekly metrics on Fridays - today's Friday
  - Traffic to downtown is heavy - leave 15 min early for your 4pm

OBSERVATIONS:
  - You mentioned following up with Mike about the API - no response yet
  - The Q1 planning doc is due Monday - 3 days left

PATTERNS NOTICED:
  - You've been spending more time on Slack #support lately
  - Your email response time has decreased this week
```

## Requirements

| Feature | Requirements |
|---------|--------------|
| Morning briefing | Gmail + Calendar connected |
| Slack sync | Slack connected |
| AI research | No additional setup |
| Pattern learning | Enabled by default |
