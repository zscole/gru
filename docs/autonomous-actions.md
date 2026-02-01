# Autonomous Actions

Gru can take real-world actions on your behalf - booking reservations, sending messages, making payments, and more.

## How It Works

1. You request an action ("Book a table at Nobu for Saturday at 7pm")
2. Gru shows you a preview of what will happen
3. You confirm or cancel
4. Gru executes the action and reports back

All actions requiring money or external changes require explicit confirmation.

## Available Actions

### Communication

| Action | Description | Example |
|--------|-------------|---------|
| `send_email` | Send email via Gmail | "Email John about the meeting tomorrow" |
| `send_slack` | Send Slack message | "Message #general that I'll be late" |
| `send_sms` | Send SMS via Twilio | "Text Sarah that I'm on my way" |

### Calendar

| Action | Description | Example |
|--------|-------------|---------|
| `create_event` | Create calendar event | "Schedule a call with Mike tomorrow at 3pm" |
| `update_event` | Modify existing event | "Move my 2pm meeting to 4pm" |
| `delete_event` | Delete event | "Cancel my dentist appointment" |

### Reservations

| Action | Description | Example |
|--------|-------------|---------|
| `opentable_reservation` | Book via OpenTable | "Book Nobu for 2 on Saturday at 7pm" |
| `resy_reservation` | Book via Resy | "Get me a table at Carbone this Friday" |

### Payments

| Action | Description | Example |
|--------|-------------|---------|
| `venmo_payment` | Send Venmo payment | "Venmo @john $25 for lunch" |

Note: Venmo payments have a $500 safety limit.

### Purchases

| Action | Description | Example |
|--------|-------------|---------|
| `doordash_order` | Order food | "Order my usual from Chipotle" |
| `amazon_order` | Order from Amazon | "Order more coffee pods" |

## Confirmation Flow

When you request an action, Gru shows:

```
ACTION REQUEST: Book Nobu for 2 on Saturday at 7:00 PM

What will happen:
  - Restaurant: Nobu
  - Date: Saturday
  - Time: 7:00 PM
  - Party size: 2

Reply 'confirm abc123' to proceed or 'cancel abc123' to cancel.
```

## Managing Actions

- `list_pending_actions` - See actions waiting for confirmation
- `confirm_action <id>` - Confirm and execute
- `cancel_action <id>` - Cancel a pending action
- `action_history` - View past actions

## Setup Requirements

| Action Type | Requirements |
|-------------|--------------|
| Email/Calendar | Google OAuth (`gru google login`) |
| Slack | Slack token (`gru slack setup`) |
| SMS | Twilio credentials in env vars |
| Reservations | Playwright + browser login |
| Payments | Playwright + browser login |
| Purchases | Playwright + browser login |

## Browser Automation

Actions for OpenTable, Resy, Venmo, DoorDash, and Amazon use browser automation via Playwright. On first use:

1. A Chrome window opens
2. Log in to the service
3. Gru saves the session for future use

Sessions are stored securely in `~/.gru/browser-profiles/`.

## Security

- All external actions require confirmation
- Payment actions have safety limits
- Browser sessions stored in user home directory
- No credentials stored in code or logs
