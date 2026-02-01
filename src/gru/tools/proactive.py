"""Proactive engine tools - reminders, triggers, observations, behavior tracking.

This module exposes the ProactiveEngine to the agent including:
- Reminders and scheduled notifications
- Behavior tracking for pattern learning
- Anticipation based on learned patterns
- Insight generation and delivery
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.proactive import ProactiveEngine

logger = logging.getLogger(__name__)

_engine: ProactiveEngine | None = None


def set_proactive_engine(engine: ProactiveEngine) -> None:
    """Set the proactive engine instance."""
    global _engine
    _engine = engine


def get_proactive_engine() -> ProactiveEngine | None:
    """Get the proactive engine instance."""
    return _engine


def _parse_time(time_str: str) -> datetime | None:
    """Parse a time string into a datetime."""
    import re

    now = datetime.now()
    time_str = time_str.lower().strip()

    # Handle "in X minutes/hours/seconds"
    match = re.search(r"in\s+(\d+)\s*(min|minute|hour|hr|second|sec)s?", time_str)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit in ("min", "minute"):
            return now + timedelta(minutes=amount)
        elif unit in ("hour", "hr"):
            return now + timedelta(hours=amount)
        elif unit in ("second", "sec"):
            return now + timedelta(seconds=amount)

    # Handle "at HH:MM" or "at H:MM AM/PM"
    match = re.search(r"(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)

        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    return None


async def set_reminder(message: str, time: str) -> dict:
    """Set a reminder using the proactive engine."""
    from gru.proactive import TriggerType

    if not _engine:
        return {"error": "Proactive engine not initialized"}

    target_time = _parse_time(time)
    if not target_time:
        return {"error": f"Couldn't parse time '{time}'. Try 'in 5 minutes' or 'at 4:30 PM'."}

    now = datetime.now()
    if target_time <= now:
        return {"error": "That time has already passed."}

    # Use add_trigger with SCHEDULED type for time-based notification
    schedule_str = target_time.strftime("%H:%M")

    trigger_id = await _engine.add_trigger(
        name=f"reminder_{target_time.strftime('%H%M')}",
        trigger_type=TriggerType.SCHEDULED,
        action=f"notify:{message}",
        schedule=schedule_str,
    )

    time_str = target_time.strftime("%I:%M %p")
    delay_seconds = int((target_time - now).total_seconds())

    return {
        "status": "scheduled",
        "reminder_id": trigger_id,
        "message": message,
        "time": time_str,
        "in_seconds": delay_seconds,
    }


async def list_reminders() -> dict:
    """List pending reminders (triggers with reminder_ prefix)."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    triggers = await _engine.list_triggers()
    reminders = [
        {
            "id": t["id"],
            "message": t["action"].replace("notify:Reminder: ", ""),
            "time": t.get("schedule", ""),
            "last_fired": t.get("last_fired"),
        }
        for t in triggers
        if t["name"].startswith("reminder_")
    ]

    return {"reminders": reminders, "count": len(reminders)}


async def cancel_reminder(reminder_id: str) -> dict:
    """Cancel a pending reminder."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    success = await _engine.remove_trigger(reminder_id)
    if success:
        return {"status": "cancelled", "reminder_id": reminder_id}
    return {"error": f"Reminder {reminder_id} not found"}


# Behavior tracking tools

async def track_action(action: str, context: dict[str, Any] | None = None) -> dict:
    """Track a user action for pattern learning.

    Call this when the user does something trackable:
    - check_email, send_message, search, make_call, check_calendar, etc.
    """
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        behavior_id = await _engine.track_behavior(action, context)
        return {"status": "tracked", "behavior_id": behavior_id, "action": action}
    except Exception as e:
        logger.error(f"Failed to track action: {e}")
        return {"error": str(e)}


async def get_learned_patterns() -> dict:
    """Get all patterns learned from user behavior."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        patterns = await _engine.list_patterns()
        return {"patterns": patterns, "count": len(patterns)}
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        return {"error": str(e)}


async def get_behavior_stats() -> dict:
    """Get statistics about tracked user behaviors."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        return await _engine.get_behavior_stats()
    except Exception as e:
        logger.error(f"Failed to get behavior stats: {e}")
        return {"error": str(e)}


async def check_anticipations() -> dict:
    """Check what the user might want based on learned patterns and current context."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        now = datetime.now()
        context = await _engine._build_context()

        anticipations = []
        for pattern in _engine._patterns.values():
            if pattern.matches_now(now, context):
                anticipations.append({
                    "action": pattern.action,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                })

        return {
            "anticipations": anticipations,
            "count": len(anticipations),
            "context_summary": await _engine.get_context_summary(),
        }
    except Exception as e:
        logger.error(f"Failed to check anticipations: {e}")
        return {"error": str(e)}


async def get_proactive_insights() -> dict:
    """Get pending proactive insights about user behavior."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        insights = await _engine.get_pending_insights()
        return {"insights": insights, "count": len(insights)}
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        return {"error": str(e)}


async def get_proactive_context() -> dict:
    """Get full proactive context for injection into conversations."""
    if not _engine:
        return {"context": ""}

    try:
        parts = []

        # Get observation summary
        obs_summary = await _engine.get_observation_summary()
        if obs_summary:
            parts.append(obs_summary)

        # Get context summary
        ctx_summary = await _engine.get_context_summary()
        if ctx_summary and ctx_summary != "No proactive context":
            parts.append(f"PROACTIVE CONTEXT: {ctx_summary}")

        # Get pending insights
        insights = await _engine.get_pending_insights()
        if insights:
            insight_str = " | ".join(i["content"][:50] for i in insights[:3])
            parts.append(f"INSIGHTS: {insight_str}")

        return {"context": "\n".join(parts) if parts else ""}
    except Exception as e:
        logger.error(f"Failed to get proactive context: {e}")
        return {"context": ""}


async def get_morning_summary() -> dict:
    """Generate and return the morning summary (email, calendar, Slack)."""
    if not _engine:
        return {"error": "Proactive engine not initialized"}

    try:
        summary = await _engine.generate_morning_summary()
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Failed to generate morning summary: {e}")
        return {"error": str(e)}


def register_proactive_tools() -> None:
    """Register proactive/reminder tools."""
    # Reminder tools
    register_tool(
        name="set_reminder",
        description="Set a reminder to send a notification at a specific time. Use when user asks to be reminded about something.",
        parameters={
            "message": {
                "type": "string",
                "description": "The reminder message",
            },
            "time": {
                "type": "string",
                "description": "When to remind (e.g., 'in 5 minutes', 'at 4:30 PM')",
            },
        },
        handler=set_reminder,
    )

    register_tool(
        name="list_reminders",
        description="List all pending reminders.",
        parameters={},
        handler=list_reminders,
    )

    register_tool(
        name="cancel_reminder",
        description="Cancel a pending reminder.",
        parameters={
            "reminder_id": {
                "type": "string",
                "description": "The ID of the reminder to cancel",
            },
        },
        handler=cancel_reminder,
    )

    # Behavior tracking tools
    register_tool(
        name="track_action",
        description="Track a user action for pattern learning. Call this when the user does something trackable like checking email, searching, sending messages, etc. This helps learn their habits and anticipate needs.",
        parameters={
            "action": {
                "type": "string",
                "description": "What the user did (check_email, search, send_message, make_call, check_calendar, etc.)",
            },
            "context": {
                "type": "object",
                "description": "Additional context like location, query, recipient, etc.",
            },
        },
        handler=track_action,
    )

    register_tool(
        name="get_patterns",
        description="Get all patterns learned from user behavior. Shows what habits have been detected like 'usually checks email at 9am'.",
        parameters={},
        handler=get_learned_patterns,
    )

    register_tool(
        name="check_anticipations",
        description="Check what the user might want to do based on learned patterns and current time/context. Use this to proactively offer help.",
        parameters={},
        handler=check_anticipations,
    )

    register_tool(
        name="get_insights",
        description="Get proactive insights about user behavior, spending patterns, time usage, etc. These are observations that might be useful to share.",
        parameters={},
        handler=get_proactive_insights,
    )

    register_tool(
        name="get_behavior_stats",
        description="Get statistics about tracked user behaviors including total actions, breakdown by type, and recent activity.",
        parameters={},
        handler=get_behavior_stats,
    )

    register_tool(
        name="get_morning_summary",
        description="Get the morning briefing summary including today's calendar, unread emails, and Slack messages. Use when user asks for their daily summary, briefing, or 'what's on today'.",
        parameters={},
        handler=get_morning_summary,
    )
