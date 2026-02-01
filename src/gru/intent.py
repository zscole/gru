"""Intent classification and action dispatch for natural language requests."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.actions.executor import ActionExecutor
    from gru.claude import ClaudeClient
    from gru.memory import MemoryStore
    from gru.proactive import ProactiveEngine

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """A classified intent from user input."""

    category: str  # food_order, search, reminder, calendar, general
    action: str | None  # Specific action to take
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    requires_action: bool = False  # Needs to execute an action
    schedule_for: datetime | None = None  # When to execute (if scheduled)
    original_text: str = ""
    needs_location: bool = False  # Requires user location to execute


# Pattern-based intent detection for common requests
INTENT_PATTERNS = [
    # Food ordering
    {
        "pattern": r"(order|get|buy)\s+(me\s+)?(some\s+)?(.+?)\s+(for\s+)?(lunch|dinner|breakfast|food)",
        "category": "food_order",
        "action": "ubereats_order",
        "extract": {"item": 4, "meal": 6},
    },
    {
        "pattern": r"(order|get|buy)\s+(me\s+)?(a\s+|some\s+)?(.+?)\s+(from|at)\s+(.+)",
        "category": "food_order",
        "action": "ubereats_order",
        "extract": {"item": 4, "restaurant": 6},
    },
    {
        "pattern": r"i('d|\s+would)\s+like\s+(a\s+|some\s+)?(.+?)\s+(for\s+)?(lunch|dinner|breakfast)",
        "category": "food_order",
        "action": "ubereats_order",
        "extract": {"item": 3, "meal": 5},
    },
    {
        "pattern": r"(find|search|look\s+for)\s+(me\s+)?(the\s+)?(best\s+)?(.+?)\s+(near|around|in)\s+(.+)",
        "category": "food_search",
        "action": "restaurant_search",
        "extract": {"query": 5, "location": 7},
    },
    {
        "pattern": r"(find|search|look\s+for)\s+(me\s+)?(the\s+)?(best\s+)?(.+?)\s+(nearby|near\s+me|in\s+the\s+area)",
        "category": "food_search",
        "action": "restaurant_search",
        "extract": {"query": 5},
    },
    {
        "pattern": r"(find|search|look\s+for)\s+(me\s+)?(the\s+)?(best\s+|closest\s+)?(.+?)\s+(closest\s+to|near|close\s+to)\s+(my\s+)?(house|home|place|location)",
        "category": "food_search",
        "action": "restaurant_search",
        "extract": {"query": 5},
        "needs_location": True,
    },
    {
        "pattern": r"(where\s+is|where's)\s+(the\s+)?(closest|nearest)\s+(.+)",
        "category": "food_search",
        "action": "restaurant_search",
        "extract": {"query": 4},
        "needs_location": True,
    },
    # Distance/directions
    {
        "pattern": r"how\s+(far|long|close)\s+(is|does\s+it\s+take)\s+(to\s+)?(.+?)\s+(from|to)\s+(my\s+)?(house|home|place|location|here)",
        "category": "distance",
        "action": "get_distance",
        "extract": {"destination": 4},
        "needs_location": True,
    },
    {
        "pattern": r"how\s+(far|long|close)\s+is\s+(.+?)\s+from\s+(.+)",
        "category": "distance",
        "action": "get_distance",
        "extract": {"destination": 2, "origin": 3},
    },
    {
        "pattern": r"(what's|what\s+is)\s+the\s+distance\s+(to|from)\s+(.+)",
        "category": "distance",
        "action": "get_distance",
        "extract": {"destination": 3},
        "needs_location": True,
    },
    {
        "pattern": r"(directions|navigate|route)\s+(to|from)\s+(.+)",
        "category": "distance",
        "action": "get_distance",
        "extract": {"destination": 3},
        "needs_location": True,
    },
    # Calendar
    {
        "pattern": r"what('s|\s+is)\s+(on\s+)?(my\s+)?(calendar|schedule|agenda)",
        "category": "calendar",
        "action": "check_calendar",
        "extract": {},
    },
    {
        "pattern": r"(when|what\s+time)\s+(is|does)\s+(my\s+)?(.+?)\s+(meeting|call|event)",
        "category": "calendar",
        "action": "check_calendar",
        "extract": {"event": 4},
    },
    # Reminders
    {
        "pattern": r"remind\s+me\s+to\s+(.+?)(?:\s+(?:at|in|on)\s+(.+))?$",
        "category": "reminder",
        "action": "add_reminder",
        "extract": {"content": 1, "time": 2},
    },
    # Google Docs
    {
        "pattern": r"(write|create|make)\s+(this|that|it|a)?\s*(to\s+)?(a\s+)?(google\s+)?doc(ument)?",
        "category": "document",
        "action": "compile_document",
        "extract": {},
    },
    {
        "pattern": r"(write|create|make)\s+(a\s+)?(prd|product\s+requirements?\s+document)",
        "category": "document",
        "action": "compile_document",
        "extract": {"doc_type": "prd"},
    },
    {
        "pattern": r"(write|create|make)\s+(a\s+)?(meeting\s+notes?|notes?\s+from\s+the\s+meeting)",
        "category": "document",
        "action": "compile_document",
        "extract": {"doc_type": "meeting_notes"},
    },
    {
        "pattern": r"(create|make|start)\s+(a\s+)?(new\s+)?(google\s+)?doc(ument)?\s+(called|named|titled)\s+(.+)",
        "category": "document",
        "action": "create_document",
        "extract": {"title": 7},
    },
    # Email
    {
        "pattern": r"(email|send)\s+(the\s+)?(link|doc|document|it)\s+to\s+(.+)",
        "category": "email",
        "action": "send_email",
        "extract": {"to": 4},
    },
    {
        "pattern": r"(email|send\s+an?\s+email)\s+to\s+(.+?)\s+(about|with|saying)\s+(.+)",
        "category": "email",
        "action": "send_email",
        "extract": {"to": 2, "subject": 4},
    },
    # Research (scheduled patterns first - more specific)
    {
        "pattern": r"by\s+(morning|tomorrow|tonight|(\d+)\s*(am|pm|hours?))\s*[,:]?\s*(i('d|\s+would)\s+like|give\s+me|prepare|have)\s+(a\s+)?(\w+\s+)?(report|research|analysis)\s+(on|about)\s+(.+)",
        "category": "research_scheduled",
        "action": "research",
        "extract": {"topic": 10, "deadline": 1},
    },
    {
        "pattern": r"(i('d|\s+would)\s+like|i\s+want|i\s+need)\s+(a\s+)?(thorough\s+|detailed\s+)?(report|research|analysis)\s+(on|about)\s+(.+)",
        "category": "research",
        "action": "research",
        "extract": {"topic": 7},
    },
    {
        "pattern": r"(write|create|generate)\s+(a\s+)?(thorough\s+|detailed\s+|comprehensive\s+)?(report|analysis)\s+(on|about)\s+(.+)",
        "category": "research",
        "action": "research",
        "extract": {"topic": 6},
    },
    {
        "pattern": r"(do|run|conduct|perform)\s+(some\s+)?(research|analysis|investigation)\s+(on|about|into)\s+(.+)",
        "category": "research",
        "action": "research",
        "extract": {"topic": 5},
    },
    {
        "pattern": r"(research|investigate|analyze|look\s+into|study)\s+(.+)",
        "category": "research",
        "action": "research",
        "extract": {"topic": 2},
    },
    {
        "pattern": r"what\s+is\s+the\s+best\s+(.+?)(\s+for\s+(.+))?$",
        "category": "research",
        "action": "research",
        "extract": {"topic": 0},  # Use full match
    },
    {
        "pattern": r"(what|which)\s+(are|is)\s+the\s+(best|top|recommended)\s+(.+)",
        "category": "research",
        "action": "research",
        "extract": {"topic": 0},
    },
    # Quick answers
    {
        "pattern": r"^(what|who|when|where|why|how)\s+.+\?$",
        "category": "question",
        "action": "quick_answer",
        "extract": {"question": 0},
    },
]

# Meal times (default)
MEAL_TIMES = {
    "breakfast": (7, 9),   # 7am-9am
    "lunch": (11, 13),     # 11am-1pm
    "dinner": (18, 20),    # 6pm-8pm
}

# Time of day keywords
TIME_OF_DAY = {
    "morning": (6, 9),     # 6am-9am
    "tonight": (20, 23),   # 8pm-11pm
    "tomorrow": None,      # Next day, handled specially
    "end of day": (17, 18), # 5pm-6pm
    "eod": (17, 18),
}


def parse_time_expression(expr: str, reference: datetime | None = None) -> datetime | None:
    """Parse a time expression into a datetime."""
    if not expr:
        return None

    reference = reference or datetime.now()
    expr = expr.lower().strip()

    # Handle "in X minutes/hours"
    match = re.search(r"in\s+(\d+)\s+(minute|hour|min|hr)s?", expr)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit in ("minute", "min"):
            return reference + timedelta(minutes=amount)
        elif unit in ("hour", "hr"):
            return reference + timedelta(hours=amount)

    # Handle "at X:XX" or "at X pm/am"
    match = re.search(r"at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?", expr)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3)

        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        result = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if result < reference:
            result += timedelta(days=1)
        return result

    # Handle meal times
    for meal, (start, end) in MEAL_TIMES.items():
        if meal in expr:
            # Use the middle of the meal window
            target_hour = (start + end) // 2
            result = reference.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if result < reference:
                result += timedelta(days=1)
            return result

    # Handle time of day keywords
    for keyword, hours in TIME_OF_DAY.items():
        if keyword in expr:
            if keyword == "tomorrow":
                # Tomorrow at 8am
                result = reference.replace(hour=8, minute=0, second=0, microsecond=0)
                result += timedelta(days=1)
                return result
            elif hours:
                start, end = hours
                target_hour = (start + end) // 2
                result = reference.replace(hour=target_hour, minute=0, second=0, microsecond=0)
                if result < reference:
                    result += timedelta(days=1)
                return result

    # Handle "by morning" - means complete by morning, so start now
    if "by morning" in expr or "by tomorrow" in expr:
        # Return tomorrow morning
        result = reference.replace(hour=7, minute=0, second=0, microsecond=0)
        result += timedelta(days=1)
        return result

    return None


class IntentClassifier:
    """Classifies user intents and dispatches to actions."""

    def __init__(
        self,
        claude: ClaudeClient,
        memory: MemoryStore | None = None,
        proactive: ProactiveEngine | None = None,
    ) -> None:
        self.claude = claude
        self.memory = memory
        self.proactive = proactive

    async def classify(self, text: str, user_context: dict[str, Any] | None = None) -> Intent:
        """Classify a user message into an intent.

        Args:
            text: User's message
            user_context: Optional context (preferences, location, etc.)

        Returns:
            Classified Intent
        """
        text_lower = text.lower().strip()
        user_context = user_context or {}

        # First try pattern matching for common intents
        for pattern_def in INTENT_PATTERNS:
            match = re.search(pattern_def["pattern"], text_lower, re.IGNORECASE)
            if match:
                params = {}
                for param_name, group_idx in pattern_def.get("extract", {}).items():
                    if isinstance(group_idx, int) and group_idx <= len(match.groups()):
                        value = match.group(group_idx)
                        if value:
                            params[param_name] = value.strip()

                # Determine scheduling
                schedule_for = None
                if "meal" in params:
                    schedule_for = parse_time_expression(params["meal"])
                elif "time" in params:
                    schedule_for = parse_time_expression(params["time"])
                elif "deadline" in params:
                    schedule_for = parse_time_expression(params["deadline"])

                return Intent(
                    category=pattern_def["category"],
                    action=pattern_def["action"],
                    parameters=params,
                    confidence=0.8,
                    requires_action=pattern_def["action"] is not None,
                    schedule_for=schedule_for,
                    original_text=text,
                    needs_location=pattern_def.get("needs_location", False),
                )

        # Fall back to LLM classification for complex intents
        return await self._llm_classify(text, user_context)

    async def _llm_classify(self, text: str, user_context: dict[str, Any]) -> Intent:
        """Use Claude to classify complex intents."""
        system_prompt = """You are an intent classifier. Analyze the user's message and extract:
1. The category (food_order, food_search, calendar, reminder, general)
2. The action to take (if any)
3. Key parameters
4. Whether it requires executing an action vs just conversation
5. Any scheduling/timing information

Respond in JSON format:
{
    "category": "string",
    "action": "string or null",
    "parameters": {},
    "requires_action": true/false,
    "schedule_time": "string description or null"
}

Available actions:
- ubereats_search: Search for food/restaurants
- ubereats_order: Order food
- restaurant_search: Find restaurants
- web_search: General web search
- check_calendar: Check calendar
- add_reminder: Add a reminder
- create_document: Create a new Google Doc
- write_document: Write content to an existing Google Doc
- send_email: Send an email via Gmail
- compile_document: Compile conversation into a formatted document (PRD, meeting notes, etc.)
- research: Conduct thorough research on a topic and generate a report
- quick_answer: Get a quick factual answer to a question

If the message is just conversation/chat, set requires_action to false."""

        context_str = ""
        if user_context:
            context_str = f"\n\nUser context: {json.dumps(user_context)}"

        try:
            response = await self.claude.send_message(
                messages=[{"role": "user", "content": f"Classify this message: {text}{context_str}"}],
                system=system_prompt,
                max_tokens=500,
            )

            # Parse JSON response
            content = response.content.strip()
            # Extract JSON from potential markdown code block
            if "```" in content:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)

            schedule_for = None
            if data.get("schedule_time"):
                schedule_for = parse_time_expression(data["schedule_time"])

            return Intent(
                category=data.get("category", "general"),
                action=data.get("action"),
                parameters=data.get("parameters", {}),
                confidence=0.7,
                requires_action=data.get("requires_action", False),
                schedule_for=schedule_for,
                original_text=text,
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return Intent(
                category="general",
                action=None,
                confidence=0.3,
                requires_action=False,
                original_text=text,
            )

    async def enrich_intent(
        self,
        intent: Intent,
        user_id: str,
        calendar_events: list[dict[str, Any]] | None = None,
    ) -> Intent:
        """Enrich an intent with user preferences and calendar context.

        Args:
            intent: The classified intent
            user_id: User ID for preference lookup
            calendar_events: Optional list of upcoming calendar events

        Returns:
            Enriched intent with additional context
        """
        if not intent.requires_action:
            return intent

        # Add location from preferences if not specified
        if self.memory and "location" not in intent.parameters:
            try:
                profile = await self.memory.get_user_profile()
                if profile.get("preferences", {}).get("location"):
                    intent.parameters["location"] = profile["preferences"]["location"]

                # Add food preferences if relevant
                if intent.category in ("food_order", "food_search"):
                    food_prefs = profile.get("preferences", {}).get("food")
                    if food_prefs:
                        intent.parameters["preferences"] = food_prefs
            except Exception as e:
                logger.warning(f"Failed to load preferences: {e}")

        # Adjust timing based on calendar
        if calendar_events and intent.schedule_for:
            intent = self._adjust_for_calendar(intent, calendar_events)

        return intent

    def _adjust_for_calendar(
        self,
        intent: Intent,
        calendar_events: list[dict[str, Any]],
    ) -> Intent:
        """Adjust intent timing based on calendar events."""
        if not intent.schedule_for:
            return intent

        target_time = intent.schedule_for

        # Find events around the target time
        for event in calendar_events:
            event_start = event.get("start_time")
            event_end = event.get("end_time")

            if not event_start:
                continue

            # Parse event times if strings
            if isinstance(event_start, str):
                try:
                    event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
                    event_start = event_start.replace(tzinfo=None)
                except Exception:
                    continue

            if isinstance(event_end, str):
                try:
                    event_end = datetime.fromisoformat(event_end.replace("Z", "+00:00"))
                    event_end = event_end.replace(tzinfo=None)
                except Exception:
                    event_end = event_start + timedelta(hours=1)

            # Check if target time conflicts with event
            if event_start <= target_time <= (event_end or event_start + timedelta(hours=1)):
                # Reschedule to after the event
                new_time = (event_end or event_start + timedelta(hours=1)) + timedelta(minutes=15)
                logger.info(f"Rescheduling from {target_time} to {new_time} due to calendar conflict")
                intent.schedule_for = new_time
                intent.parameters["rescheduled_reason"] = f"Moved after '{event.get('summary', 'event')}'"
                break

            # For food orders, check if we should order earlier for delivery
            if intent.category == "food_order":
                # If there's a meeting ending around lunch, order to arrive after
                if event_end and abs((event_end - target_time).total_seconds()) < 1800:  # 30 min
                    # Adjust delivery to arrive 15 min after meeting
                    intent.schedule_for = event_end + timedelta(minutes=15)
                    intent.parameters["timed_to_event"] = event.get("summary", "event")

        return intent


async def execute_intent(
    intent: Intent,
    executor: ActionExecutor,
    user_id: str,
) -> dict[str, Any]:
    """Execute an intent using the action executor.

    Args:
        intent: The intent to execute
        executor: Action executor
        user_id: User ID

    Returns:
        Result dictionary with response and action result
    """
    if not intent.requires_action or not intent.action:
        return {
            "executed": False,
            "reason": "No action required",
        }

    # Map intent actions to executor actions
    action_map = {
        "ubereats_search": "ubereats_search",
        "ubereats_order": "ubereats_order",
        "restaurant_search": "restaurant_search",
        "web_search": "web_search",
        "get_distance": "get_distance",
        "check_calendar": None,  # Handled separately
        "add_reminder": None,  # Handled separately
        "create_document": "create_document",
        "write_document": "write_document",
        "send_email": "send_email",
        "compile_document": "compile_document",
        "research": "research",
        "quick_answer": "quick_answer",
    }

    action_name = action_map.get(intent.action)
    if not action_name:
        return {
            "executed": False,
            "reason": f"Action {intent.action} not mapped to executor",
        }

    # Build parameters
    params = dict(intent.parameters)

    # Handle scheduling
    if intent.schedule_for:
        now = datetime.now()
        if intent.schedule_for > now + timedelta(minutes=5):
            # Schedule for later
            schedule_id = executor.schedule(
                action_name,
                execute_at=intent.schedule_for,
                user_id=user_id,
                **params
            )
            return {
                "executed": False,
                "scheduled": True,
                "schedule_id": schedule_id,
                "scheduled_for": intent.schedule_for.isoformat(),
                "message": f"Scheduled for {intent.schedule_for.strftime('%I:%M %p')}",
            }

    # Execute immediately
    result = await executor.execute(action_name, user_id=user_id, **params)

    return {
        "executed": True,
        "action": action_name,
        "result": result.to_dict(),
        "message": result.message,
    }
