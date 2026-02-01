"""Proactivity engine for agent-initiated actions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.config import Config
    from gru.db import Database
    from gru.memory import MemoryStore

logger = logging.getLogger(__name__)


# Pattern detection thresholds
MIN_OCCURRENCES_FOR_PATTERN = 3  # Need at least 3 occurrences
HOUR_TOLERANCE = 1  # Within 1 hour counts as "same time"
DAY_TOLERANCE = 1  # Within 1 day of week counts as "same day"
PATTERN_CONFIDENCE_THRESHOLD = 0.7  # 70% consistency to be a pattern


class TriggerType(Enum):
    """Types of proactive triggers."""

    SCHEDULED = "scheduled"  # Cron-like time-based
    INTERVAL = "interval"  # Every N minutes
    CONDITION = "condition"  # When a condition is met
    EVENT = "event"  # External event (webhook)
    OBSERVATION = "observation"  # When something is noticed
    PATTERN = "pattern"  # When a learned pattern predicts action


@dataclass
class BehaviorEvent:
    """A recorded user behavior for pattern learning."""

    id: str
    action: str  # What the user did (check_email, send_message, search, etc.)
    context: dict[str, Any]  # Additional context (location, device, etc.)
    timestamp: datetime
    hour: int  # 0-23
    weekday: int  # 0=Monday, 6=Sunday
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedPattern:
    """A pattern detected from user behavior."""

    id: str
    action: str  # The action this pattern relates to
    pattern_type: str  # time_of_day, day_of_week, sequence, location
    description: str  # Human readable description
    confidence: float  # 0.0 - 1.0
    occurrences: int  # How many times observed
    parameters: dict[str, Any]  # Pattern-specific params (hour, day, etc.)
    last_matched: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def matches_now(self, now: datetime, context: dict[str, Any]) -> bool:
        """Check if current time/context matches this pattern."""
        if self.pattern_type == "time_of_day":
            target_hour = self.parameters.get("hour", -1)
            return abs(now.hour - target_hour) <= HOUR_TOLERANCE

        elif self.pattern_type == "day_of_week":
            target_day = self.parameters.get("weekday", -1)
            return now.weekday() == target_day

        elif self.pattern_type == "time_and_day":
            target_hour = self.parameters.get("hour", -1)
            target_day = self.parameters.get("weekday", -1)
            return abs(now.hour - target_hour) <= HOUR_TOLERANCE and now.weekday() == target_day

        elif self.pattern_type == "location":
            target_location = self.parameters.get("location")
            current_location = context.get("location")
            return target_location and current_location == target_location

        return False


@dataclass
class Trigger:
    """A proactive trigger definition."""

    id: str
    name: str
    trigger_type: TriggerType
    config: dict[str, Any]
    action: str  # What to do when triggered
    enabled: bool = True
    last_fired: datetime | None = None
    fire_count: int = 0

    # For scheduled triggers
    schedule: str | None = None  # Cron expression or time

    # For interval triggers
    interval_minutes: int = 0

    # For condition triggers
    condition: str | None = None  # Python expression to evaluate

    def should_fire(self, now: datetime, context: dict[str, Any]) -> bool:
        """Check if this trigger should fire."""
        if not self.enabled:
            return False

        if self.trigger_type == TriggerType.INTERVAL:
            if self.last_fired is None:
                return True
            elapsed = (now - self.last_fired).total_seconds() / 60
            return elapsed >= self.interval_minutes

        elif self.trigger_type == TriggerType.SCHEDULED:
            return self._check_schedule(now)

        elif self.trigger_type == TriggerType.CONDITION:
            return self._evaluate_condition(context)

        return False

    def _check_schedule(self, now: datetime) -> bool:
        """Check if scheduled time matches."""
        if not self.schedule:
            return False

        # Simple time matching (HH:MM format)
        if ":" in self.schedule and len(self.schedule) == 5:
            try:
                hour, minute = map(int, self.schedule.split(":"))
                if now.hour == hour and now.minute == minute:
                    # Only fire once per minute
                    if self.last_fired:
                        last_min = self.last_fired.replace(second=0, microsecond=0)
                        now_min = now.replace(second=0, microsecond=0)
                        return last_min != now_min
                    return True
            except ValueError:
                pass
        return False

    def _evaluate_condition(self, context: dict[str, Any]) -> bool:
        """Safely evaluate a condition expression."""
        if not self.condition:
            return False

        try:
            # Limited safe evaluation
            allowed_names = {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "any": any,
                "all": all,
                **context,
            }
            return bool(eval(self.condition, {"__builtins__": {}}, allowed_names))
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False


@dataclass
class Observation:
    """Something the agent noticed that might be relevant later."""

    id: str
    content: str
    category: str  # reminder, deadline, follow_up, anomaly, opportunity
    importance: float  # 0.0 - 1.0
    source: str  # Where this came from
    created_at: datetime
    expires_at: datetime | None = None
    acted_on: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if observation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_message(self) -> str:
        """Format observation as a user-facing message."""
        prefix = {
            "reminder": "Reminder",
            "deadline": "Upcoming deadline",
            "follow_up": "Follow-up needed",
            "anomaly": "I noticed something",
            "opportunity": "You might want to",
        }.get(self.category, "Note")

        return f"{prefix}: {self.content}"


class ProactiveEngine:
    """Engine for agent-initiated proactive behaviors."""

    def __init__(
        self,
        config: Config,
        db: Database,
        memory: MemoryStore | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.memory = memory
        self._triggers: dict[str, Trigger] = {}
        self._observations: dict[str, Observation] = {}
        self._patterns: dict[str, LearnedPattern] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._notify_callback: Callable[[str, str], None] | None = None
        self._check_interval = 30  # seconds between checks
        self._google_connector: Any = None  # Lazy loaded
        self._slack_connector: Any = None  # Lazy loaded
        self._last_pattern_check: datetime | None = None
        self._pattern_check_interval = 300  # Check patterns every 5 minutes
        self._last_insight_generation: datetime | None = None
        self._insight_generation_interval = 3600  # Generate insights hourly

    def set_notify_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for sending notifications to user."""
        self._notify_callback = callback

    def set_google_connector(self, connector: Any) -> None:
        """Set the Google connector for calendar/email sync."""
        self._google_connector = connector

    def set_slack_connector(self, connector: Any) -> None:
        """Set the Slack connector for message sync."""
        self._slack_connector = connector

    async def initialize(self) -> None:
        """Initialize the proactive engine, loading saved state."""
        await self._migrate_tables()
        await self._load_triggers()
        await self._load_observations()
        await self._load_patterns()
        logger.info(
            f"Proactive engine initialized with {len(self._triggers)} triggers, {len(self._patterns)} learned patterns"
        )

    async def _migrate_tables(self) -> None:
        """Create proactive tables if they don't exist."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_triggers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                config JSON NOT NULL,
                action TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                schedule TEXT,
                interval_minutes INTEGER DEFAULT 0,
                condition TEXT,
                last_fired TEXT,
                fire_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_observations (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT,
                acted_on INTEGER NOT NULL DEFAULT 0,
                metadata JSON
            )
        """)
        # Behavior tracking table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_behaviors (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                context JSON,
                timestamp TEXT NOT NULL,
                hour INTEGER NOT NULL,
                weekday INTEGER NOT NULL,
                metadata JSON,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Learned patterns table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_patterns (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                occurrences INTEGER NOT NULL DEFAULT 0,
                parameters JSON NOT NULL,
                last_matched TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Insights table for tracking generated insights
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,
                content TEXT NOT NULL,
                data JSON,
                shown INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await self.db.commit()

    async def _load_triggers(self) -> None:
        """Load triggers from database."""
        rows = await self.db.fetchall("SELECT * FROM proactive_triggers WHERE enabled = 1")
        for row in rows:
            trigger = Trigger(
                id=row["id"],
                name=row["name"],
                trigger_type=TriggerType(row["trigger_type"]),
                config=json.loads(row["config"]) if row["config"] else {},
                action=row["action"],
                enabled=bool(row["enabled"]),
                schedule=row.get("schedule"),
                interval_minutes=row.get("interval_minutes", 0),
                condition=row.get("condition"),
                last_fired=datetime.fromisoformat(row["last_fired"]) if row.get("last_fired") else None,
                fire_count=row.get("fire_count", 0),
            )
            self._triggers[trigger.id] = trigger

    async def _load_observations(self) -> None:
        """Load active observations from database."""
        rows = await self.db.fetchall(
            "SELECT * FROM proactive_observations WHERE acted_on = 0 ORDER BY importance DESC"
        )
        for row in rows:
            obs = Observation(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                importance=row["importance"],
                source=row["source"],
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]) if row.get("expires_at") else None,
                acted_on=bool(row["acted_on"]),
                metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            )
            if not obs.is_expired():
                self._observations[obs.id] = obs

    async def _load_patterns(self) -> None:
        """Load learned patterns from database."""
        rows = await self.db.fetchall(
            "SELECT * FROM proactive_patterns WHERE confidence >= ?",
            (PATTERN_CONFIDENCE_THRESHOLD,),
        )
        for row in rows:
            pattern = LearnedPattern(
                id=row["id"],
                action=row["action"],
                pattern_type=row["pattern_type"],
                description=row["description"],
                confidence=row["confidence"],
                occurrences=row["occurrences"],
                parameters=json.loads(row["parameters"]) if row.get("parameters") else {},
                last_matched=datetime.fromisoformat(row["last_matched"]) if row.get("last_matched") else None,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            self._patterns[pattern.id] = pattern

    async def start(self) -> None:
        """Start the proactive monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Proactive engine started")

    async def stop(self) -> None:
        """Stop the proactive monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Proactive engine stopped")

    async def _run_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_triggers()
                await self._check_observations()
                await self._check_patterns()
                await self._maybe_generate_insights()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Proactive loop error: {e}")
                await asyncio.sleep(self._check_interval)

    async def _check_triggers(self) -> None:
        """Check all triggers and fire those that should activate."""
        now = datetime.now()
        context = await self._build_context()

        for trigger in list(self._triggers.values()):
            if trigger.should_fire(now, context):
                await self._fire_trigger(trigger, context)

    async def _check_observations(self) -> None:
        """Check observations and notify about important ones."""
        datetime.now()

        # Clean up expired observations
        expired = [obs_id for obs_id, obs in self._observations.items() if obs.is_expired()]
        for obs_id in expired:
            del self._observations[obs_id]
            await self.db.execute(
                "UPDATE proactive_observations SET acted_on = 1 WHERE id = ?",
                (obs_id,),
            )

        # Check for high-importance observations that need attention
        for obs in sorted(self._observations.values(), key=lambda o: -o.importance):
            if obs.importance >= 0.8 and not obs.acted_on:
                # High importance - notify immediately
                await self._notify(obs.to_message())
                obs.acted_on = True
                await self.db.execute(
                    "UPDATE proactive_observations SET acted_on = 1 WHERE id = ?",
                    (obs.id,),
                )
                await self.db.commit()

    async def _check_patterns(self) -> None:
        """Check if any learned patterns match current context and anticipate needs."""
        now = datetime.now()

        # Throttle pattern checks
        if self._last_pattern_check:
            elapsed = (now - self._last_pattern_check).total_seconds()
            if elapsed < self._pattern_check_interval:
                return

        self._last_pattern_check = now
        context = await self._build_context()

        for pattern in self._patterns.values():
            if pattern.matches_now(now, context):
                # Don't re-trigger if recently matched
                if pattern.last_matched:
                    since_last = (now - pattern.last_matched).total_seconds()
                    if since_last < 3600:  # 1 hour cooldown
                        continue

                await self._anticipate_from_pattern(pattern, context)
                pattern.last_matched = now
                await self.db.execute(
                    "UPDATE proactive_patterns SET last_matched = ? WHERE id = ?",
                    (now.isoformat(), pattern.id),
                )
                await self.db.commit()

    async def _anticipate_from_pattern(self, pattern: LearnedPattern, context: dict[str, Any]) -> None:
        """Generate anticipatory action based on a matched pattern."""
        action = pattern.action

        if action == "check_email":
            # Proactively summarize email
            await self._notify("You usually check email around now. Want me to summarize your inbox?")

        elif action == "check_calendar":
            # Proactively check upcoming events
            if self._google_connector:
                try:
                    events = await self._google_connector.get_upcoming_events(hours=4)
                    if events:
                        next_event = events[0]
                        await self._notify(
                            f"Your next meeting is {next_event.get('summary', 'untitled')} "
                            f"at {next_event.get('start', 'unknown time')}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to get calendar events: {e}")

        elif action == "commute":
            # Check traffic for commute
            await self._notify("You usually leave around now. Want me to check traffic?")

        elif action.startswith("search:"):
            # User often searches for this topic
            topic = action[7:]
            await self._notify(f"You often look up {topic} around this time. Want an update?")

        else:
            # Generic pattern notification
            logger.info(f"Pattern matched but no anticipation action: {pattern.description}")

    async def _maybe_generate_insights(self) -> None:
        """Periodically analyze behavior data to generate insights."""
        now = datetime.now()

        # Throttle insight generation
        if self._last_insight_generation:
            elapsed = (now - self._last_insight_generation).total_seconds()
            if elapsed < self._insight_generation_interval:
                return

        self._last_insight_generation = now

        try:
            await self._detect_patterns()
            await self._generate_spending_insights()
            await self._generate_time_insights()
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")

    async def _detect_patterns(self) -> None:
        """Analyze behavior history to detect new patterns."""
        # Get recent behaviors (last 30 days)
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        rows = await self.db.fetchall(
            "SELECT * FROM proactive_behaviors WHERE timestamp > ? ORDER BY timestamp",
            (cutoff,),
        )

        if len(rows) < MIN_OCCURRENCES_FOR_PATTERN:
            return

        # Group by action
        actions: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            actions[row["action"]].append(
                {
                    "hour": row["hour"],
                    "weekday": row["weekday"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "context": json.loads(row["context"]) if row.get("context") else {},
                }
            )

        # Detect time-of-day patterns
        for action, events in actions.items():
            if len(events) < MIN_OCCURRENCES_FOR_PATTERN:
                continue

            # Check for consistent hour patterns
            hour_counts = Counter(e["hour"] for e in events)
            for hour, count in hour_counts.most_common(3):
                if count >= MIN_OCCURRENCES_FOR_PATTERN:
                    confidence = count / len(events)
                    if confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                        await self._save_pattern(
                            action=action,
                            pattern_type="time_of_day",
                            description=f"Usually {action} around {hour}:00",
                            confidence=confidence,
                            occurrences=count,
                            parameters={"hour": hour},
                        )

            # Check for day-of-week patterns
            day_counts = Counter(e["weekday"] for e in events)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for weekday, count in day_counts.most_common(3):
                if count >= MIN_OCCURRENCES_FOR_PATTERN:
                    confidence = count / len(events)
                    if confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                        await self._save_pattern(
                            action=action,
                            pattern_type="day_of_week",
                            description=f"Usually {action} on {day_names[weekday]}s",
                            confidence=confidence,
                            occurrences=count,
                            parameters={"weekday": weekday},
                        )

    async def _save_pattern(
        self,
        action: str,
        pattern_type: str,
        description: str,
        confidence: float,
        occurrences: int,
        parameters: dict[str, Any],
    ) -> None:
        """Save or update a learned pattern."""
        # Check if pattern already exists
        existing = await self.db.fetchone(
            "SELECT id FROM proactive_patterns WHERE action = ? AND pattern_type = ? AND parameters = ?",
            (action, pattern_type, json.dumps(parameters)),
        )

        if existing:
            # Update existing pattern
            await self.db.execute(
                "UPDATE proactive_patterns SET confidence = ?, occurrences = ? WHERE id = ?",
                (confidence, occurrences, existing["id"]),
            )
        else:
            # Create new pattern
            pattern_id = str(uuid.uuid4())[:12]
            await self.db.execute(
                """
                INSERT INTO proactive_patterns (id, action, pattern_type, description, confidence, occurrences, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pattern_id, action, pattern_type, description, confidence, occurrences, json.dumps(parameters)),
            )

            # Add to in-memory cache
            self._patterns[pattern_id] = LearnedPattern(
                id=pattern_id,
                action=action,
                pattern_type=pattern_type,
                description=description,
                confidence=confidence,
                occurrences=occurrences,
                parameters=parameters,
            )
            logger.info(f"New pattern learned: {description}")

        await self.db.commit()

    async def _generate_spending_insights(self) -> None:
        """Analyze spending patterns and generate insights."""
        # This would integrate with financial data if available
        # For now, check memory for spending-related facts
        if not self.memory:
            return

        try:
            facts = await self.memory.search_facts("spending OR subscription OR payment", limit=20)
            if len(facts) >= 3:
                # Check if we've already generated this insight recently
                recent = await self.db.fetchone(
                    "SELECT id FROM proactive_insights WHERE insight_type = 'spending' AND created_at > datetime('now', '-7 days')"
                )
                if not recent:
                    await self._save_insight(
                        insight_type="spending",
                        content="You have several subscriptions and recurring payments. Want me to audit them?",
                        data={"fact_count": len(facts)},
                    )
        except Exception as e:
            logger.debug(f"Spending insight generation skipped: {e}")

    async def _generate_time_insights(self) -> None:
        """Analyze how user spends time and generate insights."""
        # Get behavior counts by action
        rows = await self.db.fetchall(
            """
            SELECT action, COUNT(*) as count, AVG(hour) as avg_hour
            FROM proactive_behaviors
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY action
            ORDER BY count DESC
            LIMIT 10
            """
        )

        if not rows:
            return

        # Find dominant activities
        total = sum(r["count"] for r in rows)
        for row in rows:
            pct = (row["count"] / total) * 100
            if pct > 40:  # Single activity dominates
                recent = await self.db.fetchone(
                    "SELECT id FROM proactive_insights WHERE insight_type = 'time_usage' AND created_at > datetime('now', '-7 days')"
                )
                if not recent:
                    await self._save_insight(
                        insight_type="time_usage",
                        content=f"You spend a lot of time on {row['action']}. Want to analyze this?",
                        data={"action": row["action"], "percentage": pct},
                    )
                break

    async def _save_insight(self, insight_type: str, content: str, data: dict[str, Any] | None = None) -> str:
        """Save a generated insight."""
        insight_id = str(uuid.uuid4())[:12]
        await self.db.execute(
            "INSERT INTO proactive_insights (id, insight_type, content, data) VALUES (?, ?, ?, ?)",
            (insight_id, insight_type, content, json.dumps(data or {})),
        )
        await self.db.commit()

        # Notify user of new insight
        await self._notify(content)
        logger.info(f"Generated insight: {content[:50]}...")
        return insight_id

    async def _build_context(self) -> dict[str, Any]:
        """Build context for condition evaluation."""
        context: dict[str, Any] = {
            "now": datetime.now(),
            "hour": datetime.now().hour,
            "minute": datetime.now().minute,
            "weekday": datetime.now().weekday(),
            "observations_count": len(self._observations),
        }

        # Add memory facts if available
        if self.memory:
            try:
                facts = await self.memory.get_facts(limit=20)
                context["facts_count"] = len(facts)
                context["has_preferences"] = any(f.fact_type == "preference" for f in facts)
            except Exception:
                pass

        return context

    async def _fire_trigger(self, trigger: Trigger, context: dict[str, Any]) -> None:
        """Execute a trigger's action."""
        logger.info(f"Firing trigger: {trigger.name}")

        trigger.last_fired = datetime.now()
        trigger.fire_count += 1

        # Update database
        await self.db.execute(
            "UPDATE proactive_triggers SET last_fired = ?, fire_count = ? WHERE id = ?",
            (trigger.last_fired.isoformat(), trigger.fire_count, trigger.id),
        )
        await self.db.commit()

        # Execute action
        action = trigger.action

        if action.startswith("notify:"):
            message = action[7:].strip()
            # Substitute context variables
            for key, value in context.items():
                message = message.replace(f"{{{key}}}", str(value))
            await self._notify(message)

        elif action.startswith("observe:"):
            # Create an observation
            content = action[8:].strip()
            await self.add_observation(
                content=content,
                category="reminder",
                importance=0.7,
                source=f"trigger:{trigger.id}",
            )

        elif action.startswith("check:"):
            # Check a condition and notify
            check_type = action[6:].strip()
            await self._run_check(check_type, context)

        elif action.startswith("research:"):
            # Run research workflow
            research_type = action[9:].strip()
            await self._run_research(research_type, trigger.config)

    async def _run_check(self, check_type: str, context: dict[str, Any]) -> None:
        """Run a specific check and create observations or notifications."""
        if check_type == "pending_observations":
            pending = [o for o in self._observations.values() if not o.acted_on]
            if pending:
                count = len(pending)
                high_priority = sum(1 for o in pending if o.importance >= 0.7)
                if high_priority > 0:
                    await self._notify(f"You have {count} pending items, {high_priority} high priority")

        elif check_type == "daily_summary":
            await self._generate_daily_summary()

        elif check_type == "google_calendar":
            if self._google_connector:
                try:
                    events = await self._google_connector.sync_calendar(self)
                    if events:
                        logger.info(f"Google Calendar sync: {len(events)} new events")
                except Exception as e:
                    logger.warning(f"Google Calendar sync failed: {e}")

        elif check_type == "google_email":
            if self._google_connector:
                try:
                    emails = await self._google_connector.sync_email(self)
                    if emails:
                        logger.info(f"Gmail sync: {len(emails)} new emails")
                except Exception as e:
                    logger.warning(f"Gmail sync failed: {e}")

        elif check_type == "slack_messages":
            if self._slack_connector:
                try:
                    messages = await self._slack_connector.sync_messages(self)
                    if messages:
                        logger.info(f"Slack sync: {len(messages)} new messages")
                except Exception as e:
                    logger.warning(f"Slack sync failed: {e}")

        elif check_type == "morning_summary":
            summary = await self.generate_morning_summary()
            await self._notify(summary)

    async def _generate_daily_summary(self) -> None:
        """Generate a daily summary notification."""
        summary_parts = []

        # Pending observations
        pending = [o for o in self._observations.values() if not o.acted_on]
        if pending:
            summary_parts.append(f"{len(pending)} items need attention")

        # Recent memory facts
        if self.memory:
            try:
                stats = await self.memory.get_stats()
                if stats["total_facts"] > 0:
                    summary_parts.append(f"{stats['total_facts']} things remembered")
            except Exception:
                pass

        if summary_parts:
            await self._notify("Daily summary: " + ", ".join(summary_parts))

    async def generate_morning_summary(self) -> str:
        """Generate a comprehensive morning summary of email, calendar, and Slack.

        Returns the summary text.
        """
        parts = []
        parts.append("Good morning. Here's your daily briefing:\n")

        # Calendar - today's schedule
        if self._google_connector:
            try:
                now = datetime.utcnow()
                time_min = now.isoformat() + "Z"
                # Get events for the rest of today
                end_of_day = now.replace(hour=23, minute=59, second=59)
                time_max = end_of_day.isoformat() + "Z"

                events_result = (
                    self._google_connector._calendar_service.events()
                    .list(
                        calendarId="primary",
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=10,
                        singleEvents=True,
                        orderBy="startTime",
                    )
                    .execute()
                )

                events = events_result.get("items", [])
                if events:
                    parts.append("CALENDAR")
                    for event in events:
                        summary = event.get("summary", "Untitled")
                        start = event.get("start", {})
                        start_time = start.get("dateTime", start.get("date", ""))

                        if "T" in start_time:
                            try:
                                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                                time_str = dt.strftime("%I:%M %p")
                            except Exception:
                                time_str = start_time
                        else:
                            time_str = "All day"

                        parts.append(f"  {time_str} - {summary}")
                    parts.append("")
                else:
                    parts.append("CALENDAR\n  No meetings scheduled today.\n")

            except Exception as e:
                logger.warning(f"Failed to get calendar for morning summary: {e}")

        # Email - unread count and important ones
        if self._google_connector:
            try:
                results = (
                    self._google_connector._gmail_service.users()
                    .messages()
                    .list(
                        userId="me",
                        q="is:unread",
                        maxResults=50,
                    )
                    .execute()
                )

                total_unread = results.get("resultSizeEstimate", 0)

                # Get important unread
                important_results = (
                    self._google_connector._gmail_service.users()
                    .messages()
                    .list(
                        userId="me",
                        q="is:unread is:important",
                        maxResults=5,
                    )
                    .execute()
                )

                important_msgs = important_results.get("messages", [])

                parts.append("EMAIL")
                parts.append(f"  {total_unread} unread emails")

                if important_msgs:
                    parts.append(f"  {len(important_msgs)} marked important:")
                    for msg_meta in important_msgs[:3]:
                        msg = (
                            self._google_connector._gmail_service.users()
                            .messages()
                            .get(
                                userId="me",
                                id=msg_meta["id"],
                                format="metadata",
                                metadataHeaders=["From", "Subject"],
                            )
                            .execute()
                        )

                        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                        from_header = headers.get("From", "Unknown")
                        subject = headers.get("Subject", "(no subject)")

                        if "<" in from_header:
                            sender = from_header.split("<")[0].strip().strip('"')
                        else:
                            sender = from_header.split("@")[0]

                        parts.append(f"    - {sender}: {subject[:50]}")

                parts.append("")

            except Exception as e:
                logger.warning(f"Failed to get email for morning summary: {e}")

        # Slack - DMs and mentions
        if self._slack_connector and self._slack_connector.is_authenticated():
            try:
                dms = await self._slack_connector.get_unread_dms(limit=5)
                mentions = await self._slack_connector.get_mentions(hours_back=12, limit=5)

                if dms or mentions:
                    parts.append("SLACK")

                    if dms:
                        parts.append(f"  {len(dms)} unread DMs:")
                        for dm in dms[:3]:
                            sender = await self._slack_connector.get_user_name(dm.get("from_user", ""))
                            text = dm.get("text", "")[:40]
                            parts.append(f"    - {sender}: {text}...")

                    if mentions:
                        parts.append(f"  {len(mentions)} mentions:")
                        for m in mentions[:3]:
                            sender = await self._slack_connector.get_user_name(m.get("from_user", ""))
                            text = m.get("text", "")[:40]
                            parts.append(f"    - {sender}: {text}...")

                    parts.append("")

            except Exception as e:
                logger.warning(f"Failed to get Slack for morning summary: {e}")

        # Pending observations
        pending = await self.get_pending_observations()
        high_priority = [o for o in pending if o.importance >= 0.8]
        if high_priority:
            parts.append("ATTENTION NEEDED")
            for obs in high_priority[:5]:
                parts.append(f"  - {obs.content}")
            parts.append("")

        summary = "\n".join(parts)
        return summary

    async def _run_research(self, research_type: str, config: dict[str, Any]) -> None:
        """Run a research workflow."""
        try:
            parts = research_type.split(":")
            workflow = parts[0] if parts else "daily"
            notify_chat_id = parts[1] if len(parts) > 1 else config.get("notify_chat_id")

            if workflow == "daily":
                from gru.tools.research import start_daily_research

                await self._notify("Starting daily AI research... I'll have your morning report ready soon.")
                result = await start_daily_research(notify_chat_id=notify_chat_id)

                if "error" in result:
                    await self._notify(f"Research failed to start: {result['error']}")
                else:
                    logger.info(f"Daily research started: agent {result.get('agent_id')}")

            elif workflow == "monitor":
                from gru.tools.research import check_breaking_news

                logger.info("Running real-time AI monitoring check...")
                result = await check_breaking_news(notify=True)

                if result.get("breaking_items", 0) > 0:
                    logger.info(f"Found {result['breaking_items']} breaking news items")
                else:
                    logger.debug("No breaking news found in this check")

            else:
                logger.warning(f"Unknown research workflow: {workflow}")

        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            await self._notify(f"Research workflow failed: {e}")

    async def _notify(self, message: str) -> None:
        """Send a notification to the user."""
        if self._notify_callback:
            try:
                self._notify_callback("proactive", message)
            except Exception as e:
                logger.error(f"Notification failed: {e}")
        else:
            logger.info(f"Proactive notification (no callback): {message}")

    # Public API for managing triggers

    async def add_trigger(
        self,
        name: str,
        trigger_type: TriggerType,
        action: str,
        schedule: str | None = None,
        interval_minutes: int = 0,
        condition: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Add a new proactive trigger."""
        trigger_id = str(uuid.uuid4())[:12]

        trigger = Trigger(
            id=trigger_id,
            name=name,
            trigger_type=trigger_type,
            config=config or {},
            action=action,
            schedule=schedule,
            interval_minutes=interval_minutes,
            condition=condition,
        )

        await self.db.execute(
            """
            INSERT INTO proactive_triggers (id, name, trigger_type, config, action, schedule,
                                           interval_minutes, condition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trigger_id,
                name,
                trigger_type.value,
                json.dumps(config or {}),
                action,
                schedule,
                interval_minutes,
                condition,
            ),
        )
        await self.db.commit()

        self._triggers[trigger_id] = trigger
        logger.info(f"Added trigger: {name} ({trigger_id})")
        return trigger_id

    async def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger."""
        if trigger_id not in self._triggers:
            return False

        del self._triggers[trigger_id]
        await self.db.execute("DELETE FROM proactive_triggers WHERE id = ?", (trigger_id,))
        await self.db.commit()
        return True

    async def list_triggers(self) -> list[dict[str, Any]]:
        """List all triggers."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "type": t.trigger_type.value,
                "action": t.action,
                "enabled": t.enabled,
                "last_fired": t.last_fired.isoformat() if t.last_fired else None,
                "fire_count": t.fire_count,
            }
            for t in self._triggers.values()
        ]

    # Public API for observations

    async def add_observation(
        self,
        content: str,
        category: str = "note",
        importance: float = 0.5,
        source: str = "agent",
        expires_in_hours: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a new observation."""
        obs_id = str(uuid.uuid4())[:12]
        now = datetime.now()
        expires_at = now + timedelta(hours=expires_in_hours) if expires_in_hours else None

        obs = Observation(
            id=obs_id,
            content=content,
            category=category,
            importance=importance,
            source=source,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        await self.db.execute(
            """
            INSERT INTO proactive_observations (id, content, category, importance, source,
                                               created_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obs_id,
                content,
                category,
                importance,
                source,
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                json.dumps(metadata or {}),
            ),
        )
        await self.db.commit()

        self._observations[obs_id] = obs

        # Store as memory fact too if memory is available
        if self.memory and importance >= 0.6:
            with contextlib.suppress(Exception):
                await self.memory.store_fact(
                    fact_type="context",
                    subject="agent",
                    predicate="noticed",
                    obj=content[:200],
                    confidence=importance,
                )

        logger.info(f"Added observation: {content[:50]}... ({obs_id})")
        return obs_id

    async def mark_observation_handled(self, obs_id: str) -> bool:
        """Mark an observation as handled."""
        if obs_id not in self._observations:
            return False

        self._observations[obs_id].acted_on = True
        await self.db.execute(
            "UPDATE proactive_observations SET acted_on = 1 WHERE id = ?",
            (obs_id,),
        )
        await self.db.commit()
        return True

    async def get_pending_observations(self) -> list[Observation]:
        """Get all pending observations sorted by importance."""
        return sorted(
            [o for o in self._observations.values() if not o.acted_on and not o.is_expired()],
            key=lambda o: -o.importance,
        )

    async def get_observation_summary(self) -> str:
        """Get a summary of pending observations for context injection."""
        pending = await self.get_pending_observations()
        if not pending:
            return ""

        lines = ["PENDING OBSERVATIONS:"]
        for obs in pending[:5]:
            priority = "!" if obs.importance >= 0.8 else ""
            lines.append(f"  {priority}[{obs.category}] {obs.content}")

        return "\n".join(lines)

    # Public API for behavior tracking

    async def track_behavior(
        self,
        action: str,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Track a user behavior for pattern learning.

        Call this whenever the user does something trackable:
        - check_email, send_message, search, make_call, etc.
        """
        behavior_id = str(uuid.uuid4())[:12]
        now = datetime.now()

        await self.db.execute(
            """
            INSERT INTO proactive_behaviors (id, action, context, timestamp, hour, weekday, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                behavior_id,
                action,
                json.dumps(context or {}),
                now.isoformat(),
                now.hour,
                now.weekday(),
                json.dumps(metadata or {}),
            ),
        )
        await self.db.commit()

        logger.debug(f"Tracked behavior: {action} at {now.strftime('%H:%M')}")
        return behavior_id

    async def get_behavior_stats(self) -> dict[str, Any]:
        """Get statistics about tracked behaviors."""
        # Total behaviors
        total = await self.db.fetchone("SELECT COUNT(*) as count FROM proactive_behaviors")

        # Behaviors by action
        by_action = await self.db.fetchall(
            """
            SELECT action, COUNT(*) as count
            FROM proactive_behaviors
            GROUP BY action
            ORDER BY count DESC
            LIMIT 10
            """
        )

        # Recent behaviors
        recent = await self.db.fetchall(
            """
            SELECT action, timestamp, hour
            FROM proactive_behaviors
            ORDER BY timestamp DESC
            LIMIT 10
            """
        )

        return {
            "total": total["count"] if total else 0,
            "by_action": [{"action": r["action"], "count": r["count"]} for r in by_action],
            "recent": [{"action": r["action"], "timestamp": r["timestamp"], "hour": r["hour"]} for r in recent],
        }

    # Public API for patterns

    async def list_patterns(self) -> list[dict[str, Any]]:
        """List all learned patterns."""
        return [
            {
                "id": p.id,
                "action": p.action,
                "type": p.pattern_type,
                "description": p.description,
                "confidence": p.confidence,
                "occurrences": p.occurrences,
                "last_matched": p.last_matched.isoformat() if p.last_matched else None,
            }
            for p in self._patterns.values()
        ]

    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a learned pattern."""
        if pattern_id not in self._patterns:
            return False

        del self._patterns[pattern_id]
        await self.db.execute("DELETE FROM proactive_patterns WHERE id = ?", (pattern_id,))
        await self.db.commit()
        return True

    async def get_pending_insights(self) -> list[dict[str, Any]]:
        """Get insights that haven't been shown to the user."""
        rows = await self.db.fetchall(
            "SELECT * FROM proactive_insights WHERE shown = 0 ORDER BY created_at DESC LIMIT 5"
        )
        return [
            {
                "id": r["id"],
                "type": r["insight_type"],
                "content": r["content"],
                "data": json.loads(r["data"]) if r.get("data") else {},
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def mark_insight_shown(self, insight_id: str) -> bool:
        """Mark an insight as shown."""
        await self.db.execute("UPDATE proactive_insights SET shown = 1 WHERE id = ?", (insight_id,))
        await self.db.commit()
        return True

    # Anticipation helpers

    async def check_upcoming_events(self, hours: int = 2) -> list[dict[str, Any]]:
        """Check for upcoming calendar events and generate anticipatory notifications."""
        if not self._google_connector:
            return []

        try:
            events = await self._google_connector.get_upcoming_events(hours=hours)

            for event in events:
                start_time = event.get("start")
                summary = event.get("summary", "Event")
                location = event.get("location")

                # Calculate time until event
                if start_time:
                    try:
                        event_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        minutes_until = (event_dt - datetime.now()).total_seconds() / 60

                        # 30 minute warning
                        if 25 <= minutes_until <= 35:
                            msg = f"Your meeting '{summary}' starts in 30 minutes"
                            if location:
                                msg += f" at {location}"
                            await self._notify(msg)

                        # 5 minute warning
                        elif 3 <= minutes_until <= 7:
                            await self._notify(f"'{summary}' starts in 5 minutes")

                    except Exception:
                        pass

            return events
        except Exception as e:
            logger.warning(f"Failed to check upcoming events: {e}")
            return []

    async def get_context_summary(self) -> str:
        """Get a summary of current context for the agent."""
        parts = []

        # Upcoming patterns
        now = datetime.now()
        context = await self._build_context()
        matching_patterns = [p for p in self._patterns.values() if p.matches_now(now, context)]
        if matching_patterns:
            patterns_str = ", ".join(p.description for p in matching_patterns[:3])
            parts.append(f"Matching patterns: {patterns_str}")

        # Pending insights
        insights = await self.get_pending_insights()
        if insights:
            parts.append(f"Pending insights: {len(insights)}")

        # Behavior summary
        stats = await self.get_behavior_stats()
        if stats["total"] > 0:
            parts.append(f"Tracked {stats['total']} behaviors")

        return " | ".join(parts) if parts else "No proactive context"


# Built-in trigger templates

BUILTIN_TRIGGERS = [
    {
        "name": "morning_briefing",
        "trigger_type": TriggerType.SCHEDULED,
        "schedule": "06:00",
        "action": "check:morning_summary",
    },
    {
        "name": "check_pending",
        "trigger_type": TriggerType.INTERVAL,
        "interval_minutes": 60,
        "action": "check:pending_observations",
    },
    {
        "name": "slack_sync",
        "trigger_type": TriggerType.INTERVAL,
        "interval_minutes": 10,
        "action": "check:slack_messages",
    },
]


async def setup_default_triggers(engine: ProactiveEngine) -> None:
    """Set up default triggers if none exist."""
    existing = await engine.list_triggers()
    if existing:
        return

    for template in BUILTIN_TRIGGERS:
        await engine.add_trigger(
            name=template["name"],
            trigger_type=template["trigger_type"],
            action=template["action"],
            schedule=template.get("schedule"),
            interval_minutes=template.get("interval_minutes", 0),
        )
