"""Conversation session management for Gru."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from gru.actions.executor import ActionExecutor
    from gru.claude import ClaudeClient
    from gru.db import Database
    from gru.intent import IntentClassifier
    from gru.memory import MemoryStore
    from gru.proactive import ProactiveEngine
    from gru.setup import ConfigManager

logger = logging.getLogger(__name__)


# Patterns that indicate user wants autonomous task execution
ESCALATION_PATTERNS = [
    r"^go\s+(build|create|make|write|implement|fix|deploy)",
    r"^(build|create|make|write|implement|fix|deploy)\s+.+\s+for\s+me",
    r"^(please\s+)?(build|create|make|write|implement|fix|deploy)\s+.+\s+(and|then)\s+(let\s+me\s+know|notify|tell\s+me|call\s+me)",
    r"(run|execute|do)\s+(this|that|it)\s+(autonomously|in\s+the\s+background|on\s+your\s+own)",
    r"^spawn\s+(an?\s+)?agent",
    r"work\s+on\s+(this|that|it)\s+and\s+(get\s+back|come\s+back|notify|ping)",
]

# Patterns for quick actions (don't need full agent)
QUICK_ACTION_PATTERNS = [
    r"^(add|create|set|schedule)\s+(a\s+)?(reminder|todo|task|event|meeting)",
    r"^(remind\s+me|don't\s+let\s+me\s+forget)",
    r"^(what's|what\s+is)\s+(on\s+)?(my\s+)?(calendar|schedule|agenda)",
    r"^(order|get|buy)\s+.+\s+(for|from)",
]


@dataclass
class Persona:
    """A persona defines how Gru behaves in conversation."""

    name: str
    description: str
    system_prompt: str
    escalation_threshold: float = 0.7  # How readily to escalate to agents
    voice_style: str = "concise"  # concise, detailed, casual, formal

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "escalation_threshold": self.escalation_threshold,
            "voice_style": self.voice_style,
        }


# Built-in personas
PERSONAS = {
    "general": Persona(
        name="general",
        description="General-purpose assistant",
        system_prompt="""You are Gru, a helpful personal assistant. You have access to the user's memory, calendar, and email.

Be direct and action-oriented. When the user asks a question, answer it. When they ask you to do something, do it. Don't ask for confirmation or permission when the intent is clear.

You remember things about the user from previous conversations. Use this context to give better answers.

For quick tasks (reminders, calendar checks, questions), handle them directly.
For complex tasks (building software, multi-step research), acknowledge and execute.

Never respond with "Would you like me to..." or "Should I..." when the user has already asked you to do something. Just do it.""",
        voice_style="concise",
    ),
    "dev": Persona(
        name="dev",
        description="Developer pair programmer",
        system_prompt="""You are Gru, a senior developer and pair programming partner. You help with:
- Code review and suggestions
- Architecture decisions
- Debugging
- Implementation planning

Be technical and precise. Don't over-explain basics. When code needs to be written, offer to spawn an agent to handle it autonomously.

You have access to the user's git repos, files, and development environment.""",
        voice_style="concise",
        escalation_threshold=0.5,  # More readily escalate to coding agents
    ),
    "exec": Persona(
        name="exec",
        description="Executive assistant",
        system_prompt="""You are Gru, an executive assistant. You help manage:
- Calendar and scheduling
- Email triage and responses
- Task management and TODO lists
- Meeting preparation
- Travel and logistics

Be proactive. Anticipate needs. Suggest optimizations to the schedule. Flag conflicts and important items.

You have access to the user's calendar, email, and task list. Use their preferences to make smart decisions.""",
        voice_style="concise",
        escalation_threshold=0.8,  # Less likely to escalate, handle more directly
    ),
    "casual": Persona(
        name="casual",
        description="Casual conversation partner",
        system_prompt="""You are Gru, a friendly assistant. Chat naturally. Be helpful but not robotic.

You know things about the user from previous conversations. Reference shared context when relevant.

Keep responses conversational. Don't be overly formal or listy unless asked.""",
        voice_style="casual",
        escalation_threshold=0.9,  # Rarely escalate
    ),
}


@dataclass
class Session:
    """A conversation session with a user."""

    id: str
    user_id: str
    channel: str  # telegram, discord, slack, cli
    persona: str = "general"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.last_active = datetime.now()

    def get_recent_messages(self, limit: int = 20) -> list[dict[str, str]]:
        """Get recent messages formatted for Claude."""
        recent = self.messages[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear_messages(self) -> None:
        """Clear conversation history."""
        self.messages = []


class SessionManager:
    """Manages conversation sessions."""

    def __init__(
        self,
        db: Database,
        claude: ClaudeClient,
        memory: MemoryStore | None = None,
        proactive: ProactiveEngine | None = None,
        action_executor: ActionExecutor | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.db = db
        self.claude = claude
        self.memory = memory
        self.proactive = proactive
        self.action_executor = action_executor
        self.data_dir = data_dir
        self._sessions: dict[str, Session] = {}
        self._user_personas: dict[str, str] = {}  # user_id -> persona name
        self._intent_classifier: IntentClassifier | None = None
        self._config_manager: ConfigManager | None = None

    def _get_intent_classifier(self) -> IntentClassifier:
        """Get or create intent classifier."""
        if self._intent_classifier is None:
            from gru.intent import IntentClassifier
            self._intent_classifier = IntentClassifier(
                claude=self.claude,
                memory=self.memory,
                proactive=self.proactive,
            )
        return self._intent_classifier

    def _get_config_manager(self) -> ConfigManager | None:
        """Get or create config manager."""
        if self._config_manager is None and self.data_dir:
            from gru.setup import get_config_manager
            self._config_manager = get_config_manager(self.data_dir)
        return self._config_manager

    async def set_action_executor(self, executor: ActionExecutor) -> None:
        """Set the action executor for handling intents."""
        self.action_executor = executor

    async def initialize(self) -> None:
        """Initialize session storage."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                persona TEXT NOT NULL DEFAULT 'general',
                messages JSON,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_active TEXT NOT NULL DEFAULT (datetime('now')),
                metadata JSON
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON conversation_sessions(user_id, channel)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                default_persona TEXT DEFAULT 'general',
                settings JSON,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await self.db.commit()
        logger.info("Session manager initialized")

    async def get_or_create_session(
        self,
        user_id: str,
        channel: str,
    ) -> Session:
        """Get existing session or create a new one."""
        session_key = f"{channel}:{user_id}"

        # Check memory cache
        if session_key in self._sessions:
            return self._sessions[session_key]

        # Check database
        row = await self.db.fetchone(
            "SELECT * FROM conversation_sessions WHERE user_id = ? AND channel = ? ORDER BY last_active DESC LIMIT 1",
            (user_id, channel),
        )

        if row:
            session = Session(
                id=row["id"],
                user_id=row["user_id"],
                channel=row["channel"],
                persona=row["persona"],
                messages=json.loads(row["messages"]) if row["messages"] else [],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_active=datetime.fromisoformat(row["last_active"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        else:
            # Create new session
            persona = await self._get_user_default_persona(user_id)
            session = Session(
                id=str(uuid.uuid4())[:12],
                user_id=user_id,
                channel=channel,
                persona=persona,
            )
            await self._save_session(session)

        self._sessions[session_key] = session
        return session

    async def _get_user_default_persona(self, user_id: str) -> str:
        """Get user's default persona."""
        if user_id in self._user_personas:
            return self._user_personas[user_id]

        row = await self.db.fetchone(
            "SELECT default_persona FROM user_preferences WHERE user_id = ?",
            (user_id,),
        )
        persona = row["default_persona"] if row else "general"
        self._user_personas[user_id] = persona
        return persona

    async def set_user_persona(self, user_id: str, persona: str) -> None:
        """Set user's default persona."""
        if persona not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona}")

        await self.db.execute(
            """
            INSERT INTO user_preferences (user_id, default_persona, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET default_persona = ?, updated_at = datetime('now')
            """,
            (user_id, persona, persona),
        )
        await self.db.commit()
        self._user_personas[user_id] = persona

        # Update active session
        session_key = None
        for key, session in self._sessions.items():
            if session.user_id == user_id:
                session.persona = persona
                await self._save_session(session)

    async def _save_session(self, session: Session) -> None:
        """Save session to database."""
        await self.db.execute(
            """
            INSERT INTO conversation_sessions (id, user_id, channel, persona, messages, created_at, last_active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                messages = ?, last_active = ?, persona = ?, metadata = ?
            """,
            (
                session.id,
                session.user_id,
                session.channel,
                session.persona,
                json.dumps(session.messages),
                session.created_at.isoformat(),
                session.last_active.isoformat(),
                json.dumps(session.metadata),
                json.dumps(session.messages),
                session.last_active.isoformat(),
                session.persona,
                json.dumps(session.metadata),
            ),
        )
        await self.db.commit()

    async def chat(
        self,
        user_id: str,
        channel: str,
        message: str,
    ) -> dict[str, Any]:
        """Process a chat message and return response.

        Returns:
            {
                "response": "text response",
                "escalate": bool,  # Should spawn agent?
                "escalate_task": str | None,  # Task for agent
                "quick_action": dict | None,  # Quick action taken
                "action_result": dict | None,  # Result from action execution
            }
        """
        session = await self.get_or_create_session(user_id, channel)
        persona = PERSONAS.get(session.persona, PERSONAS["general"])

        # Check for configuration commands or API key paste
        config_result = await self._check_config_input(message, user_id, session)
        if config_result:
            if config_result.get("handled"):
                # Already saved to session, return directly
                return {
                    "response": config_result["response"],
                    "escalate": False,
                    "escalate_task": None,
                    "quick_action": config_result,
                    "action_result": None,
                }
            else:
                session.add_message("user", message)
                session.add_message("assistant", config_result["response"])
                await self._save_session(session)
                return {
                    "response": config_result["response"],
                    "escalate": False,
                    "escalate_task": None,
                    "quick_action": config_result,
                    "action_result": None,
                }

        # Check for escalation patterns (complex dev tasks)
        should_escalate, task = self._check_escalation(message, persona)
        if should_escalate:
            session.add_message("user", message)
            await self._save_session(session)
            return {
                "response": f"I'll work on that for you. I'll let you know when it's done.",
                "escalate": True,
                "escalate_task": task or message,
                "quick_action": None,
                "action_result": None,
            }

        # Check for quick actions (reminders, calendar queries)
        quick_action = await self._check_quick_action(message, session)
        if quick_action:
            session.add_message("user", message)
            session.add_message("assistant", quick_action["response"])
            await self._save_session(session)
            return {
                "response": quick_action["response"],
                "escalate": False,
                "escalate_task": None,
                "quick_action": quick_action,
                "action_result": None,
            }

        # Try intent classification for actionable requests
        action_result = None
        if self.action_executor:
            action_result = await self._try_intent_action(message, user_id, session)
            if action_result and action_result.get("handled"):
                return action_result

        # Regular conversation
        session.add_message("user", message)

        # Build context
        system_prompt = await self._build_system_prompt(session, persona)
        messages = session.get_recent_messages()

        # Call Claude
        try:
            response = await self.claude.send_message(
                messages=messages,
                system=system_prompt,
                max_tokens=1024,
            )
            reply = response.content
        except Exception as e:
            logger.error(f"Chat error: {e}")
            reply = "Sorry, I had trouble processing that. Could you try again?"

        session.add_message("assistant", reply)
        await self._save_session(session)

        # Check if response suggests escalation
        response_escalation = self._check_response_escalation(reply)

        return {
            "response": reply,
            "escalate": response_escalation is not None,
            "escalate_task": response_escalation,
            "quick_action": None,
            "action_result": action_result,
        }

    async def _try_intent_action(
        self,
        message: str,
        user_id: str,
        session: Session,
    ) -> dict[str, Any] | None:
        """Try to classify and execute an intent-based action.

        Returns result dict with 'handled' key if action was taken.
        """
        try:
            from gru.intent import execute_intent

            classifier = self._get_intent_classifier()

            # Check if we're awaiting a location response
            provided_location = None
            if session.metadata.get("awaiting_location_for"):
                original_query = session.metadata.pop("awaiting_location_for")
                provided_location = message.strip()

                # Store the location in memory
                if self.memory:
                    try:
                        await self.memory.set_preference("location", provided_location)
                        logger.info(f"Stored location for user {user_id}: {provided_location}")
                    except Exception as e:
                        logger.warning(f"Failed to store location: {e}")

                # Use original query for classification
                message = original_query
                await self._save_session(session)

            # Get user context for classification
            user_context = {}
            if self.memory:
                try:
                    profile = await self.memory.get_user_profile()
                    user_context = {
                        "preferences": profile.get("preferences", {}),
                        "location": profile.get("preferences", {}).get("location") or provided_location,
                    }
                except Exception:
                    pass

            # If we just got a location, make sure it's in context
            if provided_location:
                user_context["location"] = provided_location

            # Classify intent
            intent = await classifier.classify(message, user_context)

            if not intent.requires_action:
                return None

            # Get calendar events for scheduling decisions
            calendar_events = []
            if self.proactive:
                try:
                    # Get pending observations that might be calendar events
                    obs = await self.proactive.get_pending_observations()
                    for o in obs:
                        if o.source == "google_calendar":
                            calendar_events.append(o.metadata or {})
                except Exception:
                    pass

            # Enrich intent with preferences and calendar
            intent = await classifier.enrich_intent(intent, user_id, calendar_events)

            # Add location to intent parameters if available
            user_location = user_context.get("location")
            if user_location and not intent.parameters.get("location"):
                intent.parameters["location"] = user_location

            # Check if we need location but don't have it
            if intent.needs_location and not intent.parameters.get("location"):
                session.add_message("user", message)
                response = "I'd need to know your address to find places nearby. What's your address? I'll remember it for next time."
                session.add_message("assistant", response)
                session.metadata["awaiting_location_for"] = intent.original_text
                await self._save_session(session)
                return {
                    "handled": True,
                    "response": response,
                    "escalate": False,
                    "escalate_task": None,
                    "quick_action": None,
                    "action_result": None,
                }

            # For research intents, run in background and respond immediately
            if intent.category in ("research", "research_scheduled"):
                import asyncio

                def strip_markdown(text: str) -> str:
                    """Remove all markdown formatting."""
                    import re
                    # Remove headers
                    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
                    # Remove bold
                    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
                    text = re.sub(r'__([^_]+)__', r'\1', text)
                    # Remove italic
                    text = re.sub(r'\*([^*]+)\*', r'\1', text)
                    text = re.sub(r'_([^_]+)_', r'\1', text)
                    # Remove any remaining asterisks
                    text = text.replace('*', '')
                    # Remove any remaining hashtags at line starts
                    text = re.sub(r'^#+ *', '', text, flags=re.MULTILINE)
                    # Remove horizontal rules
                    text = re.sub(r'^[-_]{3,}\s*$', '', text, flags=re.MULTILINE)
                    # Remove bullet markers
                    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
                    # Clean up extra blank lines
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    return text.strip()

                async def run_research_background():
                    try:
                        result = await execute_intent(intent, self.action_executor, user_id)
                        if result.get("executed"):
                            action_data = result.get("result", {}).get("data", {})
                            report = action_data.get("report", "Research complete but no report generated.")
                            # Strip markdown before sending
                            report = strip_markdown(report)
                            # Send via notify callback
                            if self.action_executor and self.action_executor._notify_callback:
                                await self.action_executor._notify_callback(user_id, report)
                    except Exception as e:
                        logger.error(f"Background research failed: {e}")
                        if self.action_executor and self.action_executor._notify_callback:
                            await self.action_executor._notify_callback(
                                user_id, f"Research failed: {e}"
                            )

                # Start background task
                asyncio.create_task(run_research_background())

                # Return varied acknowledgment based on topic and context
                import random
                topic = intent.parameters.get("topic") or intent.parameters.get("query") or message

                acknowledgments = [
                    f"On it. Looking into {topic[:50]}{'...' if len(topic) > 50 else ''}.",
                    f"Got it. I'll dig into that and get back to you.",
                    f"Researching now. Give me a moment.",
                    f"Let me look into that for you.",
                    f"Working on it. Back shortly with what I find.",
                    f"Pulling together some info on that.",
                ]
                response = random.choice(acknowledgments)

                session.add_message("user", message)
                session.add_message("assistant", response)
                await self._save_session(session)

                return {
                    "handled": True,
                    "response": response,
                    "escalate": False,
                    "escalate_task": None,
                    "quick_action": None,
                    "action_result": {"pending": True, "category": intent.category},
                }

            # Execute the intent synchronously for non-research actions
            result = await execute_intent(intent, self.action_executor, user_id)

            if result.get("executed") or result.get("scheduled"):
                # Build response message
                response_parts = []

                if result.get("scheduled"):
                    scheduled_time = result.get('scheduled_for', 'later')
                    response_parts.append(
                        f"Got it! I've scheduled that for {scheduled_time}."
                    )
                    if intent.parameters.get("timed_to_event"):
                        response_parts.append(
                            f"Timed to arrive after your {intent.parameters['timed_to_event']}."
                        )
                elif result.get("executed"):
                    # Get action result data
                    action_data = result.get("result", {}).get("data", {})

                    # Handle reports - return the full report
                    if action_data.get("report"):
                        response_parts.append(action_data["report"])
                    else:
                        response_parts.append(result.get("message", "Done!"))

                        # Add details from action result (for search results)
                        if action_data.get("results"):
                            top_results = action_data["results"][:3]
                            if top_results:
                                response_parts.append("\nTop options:")
                                for i, r in enumerate(top_results, 1):
                                    name = r.get("name", "Unknown")
                                    rating = f" ({r['rating']}*)" if r.get("rating") else ""
                                    response_parts.append(f"  {i}. {name}{rating}")

                response = "\n".join(response_parts)

                session.add_message("user", message)
                session.add_message("assistant", response)
                await self._save_session(session)

                return {
                    "handled": True,
                    "response": response,
                    "escalate": False,
                    "escalate_task": None,
                    "quick_action": None,
                    "action_result": result,
                }

        except Exception as e:
            logger.warning(f"Intent action failed: {e}")

        return None

    async def _build_system_prompt(self, session: Session, persona: Persona) -> str:
        """Build system prompt with context."""
        parts = [persona.system_prompt]

        # Add memory context
        if self.memory:
            try:
                # Get user profile
                profile = await self.memory.get_user_profile()
                if profile.get("preferences"):
                    parts.append("\nUser preferences:")
                    for key, value in list(profile["preferences"].items())[:5]:
                        parts.append(f"  - {key}: {value}")

                # Get recent relevant context
                if session.messages:
                    last_msg = session.messages[-1].get("content", "")
                    context = await self.memory.get_personalized_context(last_msg, limit=5)
                    if context:
                        parts.append(f"\n{context}")
            except Exception as e:
                logger.warning(f"Memory context failed: {e}")

        # Add pending observations
        if self.proactive:
            try:
                obs_summary = await self.proactive.get_observation_summary()
                if obs_summary:
                    parts.append(f"\n{obs_summary}")
            except Exception as e:
                logger.warning(f"Observation context failed: {e}")

        # Add current time
        now = datetime.now()
        parts.append(f"\nCurrent time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

        # Add formatting instructions based on channel
        if session.channel in ("telegram", "discord", "slack"):
            parts.append("\nFormatting: Use plain text only. No markdown, no asterisks, no bullet points. Write conversationally in natural paragraphs.")

        return "\n".join(parts)

    def _check_escalation(self, message: str, persona: Persona) -> tuple[bool, str | None]:
        """Check if message should escalate to autonomous agent."""
        message_lower = message.lower().strip()

        for pattern in ESCALATION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                # Extract the task from the message
                task = message
                return True, task

        return False, None

    def _check_response_escalation(self, response: str) -> str | None:
        """Check if Claude's response suggests it wants to escalate."""
        escalation_phrases = [
            "I'll work on that",
            "I'll handle that",
            "I'll get that done",
            "Let me work on",
            "I'll build that",
            "I'll create that",
            "working on it autonomously",
            "spawn an agent",
        ]

        response_lower = response.lower()
        for phrase in escalation_phrases:
            if phrase.lower() in response_lower:
                return response

        return None

    async def _check_config_input(
        self,
        message: str,
        user_id: str,
        session: Session,
    ) -> dict[str, Any] | None:
        """Check if message contains API keys or config commands.

        Handles:
        - Direct key paste: "sk-ant-api03-..."
        - Explicit config: "config set anthropic-key sk-ant-..."
        - Natural language: "my anthropic key is sk-ant-..."
        - Setup status: "setup status", "what's configured"

        Returns:
            Response dict if handled, None otherwise
        """
        from gru.setup import (
            detect_key_type,
            detect_multiple_keys,
            get_setup_wizard,
            KeyType,
            KEY_TYPE_NAMES,
        )

        cfg = self._get_config_manager()
        if not cfg:
            return None

        message_stripped = message.strip()
        message_lower = message_stripped.lower()

        # Check for setup/status commands
        if message_lower in ("setup", "setup status", "config status", "what's configured", "show config"):
            if self.data_dir:
                wizard = get_setup_wizard(self.data_dir)
                status_text = wizard.get_setup_instructions()
                return {
                    "type": "config_status",
                    "response": status_text,
                }

        # Check for explicit config commands
        config_match = re.match(
            r"^(?:config\s+)?set\s+([\w-]+)\s+(.+)$",
            message_stripped,
            re.IGNORECASE,
        )
        if config_match:
            key = config_match.group(1).lower()
            value = config_match.group(2).strip()
            result = cfg.set(key, value)
            key_name = KEY_TYPE_NAMES.get(result.key_type, key) if result.key_type else key
            return {
                "type": "config_set",
                "key": key,
                "response": f"Configured {key_name}. {'Saved to .env file.' if result.is_secret else ''}",
            }

        # Check for Google OAuth credentials (special handling)
        if message_lower.startswith(("google client id", "client id", "oauth client id")):
            # Extract the ID from the message
            id_match = re.search(r"(\d+-[\w]+\.apps\.googleusercontent\.com)", message_stripped)
            if id_match:
                cfg.set("google-client-id", id_match.group(1))
                return {
                    "type": "config_set",
                    "key": "google-client-id",
                    "response": "Google Client ID saved. Now send me the Client Secret.",
                }

        if message_lower.startswith(("google client secret", "client secret", "oauth client secret")):
            secret_match = re.search(r"(GOCSPX-[\w-]+)", message_stripped)
            if secret_match:
                cfg.set("google-client-secret", secret_match.group(1))
                # Check if we have both credentials
                if cfg.get("google-client-id"):
                    return {
                        "type": "config_set",
                        "key": "google-client-secret",
                        "response": "Google credentials saved! To complete setup, I need to open a browser for authentication. Run `gru google login` or tell me to authenticate with Google.",
                    }
                return {
                    "type": "config_set",
                    "key": "google-client-secret",
                    "response": "Google Client Secret saved. I still need the Client ID.",
                }

        # Try auto-detecting keys in the message
        detected = detect_multiple_keys(message_stripped)

        if detected:
            responses = []
            for key_type, value in detected:
                result = cfg.set_from_detection(value)
                if result:
                    key_name = KEY_TYPE_NAMES.get(key_type, "key")
                    responses.append(f"Detected and saved {key_name}")

                    # Check setup status after configuration
                    if self.data_dir and key_type == KeyType.ANTHROPIC:
                        wizard = get_setup_wizard(self.data_dir)
                        status = wizard.get_setup_status()
                        if not status["steps"]["messaging"]["configured"]:
                            responses.append("\nTo complete setup, I need a messaging platform token (Telegram, Discord, or Slack).")

            if responses:
                # Don't store the key in conversation history (security)
                session.add_message("user", "[API key configured]")
                session.add_message("assistant", "\n".join(responses))
                await self._save_session(session)

                return {
                    "type": "config_keys",
                    "count": len(detected),
                    "response": "\n".join(responses),
                    "handled": True,  # Mark as fully handled
                }

        # Check for preference setting via natural language
        pref_patterns = [
            (r"(?:my\s+)?location\s+is\s+(.+)", "location"),
            (r"i(?:'m|\s+am)\s+(?:in|at|from)\s+(.+)", "location"),
            (r"(?:i\s+)?prefer\s+(\w+)\s+food", "food"),
            (r"(?:i\s+)?(?:like|prefer|want)\s+(.+?)\s+(?:for\s+)?(?:food|meals|eating)", "food"),
            (r"(?:my\s+)?(?:default\s+)?budget\s+is\s+(.+)", "budget"),
        ]

        for pattern, pref_key in pref_patterns:
            match = re.search(pattern, message_lower)
            if match:
                value = match.group(1).strip()
                # Store as memory preference
                if self.memory:
                    await self.memory.set_preference(pref_key, value)
                    return {
                        "type": "preference_set",
                        "key": pref_key,
                        "value": value,
                        "response": f"Got it, I'll remember your {pref_key}: {value}",
                    }

        return None

    async def _check_quick_action(
        self,
        message: str,
        session: Session,
    ) -> dict[str, Any] | None:
        """Check for and execute quick actions."""
        message_lower = message.lower().strip()

        # Check for reminder/todo patterns
        reminder_match = re.search(
            r"(remind\s+me|add\s+(?:a\s+)?(?:reminder|todo|task))\s+(?:to\s+)?(.+?)(?:\s+(?:at|in|on|by)\s+(.+))?$",
            message_lower,
            re.IGNORECASE,
        )
        if reminder_match:
            content = reminder_match.group(2).strip()
            time_spec = reminder_match.group(3)

            if self.proactive:
                await self.proactive.add_observation(
                    content=content,
                    category="reminder",
                    importance=0.7,
                    source=f"session:{session.id}",
                    expires_in_hours=24 if not time_spec else None,
                )
                return {
                    "type": "reminder",
                    "content": content,
                    "response": f"Got it, I'll remind you: {content}",
                }

        # Check for calendar query
        if re.search(r"what('s|\s+is)\s+(on\s+)?(my\s+)?(calendar|schedule|agenda)", message_lower):
            # This would integrate with Google Calendar
            return {
                "type": "calendar_query",
                "response": "Let me check your calendar...",  # Would be filled by actual calendar check
            }

        return None

    async def reset_session(self, user_id: str, channel: str) -> None:
        """Reset/clear a session."""
        session = await self.get_or_create_session(user_id, channel)
        session.clear_messages()
        await self._save_session(session)

    async def get_session_stats(self, user_id: str) -> dict[str, Any]:
        """Get stats about user's sessions."""
        rows = await self.db.fetchall(
            "SELECT channel, persona, created_at, last_active, json_array_length(messages) as msg_count FROM conversation_sessions WHERE user_id = ?",
            (user_id,),
        )

        return {
            "sessions": [
                {
                    "channel": row["channel"],
                    "persona": row["persona"],
                    "created": row["created_at"],
                    "last_active": row["last_active"],
                    "message_count": row["msg_count"],
                }
                for row in rows
            ],
            "total_sessions": len(rows),
        }


def get_available_personas() -> list[dict[str, str]]:
    """Get list of available personas."""
    return [
        {"name": p.name, "description": p.description}
        for p in PERSONAS.values()
    ]
