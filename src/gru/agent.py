"""Conversational agent with tool use.

This module provides a simpler alternative to the intent classification
approach. Instead of pattern matching and routing, it gives Claude access
to tools and lets it decide what to use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from gru.tools import execute_tool, get_all_tools, initialize_tools
from gru.tools.memory import set_memory_store

if TYPE_CHECKING:
    from gru.claude import ClaudeClient
    from gru.memory import MemoryStore
    from gru.proactive import ProactiveEngine
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Gru, Zak's personal assistant. Talk like a human, not a bot.

HOW TO COMMUNICATE:
- Be conversational and natural, like texting a smart friend
- No bullet points, no markdown, no asterisks, no emojis
- No formal greetings or sign-offs
- Short responses unless more detail is needed
- Just answer or do the thing, don't explain what you're about to do

WHAT NOT TO DO:
- Don't say "I'll help you with that" or "Let me do that for you"
- Don't list out steps before doing them
- Don't ask for permission on clear requests
- Don't explain your capabilities, just use them
- Don't be verbose or corporate-sounding

CAPABILITIES:
- Directions, places, web search, memory, reminders, calendar, email
- Spawning agents for complex tasks (they can spawn sub-agents too)
- Task decomposition - break "plan my vacation" into research, booking, calendar, reminders
- Self-improvement via improve_gru
- AI research monitoring

For complex multi-step requests like "plan my vacation" or "research X and build a prototype":
1. Use decompose_task to break it into subtasks
2. Spawn agents for each major piece (they'll coordinate via sub-agents if needed)
3. Send progress updates so Zak knows what's happening

After using a tool, just tell Zak the result conversationally. Like "Done, reminder set for 3pm" or "It's about 20 minutes from your place" - not a formatted report.
"""


@dataclass
class Conversation:
    """A conversation with message history."""

    user_id: str
    channel: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
        self.last_active = datetime.now()

    def add_assistant_message(self, content: Any) -> None:
        """Add an assistant message (can be text or content blocks)."""
        self.messages.append({"role": "assistant", "content": content})
        self.last_active = datetime.now()

    def add_tool_results(self, results: list[dict]) -> None:
        """Add tool results as a user message."""
        self.messages.append({"role": "user", "content": results})
        self.last_active = datetime.now()

    def get_messages(self, limit: int = 50) -> list[dict]:
        """Get recent messages for API call."""
        return self.messages[-limit:]


class Agent:
    """Conversational agent that uses Claude with tools."""

    def __init__(
        self,
        claude: ClaudeClient,
        memory: MemoryStore | None = None,
        proactive: ProactiveEngine | None = None,
        orchestrator: Orchestrator | None = None,
    ):
        self.claude = claude
        self.memory = memory
        self.proactive = proactive
        self.orchestrator = orchestrator
        self._conversations: dict[str, Conversation] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent and tools."""
        if self._initialized:
            return

        # Initialize tools
        initialize_tools()

        # Set up memory for memory tools
        if self.memory:
            set_memory_store(self.memory)

        # Set up proactive engine for reminder/trigger tools
        if self.proactive:
            from gru.tools.proactive import set_proactive_engine
            set_proactive_engine(self.proactive)

        # Set up orchestrator for agent management tools
        if self.orchestrator:
            from gru.tools.orchestrator import set_orchestrator
            set_orchestrator(self.orchestrator)

        self._initialized = True
        logger.info("Agent initialized with tools")

    def _get_conversation(self, user_id: str, channel: str) -> Conversation:
        """Get or create a conversation."""
        key = f"{channel}:{user_id}"
        if key not in self._conversations:
            self._conversations[key] = Conversation(user_id=user_id, channel=channel)
        return self._conversations[key]

    async def _build_system_prompt(self, user_id: str) -> str:
        """Build system prompt with user context."""
        parts = [SYSTEM_PROMPT]

        # Add user preferences from memory
        if self.memory:
            try:
                profile = await self.memory.get_user_profile()
                prefs = profile.get("preferences", {})
                if prefs:
                    pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items())
                    parts.append(f"\nUser preferences: {pref_str}")
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")

        # Add proactive context (learned patterns, insights, anticipations)
        if self.proactive:
            try:
                proactive_context = await self.proactive.get_context_summary()
                if proactive_context and proactive_context != "No proactive context":
                    parts.append(f"\nPROACTIVE CONTEXT: {proactive_context}")

                # Check for anticipations based on current time
                now = datetime.now()
                context = await self.proactive._build_context()
                anticipations = [
                    p for p in self.proactive._patterns.values()
                    if p.matches_now(now, context)
                ]
                if anticipations:
                    ant_str = ", ".join(p.description for p in anticipations[:3])
                    parts.append(f"ANTICIPATED NEEDS: {ant_str}")
            except Exception as e:
                logger.warning(f"Failed to load proactive context: {e}")

        # Add current time
        now = datetime.now()
        parts.append(f"\nCurrent time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

        return "\n".join(parts)

    async def chat(
        self,
        user_id: str,
        message: str,
        channel: str = "cli",
    ) -> str:
        """Process a message and return a response.

        This is the main entry point. It handles the full conversation loop,
        including tool use.
        """
        await self.initialize()

        # Track user interaction for pattern learning
        await self._track_user_message(message)

        conversation = self._get_conversation(user_id, channel)
        conversation.add_user_message(message)

        system = await self._build_system_prompt(user_id)
        tools = get_all_tools()

        # Agent loop - keep going until Claude gives a final response
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            try:
                response = await self.claude.send_message(
                    messages=conversation.get_messages(),
                    system=system,
                    tools=tools,
                    max_tokens=2048,
                )
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                return f"Sorry, I had trouble processing that: {e}"

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and response.tool_uses:
                # Build content blocks for assistant message
                content_blocks = []
                if response.content:
                    content_blocks.append({"type": "text", "text": response.content})

                for tu in response.tool_uses:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tu.id,
                        "name": tu.name,
                        "input": tu.input,
                    })

                conversation.add_assistant_message(content_blocks)

                # Execute each tool call
                tool_results = []
                for tu in response.tool_uses:
                    logger.info(f"Executing tool: {tu.name} with {tu.input}")
                    result = await execute_tool(tu.name, tu.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result),
                    })

                conversation.add_tool_results(tool_results)
                continue  # Get Claude's response to tool results

            # Claude gave final response
            conversation.add_assistant_message(response.content)

            # Strip any remaining markdown
            text = response.content or ""
            text = self._strip_markdown(text)

            return text

        # Max iterations reached
        return "I got stuck in a loop. Could you try rephrasing your request?"

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        import re

        # Remove headers
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # Remove bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # Remove italic
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # Remove remaining asterisks
        text = text.replace('*', '')
        # Remove bullet markers
        text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
        # Clean up extra blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    async def _track_user_message(self, message: str) -> None:
        """Track user message for pattern learning."""
        if not self.proactive:
            return

        # Detect action type from message content
        message_lower = message.lower()

        action = None
        context: dict[str, Any] = {}

        # Email-related
        if any(kw in message_lower for kw in ["email", "mail", "inbox", "gmail"]):
            action = "check_email"

        # Calendar-related
        elif any(kw in message_lower for kw in ["calendar", "schedule", "meeting", "appointment"]):
            action = "check_calendar"

        # Search-related
        elif any(kw in message_lower for kw in ["search", "find", "look up", "google"]):
            action = "search"
            # Extract search topic if possible
            import re
            match = re.search(r"(?:search|find|look up|google)\s+(?:for\s+)?(.+)", message_lower)
            if match:
                context["query"] = match.group(1)[:50]

        # Weather-related
        elif any(kw in message_lower for kw in ["weather", "temperature", "forecast"]):
            action = "check_weather"

        # Directions/navigation
        elif any(kw in message_lower for kw in ["directions", "navigate", "how to get", "route"]):
            action = "get_directions"

        # Reminders
        elif any(kw in message_lower for kw in ["remind", "reminder"]):
            action = "set_reminder"

        # News/research
        elif any(kw in message_lower for kw in ["news", "research", "what's happening"]):
            action = "check_news"

        # General conversation
        else:
            action = "chat"

        try:
            await self.proactive.track_behavior(action, context)
        except Exception as e:
            logger.debug(f"Failed to track behavior: {e}")

    def reset_conversation(self, user_id: str, channel: str) -> None:
        """Reset a conversation."""
        key = f"{channel}:{user_id}"
        if key in self._conversations:
            del self._conversations[key]
