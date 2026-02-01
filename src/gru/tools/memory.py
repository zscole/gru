"""Memory tools for storing and retrieving user information."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.memory import MemoryStore

logger = logging.getLogger(__name__)

# Memory store reference (set by agent initialization)
_memory: MemoryStore | None = None


def set_memory_store(memory: MemoryStore) -> None:
    """Set the memory store for memory tools."""
    global _memory
    _memory = memory


def get_memory_store() -> MemoryStore | None:
    """Get the memory store."""
    return _memory


async def remember(key: str, value: str) -> dict:
    """Store a preference or fact about the user."""
    if not _memory:
        return {"error": "Memory not available"}

    try:
        await _memory.set_preference(key, value)
        return {"status": "remembered", "key": key, "value": value}
    except Exception as e:
        logger.error(f"Failed to remember {key}: {e}")
        return {"error": str(e)}


async def recall(key: str | None = None) -> dict:
    """Recall stored information about the user."""
    if not _memory:
        return {"error": "Memory not available"}

    try:
        profile = await _memory.get_user_profile()
        preferences = profile.get("preferences", {})

        if key:
            value = preferences.get(key)
            if value:
                return {"key": key, "value": value}
            else:
                return {"key": key, "value": None, "note": f"No value stored for '{key}'"}
        else:
            return {"preferences": preferences}
    except Exception as e:
        logger.error(f"Failed to recall: {e}")
        return {"error": str(e)}


async def get_user_context() -> dict:
    """Get all known context about the user."""
    if not _memory:
        return {"error": "Memory not available"}

    try:
        profile = await _memory.get_user_profile()
        return {
            "preferences": profile.get("preferences", {}),
            "facts": profile.get("facts", [])[:10],  # Last 10 facts
        }
    except Exception as e:
        logger.error(f"Failed to get user context: {e}")
        return {"error": str(e)}


def register_memory_tools() -> None:
    """Register all memory tools."""
    register_tool(
        name="remember",
        description="Store information about the user for future reference. Use this when the user tells you their preferences, location, name, or other personal info.",
        parameters={
            "key": {
                "type": "string",
                "description": "What to remember (e.g., 'location', 'name', 'food_preference')",
            },
            "value": {
                "type": "string",
                "description": "The value to store",
            },
        },
        handler=remember,
    )

    register_tool(
        name="recall",
        description="Recall stored information about the user. Use this to look up user preferences or facts you previously stored.",
        parameters={
            "key": {
                "type": "string",
                "description": "The key to look up (e.g., 'location'). If not provided, returns all preferences.",
                "optional": True,
            },
        },
        handler=recall,
    )

    register_tool(
        name="get_user_context",
        description="Get all known context about the user including preferences and facts.",
        parameters={},
        handler=get_user_context,
    )
