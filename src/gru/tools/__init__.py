"""Gru tools for Claude tool_use.

This module provides a set of tools that Claude can use to interact with
external services, remember information, and perform actions.
"""

from __future__ import annotations

import logging
from typing import Any

from gru.claude import ToolDefinition
from gru.tools.base import (
    Tool,
    ToolRegistry,
    get_registry,
    register_tool,
)

logger = logging.getLogger(__name__)

_initialized = False


def initialize_tools() -> ToolRegistry:
    """Initialize and register all tools."""
    global _initialized

    if _initialized:
        return get_registry()

    # Register tools from each module
    from gru.tools.maps import register_maps_tools
    from gru.tools.search import register_search_tools
    from gru.tools.memory import register_memory_tools
    from gru.tools.browse import register_browse_tools
    from gru.tools.provision import register_provision_tools
    from gru.tools.google import register_google_tools
    from gru.tools.slack import register_slack_tools
    from gru.tools.proactive import register_proactive_tools
    from gru.tools.orchestrator import register_orchestrator_tools
    from gru.tools.voice import register_voice_tools
    from gru.tools.research import register_research_tools
    from gru.tools.knowledge import register_knowledge_tools
    from gru.tools.actions import register_action_tools
    from gru.tools.self_heal import register_self_heal_tools

    register_maps_tools()
    register_search_tools()
    register_memory_tools()
    register_browse_tools()
    register_provision_tools()
    register_google_tools()
    register_slack_tools()
    register_proactive_tools()
    register_orchestrator_tools()
    register_voice_tools()
    register_research_tools()
    register_knowledge_tools()
    register_action_tools()
    register_self_heal_tools()

    _initialized = True
    logger.info(f"Initialized {len(get_registry().all_tools())} tools")

    return get_registry()


def get_all_tools() -> list[ToolDefinition]:
    """Get all tools as ToolDefinition objects for Claude API."""
    registry = initialize_tools()
    tools = []
    for t in registry.all_tools():
        # Build properties without the 'optional' key (not valid JSON Schema)
        properties = {}
        required = []
        for param_name, param_def in t.parameters.items():
            prop = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
            }
            if "enum" in param_def:
                prop["enum"] = param_def["enum"]
            properties[param_name] = prop

            if not param_def.get("optional", False):
                required.append(param_name)

        tool_def = ToolDefinition(
            name=t.name,
            description=t.description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
        tools.append(tool_def)
    return tools


async def execute_tool(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name."""
    registry = initialize_tools()
    return await registry.execute(name, params)


__all__ = [
    "Tool",
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "initialize_tools",
    "get_all_tools",
    "execute_tool",
]
