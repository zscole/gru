"""Base tool definitions for Claude tool_use."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A tool Claude can use."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]] | None = None

    def to_claude_format(self) -> dict[str, Any]:
        """Convert to Claude API tool format."""
        # Build properties and required list
        properties = {}
        required = []

        for param_name, param_def in self.parameters.items():
            prop = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
            }
            if "enum" in param_def:
                prop["enum"] = param_def["enum"]
            properties[param_name] = prop

            if not param_def.get("optional", False):
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry for tools available to Claude."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_claude_format(self) -> list[dict[str, Any]]:
        """Get all tools in Claude API format."""
        return [t.to_claude_format() for t in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        if not tool.handler:
            return {"error": f"Tool {name} has no handler"}

        try:
            result = await tool.handler(**params)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {"error": str(e)}


# Global registry
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable[..., Awaitable[Any]],
) -> Tool:
    """Register a tool with the global registry."""
    tool = Tool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
    )
    get_registry().register(tool)
    return tool
