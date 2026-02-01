"""Abstract base classes for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderCapability(Enum):
    """Capabilities a provider may support."""

    TOOL_USE = "tool_use"
    STREAMING = "streaming"
    VISION = "vision"
    LONG_CONTEXT = "long_context"  # >32k tokens
    UNCENSORED = "uncensored"


@dataclass
class ProviderInfo:
    """Metadata about a provider."""

    name: str
    models: list[str]
    capabilities: set[ProviderCapability]
    cost_per_1k_input: float  # USD, 0 for local
    cost_per_1k_output: float  # USD, 0 for local


@dataclass
class ToolDefinition:
    """Definition of a tool available to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolUse:
    """A tool use request from the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class Response:
    """Response from a model."""

    content: str
    tool_uses: list[ToolUse]
    stop_reason: str
    usage: dict[str, int]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        ...

    @abstractmethod
    async def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> Response:
        """Send messages to the model and get a response."""
        ...

    @abstractmethod
    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a message response from the model."""
        ...

    async def send_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[ToolResult],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> Response:
        """Send tool results back to the model.

        Default implementation appends tool results and calls send_message.
        Providers may override if they need different handling.
        """
        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_use_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in tool_results
        ]
        messages = [*messages, {"role": "user", "content": tool_result_content}]
        return await self.send_message(messages, system, model, max_tokens, tools)
