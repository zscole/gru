"""Claude API integration using Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anthropic

if TYPE_CHECKING:
    from gru.config import Config

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


@dataclass
class ToolDefinition:
    """Definition of a tool available to Claude."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolUse:
    """A tool use request from Claude."""

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
    """Response from Claude."""

    content: str
    tool_uses: list[ToolUse]
    stop_reason: str
    usage: dict[str, int]


async def retry_with_backoff(
    coro_func,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    **kwargs,
):
    """Execute a coroutine with exponential backoff retry logic.

    Args:
        coro_func: Async function to call
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        *args, **kwargs: Arguments to pass to the coroutine

    Returns:
        Result of the coroutine

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"Claude API call failed after {max_retries + 1} attempts: {e}")
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2**attempt), max_delay)
            # Add jitter (0-25% of delay) to prevent thundering herd
            jitter = delay * 0.25 * random.random()
            delay = delay + jitter

            logger.warning(
                f"Claude API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


class ClaudeClient:
    """Async client for Claude API."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> Response:
        """Send a message to Claude and get a response."""
        model = model or self.config.default_model
        max_tokens = max_tokens or self.config.max_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = await retry_with_backoff(self._client.messages.create, **kwargs)

        content = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_uses.append(
                    ToolUse(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return Response(
            content=content,
            tool_uses=tool_uses,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a message response from Claude."""
        model = model or self.config.default_model
        max_tokens = max_tokens or self.config.max_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def send_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[ToolResult],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> Response:
        """Send tool results back to Claude."""
        # Add tool results to messages
        tool_result_content = []
        for result in tool_results:
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )

        messages = messages + [{"role": "user", "content": tool_result_content}]

        return await self.send_message(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
        )


# Default tools available to agents
DEFAULT_TOOLS = [
    ToolDefinition(
        name="bash",
        description="Execute a bash command. Use for file operations, running scripts, or system tasks.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="read_file",
        description="Read the contents of a file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                }
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description="Write content to a file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="search_files",
        description="Search for files matching a pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    ),
    ToolDefinition(
        name="request_human_input",
        description="Request input or approval from the human operator.",
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or request for the human",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices for the human",
                },
            },
            "required": ["question"],
        },
    ),
    ToolDefinition(
        name="send_message_to_agent",
        description="Send a message to another agent for coordination.",
        input_schema={
            "type": "object",
            "properties": {
                "to_agent": {
                    "type": "string",
                    "description": "ID or name of the target agent",
                },
                "message": {
                    "type": "string",
                    "description": "Message content to send",
                },
                "message_type": {
                    "type": "string",
                    "enum": ["request", "response", "info", "handoff"],
                    "description": "Type of message",
                    "default": "info",
                },
            },
            "required": ["to_agent", "message"],
        },
    ),
    ToolDefinition(
        name="get_shared_context",
        description="Get shared context values for the current task.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Optional specific key to retrieve. If not provided, returns all context.",
                },
            },
        },
    ),
    ToolDefinition(
        name="set_shared_context",
        description="Set a shared context value for other agents to access.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key for the context value",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (will be JSON encoded)",
                },
            },
            "required": ["key", "value"],
        },
    ),
]
