"""Anthropic Claude provider implementation."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anthropic

from gru.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderInfo,
    Response,
    ToolDefinition,
    ToolUse,
)

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


async def retry_with_backoff(
    coro_func,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    **kwargs,
):
    """Execute a coroutine with exponential backoff retry logic."""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"Anthropic API call failed after {max_retries + 1} attempts: {e}")
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            jitter = delay * 0.25 * random.random()
            delay = delay + jitter

            logger.warning(
                f"Anthropic API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    if last_exception:
        raise last_exception


class AnthropicProvider(LLMProvider):
    """Claude via Anthropic API."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="anthropic",
            models=[
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
            ],
            capabilities={
                ProviderCapability.TOOL_USE,
                ProviderCapability.STREAMING,
                ProviderCapability.VISION,
                ProviderCapability.LONG_CONTEXT,
            },
            cost_per_1k_input=3.0,
            cost_per_1k_output=15.0,
        )

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
