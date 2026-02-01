"""Ollama provider implementation for local models."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

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


class OllamaProvider(LLMProvider):
    """Local models via Ollama."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = config.ollama_url
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="ollama",
            models=[
                "llama3:8b",
                "llama3:70b",
                "llama3.1:8b",
                "llama3.1:70b",
                "mistral",
                "mixtral",
                "dolphin-mixtral",
                "codellama",
            ],
            capabilities={
                ProviderCapability.STREAMING,
                ProviderCapability.TOOL_USE,  # Ollama 0.4+ supports tools
            },
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    async def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> Response:
        """Send a message to Ollama and get a response."""
        model = model or self.config.ollama_default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages, system),
            "stream": False,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        if tools:
            payload["tools"] = self._convert_tools(tools)

        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama connection error: {e}")
            raise

        message = data.get("message", {})
        content = message.get("content", "")
        tool_uses = self._parse_tool_calls(message)

        return Response(
            content=content,
            tool_uses=tool_uses,
            stop_reason=data.get("done_reason", "stop"),
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
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
        """Stream a message response from Ollama."""
        model = model or self.config.ollama_default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages, system),
            "stream": True,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        if tools:
            payload["tools"] = self._convert_tools(tools)

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content
                except json.JSONDecodeError:
                    continue

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> list[dict[str, Any]]:
        """Convert Anthropic message format to Ollama/OpenAI format."""
        result = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Handle Anthropic's content blocks
                text_parts = []
                tool_results = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            tool_results.append(block)
                        elif block.get("type") == "tool_use":
                            # Convert tool use to assistant message with tool call
                            pass

                if text_parts:
                    result.append({"role": role, "content": " ".join(text_parts)})
                elif tool_results:
                    # Convert tool results to Ollama format
                    for tr in tool_results:
                        result.append(
                            {
                                "role": "tool",
                                "content": tr.get("content", ""),
                            }
                        )

        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to Ollama format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _parse_tool_calls(self, message: dict[str, Any]) -> list[ToolUse]:
        """Parse tool calls from Ollama response."""
        tool_uses = []
        tool_calls = message.get("tool_calls", [])

        for call in tool_calls:
            func = call.get("function", {})
            tool_uses.append(
                ToolUse(
                    id=call.get("id", str(uuid.uuid4())),
                    name=func.get("name", ""),
                    input=func.get("arguments", {}),
                )
            )

        return tool_uses

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
