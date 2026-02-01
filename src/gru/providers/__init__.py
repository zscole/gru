"""LLM provider abstraction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gru.providers.anthropic import AnthropicProvider
from gru.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderInfo,
    Response,
    ToolDefinition,
    ToolResult,
    ToolUse,
)
from gru.providers.ollama import OllamaProvider
from gru.providers.router import ProviderRouter, RoutingDecision, TaskType

if TYPE_CHECKING:
    from gru.config import Config

__all__ = [
    "LLMProvider",
    "ProviderCapability",
    "ProviderInfo",
    "Response",
    "ToolDefinition",
    "ToolResult",
    "ToolUse",
    "AnthropicProvider",
    "OllamaProvider",
    "ProviderRouter",
    "RoutingDecision",
    "TaskType",
    "create_router",
]


def create_router(config: Config) -> ProviderRouter:
    """Factory to create configured router with available providers."""
    providers: dict[str, LLMProvider] = {
        "anthropic": AnthropicProvider(config),
    }

    # Only create OllamaProvider if ollama_url is a real string
    if isinstance(config.ollama_url, str) and config.ollama_url:
        providers["ollama"] = OllamaProvider(config)

    # Default to anthropic if default_provider is not a valid string
    default = config.default_provider
    if not isinstance(default, str) or default not in providers:
        default = "anthropic"

    return ProviderRouter(providers, default_provider=default)
