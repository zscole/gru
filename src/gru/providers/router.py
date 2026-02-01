"""Provider routing logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gru.providers.base import LLMProvider, ProviderCapability


class TaskType(Enum):
    """Categories for routing decisions."""

    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    SIMPLE_CLEANUP = "simple_cleanup"
    WEB_SEARCH = "web_search"
    UNRESTRICTED = "unrestricted"
    VISION = "vision"
    GENERAL = "general"


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    provider: str
    model: str | None
    reason: str


class ProviderRouter:
    """Decides which provider handles a task."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default_provider: str = "anthropic",
    ) -> None:
        self.providers = providers
        self.default = default_provider

    def route(
        self,
        task_type: TaskType | None = None,
        requires_tools: bool = False,
        requires_vision: bool = False,
        prefer_cheap: bool = False,
        prefer_uncensored: bool = False,
        preferred_provider: str | None = None,
        preferred_model: str | None = None,
    ) -> RoutingDecision:
        """Pick a provider based on task requirements.

        Args:
            task_type: Category of the task for routing hints
            requires_tools: Task requires tool/function calling
            requires_vision: Task requires image understanding
            prefer_cheap: Optimize for cost over capability
            prefer_uncensored: Prefer models with fewer restrictions
            preferred_provider: Explicit provider preference (overrides routing)
            preferred_model: Explicit model preference

        Returns:
            RoutingDecision with provider name, model, and reasoning
        """
        # Explicit preference takes precedence
        if preferred_provider and preferred_provider in self.providers:
            return RoutingDecision(
                preferred_provider,
                preferred_model,
                "explicit preference",
            )

        # Hard constraints - vision requires Anthropic currently
        if requires_vision:
            if not self._supports("anthropic", ProviderCapability.VISION):
                return RoutingDecision(
                    "anthropic",
                    "claude-sonnet-4-20250514",
                    "vision required, only anthropic supports it",
                )
            return RoutingDecision(
                "anthropic",
                "claude-sonnet-4-20250514",
                "vision required",
            )

        # Tool use constraint - check if ollama supports it
        if requires_tools and "ollama" in self.providers:
            if not self._supports("ollama", ProviderCapability.TOOL_USE):
                return RoutingDecision(
                    "anthropic",
                    None,
                    "tool use required, ollama doesn't support it",
                )

        # Preference-based routing
        if prefer_uncensored and "ollama" in self.providers:
            return RoutingDecision(
                "ollama",
                "dolphin-mixtral",
                "uncensored requested",
            )

        if prefer_cheap and "ollama" in self.providers:
            return RoutingDecision(
                "ollama",
                "llama3:8b",
                "cost optimization",
            )

        # Task-type based routing
        if task_type == TaskType.SIMPLE_CLEANUP and "ollama" in self.providers:
            return RoutingDecision(
                "ollama",
                "llama3:8b",
                "simple cleanup task",
            )

        if task_type == TaskType.WEB_SEARCH and "ollama" in self.providers:
            return RoutingDecision(
                "ollama",
                "llama3:8b",
                "web search task",
            )

        if task_type == TaskType.COMPLEX_REASONING:
            return RoutingDecision(
                "anthropic",
                "claude-opus-4-20250514",
                "complex reasoning task",
            )

        if task_type == TaskType.CODE_GENERATION:
            return RoutingDecision(
                "anthropic",
                "claude-sonnet-4-20250514",
                "code generation task",
            )

        if task_type == TaskType.UNRESTRICTED and "ollama" in self.providers:
            return RoutingDecision(
                "ollama",
                "dolphin-mixtral",
                "unrestricted task",
            )

        # Default fallback
        return RoutingDecision(
            self.default,
            None,
            "default routing",
        )

    def get(self, name: str) -> LLMProvider:
        """Get a provider by name."""
        if name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available: {available}")
        return self.providers[name]

    def list_providers(self) -> list[str]:
        """List available provider names."""
        return list(self.providers.keys())

    def _supports(self, provider: str, capability: ProviderCapability) -> bool:
        """Check if a provider supports a capability."""
        if provider not in self.providers:
            return False
        return capability in self.providers[provider].info.capabilities
