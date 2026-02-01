"""Action registry for registering and dispatching actions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gru.actions.base import Action, ActionContext, ActionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for action types."""

    def __init__(self) -> None:
        self._actions: dict[str, type[Action]] = {}
        self._aliases: dict[str, str] = {}  # alias -> action_name

    def register(self, action_class: type[Action], aliases: list[str] | None = None) -> None:
        """Register an action class.

        Args:
            action_class: The action class to register
            aliases: Optional list of alternative names
        """
        name = action_class.name
        if name in self._actions:
            logger.warning(f"Overwriting action: {name}")

        self._actions[name] = action_class
        logger.debug(f"Registered action: {name}")

        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> type[Action] | None:
        """Get an action class by name or alias."""
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]

        return self._actions.get(name)

    def create(self, name: str) -> Action | None:
        """Create an instance of an action by name."""
        action_class = self.get(name)
        if action_class:
            return action_class()
        return None

    def list_actions(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all registered actions.

        Args:
            category: Optional category filter

        Returns:
            List of action info dicts
        """
        actions = []
        for name, action_class in self._actions.items():
            if category and action_class.category != category:
                continue
            actions.append(
                {
                    "name": name,
                    "description": action_class.description,
                    "category": action_class.category,
                    "requires_auth": action_class.requires_auth,
                    "requires_confirmation": action_class.requires_confirmation,
                }
            )
        return actions

    def list_categories(self) -> list[str]:
        """List all action categories."""
        categories = set()
        for action_class in self._actions.values():
            categories.add(action_class.category)
        return sorted(categories)

    async def execute(
        self,
        action_name: str,
        context: ActionContext,
        **params,
    ) -> ActionResult:
        """Execute an action by name.

        Args:
            action_name: Name or alias of the action
            context: Action context
            **params: Action parameters

        Returns:
            ActionResult
        """
        action = self.create(action_name)
        if not action:
            return ActionResult.error_result(f"Unknown action: {action_name}")

        return await action.run(context, **params)


# Global registry instance
_registry: ActionRegistry | None = None


def get_registry() -> ActionRegistry:
    """Get the global action registry."""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
        _register_default_actions(_registry)
    return _registry


def _register_default_actions(registry: ActionRegistry) -> None:
    """Register default actions."""
    # Import and register service actions
    from gru.actions.services.google import (
        CompileDocumentAction,
        CreateDocumentAction,
        SendEmailAction,
        WriteDocumentAction,
    )
    from gru.actions.services.research import (
        QuickAnswerAction,
        ResearchAction,
    )
    from gru.actions.services.search import (
        DistanceAction,
        LocalSearchAction,
        RestaurantSearchAction,
        WebSearchAction,
    )
    from gru.actions.services.ubereats import (
        UberEatsCartAction,
        UberEatsOrderAction,
        UberEatsSearchAction,
    )
    from gru.actions.services.web import (
        ClickAction,
        ExtractAction,
        NavigateAction,
        ScreenshotAction,
        TypeAction,
        WaitAction,
    )

    # Web actions
    registry.register(NavigateAction, aliases=["goto", "open"])
    registry.register(ClickAction, aliases=["click"])
    registry.register(TypeAction, aliases=["type", "input"])
    registry.register(ExtractAction, aliases=["extract", "scrape"])
    registry.register(ScreenshotAction, aliases=["screenshot", "capture"])
    registry.register(WaitAction, aliases=["wait", "pause"])

    # Search actions
    registry.register(WebSearchAction, aliases=["search", "google"])
    registry.register(LocalSearchAction, aliases=["local_search", "nearby"])
    registry.register(DistanceAction, aliases=["get_distance", "directions", "distance"])
    registry.register(RestaurantSearchAction, aliases=["find_restaurant", "restaurants"])

    # Uber Eats actions
    registry.register(UberEatsSearchAction, aliases=["ubereats_search", "food_search"])
    registry.register(UberEatsOrderAction, aliases=["ubereats_order", "order_food"])
    registry.register(UberEatsCartAction, aliases=["ubereats_cart", "cart"])

    # Google actions
    registry.register(CreateDocumentAction, aliases=["create_doc", "new_doc"])
    registry.register(WriteDocumentAction, aliases=["write_doc", "update_doc"])
    registry.register(SendEmailAction, aliases=["email", "send_mail"])
    registry.register(CompileDocumentAction, aliases=["compile_doc", "write_prd"])

    # Research actions
    registry.register(ResearchAction, aliases=["research", "investigate", "analyze"])
    registry.register(QuickAnswerAction, aliases=["quick_answer", "answer", "lookup"])


def register_action(action_class: type[Action], aliases: list[str] | None = None) -> None:
    """Register an action with the global registry.

    Can be used as a decorator:
        @register_action
        class MyAction(Action):
            ...
    """
    get_registry().register(action_class, aliases)
    return action_class
