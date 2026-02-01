"""Base classes for actions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.actions.browser import Browser

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an action execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_AUTH = "needs_auth"  # Requires login/authentication
    NEEDS_CONFIRM = "needs_confirm"  # Requires user confirmation (e.g., payment)


@dataclass
class ActionResult:
    """Result of an action execution."""

    status: ActionStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    screenshot_path: str | None = None
    next_action: str | None = None  # Suggested follow-up action
    confirmation_required: dict[str, Any] | None = None  # Details for confirmation

    @property
    def success(self) -> bool:
        """Check if action completed successfully."""
        return self.status == ActionStatus.COMPLETED

    @property
    def needs_user_input(self) -> bool:
        """Check if action needs user input to proceed."""
        return self.status in (ActionStatus.NEEDS_AUTH, ActionStatus.NEEDS_CONFIRM)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "screenshot_path": self.screenshot_path,
            "next_action": self.next_action,
            "confirmation_required": self.confirmation_required,
        }

    @classmethod
    def success_result(cls, message: str, data: dict[str, Any] | None = None) -> ActionResult:
        """Create a successful result."""
        return cls(
            status=ActionStatus.COMPLETED,
            message=message,
            data=data or {},
        )

    @classmethod
    def error_result(cls, message: str, error: str | None = None) -> ActionResult:
        """Create an error result."""
        return cls(
            status=ActionStatus.FAILED,
            message=message,
            error=error or message,
        )

    @classmethod
    def auth_required(cls, service: str, login_url: str | None = None) -> ActionResult:
        """Create a result indicating authentication is needed."""
        return cls(
            status=ActionStatus.NEEDS_AUTH,
            message=f"Login required for {service}",
            data={"service": service, "login_url": login_url},
        )

    @classmethod
    def confirm_required(
        cls,
        message: str,
        details: dict[str, Any],
        action_id: str | None = None,
    ) -> ActionResult:
        """Create a result requiring user confirmation."""
        return cls(
            status=ActionStatus.NEEDS_CONFIRM,
            message=message,
            confirmation_required={
                "action_id": action_id,
                "details": details,
            },
        )


@dataclass
class ActionContext:
    """Context passed to actions during execution."""

    browser: Browser
    user_id: str
    location: dict[str, Any] | None = None  # lat, lng, address
    preferences: dict[str, Any] = field(default_factory=dict)
    memory_context: str | None = None  # Relevant memory for the action
    confirm_callback: Any = None  # Callback for confirmations
    notify_callback: Any = None  # Callback for notifications


class Action(ABC):
    """Base class for all actions."""

    # Action metadata
    name: str = "base_action"
    description: str = "Base action"
    category: str = "general"
    requires_auth: bool = False
    requires_confirmation: bool = False  # If True, always confirm before executing

    def __init__(self) -> None:
        self.created_at = datetime.now()
        self._cancelled = False

    @abstractmethod
    async def execute(self, context: ActionContext, **params) -> ActionResult:
        """Execute the action.

        Args:
            context: Action context with browser, user info, etc.
            **params: Action-specific parameters

        Returns:
            ActionResult with status and data
        """
        pass

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        """Validate action parameters before execution.

        Returns:
            (valid, error_message) tuple
        """
        return True, None

    async def pre_execute(self, context: ActionContext, **params) -> ActionResult | None:
        """Hook called before execute. Return ActionResult to skip execute."""
        return None

    async def post_execute(self, context: ActionContext, result: ActionResult, **params) -> ActionResult:
        """Hook called after execute. Can modify the result."""
        return result

    def cancel(self) -> None:
        """Cancel the action."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if action is cancelled."""
        return self._cancelled

    async def run(self, context: ActionContext, **params) -> ActionResult:
        """Run the action with validation and hooks.

        This is the main entry point for executing an action.
        """
        # Validate parameters
        valid, error = await self.validate_params(**params)
        if not valid:
            return ActionResult.error_result(f"Invalid parameters: {error}")

        # Pre-execute hook
        pre_result = await self.pre_execute(context, **params)
        if pre_result:
            return pre_result

        # Check for confirmation requirement
        if self.requires_confirmation and context.confirm_callback:
            confirmed = await self._request_confirmation(context, **params)
            if not confirmed:
                return ActionResult(
                    status=ActionStatus.CANCELLED,
                    message="Action cancelled by user",
                )

        # Execute
        try:
            result = await self.execute(context, **params)
        except Exception as e:
            logger.error(f"Action {self.name} failed: {e}")
            return ActionResult.error_result(f"Action failed: {str(e)}", str(e))

        # Post-execute hook
        result = await self.post_execute(context, result, **params)

        return result

    async def _request_confirmation(self, context: ActionContext, **params) -> bool:
        """Request user confirmation."""
        if not context.confirm_callback:
            return True

        try:
            return await context.confirm_callback(
                action=self.name,
                description=self.description,
                params=params,
            )
        except Exception as e:
            logger.error(f"Confirmation request failed: {e}")
            return False

    def get_info(self) -> dict[str, Any]:
        """Get action metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_auth": self.requires_auth,
            "requires_confirmation": self.requires_confirmation,
        }
