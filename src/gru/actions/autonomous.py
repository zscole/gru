"""Autonomous Actions - Execute real-world tasks on behalf of the user.

This module provides the framework for Gru to take autonomous actions like:
- Booking reservations
- Sending messages
- Making payments
- Ordering food/services
- Managing calendar
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from gru.db import Database

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an autonomous action."""
    PENDING = "pending"          # Waiting for user confirmation
    CONFIRMED = "confirmed"      # User confirmed, ready to execute
    EXECUTING = "executing"      # Currently running
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Execution failed
    CANCELLED = "cancelled"      # User cancelled
    EXPIRED = "expired"          # Confirmation timed out


class ActionCategory(Enum):
    """Categories of autonomous actions."""
    COMMUNICATION = "communication"  # Email, Slack, SMS
    CALENDAR = "calendar"            # Events, reminders
    RESERVATION = "reservation"      # Restaurants, hotels, etc.
    PAYMENT = "payment"              # Venmo, transfers
    PURCHASE = "purchase"            # Orders, subscriptions
    FILE = "file"                    # Send, organize documents
    TASK = "task"                    # Create tasks, reminders


@dataclass
class ActionPreview:
    """Preview of what an action will do before confirmation."""
    summary: str                     # One-line summary
    details: list[str]               # Bullet points of what will happen
    reversible: bool                 # Can this be undone?
    cost: float | None = None        # Cost if applicable
    warnings: list[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    undo_available: bool = False
    undo_data: dict[str, Any] | None = None


@dataclass
class AutonomousAction:
    """A pending or completed autonomous action."""
    id: str
    action_type: str
    category: ActionCategory
    description: str
    parameters: dict[str, Any]
    preview: ActionPreview | None = None
    status: ActionStatus = ActionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confirmed_at: str | None = None
    completed_at: str | None = None
    result: ActionResult | None = None
    agent_id: str | None = None
    expires_at: str | None = None


class ActionHandler(ABC):
    """Base class for action handlers."""

    @property
    @abstractmethod
    def action_type(self) -> str:
        """Unique identifier for this action type."""
        pass

    @property
    @abstractmethod
    def category(self) -> ActionCategory:
        """Category of this action."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @property
    def requires_confirmation(self) -> bool:
        """Whether this action requires user confirmation."""
        return True

    @abstractmethod
    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters. Returns (valid, error_message)."""
        pass

    @abstractmethod
    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        """Generate a preview of what this action will do."""
        pass

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute the action."""
        pass

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        """Undo the action if supported."""
        return ActionResult(success=False, message="Undo not supported for this action")


class ActionRegistry:
    """Registry of available action handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, handler: ActionHandler) -> None:
        """Register an action handler."""
        self._handlers[handler.action_type] = handler
        logger.debug(f"Registered action handler: {handler.action_type}")

    def get(self, action_type: str) -> ActionHandler | None:
        """Get handler for an action type."""
        return self._handlers.get(action_type)

    def list_actions(self) -> list[dict[str, Any]]:
        """List all available actions."""
        return [
            {
                "type": h.action_type,
                "category": h.category.value,
                "description": h.description,
                "requires_confirmation": h.requires_confirmation,
            }
            for h in self._handlers.values()
        ]


class AutonomousActionEngine:
    """Engine for managing autonomous actions."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.registry = ActionRegistry()
        self._pending_actions: dict[str, AutonomousAction] = {}
        self._confirmation_callback: Callable[[AutonomousAction], asyncio.Future] | None = None
        self._notification_callback: Callable[[str, str], None] | None = None

    async def initialize(self) -> None:
        """Initialize the action engine and database tables."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_actions (
                id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                parameters JSON NOT NULL,
                preview JSON,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                confirmed_at TEXT,
                completed_at TEXT,
                result JSON,
                agent_id TEXT,
                expires_at TEXT
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_status ON autonomous_actions(status)"
        )
        await self.db.commit()

        # Register built-in handlers
        self._register_builtin_handlers()

        logger.info("Autonomous action engine initialized")

    def _register_builtin_handlers(self) -> None:
        """Register built-in action handlers."""
        from gru.actions.handlers.communication import (
            SendEmailHandler,
            SendSlackMessageHandler,
            SendSMSHandler,
        )
        from gru.actions.handlers.calendar import (
            CreateEventHandler,
            UpdateEventHandler,
            DeleteEventHandler,
        )
        from gru.actions.handlers.reservations import (
            OpenTableReservationHandler,
            ResyReservationHandler,
        )
        from gru.actions.handlers.payments import (
            VenmoPaymentHandler,
        )
        from gru.actions.handlers.purchases import (
            DoorDashOrderHandler,
            AmazonOrderHandler,
        )

        handlers = [
            SendEmailHandler(),
            SendSlackMessageHandler(),
            SendSMSHandler(),
            CreateEventHandler(),
            UpdateEventHandler(),
            DeleteEventHandler(),
            OpenTableReservationHandler(),
            ResyReservationHandler(),
            VenmoPaymentHandler(),
            DoorDashOrderHandler(),
            AmazonOrderHandler(),
        ]

        for handler in handlers:
            self.registry.register(handler)

    def set_confirmation_callback(
        self, callback: Callable[[AutonomousAction], asyncio.Future]
    ) -> None:
        """Set callback for requesting user confirmation."""
        self._confirmation_callback = callback

    def set_notification_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for notifications."""
        self._notification_callback = callback

    async def request_action(
        self,
        action_type: str,
        params: dict[str, Any],
        agent_id: str | None = None,
        auto_confirm: bool = False,
    ) -> dict[str, Any]:
        """Request an autonomous action.

        This validates the action, generates a preview, and either:
        - Requests user confirmation (default)
        - Auto-executes if auto_confirm=True and action allows it

        Returns the action status and preview.
        """
        handler = self.registry.get(action_type)
        if not handler:
            return {"error": f"Unknown action type: {action_type}"}

        # Validate parameters
        valid, error = await handler.validate(params)
        if not valid:
            return {"error": f"Invalid parameters: {error}"}

        # Generate preview
        preview = await handler.preview(params)

        # Create action record
        action = AutonomousAction(
            id=str(uuid.uuid4())[:12],
            action_type=action_type,
            category=handler.category,
            description=handler.description,
            parameters=params,
            preview=preview,
            agent_id=agent_id,
        )

        # Store in database
        await self._save_action(action)
        self._pending_actions[action.id] = action

        # Check if confirmation required
        if handler.requires_confirmation and not auto_confirm:
            # Request confirmation
            if self._confirmation_callback:
                try:
                    # Send confirmation request
                    await self._request_confirmation(action)
                except Exception as e:
                    logger.error(f"Confirmation request failed: {e}")

            return {
                "status": "pending_confirmation",
                "action_id": action.id,
                "preview": {
                    "summary": preview.summary,
                    "details": preview.details,
                    "reversible": preview.reversible,
                    "cost": preview.cost,
                    "warnings": preview.warnings,
                },
                "message": f"Awaiting confirmation to: {preview.summary}",
            }
        else:
            # Execute immediately
            return await self.execute_action(action.id)

    async def _request_confirmation(self, action: AutonomousAction) -> None:
        """Send confirmation request to user."""
        if not self._notification_callback or not action.preview:
            return

        # Format confirmation message
        lines = [
            f"ACTION REQUEST: {action.preview.summary}",
            "",
            "What will happen:",
        ]
        for detail in action.preview.details:
            lines.append(f"  - {detail}")

        if action.preview.cost:
            lines.append(f"\nCost: ${action.preview.cost:.2f}")

        if action.preview.warnings:
            lines.append("\nWarnings:")
            for warning in action.preview.warnings:
                lines.append(f"  - {warning}")

        lines.append(f"\nReply 'confirm {action.id}' to proceed or 'cancel {action.id}' to cancel.")

        self._notification_callback("action", "\n".join(lines))

    async def confirm_action(self, action_id: str) -> dict[str, Any]:
        """Confirm and execute a pending action."""
        action = self._pending_actions.get(action_id)
        if not action:
            # Try loading from database
            action = await self._load_action(action_id)

        if not action:
            return {"error": f"Action not found: {action_id}"}

        if action.status != ActionStatus.PENDING:
            return {"error": f"Action is not pending (status: {action.status.value})"}

        action.status = ActionStatus.CONFIRMED
        action.confirmed_at = datetime.now().isoformat()
        await self._save_action(action)

        return await self.execute_action(action_id)

    async def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel a pending action."""
        action = self._pending_actions.get(action_id)
        if not action:
            action = await self._load_action(action_id)

        if not action:
            return {"error": f"Action not found: {action_id}"}

        if action.status not in (ActionStatus.PENDING, ActionStatus.CONFIRMED):
            return {"error": f"Cannot cancel action (status: {action.status.value})"}

        action.status = ActionStatus.CANCELLED
        await self._save_action(action)

        if action_id in self._pending_actions:
            del self._pending_actions[action_id]

        return {"status": "cancelled", "action_id": action_id}

    async def execute_action(self, action_id: str) -> dict[str, Any]:
        """Execute a confirmed action."""
        action = self._pending_actions.get(action_id)
        if not action:
            action = await self._load_action(action_id)

        if not action:
            return {"error": f"Action not found: {action_id}"}

        handler = self.registry.get(action.action_type)
        if not handler:
            return {"error": f"Handler not found for: {action.action_type}"}

        # Update status
        action.status = ActionStatus.EXECUTING
        await self._save_action(action)

        try:
            # Execute
            result = await handler.execute(action.parameters)

            # Update with result
            action.result = result
            action.status = ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
            action.completed_at = datetime.now().isoformat()
            await self._save_action(action)

            # Clean up pending
            if action_id in self._pending_actions:
                del self._pending_actions[action_id]

            # Notify
            if self._notification_callback:
                status_emoji = "Done" if result.success else "Failed"
                self._notification_callback(
                    "action",
                    f"{status_emoji}: {result.message}"
                )

            return {
                "status": action.status.value,
                "action_id": action_id,
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "undo_available": result.undo_available,
            }

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action.status = ActionStatus.FAILED
            action.result = ActionResult(success=False, message=str(e))
            await self._save_action(action)

            return {
                "status": "failed",
                "action_id": action_id,
                "error": str(e),
            }

    async def undo_action(self, action_id: str) -> dict[str, Any]:
        """Undo a completed action if supported."""
        action = await self._load_action(action_id)
        if not action:
            return {"error": f"Action not found: {action_id}"}

        if action.status != ActionStatus.COMPLETED:
            return {"error": "Can only undo completed actions"}

        if not action.result or not action.result.undo_available:
            return {"error": "Undo not available for this action"}

        handler = self.registry.get(action.action_type)
        if not handler:
            return {"error": f"Handler not found for: {action.action_type}"}

        try:
            result = await handler.undo(action.parameters, action.result.undo_data or {})
            return {
                "status": "undone" if result.success else "undo_failed",
                "message": result.message,
            }
        except Exception as e:
            return {"error": f"Undo failed: {e}"}

    async def get_pending_actions(self) -> list[dict[str, Any]]:
        """Get all pending actions."""
        rows = await self.db.fetchall(
            "SELECT * FROM autonomous_actions WHERE status = 'pending' ORDER BY created_at DESC"
        )
        return [self._row_to_dict(row) for row in rows]

    async def get_action_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent action history."""
        rows = await self.db.fetchall(
            "SELECT * FROM autonomous_actions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_dict(row) for row in rows]

    async def _save_action(self, action: AutonomousAction) -> None:
        """Save action to database."""
        preview_json = None
        if action.preview:
            preview_json = json.dumps({
                "summary": action.preview.summary,
                "details": action.preview.details,
                "reversible": action.preview.reversible,
                "cost": action.preview.cost,
                "warnings": action.preview.warnings,
            })

        result_json = None
        if action.result:
            result_json = json.dumps({
                "success": action.result.success,
                "message": action.result.message,
                "data": action.result.data,
                "undo_available": action.result.undo_available,
                "undo_data": action.result.undo_data,
            })

        await self.db.execute(
            """
            INSERT INTO autonomous_actions
            (id, action_type, category, description, parameters, preview, status,
             created_at, confirmed_at, completed_at, result, agent_id, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                confirmed_at = excluded.confirmed_at,
                completed_at = excluded.completed_at,
                result = excluded.result
            """,
            (
                action.id,
                action.action_type,
                action.category.value,
                action.description,
                json.dumps(action.parameters),
                preview_json,
                action.status.value,
                action.created_at,
                action.confirmed_at,
                action.completed_at,
                result_json,
                action.agent_id,
                action.expires_at,
            )
        )
        await self.db.commit()

    async def _load_action(self, action_id: str) -> AutonomousAction | None:
        """Load action from database."""
        row = await self.db.fetchone(
            "SELECT * FROM autonomous_actions WHERE id = ?",
            (action_id,)
        )
        if not row:
            return None

        preview = None
        if row.get("preview"):
            p = json.loads(row["preview"])
            preview = ActionPreview(
                summary=p["summary"],
                details=p["details"],
                reversible=p["reversible"],
                cost=p.get("cost"),
                warnings=p.get("warnings", []),
            )

        result = None
        if row.get("result"):
            r = json.loads(row["result"])
            result = ActionResult(
                success=r["success"],
                message=r["message"],
                data=r.get("data", {}),
                undo_available=r.get("undo_available", False),
                undo_data=r.get("undo_data"),
            )

        return AutonomousAction(
            id=row["id"],
            action_type=row["action_type"],
            category=ActionCategory(row["category"]),
            description=row["description"],
            parameters=json.loads(row["parameters"]),
            preview=preview,
            status=ActionStatus(row["status"]),
            created_at=row["created_at"],
            confirmed_at=row.get("confirmed_at"),
            completed_at=row.get("completed_at"),
            result=result,
            agent_id=row.get("agent_id"),
            expires_at=row.get("expires_at"),
        )

    def _row_to_dict(self, row: dict) -> dict[str, Any]:
        """Convert database row to dict."""
        result = dict(row)
        if result.get("parameters"):
            result["parameters"] = json.loads(result["parameters"])
        if result.get("preview"):
            result["preview"] = json.loads(result["preview"])
        if result.get("result"):
            result["result"] = json.loads(result["result"])
        return result
