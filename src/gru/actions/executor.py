"""Action executor - high-level interface for running actions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from gru.actions.base import ActionContext, ActionResult
from gru.actions.browser import Browser, BrowserConfig, get_browser, shutdown_browser
from gru.actions.registry import get_registry

if TYPE_CHECKING:
    from gru.config import Config
    from gru.memory import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ScheduledAction:
    """An action scheduled for future execution."""

    id: str
    action_name: str
    params: dict[str, Any]
    execute_at: datetime
    user_id: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def is_due(self) -> bool:
        return datetime.now() >= self.execute_at


class ActionExecutor:
    """High-level executor for actions with scheduling and context management."""

    def __init__(
        self,
        config: Config,
        memory: MemoryStore | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self._browser: Browser | None = None
        self._scheduled: dict[str, ScheduledAction] = {}
        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._notify_callback: Callable[[str, str], None] | None = None
        self._confirm_callback: Callable[..., Any] | None = None

    def set_notify_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for notifications."""
        self._notify_callback = callback

    def set_confirm_callback(self, callback: Callable[..., Any]) -> None:
        """Set callback for action confirmations."""
        self._confirm_callback = callback

    async def start(self) -> None:
        """Start the action executor."""
        if self._running:
            return

        # Initialize browser with config
        browser_config = BrowserConfig(
            headless=self.config.browser_mode == "headless",
            browser_type=self.config.browser_type,
            timeout=self.config.browser_timeout,
            storage_dir=self.config.data_dir / "browser_sessions",
            screenshots_dir=self.config.data_dir / "screenshots",
        )

        self._browser = await get_browser(browser_config)
        await self._browser.start()

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("Action executor started")

    async def stop(self) -> None:
        """Stop the action executor."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

        await shutdown_browser()
        self._browser = None

        logger.info("Action executor stopped")

    async def _scheduler_loop(self) -> None:
        """Background loop to execute scheduled actions."""
        while self._running:
            try:
                datetime.now()

                # Find due actions
                due = [action for action in self._scheduled.values() if action.is_due]

                for action in due:
                    logger.info(f"Executing scheduled action: {action.action_name}")
                    result = await self.execute(action.action_name, user_id=action.user_id, **action.params)

                    # Notify on completion
                    if self._notify_callback:
                        status = "completed" if result.success else "failed"
                        self._notify_callback(
                            action.user_id, f"Scheduled action {action.action_name} {status}: {result.message}"
                        )

                    # Remove from scheduled
                    del self._scheduled[action.id]

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    async def execute(
        self, action_name: str, user_id: str = "default", location: dict[str, Any] | None = None, **params
    ) -> ActionResult:
        """Execute an action.

        Args:
            action_name: Name of the action to execute
            user_id: User ID for context and preferences
            location: Optional location override
            **params: Action-specific parameters

        Returns:
            ActionResult
        """
        if not self._browser:
            await self.start()

        # Build context
        preferences = {}
        memory_context = None

        if self.memory:
            try:
                profile = await self.memory.get_user_profile()
                preferences = profile.get("preferences", {})

                # Get relevant memory for this action
                memory_context = await self.memory.get_relevant_context(
                    f"{action_name} {' '.join(str(v) for v in params.values())}"
                )
            except Exception as e:
                logger.warning(f"Failed to load memory context: {e}")

        context = ActionContext(
            browser=self._browser,
            user_id=user_id,
            location=location or preferences.get("location"),
            preferences=preferences,
            memory_context=memory_context,
            confirm_callback=self._confirm_callback,
            notify_callback=self._notify_callback,
        )

        # Get and execute action
        registry = get_registry()
        return await registry.execute(action_name, context, **params)

    def schedule(self, action_name: str, execute_at: datetime | timedelta, user_id: str = "default", **params) -> str:
        """Schedule an action for future execution.

        Args:
            action_name: Name of the action
            execute_at: When to execute (datetime or timedelta from now)
            user_id: User ID
            **params: Action parameters

        Returns:
            Schedule ID
        """
        import uuid

        if isinstance(execute_at, timedelta):
            execute_at = datetime.now() + execute_at

        schedule_id = str(uuid.uuid4())[:8]
        self._scheduled[schedule_id] = ScheduledAction(
            id=schedule_id,
            action_name=action_name,
            params=params,
            execute_at=execute_at,
            user_id=user_id,
        )

        logger.info(f"Scheduled action {action_name} for {execute_at}")
        return schedule_id

    def cancel_scheduled(self, schedule_id: str) -> bool:
        """Cancel a scheduled action."""
        if schedule_id in self._scheduled:
            del self._scheduled[schedule_id]
            return True
        return False

    def list_scheduled(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List scheduled actions."""
        actions = []
        for action in self._scheduled.values():
            if user_id and action.user_id != user_id:
                continue
            actions.append(
                {
                    "id": action.id,
                    "action": action.action_name,
                    "execute_at": action.execute_at.isoformat(),
                    "params": action.params,
                    "user_id": action.user_id,
                }
            )
        return sorted(actions, key=lambda x: x["execute_at"])

    def list_actions(self, category: str | None = None) -> list[dict[str, Any]]:
        """List available actions."""
        return get_registry().list_actions(category)

    async def get_browser_status(self) -> dict[str, Any]:
        """Get browser status."""
        if not self._browser:
            return {"running": False}

        contexts = await self._browser.list_contexts()
        return {
            "running": self._browser.is_running(),
            "headless": self._browser.config.headless,
            "browser_type": self._browser.config.browser_type,
            "contexts": contexts,
        }
