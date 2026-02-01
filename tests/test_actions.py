"""Tests for the action layer."""

from __future__ import annotations

from unittest.mock import MagicMock

from gru.actions.base import (
    Action,
    ActionContext,
    ActionResult,
    ActionStatus,
)
from gru.actions.browser import Browser, BrowserConfig

# Import registry module but don't call get_registry() which loads services
from gru.actions.registry import ActionRegistry


class TestActionResult:
    """Tests for ActionResult."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ActionResult.success_result("Done", {"key": "value"})
        assert result.success is True
        assert result.status == ActionStatus.COMPLETED
        assert result.message == "Done"
        assert result.data == {"key": "value"}

    def test_error_result(self):
        """Test creating an error result."""
        result = ActionResult.error_result("Failed", "Something broke")
        assert result.success is False
        assert result.status == ActionStatus.FAILED
        assert result.error == "Something broke"

    def test_auth_required(self):
        """Test auth required result."""
        result = ActionResult.auth_required("ubereats", "https://ubereats.com/login")
        assert result.status == ActionStatus.NEEDS_AUTH
        assert result.needs_user_input is True
        assert result.data["service"] == "ubereats"

    def test_confirm_required(self):
        """Test confirmation required result."""
        result = ActionResult.confirm_required("Confirm order", {"total": "$25.00"}, "order_123")
        assert result.status == ActionStatus.NEEDS_CONFIRM
        assert result.needs_user_input is True
        assert result.confirmation_required["details"]["total"] == "$25.00"

    def test_to_dict(self):
        """Test serialization."""
        result = ActionResult.success_result("Done", {"key": "value"})
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["message"] == "Done"
        assert d["data"] == {"key": "value"}


class TestBrowserConfig:
    """Tests for BrowserConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = BrowserConfig()
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.timeout == 30000
        assert config.viewport_width == 1280

    def test_from_env(self, monkeypatch):
        """Test loading from environment."""
        monkeypatch.setenv("GRU_BROWSER_MODE", "headed")
        monkeypatch.setenv("GRU_BROWSER_TYPE", "firefox")
        monkeypatch.setenv("GRU_BROWSER_TIMEOUT", "60000")

        config = BrowserConfig.from_env()
        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.timeout == 60000

    def test_from_env_with_data_dir(self, tmp_path):
        """Test that data_dir sets storage paths."""
        config = BrowserConfig.from_env(tmp_path)
        assert config.storage_dir == tmp_path / "browser_sessions"
        assert config.screenshots_dir == tmp_path / "screenshots"


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_register_action(self):
        """Test registering an action."""
        registry = ActionRegistry()

        class TestAction(Action):
            name = "test_action"
            description = "A test action"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Test done")

        registry.register(TestAction)
        assert registry.get("test_action") is TestAction

    def test_register_with_aliases(self):
        """Test registering with aliases."""
        registry = ActionRegistry()

        class TestAction(Action):
            name = "test_action"
            description = "Test"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        registry.register(TestAction, aliases=["test", "t"])
        assert registry.get("test_action") is TestAction
        assert registry.get("test") is TestAction
        assert registry.get("t") is TestAction

    def test_create_action(self):
        """Test creating an action instance."""
        registry = ActionRegistry()

        class TestAction(Action):
            name = "test_action"
            description = "Test"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        registry.register(TestAction)
        action = registry.create("test_action")
        assert action is not None
        assert isinstance(action, TestAction)

    def test_list_actions(self):
        """Test listing actions."""
        registry = ActionRegistry()

        class TestAction1(Action):
            name = "action1"
            description = "First action"
            category = "cat1"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        class TestAction2(Action):
            name = "action2"
            description = "Second action"
            category = "cat2"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        registry.register(TestAction1)
        registry.register(TestAction2)

        all_actions = registry.list_actions()
        assert len(all_actions) == 2

        cat1_actions = registry.list_actions(category="cat1")
        assert len(cat1_actions) == 1
        assert cat1_actions[0]["name"] == "action1"

    def test_list_categories(self):
        """Test listing categories."""
        registry = ActionRegistry()

        class TestAction1(Action):
            name = "action1"
            description = "First"
            category = "cat_a"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        class TestAction2(Action):
            name = "action2"
            description = "Second"
            category = "cat_b"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        registry.register(TestAction1)
        registry.register(TestAction2)

        categories = registry.list_categories()
        assert "cat_a" in categories
        assert "cat_b" in categories


class TestAction:
    """Tests for the Action base class."""

    async def test_action_run_success(self):
        """Test running an action successfully."""

        class SimpleAction(Action):
            name = "simple"
            description = "Simple action"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done", {"param": params.get("test")})

        action = SimpleAction()
        context = MagicMock(spec=ActionContext)

        result = await action.run(context, test="value")
        assert result.success is True
        assert result.data["param"] == "value"

    async def test_action_validation(self):
        """Test parameter validation."""

        class ValidatedAction(Action):
            name = "validated"
            description = "Validated action"
            category = "test"

            async def validate_params(self, **params):
                if "required" not in params:
                    return False, "required param is missing"
                return True, None

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        action = ValidatedAction()
        context = MagicMock(spec=ActionContext)

        # Without required param
        result = await action.run(context)
        assert result.success is False
        assert "required" in result.message

        # With required param
        result = await action.run(context, required="value")
        assert result.success is True

    async def test_action_pre_execute_hook(self):
        """Test pre-execute hook."""

        class HookedAction(Action):
            name = "hooked"
            description = "Hooked action"
            category = "test"

            async def pre_execute(self, context, **params):
                if params.get("skip"):
                    return ActionResult.success_result("Skipped in pre-execute")
                return None

            async def execute(self, context, **params):
                return ActionResult.success_result("Executed")

        action = HookedAction()
        context = MagicMock(spec=ActionContext)

        # With skip=True, should return from pre_execute
        result = await action.run(context, skip=True)
        assert result.message == "Skipped in pre-execute"

        # Without skip, should execute normally
        result = await action.run(context)
        assert result.message == "Executed"

    async def test_action_post_execute_hook(self):
        """Test post-execute hook."""

        class HookedAction(Action):
            name = "hooked"
            description = "Hooked action"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Original")

            async def post_execute(self, context, result, **params):
                result.message = "Modified by post_execute"
                return result

        action = HookedAction()
        context = MagicMock(spec=ActionContext)

        result = await action.run(context)
        assert result.message == "Modified by post_execute"

    async def test_action_exception_handling(self):
        """Test that exceptions are caught."""

        class FailingAction(Action):
            name = "failing"
            description = "Failing action"
            category = "test"

            async def execute(self, context, **params):
                raise ValueError("Something went wrong")

        action = FailingAction()
        context = MagicMock(spec=ActionContext)

        result = await action.run(context)
        assert result.success is False
        assert "Something went wrong" in result.error

    async def test_action_cancellation(self):
        """Test action cancellation."""

        class CancellableAction(Action):
            name = "cancellable"
            description = "Cancellable action"
            category = "test"

            async def execute(self, context, **params):
                return ActionResult.success_result("Done")

        action = CancellableAction()
        assert action.is_cancelled is False

        action.cancel()
        assert action.is_cancelled is True


class TestActionContext:
    """Tests for ActionContext."""

    def test_context_creation(self):
        """Test creating an action context."""
        browser = MagicMock(spec=Browser)
        context = ActionContext(
            browser=browser,
            user_id="user123",
            location={"lat": 37.7749, "lng": -122.4194, "address": "San Francisco"},
            preferences={"food": "burger"},
        )

        assert context.browser is browser
        assert context.user_id == "user123"
        assert context.location["address"] == "San Francisco"
        assert context.preferences["food"] == "burger"
