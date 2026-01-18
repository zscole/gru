"""Tests for Telegram bot interface."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gru.config import Config
from gru.telegram_bot import TelegramBot


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        telegram_token="test_token",
        telegram_admin_ids=[123, 456],
        anthropic_api_key="test_api_key",
        default_model="test-model",
        max_tokens=1000,
    )


@pytest.fixture
def orchestrator():
    """Create mock orchestrator."""
    mock = MagicMock()
    mock.spawn_agent = AsyncMock(return_value={"id": "agent1", "workdir": "/test/workdir"})
    mock.get_agent = AsyncMock(return_value=None)
    mock.list_agents = AsyncMock(return_value=[])
    mock.get_status = AsyncMock(
        return_value={
            "running": True,
            "agents": {"total": 5, "running": 2, "paused": 1},
            "scheduler": {"queued": 3},
        }
    )
    mock.get_pending_approvals = AsyncMock(return_value=[])
    mock.pause_agent = AsyncMock(return_value=True)
    mock.resume_agent = AsyncMock(return_value=True)
    mock.terminate_agent = AsyncMock(return_value=True)
    mock.nudge_agent = AsyncMock(return_value=True)
    mock.approve = AsyncMock(return_value=True)
    mock.db = MagicMock()
    mock.db.get_conversation = AsyncMock(return_value=[])
    mock.db.save_template = AsyncMock()
    mock.db.list_templates = AsyncMock(return_value=[])
    mock.db.get_template = AsyncMock(return_value=None)
    mock.db.delete_template = AsyncMock(return_value=True)
    mock.secrets = MagicMock()
    mock.secrets.set = AsyncMock()
    mock.secrets.get = AsyncMock(return_value=None)
    mock.secrets.list_keys = AsyncMock(return_value=[])
    mock.secrets.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def bot(config, orchestrator):
    """Create TelegramBot instance."""
    return TelegramBot(config, orchestrator)


@pytest.fixture
def mock_update():
    """Create mock Telegram Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 123  # Admin user
    update.effective_chat = MagicMock()
    update.effective_chat.id = 123
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.text = ""
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    context = MagicMock()
    context.args = []
    return context


class TestTelegramBotAuth:
    """Tests for authentication."""

    def test_is_admin_true(self, bot):
        """Test admin check returns true for admin."""
        assert bot._is_admin(123) is True
        assert bot._is_admin(456) is True

    def test_is_admin_false(self, bot):
        """Test admin check returns false for non-admin."""
        assert bot._is_admin(999) is False
        assert bot._is_admin(0) is False

    @pytest.mark.asyncio
    async def test_check_admin_authorized(self, bot, mock_update):
        """Test check_admin passes for admin."""
        result = await bot._check_admin(mock_update)
        assert result is True
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_admin_unauthorized(self, bot, mock_update):
        """Test check_admin fails for non-admin."""
        mock_update.effective_user.id = 999
        result = await bot._check_admin(mock_update)
        assert result is False
        mock_update.message.reply_text.assert_called_once_with("Unauthorized")

    @pytest.mark.asyncio
    async def test_check_admin_no_user(self, bot, mock_update):
        """Test check_admin fails when no user."""
        mock_update.effective_user = None
        result = await bot._check_admin(mock_update)
        assert result is False


class TestMessageSplitting:
    """Tests for message splitting logic."""

    def test_split_short_message(self, bot):
        """Test splitting a short message."""
        chunks = bot._split_message("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_split_long_message_at_newlines(self, bot):
        """Test splitting long message at line boundaries."""
        # Create enough lines to exceed MAX_MESSAGE_LENGTH (4096)
        lines = ["Line " + str(i) + " with some extra padding text" for i in range(500)]
        text = "\n".join(lines)
        assert len(text) > bot.MAX_MESSAGE_LENGTH  # Verify setup
        chunks = bot._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= bot.MAX_MESSAGE_LENGTH

    def test_split_long_single_line(self, bot):
        """Test splitting a very long single line."""
        text = "x" * 10000
        chunks = bot._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= bot.MAX_MESSAGE_LENGTH

    def test_split_empty_message(self, bot):
        """Test splitting empty message."""
        chunks = bot._split_message("")
        assert chunks == []


class TestSendOutput:
    """Tests for send_output method."""

    @pytest.mark.asyncio
    async def test_send_output_no_app(self, bot):
        """Test send_output with no app initialized."""
        await bot.send_output(123, "Hello")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_output_short_message(self, bot):
        """Test sending a short message."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        await bot.send_output(123, "Hello")

        bot._app.bot.send_message.assert_called_once_with(123, "Hello")

    @pytest.mark.asyncio
    async def test_send_output_force_file(self, bot):
        """Test force sending as file."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_document = AsyncMock()

        await bot.send_output(123, "Hello", filename="test.txt", force_file=True)

        bot._app.bot.send_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_output_long_message_as_file(self, bot):
        """Test long message sent as file."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_document = AsyncMock()

        # Create message longer than threshold
        text = "x" * (bot.MAX_MESSAGE_LENGTH * bot.SPLIT_THRESHOLD + 1)
        await bot.send_output(123, text, filename="output.txt")

        bot._app.bot.send_document.assert_called_once()


class TestCommandHandlers:
    """Tests for command handlers."""

    @pytest.mark.asyncio
    async def test_cmd_start(self, bot, mock_update, mock_context):
        """Test /start command shows welcome message."""
        await bot.cmd_start(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Welcome to Gru" in args
        assert "/gru examples" in args

    @pytest.mark.asyncio
    async def test_cmd_start_unauthorized(self, bot, mock_update, mock_context):
        """Test /start command for unauthorized user."""
        mock_update.effective_user.id = 999
        await bot.cmd_start(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once_with("Unauthorized")

    @pytest.mark.asyncio
    async def test_cmd_gru_no_args(self, bot, mock_update, mock_context):
        """Test /gru with no arguments."""
        await bot.cmd_gru(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_cmd_gru_unknown_command(self, bot, mock_update, mock_context):
        """Test /gru with unknown command."""
        mock_context.args = ["unknown"]
        await bot.cmd_gru(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Unknown command" in args

    @pytest.mark.asyncio
    async def test_cmd_help(self, bot, mock_update, mock_context):
        """Test /gru help command."""
        mock_context.args = ["help"]
        await bot.cmd_gru(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Agent Management" in args
        assert "spawn" in args


class TestSpawnCommand:
    """Tests for spawn command."""

    @pytest.mark.asyncio
    async def test_spawn_no_task(self, bot, mock_update):
        """Test spawn without task."""
        await bot._cmd_spawn(mock_update, [])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_spawn_basic(self, bot, mock_update, orchestrator):
        """Test basic spawn."""
        await bot._cmd_spawn(mock_update, ["run", "tests"])
        orchestrator.spawn_agent.assert_called_once()
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["task"] == "run tests"
        assert call_kwargs["supervised"] is True

    @pytest.mark.asyncio
    async def test_spawn_unsupervised(self, bot, mock_update, orchestrator):
        """Test spawn with --unsupervised."""
        await bot._cmd_spawn(mock_update, ["--unsupervised", "run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["supervised"] is False

    @pytest.mark.asyncio
    async def test_spawn_oneshot(self, bot, mock_update, orchestrator):
        """Test spawn with --oneshot."""
        await bot._cmd_spawn(mock_update, ["--oneshot", "run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["supervised"] is False
        assert call_kwargs["timeout_mode"] == "auto"

    @pytest.mark.asyncio
    async def test_spawn_with_priority(self, bot, mock_update, orchestrator):
        """Test spawn with --priority."""
        await bot._cmd_spawn(mock_update, ["--priority", "high", "run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["priority"] == "high"

    @pytest.mark.asyncio
    async def test_spawn_with_workdir(self, bot, mock_update, orchestrator):
        """Test spawn with --workdir."""
        await bot._cmd_spawn(mock_update, ["--workdir", "/test/path", "run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["workdir"] == "/test/path"

    @pytest.mark.asyncio
    async def test_spawn_error_handling(self, bot, mock_update, orchestrator):
        """Test spawn error handling."""
        orchestrator.spawn_agent.side_effect = ValueError("Task too long")
        await bot._cmd_spawn(mock_update, ["test", "task"])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Error:" in args


class TestStatusCommand:
    """Tests for status command."""

    @pytest.mark.asyncio
    async def test_status_overall(self, bot, mock_update, orchestrator):
        """Test overall status."""
        await bot._cmd_status(mock_update, [])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Orchestrator Status" in args

    @pytest.mark.asyncio
    async def test_status_agent_not_found(self, bot, mock_update, orchestrator):
        """Test agent status when not found."""
        await bot._cmd_status(mock_update, ["nonexistent"])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "not found" in args

    @pytest.mark.asyncio
    async def test_status_agent_found(self, bot, mock_update, orchestrator):
        """Test agent status when found."""
        orchestrator.get_agent.return_value = {
            "id": "agent1",
            "status": "running",
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "workdir": "/test",
            "created_at": "2024-01-01",
        }
        await bot._cmd_status(mock_update, ["agent1"])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "agent1" in args
        assert "running" in args


class TestListCommand:
    """Tests for list command."""

    @pytest.mark.asyncio
    async def test_list_empty(self, bot, mock_update, orchestrator):
        """Test list with no agents."""
        await bot._cmd_list(mock_update, [])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "No agents found" in args

    @pytest.mark.asyncio
    async def test_list_with_agents(self, bot, mock_update, orchestrator):
        """Test list with agents."""
        orchestrator.list_agents.return_value = [
            {"id": "agent1", "status": "running", "task": "Task 1"},
            {"id": "agent2", "status": "paused", "task": "Task 2"},
        ]
        await bot._cmd_list(mock_update, [])
        mock_update.message.reply_text.assert_called_once()
        args = mock_update.message.reply_text.call_args[0][0]
        assert "agent1" in args
        assert "agent2" in args

    @pytest.mark.asyncio
    async def test_list_with_filter(self, bot, mock_update, orchestrator):
        """Test list with status filter."""
        await bot._cmd_list(mock_update, ["running"])
        orchestrator.list_agents.assert_called_once_with("running")


class TestAgentControlCommands:
    """Tests for agent control commands."""

    @pytest.mark.asyncio
    async def test_pause_no_args(self, bot, mock_update):
        """Test pause without agent ID."""
        await bot._cmd_pause(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_pause_success(self, bot, mock_update, orchestrator):
        """Test successful pause."""
        await bot._cmd_pause(mock_update, ["agent1"])
        orchestrator.pause_agent.assert_called_once_with("agent1")
        args = mock_update.message.reply_text.call_args[0][0]
        assert "paused" in args

    @pytest.mark.asyncio
    async def test_pause_failure(self, bot, mock_update, orchestrator):
        """Test failed pause."""
        orchestrator.pause_agent.return_value = False
        await bot._cmd_pause(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Could not pause" in args

    @pytest.mark.asyncio
    async def test_resume_success(self, bot, mock_update, orchestrator):
        """Test successful resume."""
        await bot._cmd_resume(mock_update, ["agent1"])
        orchestrator.resume_agent.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_terminate_success(self, bot, mock_update, orchestrator):
        """Test successful terminate."""
        await bot._cmd_terminate(mock_update, ["agent1"])
        orchestrator.terminate_agent.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_nudge_no_message(self, bot, mock_update):
        """Test nudge without message."""
        await bot._cmd_nudge(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_nudge_success(self, bot, mock_update, orchestrator):
        """Test successful nudge."""
        await bot._cmd_nudge(mock_update, ["agent1", "please", "continue"])
        orchestrator.nudge_agent.assert_called_once_with("agent1", "please continue")


class TestApprovalCommands:
    """Tests for approval commands."""

    @pytest.mark.asyncio
    async def test_approve_no_args(self, bot, mock_update):
        """Test approve without ID."""
        await bot._cmd_approve(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_approve_success(self, bot, mock_update, orchestrator):
        """Test successful approval."""
        await bot._cmd_approve(mock_update, ["approval1"])
        orchestrator.approve.assert_called_once_with("approval1", approved=True)

    @pytest.mark.asyncio
    async def test_approve_with_pending_future(self, bot, mock_update, orchestrator):
        """Test approval resolves pending future."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot._cmd_approve(mock_update, ["approval1"])

        assert "approval1" not in bot._pending_approvals
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_reject_success(self, bot, mock_update, orchestrator):
        """Test successful rejection."""
        await bot._cmd_reject(mock_update, ["approval1"])
        orchestrator.approve.assert_called_once_with("approval1", approved=False)

    @pytest.mark.asyncio
    async def test_pending_empty(self, bot, mock_update, orchestrator):
        """Test pending with no approvals."""
        await bot._cmd_pending(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "No pending" in args

    @pytest.mark.asyncio
    async def test_pending_with_approvals(self, bot, mock_update, orchestrator):
        """Test pending with approvals."""
        orchestrator.get_pending_approvals.return_value = [
            {"id": "app1", "action_type": "bash", "action_details": {"cmd": "ls"}, "agent_id": "a1"},
        ]
        await bot._cmd_pending(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "app1" in args


class TestSecretCommands:
    """Tests for secret commands."""

    @pytest.mark.asyncio
    async def test_secret_no_args(self, bot, mock_update):
        """Test secret without subcommand."""
        await bot._cmd_secret(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_secret_set(self, bot, mock_update, orchestrator):
        """Test setting a secret."""
        await bot._cmd_secret(mock_update, ["set", "MY_KEY", "my_value"])
        orchestrator.secrets.set.assert_called_once_with("MY_KEY", "my_value")

    @pytest.mark.asyncio
    async def test_secret_get_exists(self, bot, mock_update, orchestrator):
        """Test getting existing secret."""
        orchestrator.secrets.get.return_value = "secret_value"
        await bot._cmd_secret(mock_update, ["get", "MY_KEY"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "exists" in args
        assert "12 chars" in args

    @pytest.mark.asyncio
    async def test_secret_get_not_found(self, bot, mock_update, orchestrator):
        """Test getting non-existent secret."""
        await bot._cmd_secret(mock_update, ["get", "MISSING"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "not found" in args

    @pytest.mark.asyncio
    async def test_secret_list(self, bot, mock_update, orchestrator):
        """Test listing secrets."""
        orchestrator.secrets.list_keys.return_value = ["KEY1", "KEY2"]
        await bot._cmd_secret(mock_update, ["list"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "KEY1" in args
        assert "KEY2" in args

    @pytest.mark.asyncio
    async def test_secret_delete(self, bot, mock_update, orchestrator):
        """Test deleting secret."""
        await bot._cmd_secret(mock_update, ["delete", "MY_KEY"])
        orchestrator.secrets.delete.assert_called_once_with("MY_KEY")


class TestTemplateCommands:
    """Tests for template commands."""

    @pytest.mark.asyncio
    async def test_template_no_args(self, bot, mock_update):
        """Test template without subcommand."""
        await bot._cmd_template(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_template_save(self, bot, mock_update, orchestrator):
        """Test saving a template."""
        await bot._cmd_template(mock_update, ["save", "my_template", "run", "tests"])
        orchestrator.db.save_template.assert_called_once_with("my_template", "run tests")

    @pytest.mark.asyncio
    async def test_template_list_empty(self, bot, mock_update, orchestrator):
        """Test listing templates when empty."""
        await bot._cmd_template(mock_update, ["list"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "No templates" in args

    @pytest.mark.asyncio
    async def test_template_list_with_templates(self, bot, mock_update, orchestrator):
        """Test listing templates."""
        orchestrator.db.list_templates.return_value = [
            {"name": "t1", "task": "Task 1"},
            {"name": "t2", "task": "Task 2"},
        ]
        await bot._cmd_template(mock_update, ["list"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "t1" in args
        assert "t2" in args

    @pytest.mark.asyncio
    async def test_template_use_not_found(self, bot, mock_update, orchestrator):
        """Test using non-existent template."""
        await bot._cmd_template(mock_update, ["use", "missing"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "not found" in args

    @pytest.mark.asyncio
    async def test_template_use_success(self, bot, mock_update, orchestrator):
        """Test using template successfully."""
        orchestrator.db.get_template.return_value = {
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "priority": "normal",
        }
        await bot._cmd_template(mock_update, ["use", "my_template"])
        orchestrator.spawn_agent.assert_called_once()


class TestCallbackHandler:
    """Tests for callback handlers."""

    @pytest.mark.asyncio
    async def test_callback_approval_no_query(self, bot, mock_context):
        """Test callback with no query."""
        update = MagicMock()
        update.callback_query = None
        await bot.callback_approval(update, mock_context)  # Should not raise

    @pytest.mark.asyncio
    async def test_callback_approval_invalid_data(self, bot, mock_context, orchestrator):
        """Test callback with invalid data."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "invalid"
        update.callback_query.answer = AsyncMock()
        await bot.callback_approval(update, mock_context)
        orchestrator.approve.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_approval_approve(self, bot, mock_context, orchestrator):
        """Test approve callback."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "approve:approval1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        # Set up pending approval
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot.callback_approval(update, mock_context)

        orchestrator.approve.assert_called_once_with("approval1", True)

    @pytest.mark.asyncio
    async def test_callback_approval_reject(self, bot, mock_context, orchestrator):
        """Test reject callback."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "reject:approval1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        # Set up pending approval
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot.callback_approval(update, mock_context)

        orchestrator.approve.assert_called_once_with("approval1", False)

    @pytest.mark.asyncio
    async def test_callback_approval_expired(self, bot, mock_context, orchestrator):
        """Test clicking button after approval expired."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "approve:approval1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        # No pending approval - simulates expired/timed out

        await bot.callback_approval(update, mock_context)

        # Should not call orchestrator.approve
        orchestrator.approve.assert_not_called()
        # Should show expired message
        update.callback_query.edit_message_text.assert_called_once()
        assert "Expired" in update.callback_query.edit_message_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_callback_approval_option_selection(self, bot, mock_context, orchestrator):
        """Test option selection callback."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "option:approval1:0"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        # Add pending approval and options
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Option A", "Option B", "Option C"]

        await bot.callback_approval(update, mock_context)

        # Should have resolved the future with the full option text
        assert future.done()
        assert future.result() == "Option A"
        assert "approval1" not in bot._pending_approvals
        assert "approval1" not in bot._pending_options
        update.callback_query.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_approval_option_preserves_full_text(self, bot, mock_context, orchestrator):
        """Test option selection preserves full option text even if long."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "option:approval1:1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        long_option = "This is a very long option text that exceeds the display limit " * 3

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Short", long_option, "Another"]

        await bot.callback_approval(update, mock_context)

        # Should return the full original text, not truncated
        assert future.result() == long_option

    @pytest.mark.asyncio
    async def test_callback_approval_option_invalid_index(self, bot, mock_context, orchestrator):
        """Test option selection with invalid index."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "option:approval1:99"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Option A", "Option B"]

        await bot.callback_approval(update, mock_context)

        # Future should not be resolved due to invalid index
        assert not future.done()

    @pytest.mark.asyncio
    async def test_callback_approval_option_non_numeric_index(self, bot, mock_context, orchestrator):
        """Test option selection with non-numeric index."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "option:approval1:notanumber"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Option A"]

        await bot.callback_approval(update, mock_context)

        # Future should not be resolved due to invalid index
        assert not future.done()

    @pytest.mark.asyncio
    async def test_callback_approval_pending_future_approve(self, bot, mock_context, orchestrator):
        """Test approve callback resolves pending future with 'Confirmed'."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "approve:approval1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot.callback_approval(update, mock_context)

        assert future.done()
        assert future.result() == "Confirmed"

    @pytest.mark.asyncio
    async def test_callback_approval_pending_future_reject(self, bot, mock_context, orchestrator):
        """Test reject callback resolves pending future with None."""
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "reject:approval1"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot.callback_approval(update, mock_context)

        assert future.done()
        assert future.result() is None


class TestNotifyCallback:
    """Tests for notification callback."""

    def test_notify_callback_no_app(self, bot):
        """Test notify callback with no app."""
        bot.notify_callback("agent1", "Test message")  # Should not raise

    def test_notify_callback_with_app(self, bot):
        """Test notify callback with app."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()

        # This creates a task but we can't easily test async behavior
        # Just verify it doesn't raise
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.side_effect = RuntimeError("No running loop")
            bot.notify_callback("agent1", "Test message")


class TestApprovalCallback:
    """Tests for approval callback."""

    @pytest.mark.asyncio
    async def test_approval_callback(self, bot):
        """Test approval callback creates future."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        future = bot.approval_callback("approval1", {"action": "bash"})
        assert "approval1" in bot._pending_approvals
        assert isinstance(future, asyncio.Future)

    @pytest.mark.asyncio
    async def test_approval_callback_with_options(self, bot):
        """Test approval callback with options generates option buttons."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        options = ["Option A", "Option B", "Option C"]
        future = bot.approval_callback("approval1", {"question": "Which option?", "options": options})

        assert "approval1" in bot._pending_approvals
        assert isinstance(future, asyncio.Future)

        # Wait for the task to complete
        await asyncio.sleep(0.01)

        # Verify options are stored for later lookup
        assert "approval1" in bot._pending_options
        assert bot._pending_options["approval1"] == options

        # Verify send_message was called
        assert bot._app.bot.send_message.called

        # Get args - send_message(admin_id, text, reply_markup=keyboard)
        call_args = bot._app.bot.send_message.call_args
        text = call_args.args[1]
        markup = call_args.kwargs["reply_markup"]

        # Check the text contains the question
        assert "Which option?" in text

        # Check reply_markup has option buttons
        inline_keyboard = markup.inline_keyboard
        # Should have 4 rows: 3 options + decline
        assert len(inline_keyboard) == 4

        # First three rows should be options with index-based callback_data
        assert inline_keyboard[0][0].text == "Option A"
        assert inline_keyboard[0][0].callback_data == "option:approval1:0"
        assert inline_keyboard[1][0].text == "Option B"
        assert inline_keyboard[1][0].callback_data == "option:approval1:1"
        assert inline_keyboard[2][0].text == "Option C"
        assert inline_keyboard[2][0].callback_data == "option:approval1:2"

        # Last row should be decline
        assert "Decline" in inline_keyboard[3][0].text
        assert inline_keyboard[3][0].callback_data == "reject:approval1"

    @pytest.mark.asyncio
    async def test_approval_callback_stores_message_info(self, bot):
        """Test approval callback stores message info for later cleanup."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()

        # Mock send_message to return a message with message_id
        mock_message = MagicMock()
        mock_message.message_id = 12345
        bot._app.bot.send_message = AsyncMock(return_value=mock_message)

        bot.approval_callback("approval1", {"action": "test"})

        # Wait for the task to complete
        await asyncio.sleep(0.01)

        # Verify message info is stored
        assert "approval1" in bot._pending_messages
        assert len(bot._pending_messages["approval1"]) == len(bot.config.telegram_admin_ids)
        # Check first admin's message info
        chat_id, msg_id = bot._pending_messages["approval1"][0]
        assert chat_id == bot.config.telegram_admin_ids[0]
        assert msg_id == 12345


class TestCancelApproval:
    """Tests for cancel_approval method."""

    @pytest.mark.asyncio
    async def test_cancel_approval_cleans_up_state(self, bot):
        """Test cancel_approval removes pending state."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Option A"]
        bot._pending_messages["approval1"] = [(123, 456)]

        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.edit_message_text = AsyncMock()

        await bot.cancel_approval("approval1")

        assert "approval1" not in bot._pending_approvals
        assert "approval1" not in bot._pending_options
        assert "approval1" not in bot._pending_messages

    @pytest.mark.asyncio
    async def test_cancel_approval_edits_messages(self, bot):
        """Test cancel_approval edits messages to show expired."""
        bot._pending_messages["approval1"] = [(123, 456), (789, 101)]

        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.edit_message_text = AsyncMock()

        await bot.cancel_approval("approval1")

        # Should have edited both messages
        assert bot._app.bot.edit_message_text.call_count == 2
        calls = bot._app.bot.edit_message_text.call_args_list
        assert calls[0].kwargs["chat_id"] == 123
        assert calls[0].kwargs["message_id"] == 456
        assert "Expired" in calls[0].kwargs["text"]

    @pytest.mark.asyncio
    async def test_cancel_approval_handles_missing_approval(self, bot):
        """Test cancel_approval handles non-existent approval gracefully."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()

        # Should not raise
        await bot.cancel_approval("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_approval_handles_edit_failure(self, bot):
        """Test cancel_approval handles message edit failures gracefully."""
        from telegram.error import BadRequest

        bot._pending_messages["approval1"] = [(123, 456)]
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message not found"))

        # Should not raise
        await bot.cancel_approval("approval1")

        # State should still be cleaned up
        assert "approval1" not in bot._pending_messages


class TestLogsCommand:
    """Tests for logs command."""

    @pytest.mark.asyncio
    async def test_logs_no_args(self, bot, mock_update):
        """Test logs without agent ID."""
        await bot._cmd_logs(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_logs_no_conversation(self, bot, mock_update, orchestrator):
        """Test logs when no conversation."""
        await bot._cmd_logs(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "No logs found" in args

    @pytest.mark.asyncio
    async def test_logs_with_conversation(self, bot, mock_update, orchestrator):
        """Test logs with conversation."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        orchestrator.db.get_conversation.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        await bot._cmd_logs(mock_update, ["agent1"])
        bot._app.bot.send_message.assert_called()


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_initial_requests(self):
        """Test rate limiter allows initial requests."""
        from gru.telegram_bot import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed(123) is True

    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests."""
        from gru.telegram_bot import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        # First 3 should pass
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is True
        # 4th should fail
        assert limiter.is_allowed(123) is False

    def test_rate_limiter_tracks_separate_users(self):
        """Test rate limiter tracks users separately."""
        from gru.telegram_bot import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed(111) is True
        assert limiter.is_allowed(111) is True
        assert limiter.is_allowed(111) is False
        # Different user should still be allowed
        assert limiter.is_allowed(222) is True

    def test_rate_limiter_cleanup_triggered(self):
        """Test rate limiter cleanup is triggered."""
        from gru.telegram_bot import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        # Add many users to trigger cleanup
        for i in range(limiter.CLEANUP_THRESHOLD + 10):
            limiter.is_allowed(i)
        # Should still function correctly
        assert limiter.is_allowed(1) is True


# =============================================================================
# Agent References Tests
# =============================================================================


class TestAgentReferences:
    """Tests for agent number and nickname handling."""

    def test_assign_agent_number(self, bot):
        """Test assigning agent numbers."""
        num1 = bot._assign_agent_number("agent1")
        num2 = bot._assign_agent_number("agent2")
        assert num1 == 1
        assert num2 == 2
        # Same agent should get same number
        assert bot._assign_agent_number("agent1") == 1

    def test_resolve_agent_ref_by_number(self, bot):
        """Test resolving agent by number."""
        bot._assign_agent_number("agent1")
        assert bot._resolve_agent_ref("1") == "agent1"
        assert bot._resolve_agent_ref("999") is None

    def test_resolve_agent_ref_by_nickname(self, bot):
        """Test resolving agent by nickname."""
        bot._set_agent_nickname("agent1", "bugfixer")
        assert bot._resolve_agent_ref("bugfixer") == "agent1"

    def test_resolve_agent_ref_by_id(self, bot):
        """Test resolving agent by full ID."""
        assert bot._resolve_agent_ref("abc123") == "abc123"

    def test_set_agent_nickname(self, bot):
        """Test setting agent nickname."""
        assert bot._set_agent_nickname("agent1", "tester") is True
        assert bot._agent_nicknames["agent1"] == "tester"
        assert bot._nickname_to_agent["tester"] == "agent1"

    def test_set_agent_nickname_replace(self, bot):
        """Test replacing agent nickname."""
        bot._set_agent_nickname("agent1", "old_nick")
        bot._set_agent_nickname("agent1", "new_nick")
        assert bot._agent_nicknames["agent1"] == "new_nick"
        assert "old_nick" not in bot._nickname_to_agent
        assert bot._nickname_to_agent["new_nick"] == "agent1"

    def test_set_agent_nickname_taken(self, bot):
        """Test setting nickname that's taken by another agent."""
        bot._set_agent_nickname("agent1", "nick")
        result = bot._set_agent_nickname("agent2", "nick")
        assert result is False
        # Original mapping should remain
        assert bot._nickname_to_agent["nick"] == "agent1"

    def test_get_agent_display_with_number(self, bot):
        """Test agent display with just number."""
        bot._assign_agent_number("agent1")
        assert bot._get_agent_display("agent1") == "[1]"

    def test_get_agent_display_with_nickname(self, bot):
        """Test agent display with number and nickname."""
        bot._assign_agent_number("agent1")
        bot._set_agent_nickname("agent1", "tester")
        assert bot._get_agent_display("agent1") == "[1:tester]"

    def test_get_agent_display_no_assignment(self, bot):
        """Test agent display without any assignment."""
        assert bot._get_agent_display("unknown") == "unknown"


# =============================================================================
# Rate Limit Check Tests
# =============================================================================


class TestRateLimitCheck:
    """Tests for rate limit checking in handlers."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_user(self, bot, mock_update):
        """Test rate limit check with no user."""
        mock_update.effective_user = None
        result = await bot._check_rate_limit(mock_update)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, bot, mock_update):
        """Test rate limit exceeded message."""
        # Exhaust rate limit
        for _ in range(25):
            bot._rate_limiter.is_allowed(mock_update.effective_user.id)

        result = await bot._check_rate_limit(mock_update)
        assert result is False
        mock_update.message.reply_text.assert_called_with("Rate limit exceeded. Please wait.")


# =============================================================================
# Send Output Extended Tests
# =============================================================================


class TestSendOutputExtended:
    """Extended tests for send_output."""

    @pytest.mark.asyncio
    async def test_send_output_split_messages(self, bot):
        """Test send_output splits into multiple messages."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        # Create text that needs splitting into 2 chunks
        text = "x" * (bot.MAX_MESSAGE_LENGTH + 100)
        await bot.send_output(123, text)

        # Should have called send_message multiple times
        assert bot._app.bot.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_output_many_chunks_as_file(self, bot):
        """Test send_output sends file for many chunks."""
        bot._app = MagicMock()
        bot._app.bot = MagicMock()
        bot._app.bot.send_message = AsyncMock()
        bot._app.bot.send_document = AsyncMock()

        # Create text that would need many chunks but exceeds threshold
        text = "x\n" * (bot.MAX_MESSAGE_LENGTH * 5)
        await bot.send_output(123, text, filename="output.txt")

        # Should send as document
        bot._app.bot.send_document.assert_called_once()


# =============================================================================
# Additional Command Tests
# =============================================================================


class TestSearchCommand:
    """Tests for search command."""

    @pytest.mark.asyncio
    async def test_search_no_args(self, bot, mock_update):
        """Test search without query."""
        await bot._cmd_search(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_search_with_results(self, bot, mock_update, orchestrator):
        """Test search with results."""
        orchestrator.search_agents = AsyncMock(
            return_value=[
                {"id": "agent1", "status": "running", "task": "Fix bug"},
            ]
        )
        await bot._cmd_search(mock_update, ["bug"])
        orchestrator.search_agents.assert_called_once_with("bug")
        args = mock_update.message.reply_text.call_args[0][0]
        assert "agent1" in args


class TestCostCommand:
    """Tests for cost command."""

    @pytest.mark.asyncio
    async def test_cost_no_args_no_running(self, bot, mock_update, orchestrator):
        """Test cost without agent ID and no running agents."""
        await bot._cmd_cost(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "No running agents" in args

    @pytest.mark.asyncio
    async def test_cost_no_args_with_running(self, bot, mock_update, orchestrator):
        """Test cost without agent ID with running agents."""
        orchestrator.list_agents.return_value = [
            {"id": "agent1", "status": "running", "input_tokens": 1000, "output_tokens": 500}
        ]
        bot._assign_agent_number("agent1")
        await bot._cmd_cost(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "1,500" in args

    @pytest.mark.asyncio
    async def test_cost_active_agent(self, bot, mock_update, orchestrator):
        """Test cost for active agent."""
        orchestrator.get_agent_cost = MagicMock(return_value=(1000, 500, "0.0075"))
        await bot._cmd_cost(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "1,000" in args or "1000" in args
        assert "$0.0075" in args

    @pytest.mark.asyncio
    async def test_cost_from_db(self, bot, mock_update, orchestrator):
        """Test cost for inactive agent from DB."""
        orchestrator.get_agent_cost = MagicMock(return_value=None)
        orchestrator.get_agent_cost_from_db = AsyncMock(return_value=(2000, 1000, "0.0150"))
        await bot._cmd_cost(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "$0.0150" in args

    @pytest.mark.asyncio
    async def test_cost_agent_not_found(self, bot, mock_update, orchestrator):
        """Test cost for agent not found."""
        orchestrator.get_agent_cost = MagicMock(return_value=None)
        orchestrator.get_agent_cost_from_db = AsyncMock(return_value=None)
        await bot._cmd_cost(mock_update, ["unknown"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "not found" in args


class TestResumeCommand:
    """Tests for resume command."""

    @pytest.mark.asyncio
    async def test_resume_no_args(self, bot, mock_update):
        """Test resume without agent ID."""
        await bot._cmd_resume(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_resume_failure(self, bot, mock_update, orchestrator):
        """Test failed resume."""
        orchestrator.resume_agent.return_value = False
        await bot._cmd_resume(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Could not resume" in args


class TestTerminateCommand:
    """Tests for terminate command."""

    @pytest.mark.asyncio
    async def test_terminate_no_args(self, bot, mock_update):
        """Test terminate without agent ID."""
        await bot._cmd_terminate(mock_update, [])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in args

    @pytest.mark.asyncio
    async def test_terminate_failure(self, bot, mock_update, orchestrator):
        """Test failed terminate."""
        orchestrator.terminate_agent.return_value = False
        await bot._cmd_terminate(mock_update, ["agent1"])
        args = mock_update.message.reply_text.call_args[0][0]
        assert "Could not terminate" in args


class TestSpawnLiveOutput:
    """Tests for spawn with live output."""

    @pytest.mark.asyncio
    async def test_spawn_live_output(self, bot, mock_update, orchestrator):
        """Test spawn with --live flag."""
        await bot._cmd_spawn(mock_update, ["--live", "run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs.get("live_output") is True

    @pytest.mark.asyncio
    async def test_spawn_no_live_output(self, bot, mock_update, orchestrator):
        """Test spawn without --live flag."""
        await bot._cmd_spawn(mock_update, ["run", "tests"])
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs.get("live_output", False) is False
