"""Tests for Discord bot interface."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gru.config import Config
from gru.discord_bot import ApprovalView, DiscordBot, RateLimiter


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        discord_token="test_token",
        discord_admin_ids=[123, 456],
        discord_guild_id=789,
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
    mock.set_notify_callback = MagicMock()
    mock.set_approval_callback = MagicMock()
    mock.set_cancel_approval_callback = MagicMock()
    return mock


@pytest.fixture
def bot(config, orchestrator):
    """Create DiscordBot instance."""
    with patch("gru.discord_bot.commands.Bot"):
        return DiscordBot(config, orchestrator)


@pytest.fixture
def mock_interaction():
    """Create mock Discord Interaction."""
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.id = 123  # Admin user
    interaction.channel = MagicMock()
    interaction.channel.id = 999
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    return interaction


class TestDiscordBotAuth:
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
    async def test_check_admin_authorized(self, bot, mock_interaction):
        """Test check_admin passes for admin."""
        result = await bot._check_admin(mock_interaction)
        assert result is True
        mock_interaction.response.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_admin_unauthorized(self, bot, mock_interaction):
        """Test check_admin fails for non-admin."""
        mock_interaction.user.id = 999
        result = await bot._check_admin(mock_interaction)
        assert result is False
        mock_interaction.response.send_message.assert_called_once_with("Unauthorized", ephemeral=True)


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_allows_requests_under_limit(self):
        """Test rate limiter allows requests under limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed(123) is True

    def test_blocks_requests_over_limit(self):
        """Test rate limiter blocks requests over limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is False

    def test_separate_limits_per_user(self):
        """Test rate limiter has separate limits per user."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is True
        assert limiter.is_allowed(123) is False
        # Different user should not be affected
        assert limiter.is_allowed(456) is True

    def test_cleanup_removes_stale_entries(self):
        """Test cleanup removes stale entries."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        limiter.CLEANUP_THRESHOLD = 1  # Force cleanup on 2nd user

        limiter.is_allowed(1)
        limiter._requests[1] = [0.0]  # Stale entry
        limiter.is_allowed(2)  # Triggers cleanup

        assert 1 not in limiter._requests or len(limiter._requests[1]) == 0


class TestMessageSplitting:
    """Tests for message splitting logic."""

    def test_split_short_message(self, bot):
        """Test splitting a short message."""
        chunks = bot._split_message("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_split_long_message_at_newlines(self, bot):
        """Test splitting long message at line boundaries."""
        # Create enough lines to exceed MAX_MESSAGE_LENGTH (2000 for Discord)
        lines = ["Line " + str(i) + " with some extra padding text" for i in range(200)]
        text = "\n".join(lines)
        assert len(text) > bot.MAX_MESSAGE_LENGTH  # Verify setup
        chunks = bot._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= bot.MAX_MESSAGE_LENGTH

    def test_split_long_single_line(self, bot):
        """Test splitting a very long single line."""
        text = "x" * 5000
        chunks = bot._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= bot.MAX_MESSAGE_LENGTH

    def test_split_empty_message(self, bot):
        """Test splitting empty message."""
        chunks = bot._split_message("")
        assert chunks == []


class TestApprovalView:
    """Tests for approval view buttons."""

    @pytest.mark.asyncio
    async def test_approval_view_creates_approve_reject_buttons(self, bot):
        """Test approval view creates approve/reject buttons."""
        view = ApprovalView("test-approval", bot)
        # View should have 2 children (approve and reject buttons)
        assert len(view.children) == 2

    @pytest.mark.asyncio
    async def test_approval_view_creates_option_buttons(self, bot):
        """Test approval view creates option buttons."""
        options = ["Option 1", "Option 2", "Option 3"]
        view = ApprovalView("test-approval", bot, options=options)
        # Should have options + decline button
        assert len(view.children) == len(options) + 1


class TestNotifyCallback:
    """Tests for notify callback."""

    def test_notify_callback_no_loop(self, bot):
        """Test notify_callback handles no running loop."""
        # Should not raise
        bot.notify_callback("agent1", "Test message")


class TestApprovalCallback:
    """Tests for approval callback."""

    @pytest.mark.asyncio
    async def test_approval_callback_returns_future(self, bot):
        """Test approval_callback returns a future."""
        future = bot.approval_callback("test-approval", {"action": "test"})
        assert isinstance(future, asyncio.Future)
        assert "test-approval" in bot._pending_approvals

    @pytest.mark.asyncio
    async def test_approval_callback_with_options(self, bot):
        """Test approval_callback with options creates future."""
        options = ["Option 1", "Option 2"]
        future = bot.approval_callback("test-approval", {"options": options})
        assert isinstance(future, asyncio.Future)
        assert "test-approval" in bot._pending_approvals
        # Note: _pending_options is set in background task, not checked here


class TestCancelApproval:
    """Tests for cancel approval."""

    @pytest.mark.asyncio
    async def test_cancel_approval_cleans_up_state(self, bot):
        """Test cancel_approval cleans up pending state."""
        bot._pending_approvals["test-approval"] = asyncio.Future()
        bot._pending_options["test-approval"] = ["opt1"]
        bot._pending_messages["test-approval"] = [(123, 456)]

        await bot.cancel_approval("test-approval")

        assert "test-approval" not in bot._pending_approvals
        assert "test-approval" not in bot._pending_options
        assert "test-approval" not in bot._pending_messages

    @pytest.mark.asyncio
    async def test_cancel_approval_handles_missing(self, bot):
        """Test cancel_approval handles missing approval."""
        # Should not raise
        await bot.cancel_approval("nonexistent")


class TestConfigValidation:
    """Tests for config validation with Discord."""

    def test_config_valid_with_discord_only(self):
        """Test config is valid with only Discord configured."""
        config = Config(
            discord_token="test_token",
            discord_admin_ids=[123],
            anthropic_api_key="test_key",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_config_valid_with_both(self):
        """Test config is valid with both Telegram and Discord."""
        config = Config(
            telegram_token="tg_token",
            telegram_admin_ids=[123],
            discord_token="discord_token",
            discord_admin_ids=[456],
            anthropic_api_key="test_key",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_config_invalid_discord_partial(self):
        """Test config is invalid with partial Discord config."""
        config = Config(
            discord_token="discord_token",
            # Missing discord_admin_ids
            anthropic_api_key="test_key",
        )
        errors = config.validate()
        assert any("GRU_DISCORD_ADMIN_IDS" in e for e in errors)

    def test_config_invalid_no_bot(self):
        """Test config is invalid with no bot configured."""
        config = Config(
            anthropic_api_key="test_key",
        )
        errors = config.validate()
        assert any("At least one bot interface" in e for e in errors)


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
# Button Callback Tests
# =============================================================================


class TestApproveButtonCallback:
    """Tests for ApproveButton callback."""

    @pytest.mark.asyncio
    async def test_approve_button_expired(self, bot, mock_interaction):
        """Test approve button when approval has expired."""
        from gru.discord_bot import ApproveButton, ApprovalView

        view = ApprovalView("test-approval", bot)
        button = ApproveButton(view)

        # No pending approval - simulates expired
        await button.callback(mock_interaction)

        mock_interaction.response.edit_message.assert_called_once()
        args = mock_interaction.response.edit_message.call_args
        assert "Expired" in args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_approve_button_success(self, bot, mock_interaction, orchestrator):
        """Test approve button success."""
        from gru.discord_bot import ApproveButton, ApprovalView

        view = ApprovalView("test-approval", bot)
        button = ApproveButton(view)

        # Set up pending approval
        future = asyncio.get_event_loop().create_future()
        bot._pending_approvals["test-approval"] = future

        await button.callback(mock_interaction)

        assert future.done()
        assert future.result() == "Confirmed"
        assert "test-approval" not in bot._pending_approvals
        orchestrator.approve.assert_called_once_with("test-approval", approved=True)


class TestRejectButtonCallback:
    """Tests for RejectButton callback."""

    @pytest.mark.asyncio
    async def test_reject_button_expired(self, bot, mock_interaction):
        """Test reject button when approval has expired."""
        from gru.discord_bot import RejectButton, ApprovalView

        view = ApprovalView("test-approval", bot)
        button = RejectButton(view)

        await button.callback(mock_interaction)

        mock_interaction.response.edit_message.assert_called_once()
        args = mock_interaction.response.edit_message.call_args
        assert "Expired" in args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_reject_button_success(self, bot, mock_interaction, orchestrator):
        """Test reject button success."""
        from gru.discord_bot import RejectButton, ApprovalView

        view = ApprovalView("test-approval", bot)
        button = RejectButton(view)

        future = asyncio.get_event_loop().create_future()
        bot._pending_approvals["test-approval"] = future

        await button.callback(mock_interaction)

        assert future.done()
        assert future.result() is None
        assert "test-approval" not in bot._pending_approvals
        orchestrator.approve.assert_called_once_with("test-approval", approved=False)


class TestOptionButtonCallback:
    """Tests for OptionButton callback."""

    @pytest.mark.asyncio
    async def test_option_button_expired(self, bot, mock_interaction):
        """Test option button when approval has expired."""
        from gru.discord_bot import OptionButton, ApprovalView

        options = ["Option A", "Option B"]
        view = ApprovalView("test-approval", bot, options=options)
        button = OptionButton(view, 0, "Option A")

        await button.callback(mock_interaction)

        mock_interaction.response.edit_message.assert_called_once()
        args = mock_interaction.response.edit_message.call_args
        assert "Expired" in args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_option_button_success(self, bot, mock_interaction):
        """Test option button selection success."""
        from gru.discord_bot import OptionButton, ApprovalView

        options = ["Option A", "Option B"]
        view = ApprovalView("test-approval", bot, options=options)
        button = OptionButton(view, 0, "Option A")

        future = asyncio.get_event_loop().create_future()
        bot._pending_approvals["test-approval"] = future

        await button.callback(mock_interaction)

        assert future.done()
        assert future.result() == "Option A"
        assert "test-approval" not in bot._pending_approvals


# =============================================================================
# Rate Limit Check Tests
# =============================================================================


class TestRateLimitCheck:
    """Tests for rate limit checking in handlers."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, bot, mock_interaction):
        """Test rate limit exceeded message."""
        # Exhaust rate limit
        for _ in range(25):
            bot._rate_limiter.is_allowed(mock_interaction.user.id)

        result = await bot._check_rate_limit(mock_interaction)
        assert result is False
        mock_interaction.response.send_message.assert_called_with(
            "Rate limit exceeded. Please wait.", ephemeral=True
        )

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, bot, mock_interaction):
        """Test rate limit check passes."""
        result = await bot._check_rate_limit(mock_interaction)
        assert result is True
        mock_interaction.response.send_message.assert_not_called()


# =============================================================================
# Format Log Entry Tests
# =============================================================================


class TestFormatLogEntry:
    """Tests for log entry formatting."""

    def test_format_log_entry_string_content(self, bot):
        """Test formatting string content."""
        msg = {"role": "user", "content": "Hello world"}
        result = bot._format_log_entry(msg)
        assert "[user]" in result
        assert "Hello world" in result

    def test_format_log_entry_long_string_truncated(self, bot):
        """Test long string content is truncated."""
        msg = {"role": "assistant", "content": "x" * 300}
        result = bot._format_log_entry(msg)
        assert len(result) < 300

    def test_format_log_entry_list_with_text(self, bot):
        """Test formatting list content with text."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Some text here"}],
        }
        result = bot._format_log_entry(msg)
        assert "[assistant]" in result
        assert "Some text" in result

    def test_format_log_entry_list_with_tool_use(self, bot):
        """Test formatting list content with tool_use."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "bash", "input": {"command": "ls"}}
            ],
        }
        result = bot._format_log_entry(msg)
        assert "[tool]" in result
        assert "bash" in result

    def test_format_log_entry_list_with_tool_result(self, bot):
        """Test formatting list content with tool_result."""
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "file.txt", "is_error": False}
            ],
        }
        result = bot._format_log_entry(msg)
        assert "[result]" in result

    def test_format_log_entry_list_with_error_result(self, bot):
        """Test formatting list content with error tool_result."""
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "Command failed", "is_error": True}
            ],
        }
        result = bot._format_log_entry(msg)
        assert "[error]" in result

    def test_format_log_entry_empty_list(self, bot):
        """Test formatting empty list content."""
        msg = {"role": "assistant", "content": []}
        result = bot._format_log_entry(msg)
        assert "(empty)" in result

    def test_format_log_entry_non_string_non_list(self, bot):
        """Test formatting other content types."""
        msg = {"role": "system", "content": {"key": "value"}}
        result = bot._format_log_entry(msg)
        assert "[system]" in result


# =============================================================================
# Summarize Tool Input Tests
# =============================================================================


class TestSummarizeToolInput:
    """Tests for tool input summarization."""

    def test_summarize_bash(self, bot):
        """Test summarizing bash command."""
        result = bot._summarize_tool_input("bash", {"command": "ls -la"})
        assert "ls -la" in result

    def test_summarize_bash_long_command(self, bot):
        """Test summarizing long bash command is truncated."""
        long_cmd = "x" * 50
        result = bot._summarize_tool_input("bash", {"command": long_cmd})
        assert "..." in result
        assert len(result) < 50

    def test_summarize_read_file(self, bot):
        """Test summarizing read_file."""
        result = bot._summarize_tool_input("read_file", {"path": "/test/file.txt"})
        assert "/test/file.txt" in result

    def test_summarize_write_file(self, bot):
        """Test summarizing write_file."""
        result = bot._summarize_tool_input("write_file", {"path": "/output.txt"})
        assert "/output.txt" in result

    def test_summarize_search_files(self, bot):
        """Test summarizing search_files."""
        result = bot._summarize_tool_input("search_files", {"pattern": "*.py"})
        assert "*.py" in result

    def test_summarize_unknown_tool(self, bot):
        """Test summarizing unknown tool."""
        result = bot._summarize_tool_input("custom_tool", {"arg1": "value1"})
        assert "arg1" in result or "value1" in result


# =============================================================================
# Send Output Tests
# =============================================================================


class TestSendOutput:
    """Tests for send_output method."""

    @pytest.mark.asyncio
    async def test_send_output_short_message(self, bot):
        """Test sending a short message."""
        channel = MagicMock()
        channel.send = AsyncMock()

        await bot.send_output(channel, "Hello world")

        channel.send.assert_called_once_with("Hello world")

    @pytest.mark.asyncio
    async def test_send_output_force_file(self, bot):
        """Test force sending as file."""
        channel = MagicMock()
        channel.send = AsyncMock()

        await bot.send_output(channel, "Hello", filename="test.txt", force_file=True)

        channel.send.assert_called_once()
        # Check that file was sent
        call_args = channel.send.call_args
        assert "file" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_send_output_long_message_as_file(self, bot):
        """Test long message sent as file."""
        channel = MagicMock()
        channel.send = AsyncMock()

        # Create message longer than threshold
        text = "x" * (bot.MAX_MESSAGE_LENGTH * bot.SPLIT_THRESHOLD + 1)
        await bot.send_output(channel, text, filename="output.txt")

        # Should send as file
        call_args = channel.send.call_args
        assert "file" in call_args.kwargs


# =============================================================================
# Additional Tests
# =============================================================================


class TestApprovalViewOptions:
    """Tests for ApprovalView with options."""

    @pytest.mark.asyncio
    async def test_creates_option_buttons(self, bot):
        """Test view creates option buttons with truncated labels."""
        from gru.discord_bot import ApprovalView

        long_option = "x" * 100
        options = [long_option, "Short"]
        view = ApprovalView("test", bot, options=options)

        # Should have 3 buttons: 2 options + decline
        assert len(view.children) == 3

    @pytest.mark.asyncio
    async def test_creates_standard_buttons(self, bot):
        """Test view creates standard approve/reject buttons."""
        from gru.discord_bot import ApprovalView

        view = ApprovalView("test", bot, options=None)

        # Should have 2 buttons: approve + reject
        assert len(view.children) == 2