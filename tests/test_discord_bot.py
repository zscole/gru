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
