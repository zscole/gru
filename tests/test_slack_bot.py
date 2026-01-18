"""Tests for Slack bot interface."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gru.config import Config
from gru.slack_bot import RateLimiter, SlackBot


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        slack_bot_token="xoxb-test-token",
        slack_app_token="xapp-test-token",
        slack_admin_ids=["U123", "U456"],
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
    mock.search_agents = AsyncMock(return_value=[])
    mock.get_agent_cost = MagicMock(return_value=None)
    mock.get_agent_cost_from_db = AsyncMock(return_value=None)
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
    mock.claude = MagicMock()
    mock.claude.send_message = AsyncMock()
    return mock


@pytest.fixture
def bot(config, orchestrator):
    """Create SlackBot instance with mocked Slack components."""
    with patch("gru.slack_bot.AsyncApp") as mock_app_class:
        mock_app = MagicMock()
        mock_app.client = MagicMock()
        mock_app.command = MagicMock(return_value=lambda f: f)
        mock_app.action = MagicMock(return_value=lambda f: f)
        mock_app.event = MagicMock(return_value=lambda f: f)
        mock_app_class.return_value = mock_app
        bot = SlackBot(config, orchestrator)
        bot._client = MagicMock()
        bot._client.chat_postMessage = AsyncMock(return_value={"ok": True, "channel": "C123", "ts": "1234.5678"})
        bot._client.chat_update = AsyncMock()
        return bot


@pytest.fixture
def mock_respond():
    """Create mock respond function."""
    return AsyncMock()


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_requests_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("user1") is True

    def test_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

    def test_separate_limits_per_user(self):
        """Test that limits are tracked per user."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        # Different user should have their own limit
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user2") is False

    def test_window_expiry_allows_new_requests(self):
        """Test that requests are allowed after window expires."""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        # Wait for window to expire
        time.sleep(1.1)
        assert limiter.is_allowed("user1") is True

    def test_cleanup_removes_stale_entries(self):
        """Test that cleanup removes old entries."""
        limiter = RateLimiter(max_requests=10, window_seconds=1)
        limiter.CLEANUP_THRESHOLD = 2  # Lower threshold for testing

        # Add entries for multiple users
        limiter.is_allowed("user1")
        limiter.is_allowed("user2")
        time.sleep(1.1)  # Wait for window to expire

        # Adding a third user should trigger cleanup
        limiter.is_allowed("user3")

        # After cleanup, only user3 should remain (others expired)
        assert "user3" in limiter._requests
        # user1 and user2 should be cleaned up
        assert "user1" not in limiter._requests or len(limiter._requests.get("user1", [])) == 0


# =============================================================================
# SlackBot Auth Tests
# =============================================================================


class TestSlackBotAuth:
    """Tests for authentication."""

    def test_is_admin_true(self, bot):
        """Test admin check returns true for admin."""
        assert bot._is_admin("U123") is True
        assert bot._is_admin("U456") is True

    def test_is_admin_false(self, bot):
        """Test admin check returns false for non-admin."""
        assert bot._is_admin("U999") is False
        assert bot._is_admin("") is False


# =============================================================================
# Message Formatting Tests
# =============================================================================


class TestMessageFormatting:
    """Tests for message formatting utilities."""

    def test_split_short_message(self, bot):
        """Test splitting a short message."""
        chunks = bot._split_message("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_split_long_message_at_newlines(self, bot):
        """Test splitting long message at line boundaries."""
        lines = ["Line " + str(i) + " with some extra padding text" for i in range(200)]
        text = "\n".join(lines)
        assert len(text) > bot.MAX_MESSAGE_LENGTH
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

    def test_format_log_entry_string_content(self, bot):
        """Test formatting log entry with string content."""
        msg = {"role": "user", "content": "Hello world"}
        result = bot._format_log_entry(msg)
        assert "[user]" in result
        assert "Hello world" in result

    def test_format_log_entry_list_content_text(self, bot):
        """Test formatting log entry with list content containing text."""
        msg = {"role": "assistant", "content": [{"type": "text", "text": "Response text"}]}
        result = bot._format_log_entry(msg)
        assert "[assistant]" in result
        assert "Response text" in result

    def test_format_log_entry_list_content_tool_use(self, bot):
        """Test formatting log entry with tool use."""
        msg = {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls -la"}}],
        }
        result = bot._format_log_entry(msg)
        assert "[tool]" in result
        assert "bash" in result

    def test_format_log_entry_list_content_tool_result(self, bot):
        """Test formatting log entry with tool result."""
        msg = {"role": "user", "content": [{"type": "tool_result", "content": "Output here", "is_error": False}]}
        result = bot._format_log_entry(msg)
        assert "[result]" in result

    def test_format_log_entry_list_content_tool_error(self, bot):
        """Test formatting log entry with tool error."""
        msg = {"role": "user", "content": [{"type": "tool_result", "content": "Error msg", "is_error": True}]}
        result = bot._format_log_entry(msg)
        assert "[error]" in result

    def test_summarize_tool_input_bash(self, bot):
        """Test summarizing bash tool input."""
        result = bot._summarize_tool_input("bash", {"command": "echo hello"})
        assert "echo hello" in result

    def test_summarize_tool_input_bash_long(self, bot):
        """Test summarizing long bash command."""
        long_cmd = "x" * 50
        result = bot._summarize_tool_input("bash", {"command": long_cmd})
        assert len(result) <= 33  # 30 + "..."
        assert "..." in result

    def test_summarize_tool_input_file(self, bot):
        """Test summarizing file tool input."""
        result = bot._summarize_tool_input("read_file", {"path": "/test/file.txt"})
        assert "/test/file.txt" in result

    def test_summarize_tool_input_other(self, bot):
        """Test summarizing other tool input."""
        result = bot._summarize_tool_input("unknown_tool", {"key": "value"})
        assert "key" in result or "value" in result


# =============================================================================
# Agent Reference Tests
# =============================================================================


class TestAgentReferences:
    """Tests for agent reference management."""

    def test_assign_agent_number(self, bot):
        """Test assigning agent numbers."""
        num1 = bot._assign_agent_number("agent-abc")
        num2 = bot._assign_agent_number("agent-def")
        assert num1 == 1
        assert num2 == 2
        # Same agent should get same number
        assert bot._assign_agent_number("agent-abc") == 1

    def test_resolve_agent_ref_by_number(self, bot):
        """Test resolving agent by number."""
        bot._assign_agent_number("agent-abc")
        result = bot._resolve_agent_ref("1")
        assert result == "agent-abc"

    def test_resolve_agent_ref_by_nickname(self, bot):
        """Test resolving agent by nickname."""
        bot._assign_agent_number("agent-abc")
        bot._set_agent_nickname("agent-abc", "myagent")
        result = bot._resolve_agent_ref("myagent")
        assert result == "agent-abc"

    def test_resolve_agent_ref_by_full_id(self, bot):
        """Test resolving agent by full ID."""
        result = bot._resolve_agent_ref("agent-full-id")
        assert result == "agent-full-id"

    def test_resolve_agent_ref_unknown_number(self, bot):
        """Test resolving unknown number."""
        result = bot._resolve_agent_ref("999")
        assert result is None

    def test_set_agent_nickname_success(self, bot):
        """Test setting agent nickname."""
        bot._assign_agent_number("agent-abc")
        result = bot._set_agent_nickname("agent-abc", "myagent")
        assert result is True
        assert bot._agent_nicknames["agent-abc"] == "myagent"

    def test_set_agent_nickname_duplicate(self, bot):
        """Test setting duplicate nickname fails."""
        bot._assign_agent_number("agent-abc")
        bot._assign_agent_number("agent-def")
        bot._set_agent_nickname("agent-abc", "myagent")
        result = bot._set_agent_nickname("agent-def", "myagent")
        assert result is False

    def test_set_agent_nickname_update(self, bot):
        """Test updating agent nickname."""
        bot._assign_agent_number("agent-abc")
        bot._set_agent_nickname("agent-abc", "oldname")
        bot._set_agent_nickname("agent-abc", "newname")
        assert bot._agent_nicknames["agent-abc"] == "newname"
        # Old nickname should be removed
        assert "oldname" not in bot._nickname_to_agent

    def test_get_agent_display_with_number(self, bot):
        """Test getting agent display with number."""
        bot._assign_agent_number("agent-abc")
        result = bot._get_agent_display("agent-abc")
        assert result == "[1]"

    def test_get_agent_display_with_nickname(self, bot):
        """Test getting agent display with nickname."""
        bot._assign_agent_number("agent-abc")
        bot._set_agent_nickname("agent-abc", "myagent")
        result = bot._get_agent_display("agent-abc")
        assert result == "[1:myagent]"

    def test_get_agent_display_unknown(self, bot):
        """Test getting agent display for unknown agent."""
        result = bot._get_agent_display("unknown-agent")
        assert result == "unknown-agent"


# =============================================================================
# Command Handler Tests
# =============================================================================


class TestCommandHandlers:
    """Tests for slash command handlers."""

    @pytest.mark.asyncio
    async def test_cmd_help(self, bot, mock_respond):
        """Test help command."""
        await bot._cmd_help(mock_respond, [], "U123")
        mock_respond.assert_called_once()
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Agent Management" in text
        assert "spawn" in text

    @pytest.mark.asyncio
    async def test_cmd_spawn_no_task(self, bot, mock_respond):
        """Test spawn without task."""
        await bot._cmd_spawn(mock_respond, [], "U123")
        mock_respond.assert_called_once()
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_spawn_basic(self, bot, mock_respond, orchestrator):
        """Test basic spawn."""
        await bot._cmd_spawn(mock_respond, ["run", "tests"], "U123")
        orchestrator.spawn_agent.assert_called_once()
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["task"] == "run tests"
        assert call_kwargs["supervised"] is True

    @pytest.mark.asyncio
    async def test_cmd_spawn_unsupervised(self, bot, mock_respond, orchestrator):
        """Test spawn with --unsupervised."""
        await bot._cmd_spawn(mock_respond, ["--unsupervised", "run", "tests"], "U123")
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["supervised"] is False

    @pytest.mark.asyncio
    async def test_cmd_spawn_oneshot(self, bot, mock_respond, orchestrator):
        """Test spawn with --oneshot."""
        await bot._cmd_spawn(mock_respond, ["--oneshot", "run", "tests"], "U123")
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["supervised"] is False
        assert call_kwargs["timeout_mode"] == "auto"

    @pytest.mark.asyncio
    async def test_cmd_spawn_with_live(self, bot, mock_respond, orchestrator):
        """Test spawn with --live."""
        await bot._cmd_spawn(mock_respond, ["--live", "run", "tests"], "U123")
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["live_output"] is True

    @pytest.mark.asyncio
    async def test_cmd_spawn_with_priority(self, bot, mock_respond, orchestrator):
        """Test spawn with --priority."""
        await bot._cmd_spawn(mock_respond, ["--priority", "high", "run", "tests"], "U123")
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["priority"] == "high"

    @pytest.mark.asyncio
    async def test_cmd_spawn_with_workdir(self, bot, mock_respond, orchestrator):
        """Test spawn with --workdir."""
        await bot._cmd_spawn(mock_respond, ["--workdir", "/test/path", "run", "tests"], "U123")
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["workdir"] == "/test/path"

    @pytest.mark.asyncio
    async def test_cmd_spawn_error_handling(self, bot, mock_respond, orchestrator):
        """Test spawn error handling."""
        orchestrator.spawn_agent.side_effect = ValueError("Task too long")
        await bot._cmd_spawn(mock_respond, ["test", "task"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Error:" in text

    @pytest.mark.asyncio
    async def test_cmd_status_overall(self, bot, mock_respond, orchestrator):
        """Test overall status."""
        await bot._cmd_status(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Orchestrator Status" in text

    @pytest.mark.asyncio
    async def test_cmd_status_agent_not_found(self, bot, mock_respond, orchestrator):
        """Test agent status when not found."""
        await bot._cmd_status(mock_respond, ["nonexistent"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_cmd_status_agent_found(self, bot, mock_respond, orchestrator):
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
        await bot._cmd_status(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "agent1" in text
        assert "running" in text

    @pytest.mark.asyncio
    async def test_cmd_list_empty(self, bot, mock_respond, orchestrator):
        """Test list with no agents."""
        await bot._cmd_list(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "No agents found" in text

    @pytest.mark.asyncio
    async def test_cmd_list_with_agents(self, bot, mock_respond, orchestrator):
        """Test list with agents."""
        orchestrator.list_agents.return_value = [
            {"id": "agent1", "status": "running", "task": "Task 1"},
            {"id": "agent2", "status": "paused", "task": "Task 2"},
        ]
        await bot._cmd_list(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "agent1" in text
        assert "agent2" in text

    @pytest.mark.asyncio
    async def test_cmd_list_with_filter(self, bot, mock_respond, orchestrator):
        """Test list with status filter."""
        await bot._cmd_list(mock_respond, ["running"], "U123")
        orchestrator.list_agents.assert_called_once_with("running")


# =============================================================================
# Agent Control Command Tests
# =============================================================================


class TestAgentControlCommands:
    """Tests for agent control commands."""

    @pytest.mark.asyncio
    async def test_cmd_pause_no_args(self, bot, mock_respond):
        """Test pause without agent ID."""
        await bot._cmd_pause(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_pause_success(self, bot, mock_respond, orchestrator):
        """Test successful pause."""
        await bot._cmd_pause(mock_respond, ["agent1"], "U123")
        orchestrator.pause_agent.assert_called_once_with("agent1")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "paused" in text

    @pytest.mark.asyncio
    async def test_cmd_pause_failure(self, bot, mock_respond, orchestrator):
        """Test failed pause."""
        orchestrator.pause_agent.return_value = False
        await bot._cmd_pause(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Could not pause" in text

    @pytest.mark.asyncio
    async def test_cmd_resume_no_args(self, bot, mock_respond):
        """Test resume without agent ID."""
        await bot._cmd_resume(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_resume_success(self, bot, mock_respond, orchestrator):
        """Test successful resume."""
        await bot._cmd_resume(mock_respond, ["agent1"], "U123")
        orchestrator.resume_agent.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_cmd_terminate_no_args(self, bot, mock_respond):
        """Test terminate without agent ID."""
        await bot._cmd_terminate(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_terminate_success(self, bot, mock_respond, orchestrator):
        """Test successful terminate."""
        await bot._cmd_terminate(mock_respond, ["agent1"], "U123")
        orchestrator.terminate_agent.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_cmd_nudge_no_message(self, bot, mock_respond):
        """Test nudge without message."""
        await bot._cmd_nudge(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_nudge_success(self, bot, mock_respond, orchestrator):
        """Test successful nudge."""
        await bot._cmd_nudge(mock_respond, ["agent1", "please", "continue"], "U123")
        orchestrator.nudge_agent.assert_called_once_with("agent1", "please continue")


# =============================================================================
# Approval Command Tests
# =============================================================================


class TestApprovalCommands:
    """Tests for approval commands."""

    @pytest.mark.asyncio
    async def test_cmd_approve_no_args(self, bot, mock_respond):
        """Test approve without ID."""
        await bot._cmd_approve(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_approve_success(self, bot, mock_respond, orchestrator):
        """Test successful approval."""
        await bot._cmd_approve(mock_respond, ["approval1"], "U123")
        orchestrator.approve.assert_called_once_with("approval1", approved=True)

    @pytest.mark.asyncio
    async def test_cmd_approve_with_pending_future(self, bot, mock_respond, orchestrator):
        """Test approval resolves pending future."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot._cmd_approve(mock_respond, ["approval1"], "U123")

        assert "approval1" not in bot._pending_approvals
        assert future.result() == "Confirmed"

    @pytest.mark.asyncio
    async def test_cmd_reject_no_args(self, bot, mock_respond):
        """Test reject without ID."""
        await bot._cmd_reject(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_reject_success(self, bot, mock_respond, orchestrator):
        """Test successful rejection."""
        await bot._cmd_reject(mock_respond, ["approval1"], "U123")
        orchestrator.approve.assert_called_once_with("approval1", approved=False)

    @pytest.mark.asyncio
    async def test_cmd_reject_with_pending_future(self, bot, mock_respond, orchestrator):
        """Test rejection resolves pending future with None."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future

        await bot._cmd_reject(mock_respond, ["approval1"], "U123")

        assert "approval1" not in bot._pending_approvals
        assert future.result() is None

    @pytest.mark.asyncio
    async def test_cmd_pending_empty(self, bot, mock_respond, orchestrator):
        """Test pending with no approvals."""
        await bot._cmd_pending(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "No pending" in text

    @pytest.mark.asyncio
    async def test_cmd_pending_with_approvals(self, bot, mock_respond, orchestrator):
        """Test pending with approvals."""
        orchestrator.get_pending_approvals.return_value = [
            {"id": "app1", "action_type": "bash", "action_details": {"cmd": "ls"}},
        ]
        await bot._cmd_pending(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "app1" in text


# =============================================================================
# Logs Command Tests
# =============================================================================


class TestLogsCommand:
    """Tests for logs command."""

    @pytest.mark.asyncio
    async def test_cmd_logs_no_args(self, bot, mock_respond):
        """Test logs without agent ID."""
        await bot._cmd_logs(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_logs_no_conversation(self, bot, mock_respond, orchestrator):
        """Test logs when no conversation."""
        await bot._cmd_logs(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "No logs found" in text

    @pytest.mark.asyncio
    async def test_cmd_logs_with_conversation(self, bot, mock_respond, orchestrator):
        """Test logs with conversation."""
        orchestrator.db.get_conversation.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await bot._cmd_logs(mock_respond, ["agent1"], "U123")
        # Should call respond at least once
        assert mock_respond.call_count >= 1


# =============================================================================
# Search Command Tests
# =============================================================================


class TestSearchCommand:
    """Tests for search command."""

    @pytest.mark.asyncio
    async def test_cmd_search_no_query(self, bot, mock_respond):
        """Test search without query."""
        await bot._cmd_search(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_search_no_results(self, bot, mock_respond, orchestrator):
        """Test search with no results."""
        await bot._cmd_search(mock_respond, ["test"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "No agents found" in text

    @pytest.mark.asyncio
    async def test_cmd_search_with_results(self, bot, mock_respond, orchestrator):
        """Test search with results."""
        orchestrator.search_agents.return_value = [
            {"id": "agent1", "status": "completed", "task": "Test task", "input_tokens": 100, "output_tokens": 50},
        ]
        await bot._cmd_search(mock_respond, ["test"], "U123")
        orchestrator.search_agents.assert_called_once_with("test")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "agent1" in text


# =============================================================================
# Cost Command Tests
# =============================================================================


class TestCostCommand:
    """Tests for cost command."""

    @pytest.mark.asyncio
    async def test_cmd_cost_specific_agent_not_found(self, bot, mock_respond, orchestrator):
        """Test cost for specific agent not found."""
        await bot._cmd_cost(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_cmd_cost_specific_agent_found(self, bot, mock_respond, orchestrator):
        """Test cost for specific agent found."""
        orchestrator.get_agent_cost.return_value = (1000, 500, "$0.05")
        await bot._cmd_cost(mock_respond, ["agent1"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "1,000" in text
        assert "500" in text
        assert "$0.05" in text

    @pytest.mark.asyncio
    async def test_cmd_cost_all_agents(self, bot, mock_respond, orchestrator):
        """Test cost for all agents."""
        orchestrator.list_agents.return_value = [
            {"id": "agent1", "status": "completed", "input_tokens": 100, "output_tokens": 50},
            {"id": "agent2", "status": "running", "input_tokens": 200, "output_tokens": 100},
        ]
        orchestrator.get_agent_cost.return_value = (100, 50, "$0.01")
        await bot._cmd_cost(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Token Usage Summary" in text


# =============================================================================
# Secret Command Tests
# =============================================================================


class TestSecretCommands:
    """Tests for secret commands."""

    @pytest.mark.asyncio
    async def test_cmd_secret_no_args(self, bot, mock_respond):
        """Test secret without subcommand."""
        await bot._cmd_secret(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_secret_set(self, bot, mock_respond, orchestrator):
        """Test setting a secret."""
        await bot._cmd_secret(mock_respond, ["set", "MY_KEY", "my_value"], "U123")
        orchestrator.secrets.set.assert_called_once_with("MY_KEY", "my_value")

    @pytest.mark.asyncio
    async def test_cmd_secret_get_exists(self, bot, mock_respond, orchestrator):
        """Test getting existing secret."""
        orchestrator.secrets.get.return_value = "secret_value"
        await bot._cmd_secret(mock_respond, ["get", "MY_KEY"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "exists" in text

    @pytest.mark.asyncio
    async def test_cmd_secret_get_not_found(self, bot, mock_respond, orchestrator):
        """Test getting non-existent secret."""
        await bot._cmd_secret(mock_respond, ["get", "MISSING"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_cmd_secret_list(self, bot, mock_respond, orchestrator):
        """Test listing secrets."""
        orchestrator.secrets.list_keys.return_value = ["KEY1", "KEY2"]
        await bot._cmd_secret(mock_respond, ["list"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "KEY1" in text
        assert "KEY2" in text

    @pytest.mark.asyncio
    async def test_cmd_secret_delete(self, bot, mock_respond, orchestrator):
        """Test deleting secret."""
        await bot._cmd_secret(mock_respond, ["delete", "MY_KEY"], "U123")
        orchestrator.secrets.delete.assert_called_once_with("MY_KEY")


# =============================================================================
# Template Command Tests
# =============================================================================


class TestTemplateCommands:
    """Tests for template commands."""

    @pytest.mark.asyncio
    async def test_cmd_template_no_args(self, bot, mock_respond):
        """Test template without subcommand."""
        await bot._cmd_template(mock_respond, [], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "Usage:" in text

    @pytest.mark.asyncio
    async def test_cmd_template_save(self, bot, mock_respond, orchestrator):
        """Test saving a template."""
        await bot._cmd_template(mock_respond, ["save", "my_template", "run", "tests"], "U123")
        orchestrator.db.save_template.assert_called_once_with("my_template", "run tests")

    @pytest.mark.asyncio
    async def test_cmd_template_list_empty(self, bot, mock_respond, orchestrator):
        """Test listing templates when empty."""
        await bot._cmd_template(mock_respond, ["list"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "No templates" in text

    @pytest.mark.asyncio
    async def test_cmd_template_list_with_templates(self, bot, mock_respond, orchestrator):
        """Test listing templates."""
        orchestrator.db.list_templates.return_value = [
            {"name": "t1", "task": "Task 1"},
            {"name": "t2", "task": "Task 2"},
        ]
        await bot._cmd_template(mock_respond, ["list"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "t1" in text
        assert "t2" in text

    @pytest.mark.asyncio
    async def test_cmd_template_use_not_found(self, bot, mock_respond, orchestrator):
        """Test using non-existent template."""
        await bot._cmd_template(mock_respond, ["use", "missing"], "U123")
        args = mock_respond.call_args
        text = args.kwargs.get("text") or args.args[0]
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_cmd_template_use_success(self, bot, mock_respond, orchestrator):
        """Test using template successfully."""
        orchestrator.db.get_template.return_value = {
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "priority": "normal",
        }
        await bot._cmd_template(mock_respond, ["use", "my_template"], "U123")
        orchestrator.spawn_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_template_delete_success(self, bot, mock_respond, orchestrator):
        """Test deleting template."""
        await bot._cmd_template(mock_respond, ["delete", "my_template"], "U123")
        orchestrator.db.delete_template.assert_called_once_with("my_template")


# =============================================================================
# Callback Tests
# =============================================================================


class TestNotifyCallback:
    """Tests for notification callback."""

    def test_notify_callback_no_running_loop(self, bot):
        """Test notify callback with no running loop."""
        # Should not raise when no event loop is running
        bot.notify_callback("agent1", "Test message")

    @pytest.mark.asyncio
    async def test_notify_callback_sends_messages(self, bot):
        """Test notify callback sends messages to admins."""
        bot.notify_callback("agent1", "Test message")
        # Allow async task to run
        await asyncio.sleep(0.01)
        # Should have attempted to send to admins
        assert bot._client.chat_postMessage.called


class TestApprovalCallback:
    """Tests for approval callback."""

    @pytest.mark.asyncio
    async def test_approval_callback_creates_future(self, bot):
        """Test approval callback creates future."""
        future = bot.approval_callback("approval1", {"action": "bash"})
        assert "approval1" in bot._pending_approvals
        assert isinstance(future, asyncio.Future)

    @pytest.mark.asyncio
    async def test_approval_callback_with_options(self, bot):
        """Test approval callback with options."""
        options = ["Option A", "Option B", "Option C"]
        future = bot.approval_callback("approval1", {"question": "Which option?", "options": options})

        assert "approval1" in bot._pending_approvals
        await asyncio.sleep(0.01)

        # Options should be stored
        assert "approval1" in bot._pending_options
        assert bot._pending_options["approval1"] == options


class TestCancelApproval:
    """Tests for cancel_approval method."""

    @pytest.mark.asyncio
    async def test_cancel_approval_cleans_up_state(self, bot):
        """Test cancel_approval removes pending state."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        bot._pending_approvals["approval1"] = future
        bot._pending_options["approval1"] = ["Option A"]
        bot._pending_messages["approval1"] = [("C123", "1234.5678")]

        await bot.cancel_approval("approval1")

        assert "approval1" not in bot._pending_approvals
        assert "approval1" not in bot._pending_options
        assert "approval1" not in bot._pending_messages

    @pytest.mark.asyncio
    async def test_cancel_approval_updates_messages(self, bot):
        """Test cancel_approval updates messages to show expired."""
        bot._pending_messages["approval1"] = [("C123", "1234.5678")]

        await bot.cancel_approval("approval1")

        bot._client.chat_update.assert_called()
        call_kwargs = bot._client.chat_update.call_args.kwargs
        assert "Expired" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_cancel_approval_handles_missing(self, bot):
        """Test cancel_approval handles non-existent approval."""
        # Should not raise
        await bot.cancel_approval("nonexistent")


# =============================================================================
# Tool Handling Tests
# =============================================================================


class TestToolHandling:
    """Tests for chat tool handling."""

    @pytest.mark.asyncio
    async def test_handle_tool_use_spawn_agent(self, bot, orchestrator):
        """Test handling spawn_agent tool."""
        result = await bot._handle_tool_use("spawn_agent", {"task": "Test task"})
        orchestrator.spawn_agent.assert_called_once()
        assert "spawned" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_terminate_agent(self, bot, orchestrator):
        """Test handling terminate_agent tool."""
        bot._assign_agent_number("agent-abc")
        result = await bot._handle_tool_use("terminate_agent", {"agent_ref": "1"})
        orchestrator.terminate_agent.assert_called_once_with("agent-abc")
        assert "terminated" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_pause_agent(self, bot, orchestrator):
        """Test handling pause_agent tool."""
        bot._assign_agent_number("agent-abc")
        result = await bot._handle_tool_use("pause_agent", {"agent_ref": "1"})
        orchestrator.pause_agent.assert_called_once_with("agent-abc")
        assert "paused" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_resume_agent(self, bot, orchestrator):
        """Test handling resume_agent tool."""
        bot._assign_agent_number("agent-abc")
        result = await bot._handle_tool_use("resume_agent", {"agent_ref": "1"})
        orchestrator.resume_agent.assert_called_once_with("agent-abc")
        assert "resumed" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_get_status(self, bot, orchestrator):
        """Test handling get_status tool."""
        result = await bot._handle_tool_use("get_status", {})
        assert "Orchestrator Status" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_list_agents(self, bot, orchestrator):
        """Test handling list_agents tool."""
        result = await bot._handle_tool_use("list_agents", {})
        assert "No agents found" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_get_pending_approvals(self, bot, orchestrator):
        """Test handling get_pending_approvals tool."""
        result = await bot._handle_tool_use("get_pending_approvals", {})
        assert "No pending" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_approve_action(self, bot, orchestrator):
        """Test handling approve_action tool."""
        result = await bot._handle_tool_use("approve_action", {"approval_id": "app1"})
        orchestrator.approve.assert_called_once_with("app1", approved=True)
        assert "Approved" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_reject_action(self, bot, orchestrator):
        """Test handling reject_action tool."""
        result = await bot._handle_tool_use("reject_action", {"approval_id": "app1"})
        orchestrator.approve.assert_called_once_with("app1", approved=False)
        assert "Rejected" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_nudge_agent(self, bot, orchestrator):
        """Test handling nudge_agent tool."""
        bot._assign_agent_number("agent-abc")
        result = await bot._handle_tool_use("nudge_agent", {"agent_ref": "1", "message": "status?"})
        orchestrator.nudge_agent.assert_called_once()
        assert "Nudge sent" in result

    @pytest.mark.asyncio
    async def test_handle_tool_use_nickname_agent(self, bot):
        """Test handling nickname_agent tool."""
        bot._assign_agent_number("agent-abc")
        result = await bot._handle_tool_use("nickname_agent", {"agent_ref": "1", "nickname": "myagent"})
        assert "nicknamed" in result
        assert bot._agent_nicknames["agent-abc"] == "myagent"

    @pytest.mark.asyncio
    async def test_handle_tool_use_unknown(self, bot):
        """Test handling unknown tool."""
        result = await bot._handle_tool_use("unknown_tool", {})
        assert "Unknown tool" in result
