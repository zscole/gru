"""Tests for orchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gru.claude import Response, ToolUse
from gru.config import Config
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.orchestrator import Agent, Orchestrator


@pytest.fixture
async def test_config():
    """Create test config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            data_dir=Path(tmpdir),
            telegram_token="test_token",
            telegram_admin_ids=[123],
            anthropic_api_key="test_key",
            default_model="test-model",
            max_tokens=1000,
            default_timeout=10,
            max_concurrent_agents=5,
        )
        yield config


@pytest.fixture
async def test_db(test_config):
    """Create test database."""
    db = Database(test_config.db_path)
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def test_secrets(test_db, test_config):
    """Create test secret store."""
    crypto = CryptoManager(test_config.data_dir, iterations=1000)
    crypto.initialize("test_password")
    return SecretStore(test_db, crypto)


@pytest.fixture
async def orchestrator(test_config, test_db, test_secrets):
    """Create test orchestrator."""
    orch = Orchestrator(test_config, test_db, test_secrets)
    yield orch
    await orch.stop()


@pytest.mark.asyncio
async def test_spawn_agent(orchestrator):
    """Test spawning an agent."""
    agent = await orchestrator.spawn_agent(
        task="Test task",
        supervised=True,
        priority="normal",
    )

    assert agent["id"] is not None
    assert agent["task"] == "Test task"
    assert agent["status"] == "idle"
    assert agent["supervised"] == 1


@pytest.mark.asyncio
async def test_spawn_agent_with_workdir(orchestrator, test_config):
    """Test spawning agent with custom workdir."""
    workdir = str(test_config.data_dir / "custom_workdir")
    agent = await orchestrator.spawn_agent(
        task="Test task",
        workdir=workdir,
    )

    assert agent["workdir"] == workdir
    assert Path(workdir).exists()


@pytest.mark.asyncio
async def test_get_agent(orchestrator):
    """Test getting an agent."""
    spawned = await orchestrator.spawn_agent(task="Test task")
    agent = await orchestrator.get_agent(spawned["id"])

    assert agent is not None
    assert agent["id"] == spawned["id"]


@pytest.mark.asyncio
async def test_get_nonexistent_agent(orchestrator):
    """Test getting nonexistent agent returns None."""
    agent = await orchestrator.get_agent("nonexistent")
    assert agent is None


@pytest.mark.asyncio
async def test_list_agents(orchestrator):
    """Test listing agents."""
    await orchestrator.spawn_agent(task="Task 1")
    await orchestrator.spawn_agent(task="Task 2")

    agents = await orchestrator.list_agents()
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_list_agents_by_status(orchestrator):
    """Test listing agents filtered by status."""
    await orchestrator.spawn_agent(task="Task 1")

    idle_agents = await orchestrator.list_agents("idle")
    assert len(idle_agents) == 1

    running_agents = await orchestrator.list_agents("running")
    assert len(running_agents) == 0


@pytest.mark.asyncio
async def test_pause_agent(orchestrator):
    """Test pausing an agent."""
    agent = await orchestrator.spawn_agent(task="Test task")

    # Agent must be in _agents dict to be paused
    orchestrator._agents[agent["id"]] = Agent(
        agent_id=agent["id"],
        task="Test task",
        model="test-model",
        supervised=True,
        timeout_mode="block",
        workdir="/tmp",
        orchestrator=orchestrator,
    )

    success = await orchestrator.pause_agent(agent["id"])
    assert success

    updated = await orchestrator.get_agent(agent["id"])
    assert updated["status"] == "paused"


@pytest.mark.asyncio
async def test_pause_nonexistent_agent(orchestrator):
    """Test pausing nonexistent agent fails."""
    success = await orchestrator.pause_agent("nonexistent")
    assert not success


@pytest.mark.asyncio
async def test_resume_agent(orchestrator):
    """Test resuming a paused agent."""
    agent = await orchestrator.spawn_agent(task="Test task")

    # First pause it
    orchestrator._agents[agent["id"]] = Agent(
        agent_id=agent["id"],
        task="Test task",
        model="test-model",
        supervised=True,
        timeout_mode="block",
        workdir="/tmp",
        orchestrator=orchestrator,
    )
    await orchestrator.pause_agent(agent["id"])

    # Then resume
    success = await orchestrator.resume_agent(agent["id"])
    assert success

    updated = await orchestrator.get_agent(agent["id"])
    assert updated["status"] == "running"


@pytest.mark.asyncio
async def test_terminate_agent(orchestrator):
    """Test terminating an agent."""
    agent = await orchestrator.spawn_agent(task="Test task")

    orchestrator._agents[agent["id"]] = Agent(
        agent_id=agent["id"],
        task="Test task",
        model="test-model",
        supervised=True,
        timeout_mode="block",
        workdir="/tmp",
        orchestrator=orchestrator,
    )

    success = await orchestrator.terminate_agent(agent["id"])
    assert success

    updated = await orchestrator.get_agent(agent["id"])
    assert updated["status"] == "terminated"
    assert agent["id"] not in orchestrator._agents


@pytest.mark.asyncio
async def test_nudge_agent(orchestrator):
    """Test nudging an agent with a message."""
    agent = await orchestrator.spawn_agent(task="Test task")

    agent_obj = Agent(
        agent_id=agent["id"],
        task="Test task",
        model="test-model",
        supervised=True,
        timeout_mode="block",
        workdir="/tmp",
        orchestrator=orchestrator,
    )
    orchestrator._agents[agent["id"]] = agent_obj

    success = await orchestrator.nudge_agent(agent["id"], "Hey, update?")
    assert success
    assert len(agent_obj.messages) == 1
    assert agent_obj.messages[0]["content"] == "Hey, update?"


@pytest.mark.asyncio
async def test_get_status(orchestrator):
    """Test getting orchestrator status."""
    await orchestrator.spawn_agent(task="Task 1")

    status = await orchestrator.get_status()

    assert "running" in status
    assert "agents" in status
    assert "scheduler" in status
    assert status["agents"]["total"] == 1


@pytest.mark.asyncio
async def test_run_agent_completion(orchestrator, test_config):
    """Test running an agent to completion."""
    agent_data = await orchestrator.spawn_agent(task="Say hello")

    agent = Agent(
        agent_id=agent_data["id"],
        task="Say hello",
        model="test-model",
        supervised=False,
        timeout_mode="block",
        workdir=str(test_config.data_dir),
        orchestrator=orchestrator,
    )
    orchestrator._agents[agent_data["id"]] = agent

    # Mock Claude to return a simple response
    mock_response = Response(
        content="Hello!",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    with patch.object(orchestrator.claude, "send_message", return_value=mock_response):
        await orchestrator.run_agent(agent, "task123")

    updated = await orchestrator.get_agent(agent_data["id"])
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_run_agent_with_tool_use(orchestrator, test_config):
    """Test running agent that uses tools."""
    agent_data = await orchestrator.spawn_agent(task="Read a file")

    agent = Agent(
        agent_id=agent_data["id"],
        task="Read a file",
        model="test-model",
        supervised=False,
        timeout_mode="block",
        workdir=str(test_config.data_dir),
        orchestrator=orchestrator,
    )
    orchestrator._agents[agent_data["id"]] = agent

    # Create a test file
    test_file = test_config.data_dir / "test.txt"
    test_file.write_text("test content")

    # First response: tool use
    tool_response = Response(
        content="I'll read the file.",
        tool_uses=[ToolUse(id="tu1", name="read_file", input={"path": str(test_file)})],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    # Second response: completion
    final_response = Response(
        content="The file contains: test content",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 20, "output_tokens": 10},
    )

    call_count = 0

    async def mock_send(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    with patch.object(orchestrator.claude, "send_message", side_effect=mock_send):
        await orchestrator.run_agent(agent, "task123")

    updated = await orchestrator.get_agent(agent_data["id"])
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_run_agent_failure(orchestrator, test_config):
    """Test agent failure handling."""
    agent_data = await orchestrator.spawn_agent(task="Fail")

    agent = Agent(
        agent_id=agent_data["id"],
        task="Fail",
        model="test-model",
        supervised=False,
        timeout_mode="block",
        workdir=str(test_config.data_dir),
        orchestrator=orchestrator,
    )
    orchestrator._agents[agent_data["id"]] = agent

    with patch.object(orchestrator.claude, "send_message", side_effect=Exception("API Error")):
        await orchestrator.run_agent(agent, "task123")

    updated = await orchestrator.get_agent(agent_data["id"])
    assert updated["status"] == "failed"
    assert "API Error" in updated["error"]


@pytest.mark.asyncio
async def test_execute_bash(orchestrator, test_config):
    """Test bash execution."""
    result = await orchestrator._execute_bash("echo hello", str(test_config.data_dir))
    assert "hello" in result


@pytest.mark.asyncio
async def test_execute_bash_timeout(orchestrator, test_config):
    """Test bash execution timeout."""
    result = await orchestrator._execute_bash("sleep 100", str(test_config.data_dir))
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_read_file(orchestrator, test_config):
    """Test file reading."""
    test_file = test_config.data_dir / "test.txt"
    test_file.write_text("file content")

    result = await orchestrator._read_file(str(test_file), str(test_config.data_dir))
    assert result == "file content"


@pytest.mark.asyncio
async def test_read_file_not_found(orchestrator, test_config):
    """Test reading nonexistent file."""
    result = await orchestrator._read_file("nonexistent.txt", str(test_config.data_dir))
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_write_file(orchestrator, test_config):
    """Test file writing."""
    test_path = test_config.data_dir / "output.txt"

    workdir = str(test_config.data_dir)
    result = await orchestrator._write_file(str(test_path), "new content", workdir)

    assert "successfully" in result.lower()
    assert test_path.read_text() == "new content"


@pytest.mark.asyncio
async def test_search_files(orchestrator, test_config):
    """Test file search."""
    (test_config.data_dir / "file1.txt").write_text("a")
    (test_config.data_dir / "file2.txt").write_text("b")

    result = await orchestrator._search_files("*.txt", ".", str(test_config.data_dir))

    assert "file1.txt" in result
    assert "file2.txt" in result


@pytest.mark.asyncio
async def test_approval_auto_approve_without_callback(orchestrator, test_config):
    """Test that actions auto-approve when no callback is set."""
    agent_data = await orchestrator.spawn_agent(task="Test")

    agent = Agent(
        agent_id=agent_data["id"],
        task="Test",
        model="test-model",
        supervised=True,
        timeout_mode="block",
        workdir=str(test_config.data_dir),
        orchestrator=orchestrator,
    )

    # No approval callback set, should auto-approve
    result = await orchestrator._request_approval(agent, "bash", {"command": "ls"}, "task1")
    assert result is True


@pytest.mark.asyncio
async def test_pending_approvals(orchestrator):
    """Test getting pending approvals."""
    # Create an agent first (FK constraint)
    agent = await orchestrator.spawn_agent(task="Test")

    await orchestrator.db.create_approval(
        approval_id="appr1",
        agent_id=agent["id"],
        action_type="bash",
        action_details={"command": "rm -rf /"},
    )

    pending = await orchestrator.get_pending_approvals()
    assert len(pending) == 1
    assert pending[0]["id"] == "appr1"


@pytest.mark.asyncio
async def test_approve_action(orchestrator):
    """Test approving an action."""
    agent = await orchestrator.spawn_agent(task="Test")

    await orchestrator.db.create_approval(
        approval_id="appr1",
        agent_id=agent["id"],
        action_type="bash",
        action_details={"command": "ls"},
    )

    success = await orchestrator.approve("appr1", approved=True)
    assert success

    approval = await orchestrator.db.get_approval("appr1")
    assert approval["status"] == "approved"


@pytest.mark.asyncio
async def test_reject_action(orchestrator):
    """Test rejecting an action."""
    agent = await orchestrator.spawn_agent(task="Test")

    await orchestrator.db.create_approval(
        approval_id="appr1",
        agent_id=agent["id"],
        action_type="bash",
        action_details={"command": "rm -rf /"},
    )

    success = await orchestrator.approve("appr1", approved=False)
    assert success

    approval = await orchestrator.db.get_approval("appr1")
    assert approval["status"] == "rejected"


# =============================================================================
# Agent Class Tests
# =============================================================================


class TestAgent:
    """Tests for Agent class methods."""

    @pytest.fixture
    def mock_orchestrator(self, orchestrator):
        """Return the orchestrator fixture."""
        return orchestrator

    @pytest.fixture
    def agent(self, mock_orchestrator, test_config):
        """Create an Agent for testing."""
        return Agent(
            agent_id="test123",
            task="Test task",
            model="test-model",
            supervised=True,
            timeout_mode="block",
            workdir=str(test_config.data_dir),
            orchestrator=mock_orchestrator,
        )

    def test_runtime_seconds_not_started(self, agent):
        """Test runtime_seconds returns 0 when not started."""
        assert agent.runtime_seconds == 0

    def test_runtime_seconds_after_start(self, agent):
        """Test runtime_seconds increases after start."""
        agent.start()
        import time
        time.sleep(0.01)
        assert agent.runtime_seconds > 0

    def test_should_send_progress_report_disabled(self, agent):
        """Test progress report disabled when interval is 0."""
        agent.start()
        assert not agent.should_send_progress_report(0)

    def test_should_send_progress_report_not_started(self, agent):
        """Test progress report when agent not started."""
        assert not agent.should_send_progress_report(1)

    def test_should_send_progress_report_too_early(self, agent):
        """Test progress report before interval elapsed."""
        agent.start()
        assert not agent.should_send_progress_report(60)  # 60 min not elapsed

    def test_mark_report_sent(self, agent):
        """Test mark_report_sent updates time and clears tools."""
        agent.start()
        agent._recent_tools = ["bash: ls", "read_file: test.txt"]
        agent.mark_report_sent()
        assert agent._last_report_time is not None
        assert agent._recent_tools == []

    def test_add_tool_call(self, agent):
        """Test add_tool_call tracks tools and resets stuck counter."""
        agent._turns_since_tool = 5
        agent.add_tool_call("bash", "ls -la")
        assert "bash: ls -la" in agent._recent_tools
        assert agent._turns_since_tool == 0

    def test_increment_turns_since_tool(self, agent):
        """Test increment_turns_since_tool."""
        assert agent._turns_since_tool == 0
        agent.increment_turns_since_tool()
        assert agent._turns_since_tool == 1

    def test_is_stuck_disabled(self, agent):
        """Test is_stuck with threshold 0."""
        agent._turns_since_tool = 100
        assert not agent.is_stuck(0)

    def test_is_stuck_true(self, agent):
        """Test is_stuck returns true when threshold exceeded."""
        agent._turns_since_tool = 5
        assert agent.is_stuck(5)

    def test_is_stuck_false(self, agent):
        """Test is_stuck returns false below threshold."""
        agent._turns_since_tool = 4
        assert not agent.is_stuck(5)

    def test_add_tokens(self, agent):
        """Test add_tokens accumulates."""
        agent.add_tokens(100, 50)
        agent.add_tokens(200, 100)
        assert agent._total_input_tokens == 300
        assert agent._total_output_tokens == 150
        assert agent.total_tokens == 450

    def test_should_alert_token_burn_disabled(self, agent):
        """Test token burn alert disabled when threshold 0."""
        agent.add_tokens(1000000, 1000000)
        assert not agent.should_alert_token_burn(0)

    def test_should_alert_token_burn_below_threshold(self, agent):
        """Test token burn alert not triggered below threshold."""
        agent.add_tokens(100, 50)
        assert not agent.should_alert_token_burn(1000)

    def test_should_alert_token_burn_triggers_once(self, agent):
        """Test token burn alert triggers once."""
        agent.add_tokens(500, 600)
        assert agent.should_alert_token_burn(1000)  # First time - True
        assert not agent.should_alert_token_burn(1000)  # Second time - False

    def test_should_alert_stuck_not_stuck(self, agent):
        """Test stuck alert when not stuck."""
        agent._turns_since_tool = 2
        assert not agent.should_alert_stuck(5)

    def test_should_alert_stuck_triggers_once(self, agent):
        """Test stuck alert triggers once."""
        agent._turns_since_tool = 5
        assert agent.should_alert_stuck(5)  # First time - True
        assert not agent.should_alert_stuck(5)  # Second time - False (already sent)

    def test_get_progress_summary_basic(self, agent):
        """Test progress summary without tool calls."""
        agent.start()
        agent._turn_count = 5
        summary = agent.get_progress_summary()
        assert "Turn 5" in summary
        assert "running for 0m" in summary

    def test_get_progress_summary_with_tools(self, agent):
        """Test progress summary with recent tool calls."""
        agent.start()
        agent._turn_count = 10
        agent._recent_tools = ["bash: ls", "read_file: test.py", "write_file: out.txt"]
        summary = agent.get_progress_summary()
        assert "Turn 10" in summary
        assert "Recent:" in summary
        assert "bash: ls" in summary


# =============================================================================
# Orchestrator Callback Tests
# =============================================================================


class TestOrchestratorCallbacks:
    """Tests for orchestrator callback methods."""

    @pytest.mark.asyncio
    async def test_set_notify_callback(self, orchestrator):
        """Test setting notify callback."""
        callback = lambda agent_id, msg: None
        orchestrator.set_notify_callback(callback)
        assert orchestrator._notify_callback == callback

    @pytest.mark.asyncio
    async def test_set_approval_callback(self, orchestrator):
        """Test setting approval callback."""
        callback = lambda approval_id, details: asyncio.Future()
        orchestrator.set_approval_callback(callback)
        assert orchestrator._approval_callback == callback

    @pytest.mark.asyncio
    async def test_set_cancel_approval_callback(self, orchestrator):
        """Test setting cancel approval callback."""
        callback = lambda approval_id: None
        orchestrator.set_cancel_approval_callback(callback)
        assert orchestrator._cancel_approval_callback == callback

    @pytest.mark.asyncio
    async def test_notify_with_callback(self, orchestrator):
        """Test notify calls callback."""
        called = {}
        def callback(agent_id, msg):
            called["agent_id"] = agent_id
            called["msg"] = msg
        orchestrator.set_notify_callback(callback)
        await orchestrator.notify("agent1", "Hello")
        assert called["agent_id"] == "agent1"
        assert called["msg"] == "Hello"

    @pytest.mark.asyncio
    async def test_notify_without_callback(self, orchestrator):
        """Test notify without callback does not raise."""
        await orchestrator.notify("agent1", "Hello")  # Should not raise

    @pytest.mark.asyncio
    async def test_notify_callback_error(self, orchestrator):
        """Test notify handles callback errors."""
        def bad_callback(agent_id, msg):
            raise RuntimeError("Callback failed")
        orchestrator.set_notify_callback(bad_callback)
        # Should not raise
        await orchestrator.notify("agent1", "Hello")


# =============================================================================
# Conversation Truncation Tests
# =============================================================================


class TestConversationTruncation:
    """Tests for conversation truncation."""

    @pytest.mark.asyncio
    async def test_truncate_short_conversation(self, orchestrator):
        """Test short conversation not truncated."""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        result = orchestrator._truncate_conversation(messages)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_truncate_long_conversation(self, orchestrator):
        """Test long conversation is truncated."""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(200)]
        result = orchestrator._truncate_conversation(messages)
        # First message + truncation notice + recent messages
        assert len(result) <= orchestrator.config.max_conversation_messages + 1
        assert result[0]["content"] == "msg0"  # First message preserved
        assert "truncated" in result[1]["content"].lower()


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostEstimation:
    """Tests for cost estimation."""

    @pytest.mark.asyncio
    async def test_estimate_cost_sonnet(self, orchestrator, test_config):
        """Test cost estimation for sonnet model."""
        agent = Agent(
            agent_id="test1",
            task="Test",
            model="claude-sonnet-4-20250514",
            supervised=False,
            timeout_mode="block",
            workdir=str(test_config.data_dir),
            orchestrator=orchestrator,
        )
        agent.add_tokens(1000000, 500000)
        cost = orchestrator._estimate_cost(agent)
        # Input: 1M * $3 = $3, Output: 0.5M * $15 = $7.50
        assert cost == "10.5000"

    @pytest.mark.asyncio
    async def test_estimate_cost_unknown_model(self, orchestrator, test_config):
        """Test cost estimation uses default for unknown model."""
        agent = Agent(
            agent_id="test1",
            task="Test",
            model="unknown-model",
            supervised=False,
            timeout_mode="block",
            workdir=str(test_config.data_dir),
            orchestrator=orchestrator,
        )
        agent.add_tokens(1000000, 500000)
        cost = orchestrator._estimate_cost(agent)
        # Uses default rates (sonnet pricing)
        assert cost == "10.5000"


# =============================================================================
# Task Validation Tests
# =============================================================================


class TestTaskValidation:
    """Tests for task validation."""

    @pytest.mark.asyncio
    async def test_spawn_agent_empty_task(self, orchestrator):
        """Test spawning with empty task raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await orchestrator.spawn_agent(task="   ")

    @pytest.mark.asyncio
    async def test_spawn_agent_task_too_long(self, orchestrator):
        """Test spawning with task too long raises error."""
        long_task = "x" * (orchestrator.config.max_task_length + 1)
        with pytest.raises(ValueError, match="Task too long"):
            await orchestrator.spawn_agent(task=long_task)


# =============================================================================
# Agent Control Extended Tests
# =============================================================================


class TestAgentControlExtended:
    """Extended tests for agent control operations."""

    @pytest.mark.asyncio
    async def test_resume_agent_not_paused(self, orchestrator):
        """Test resuming agent that's not paused fails."""
        agent = await orchestrator.spawn_agent(task="Test task")
        # Agent is idle, not paused
        success = await orchestrator.resume_agent(agent["id"])
        assert not success

    @pytest.mark.asyncio
    async def test_nudge_nonexistent_agent(self, orchestrator):
        """Test nudging nonexistent agent fails."""
        success = await orchestrator.nudge_agent("nonexistent", "Hello")
        assert not success

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_agent(self, orchestrator):
        """Test terminating nonexistent agent fails."""
        success = await orchestrator.terminate_agent("nonexistent")
        assert not success


# =============================================================================
# Tool Execution Extended Tests
# =============================================================================


class TestToolExecutionExtended:
    """Extended tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, orchestrator, test_config):
        """Test executing unknown tool raises error."""
        agent = Agent(
            agent_id="test1",
            task="Test",
            model="test-model",
            supervised=False,
            timeout_mode="block",
            workdir=str(test_config.data_dir),
            orchestrator=orchestrator,
        )
        with pytest.raises(ValueError, match="Unknown tool"):
            await orchestrator._execute_tool(agent, "unknown_tool", {}, "task1")

    @pytest.mark.asyncio
    async def test_summarize_tool_input_search(self, orchestrator):
        """Test tool input summary for search_files."""
        summary = orchestrator._summarize_tool_input("search_files", {"pattern": "*.py"})
        assert summary == "*.py"

    @pytest.mark.asyncio
    async def test_summarize_tool_input_unknown(self, orchestrator):
        """Test tool input summary for unknown tool."""
        summary = orchestrator._summarize_tool_input("some_tool", {})
        assert summary == "some_tool"

    @pytest.mark.asyncio
    async def test_execute_bash_with_stderr(self, orchestrator, test_config):
        """Test bash execution with stderr output."""
        result = await orchestrator._execute_bash(
            "echo error >&2",
            str(test_config.data_dir),
        )
        assert "STDERR" in result or "error" in result

    @pytest.mark.asyncio
    async def test_execute_bash_nonzero_exit(self, orchestrator, test_config):
        """Test bash execution with non-zero exit code."""
        result = await orchestrator._execute_bash(
            "exit 1",
            str(test_config.data_dir),
        )
        assert "Exit code: 1" in result

    @pytest.mark.asyncio
    async def test_execute_bash_exception(self, orchestrator, test_config):
        """Test bash execution with exception."""
        result = await orchestrator._execute_bash(
            "invalid_command_xyz",
            str(test_config.data_dir),
        )
        # Should return error output, not raise
        assert "invalid_command_xyz" in result.lower() or "not found" in result.lower() or "exit code" in result.lower()

    @pytest.mark.asyncio
    async def test_read_file_truncated(self, orchestrator, test_config):
        """Test reading large file is truncated."""
        # Create a large file
        large_file = test_config.data_dir / "large.txt"
        large_file.write_text("x" * 200000)
        result = await orchestrator._read_file(str(large_file), str(test_config.data_dir))
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_read_file_error(self, orchestrator, test_config):
        """Test reading directory returns error."""
        result = await orchestrator._read_file(str(test_config.data_dir), str(test_config.data_dir))
        assert "error" in result.lower() or "not found" in result.lower() or "directory" in result.lower()

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, orchestrator, test_config):
        """Test write_file creates parent directories."""
        deep_path = test_config.data_dir / "deep" / "nested" / "file.txt"
        result = await orchestrator._write_file(str(deep_path), "content", str(test_config.data_dir))
        assert "successfully" in result.lower()
        assert deep_path.exists()

    @pytest.mark.asyncio
    async def test_search_files_no_matches(self, orchestrator, test_config):
        """Test search_files with no matches."""
        result = await orchestrator._search_files(
            "*.nonexistent",
            ".",
            str(test_config.data_dir),
        )
        assert "no files found" in result.lower()


# =============================================================================
# Search and Cost Tests
# =============================================================================


class TestSearchAndCost:
    """Tests for search and cost methods."""

    @pytest.mark.asyncio
    async def test_search_agents(self, orchestrator):
        """Test searching agents."""
        await orchestrator.spawn_agent(task="Find the bug in login", name="bugfixer")
        await orchestrator.spawn_agent(task="Write unit tests")

        results = await orchestrator.search_agents("bug")
        assert len(results) >= 1
        assert any("bug" in r.get("task", "").lower() or "bug" in (r.get("name") or "").lower() for r in results)

    @pytest.mark.asyncio
    async def test_get_agent_cost_active(self, orchestrator, test_config):
        """Test getting cost for active agent."""
        agent_data = await orchestrator.spawn_agent(task="Test")
        agent = orchestrator._agents.get(agent_data["id"])
        agent.add_tokens(1000, 500)

        result = orchestrator.get_agent_cost(agent_data["id"])
        assert result is not None
        input_tokens, output_tokens, cost = result
        assert input_tokens == 1000
        assert output_tokens == 500

    @pytest.mark.asyncio
    async def test_get_agent_cost_nonexistent(self, orchestrator):
        """Test getting cost for nonexistent agent."""
        result = orchestrator.get_agent_cost("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_cost_from_db(self, orchestrator):
        """Test getting cost from database."""
        agent_data = await orchestrator.spawn_agent(task="Test")
        # Add tokens to database
        await orchestrator.db.add_tokens(agent_data["id"], 5000, 2000)

        result = await orchestrator.get_agent_cost_from_db(agent_data["id"])
        assert result is not None
        input_tokens, output_tokens, cost = result
        assert input_tokens == 5000
        assert output_tokens == 2000

    @pytest.mark.asyncio
    async def test_get_agent_cost_from_db_nonexistent(self, orchestrator):
        """Test getting cost from db for nonexistent agent."""
        result = await orchestrator.get_agent_cost_from_db("nonexistent")
        assert result is None


# =============================================================================
# Approval Extended Tests
# =============================================================================


class TestApprovalExtended:
    """Extended tests for approval handling."""

    @pytest.mark.asyncio
    async def test_approve_nonexistent(self, orchestrator):
        """Test approving nonexistent approval fails."""
        success = await orchestrator.approve("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_approve_already_resolved(self, orchestrator):
        """Test approving already resolved approval fails."""
        agent = await orchestrator.spawn_agent(task="Test")
        await orchestrator.db.create_approval(
            approval_id="appr1",
            agent_id=agent["id"],
            action_type="bash",
            action_details={"command": "ls"},
        )
        # Resolve it first
        await orchestrator.db.resolve_approval("appr1", "approved", "user")

        # Try to approve again
        success = await orchestrator.approve("appr1")
        assert not success
