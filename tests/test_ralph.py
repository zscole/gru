"""Tests for Ralph Wiggum iterative loop functionality."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from gru.cli import cli
from gru.orchestrator import Orchestrator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator with Ralph methods."""
    mock = MagicMock()
    mock.spawn_ralph_loop = AsyncMock(
        return_value={
            "id": "ralph-123",
            "task": "test task",
            "model": "claude-3-5-sonnet-20241022",
            "status": "running",
        }
    )
    mock.cancel_ralph_loop = AsyncMock(return_value=True)
    mock.get_agent = AsyncMock(
        return_value={
            "id": "ralph-123",
            "task": "test task",
            "status": "running",
            "model": "claude-3-5-sonnet-20241022",
        }
    )
    return mock


@pytest.fixture
def mock_db():
    """Create mock database."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.get_conversation = AsyncMock(return_value=[])
    mock.create_agent = AsyncMock(
        return_value={
            "id": "ralph-123",
            "task": "test task",
            "status": "running",
        }
    )
    mock.update_agent = AsyncMock()
    mock.get_agent = AsyncMock(
        return_value={
            "id": "ralph-123",
            "task": "test task",
            "status": "completed",
        }
    )
    return mock


# =============================================================================
# CLI Tests
# =============================================================================


def test_ralph_command_basic(runner, mock_orchestrator, mock_db):
    """Test basic ralph command."""
    with (
        patch("gru.cli.get_orchestrator", return_value=mock_orchestrator),
        patch("gru.cli.run_async", side_effect=lambda x: asyncio.run(x) if asyncio.iscoroutine(x) else x),
        patch("gru.cli.Database", return_value=mock_db),
    ):
        result = runner.invoke(cli, ["ralph", "Write a hello world program"])

        assert result.exit_code == 0
        assert "Ralph loop started: ralph-123" in result.output
        assert "Task: Write a hello world program" in result.output
        assert "Max iterations: 20" in result.output  # Default
        mock_orchestrator.spawn_ralph_loop.assert_called_once()


def test_ralph_command_with_options(runner, mock_orchestrator, mock_db):
    """Test ralph command with all options."""
    with (
        patch("gru.cli.get_orchestrator", return_value=mock_orchestrator),
        patch("gru.cli.run_async", side_effect=lambda x: asyncio.run(x) if asyncio.iscoroutine(x) else x),
        patch("gru.cli.Database", return_value=mock_db),
    ):
        result = runner.invoke(
            cli,
            [
                "ralph",
                "Write tests",
                "--max-iterations",
                "10",
                "--completion-promise",
                "DONE",
                "--name",
                "test-ralph",
                "--model",
                "claude-3-5-sonnet-20241022",
                "--priority",
                "high",
            ],
        )

        assert result.exit_code == 0
        assert "Max iterations: 10" in result.output
        assert "Completion promise: DONE" in result.output
        assert "Priority: high" in result.output

        mock_orchestrator.spawn_ralph_loop.assert_called_once_with(
            task="Write tests",
            max_iterations=10,
            completion_promise="DONE",
            name="test-ralph",
            model="claude-3-5-sonnet-20241022",
            priority="high",
        )


def test_cancel_ralph_command(runner, mock_orchestrator, mock_db):
    """Test cancel-ralph command."""
    with (
        patch("gru.cli.get_orchestrator", return_value=mock_orchestrator),
        patch("gru.cli.run_async", side_effect=lambda x: asyncio.run(x) if asyncio.iscoroutine(x) else x),
        patch("gru.cli.Database", return_value=mock_db),
    ):
        result = runner.invoke(cli, ["cancel-ralph", "ralph-123"])

        assert result.exit_code == 0
        assert "Ralph loop ralph-123 cancelled" in result.output
        mock_orchestrator.cancel_ralph_loop.assert_called_once_with("ralph-123")


def test_cancel_ralph_command_failure(runner, mock_orchestrator, mock_db):
    """Test cancel-ralph command when cancellation fails."""
    mock_orchestrator.cancel_ralph_loop = AsyncMock(return_value=False)

    with (
        patch("gru.cli.get_orchestrator", return_value=mock_orchestrator),
        patch("gru.cli.run_async", side_effect=lambda x: asyncio.run(x) if asyncio.iscoroutine(x) else x),
        patch("gru.cli.Database", return_value=mock_db),
    ):
        result = runner.invoke(cli, ["cancel-ralph", "ralph-123"])

        assert result.exit_code == 1
        assert "Could not cancel Ralph loop ralph-123" in result.output


# =============================================================================
# Orchestrator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_spawn_ralph_loop():
    """Test spawning a Ralph loop in the orchestrator."""
    # Create mock dependencies
    config = MagicMock()
    config.max_task_length = 10000
    config.default_model = "claude-3-5-sonnet-20241022"
    config.default_workdir = "/tmp/work"
    config.enable_worktrees = False

    db = MagicMock()
    db.create_agent = AsyncMock(return_value={"id": "ralph-abc", "task": "test", "status": "running"})
    db.create_task = AsyncMock()

    secrets = MagicMock()
    scheduler = MagicMock()
    scheduler.enqueue = AsyncMock()

    # Create orchestrator
    orchestrator = Orchestrator(config, db, secrets)
    orchestrator.scheduler = scheduler

    # Test spawn_ralph_loop
    with patch("gru.orchestrator.uuid.uuid4", return_value="12345678-1234-5678-1234-567812345678"):
        result = await orchestrator.spawn_ralph_loop(
            task="Create a web server",
            max_iterations=5,
            completion_promise="SERVER_READY",
            name="my-ralph",
            model="claude-3-5-sonnet-20241022",
            priority="high",
        )

    # Verify agent was created
    assert result["id"] == "ralph-abc"

    # Verify Ralph metadata was stored (using the actual agent ID returned)
    assert "ralph-abc" in orchestrator._ralph_loops
    ralph_meta = orchestrator._ralph_loops["ralph-abc"]
    assert ralph_meta["is_ralph_loop"] is True
    assert ralph_meta["max_iterations"] == 5
    assert ralph_meta["current_iteration"] == 1
    assert ralph_meta["completion_promise"] == "SERVER_READY"
    assert ralph_meta["original_task"] == "Create a web server"

    # Verify agent creation was called with Ralph task
    db.create_agent.assert_called_once()
    call_args = db.create_agent.call_args[1]
    assert "RALPH LOOP TASK" in call_args["task"]
    assert "Create a web server" in call_args["task"]
    assert "Current iteration: 1/5" in call_args["task"]
    assert call_args["name"] == "my-ralph"
    assert call_args["supervised"] is False  # Ralph loops are unsupervised


@pytest.mark.asyncio
async def test_ralph_loop_iteration():
    """Test Ralph loop iteration behavior."""
    # Create mock dependencies
    config = MagicMock()
    config.max_task_length = 10000
    config.default_model = "claude-3-5-sonnet-20241022"
    config.default_workdir = "/tmp/work"
    config.enable_worktrees = False

    db = MagicMock()
    db.get_conversation = AsyncMock(
        return_value=[
            {"role": "user", "content": "Create a calculator"},
            {"role": "assistant", "content": "I've created a basic calculator."},
        ]
    )

    secrets = MagicMock()

    # Create orchestrator
    orchestrator = Orchestrator(config, db, secrets)

    # Set up Ralph loop metadata
    orchestrator._ralph_loops["test-123"] = {
        "is_ralph_loop": True,
        "max_iterations": 3,
        "current_iteration": 1,
        "completion_promise": "CALCULATOR_DONE",
        "original_task": "Create a calculator",
    }

    # Mock get_agent to return completed status
    orchestrator.get_agent = AsyncMock(return_value={"id": "test-123", "status": "completed", "name": "ralph-test"})

    # Mock spawn_agent
    orchestrator.spawn_agent = AsyncMock()

    # Mock notify
    orchestrator.notify = AsyncMock()

    # Run the monitor loop with a short timeout
    monitor_task = asyncio.create_task(orchestrator._monitor_ralph_loop("test-123"))

    # Wait a bit for the loop to process
    await asyncio.sleep(6)

    # Cancel the monitor task
    monitor_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await monitor_task

    # Verify next iteration was spawned
    orchestrator.spawn_agent.assert_called()
    call_args = orchestrator.spawn_agent.call_args[1]
    assert "RALPH LOOP CONTINUATION" in call_args["task"]
    # The metadata shows current_iteration was incremented to 2
    assert orchestrator._ralph_loops["test-123"]["current_iteration"] == 2
    assert "Create a calculator" in call_args["task"]
    assert "PREVIOUS WORK SUMMARY" in call_args["task"]


@pytest.mark.asyncio
async def test_ralph_loop_completion_promise():
    """Test Ralph loop completion via promise detection."""
    # Create mock dependencies
    config = MagicMock()
    db = MagicMock()
    db.get_conversation = AsyncMock(
        return_value=[
            {"role": "user", "content": "Create a calculator"},
            {"role": "assistant", "content": "Calculator created. CALCULATOR_DONE"},
        ]
    )

    secrets = MagicMock()

    # Create orchestrator
    orchestrator = Orchestrator(config, db, secrets)

    # Set up Ralph loop metadata
    orchestrator._ralph_loops["test-456"] = {
        "is_ralph_loop": True,
        "max_iterations": 10,
        "current_iteration": 2,
        "completion_promise": "CALCULATOR_DONE",
        "original_task": "Create a calculator",
    }

    # Mock get_agent to return completed status
    orchestrator.get_agent = AsyncMock(return_value={"id": "test-456", "status": "completed"})

    # Mock notify
    orchestrator.notify = AsyncMock()

    # Run monitor loop - should detect completion promise
    await orchestrator._monitor_ralph_loop("test-456")
    await asyncio.sleep(6)

    # Verify Ralph loop was marked complete
    assert "test-456" not in orchestrator._ralph_loops

    # Verify completion notification
    orchestrator.notify.assert_called_with("test-456", "Ralph loop completed: CALCULATOR_DONE detected")


@pytest.mark.asyncio
async def test_cancel_ralph_loop():
    """Test cancelling a Ralph loop."""
    # Create mock dependencies
    config = MagicMock()
    db = MagicMock()
    secrets = MagicMock()

    # Create orchestrator
    orchestrator = Orchestrator(config, db, secrets)

    # Set up Ralph loop
    orchestrator._ralph_loops["test-789"] = {
        "is_ralph_loop": True,
        "max_iterations": 10,
        "current_iteration": 3,
    }

    # Mock terminate_agent
    orchestrator.terminate_agent = AsyncMock(return_value=True)

    # Mock notify
    orchestrator.notify = AsyncMock()

    # Cancel the Ralph loop
    result = await orchestrator.cancel_ralph_loop("test-789")

    assert result is True
    assert "test-789" not in orchestrator._ralph_loops
    orchestrator.terminate_agent.assert_called_once_with("test-789")
    orchestrator.notify.assert_called_once_with("test-789", "Ralph loop test-789 cancelled")


@pytest.mark.asyncio
async def test_cancel_nonexistent_ralph_loop():
    """Test cancelling a non-existent Ralph loop."""
    # Create mock dependencies
    config = MagicMock()
    db = MagicMock()
    secrets = MagicMock()

    # Create orchestrator
    orchestrator = Orchestrator(config, db, secrets)

    # Try to cancel non-existent loop
    result = await orchestrator.cancel_ralph_loop("nonexistent")

    assert result is False
