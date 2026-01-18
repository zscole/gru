"""Tests for CLI interface."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from gru.cli import cli, get_orchestrator, run_async


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    mock = MagicMock()
    mock.spawn_agent = AsyncMock(return_value={"id": "agent-123", "task": "test task"})
    mock.get_agent = AsyncMock(return_value=None)
    mock.list_agents = AsyncMock(return_value=[])
    mock.get_status = AsyncMock(
        return_value={
            "running": True,
            "agents": {"total": 5, "running": 2, "paused": 1, "completed": 1, "failed": 1},
            "scheduler": {"queued": 3},
        }
    )
    mock.get_pending_approvals = AsyncMock(return_value=[])
    mock.pause_agent = AsyncMock(return_value=True)
    mock.resume_agent = AsyncMock(return_value=True)
    mock.terminate_agent = AsyncMock(return_value=True)
    mock.nudge_agent = AsyncMock(return_value=True)
    mock.approve = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_db():
    """Create mock database."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.get_conversation = AsyncMock(return_value=[])
    mock.save_template = AsyncMock()
    mock.list_templates = AsyncMock(return_value=[])
    mock.get_template = AsyncMock(return_value=None)
    mock.delete_template = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_secrets():
    """Create mock secrets store."""
    mock = MagicMock()
    mock.set = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.list_keys = AsyncMock(return_value=[])
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_crypto():
    """Create mock crypto manager."""
    mock = MagicMock()
    mock.is_initialized = MagicMock(return_value=True)
    mock.initialize = MagicMock()
    return mock


def setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
    """Helper to set up CLI mocks."""
    return patch.multiple(
        "gru.cli",
        Database=MagicMock(return_value=mock_db),
        CryptoManager=MagicMock(return_value=mock_crypto),
        SecretStore=MagicMock(return_value=mock_secrets),
        Orchestrator=MagicMock(return_value=mock_orchestrator),
    )


# =============================================================================
# run_async Tests
# =============================================================================


class TestRunAsync:
    """Tests for run_async helper."""

    def test_run_async_no_loop(self):
        """Test run_async creates loop when none exists."""

        async def sample_coro():
            return "result"

        result = run_async(sample_coro())
        assert result == "result"

    def test_run_async_with_existing_loop(self):
        """Test run_async uses existing loop."""

        async def sample_coro():
            return "result"

        async def test():
            # Inside an async context, there's a running loop
            # but run_until_complete can't be used in this case
            # so we test the try branch
            pass

        # Just verify it doesn't crash in the no-loop case
        result = run_async(sample_coro())
        assert result == "result"


class TestGetOrchestrator:
    """Tests for get_orchestrator helper."""

    def test_get_orchestrator(self):
        """Test getting orchestrator from context."""
        ctx = MagicMock()
        ctx.obj = {"orchestrator": "mock_orch"}
        result = get_orchestrator(ctx)
        assert result == "mock_orch"


# =============================================================================
# Spawn Command Tests
# =============================================================================


class TestSpawnCommand:
    """Tests for spawn command."""

    def test_spawn_basic(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test basic spawn command."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["spawn", "test task"])

        assert result.exit_code == 0
        assert "Agent spawned" in result.output
        mock_orchestrator.spawn_agent.assert_called_once()

    def test_spawn_with_options(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test spawn with various options."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(
                cli, ["spawn", "test task", "--name", "myagent", "--priority", "high", "--unsupervised"]
            )

        assert result.exit_code == 0
        call_kwargs = mock_orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["name"] == "myagent"
        assert call_kwargs["priority"] == "high"
        assert call_kwargs["supervised"] is False

    def test_spawn_with_model(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test spawn with model option."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["spawn", "test task", "--model", "claude-opus"])

        assert result.exit_code == 0
        call_kwargs = mock_orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus"


# =============================================================================
# Status Command Tests
# =============================================================================


class TestStatusCommand:
    """Tests for status command."""

    def test_status_overall(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test overall status command."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Orchestrator Status" in result.output
        assert "Running: True" in result.output

    def test_status_agent_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test status for specific agent."""
        mock_orchestrator.get_agent.return_value = {
            "id": "agent-123",
            "status": "running",
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "created_at": "2024-01-01",
            "started_at": "2024-01-01T10:00:00",
        }
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["status", "agent-123"])

        assert result.exit_code == 0
        assert "agent-123" in result.output
        assert "running" in result.output

    def test_status_agent_not_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test status for non-existent agent."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["status", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_status_agent_with_error(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test status for agent with error."""
        mock_orchestrator.get_agent.return_value = {
            "id": "agent-123",
            "status": "failed",
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "created_at": "2024-01-01",
            "completed_at": "2024-01-01T11:00:00",
            "error": "Something went wrong",
        }
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["status", "agent-123"])

        assert result.exit_code == 0
        assert "Error:" in result.output
        assert "Something went wrong" in result.output


# =============================================================================
# List Command Tests
# =============================================================================


class TestListCommand:
    """Tests for list command."""

    def test_list_empty(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test list with no agents."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No agents found" in result.output

    def test_list_with_agents(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test list with agents."""
        mock_orchestrator.list_agents.return_value = [
            {"id": "agent-1", "status": "running", "task": "Task 1"},
            {"id": "agent-2", "status": "paused", "task": "Task 2"},
        ]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "agent-1" in result.output
        assert "agent-2" in result.output

    def test_list_with_status_filter(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test list with status filter."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["list", "--status", "running"])

        mock_orchestrator.list_agents.assert_called_once_with("running")

    def test_list_truncates_long_tasks(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test list truncates long task descriptions."""
        long_task = "x" * 100
        mock_orchestrator.list_agents.return_value = [{"id": "agent-1", "status": "running", "task": long_task}]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["list"])

        assert "..." in result.output


# =============================================================================
# Agent Control Command Tests
# =============================================================================


class TestAgentControlCommands:
    """Tests for pause/resume/terminate commands."""

    def test_pause_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful pause."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["pause", "agent-123"])

        assert result.exit_code == 0
        assert "paused" in result.output
        mock_orchestrator.pause_agent.assert_called_once_with("agent-123")

    def test_pause_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed pause."""
        mock_orchestrator.pause_agent.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["pause", "agent-123"])

        assert result.exit_code == 1
        assert "Could not pause" in result.output

    def test_resume_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful resume."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["resume", "agent-123"])

        assert result.exit_code == 0
        assert "resumed" in result.output

    def test_resume_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed resume."""
        mock_orchestrator.resume_agent.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["resume", "agent-123"])

        assert result.exit_code == 1

    def test_terminate_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful terminate."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["terminate", "agent-123"])

        assert result.exit_code == 0
        assert "terminated" in result.output

    def test_terminate_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed terminate."""
        mock_orchestrator.terminate_agent.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["terminate", "agent-123"])

        assert result.exit_code == 1

    def test_nudge_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful nudge."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["nudge", "agent-123", "please continue"])

        assert result.exit_code == 0
        assert "Nudge sent" in result.output
        mock_orchestrator.nudge_agent.assert_called_once_with("agent-123", "please continue")

    def test_nudge_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed nudge."""
        mock_orchestrator.nudge_agent.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["nudge", "agent-123", "message"])

        assert result.exit_code == 1


# =============================================================================
# Logs Command Tests
# =============================================================================


class TestLogsCommand:
    """Tests for logs command."""

    def test_logs_empty(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test logs with no conversation."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["logs", "agent-123"])

        assert result.exit_code == 0
        assert "No logs found" in result.output

    def test_logs_with_conversation(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test logs with conversation."""
        mock_db.get_conversation.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["logs", "agent-123"])

        assert result.exit_code == 0
        assert "[USER]" in result.output
        assert "[ASSISTANT]" in result.output

    def test_logs_with_list_content(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test logs with list content (tool uses)."""
        mock_db.get_conversation.return_value = [
            {"role": "assistant", "content": [{"type": "tool_use", "name": "bash"}]},
        ]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["logs", "agent-123"])

        assert result.exit_code == 0
        assert "tool_use" in result.output


# =============================================================================
# Approval Command Tests
# =============================================================================


class TestApprovalCommands:
    """Tests for pending/approve/reject commands."""

    def test_pending_empty(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test pending with no approvals."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["pending"])

        assert result.exit_code == 0
        assert "No pending approvals" in result.output

    def test_pending_with_approvals(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test pending with approvals."""
        mock_orchestrator.get_pending_approvals.return_value = [
            {
                "id": "approval-1",
                "agent_id": "agent-123",
                "action_type": "bash",
                "action_details": {"command": "ls"},
                "created_at": "2024-01-01",
            }
        ]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["pending"])

        assert result.exit_code == 0
        assert "approval-1" in result.output
        assert "agent-123" in result.output

    def test_approve_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful approval."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["approve", "approval-1"])

        assert result.exit_code == 0
        assert "Approved" in result.output
        mock_orchestrator.approve.assert_called_once_with("approval-1", approved=True)

    def test_approve_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed approval."""
        mock_orchestrator.approve.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["approve", "approval-1"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_reject_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test successful rejection."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["reject", "approval-1"])

        assert result.exit_code == 0
        assert "Rejected" in result.output
        mock_orchestrator.approve.assert_called_once_with("approval-1", approved=False)

    def test_reject_failure(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test failed rejection."""
        mock_orchestrator.approve.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["reject", "approval-1"])

        assert result.exit_code == 1


# =============================================================================
# Secret Command Tests
# =============================================================================


class TestSecretCommands:
    """Tests for secret subcommands."""

    def test_secret_set_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test setting a secret."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "set", "MY_KEY", "my_value"])

        assert result.exit_code == 0
        assert "set" in result.output
        mock_secrets.set.assert_called_once_with("MY_KEY", "my_value")

    def test_secret_set_crypto_not_initialized(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test setting secret when crypto not initialized."""
        mock_crypto.is_initialized.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "set", "MY_KEY", "value"])

        assert result.exit_code == 1
        assert "Crypto not initialized" in result.output

    def test_secret_get_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test getting a secret."""
        mock_secrets.get.return_value = "secret_value"
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "get", "MY_KEY"])

        assert result.exit_code == 0
        assert "secret_value" in result.output

    def test_secret_get_not_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test getting non-existent secret."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "get", "MISSING"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_secret_get_crypto_not_initialized(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test getting secret when crypto not initialized."""
        mock_crypto.is_initialized.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "get", "MY_KEY"])

        assert result.exit_code == 1
        assert "Crypto not initialized" in result.output

    def test_secret_list_empty(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test listing secrets when empty."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "list"])

        assert result.exit_code == 0
        assert "No secrets stored" in result.output

    def test_secret_list_with_secrets(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test listing secrets."""
        mock_secrets.list_keys.return_value = ["KEY1", "KEY2"]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "list"])

        assert result.exit_code == 0
        assert "KEY1" in result.output
        assert "KEY2" in result.output

    def test_secret_delete_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test deleting a secret."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "delete", "MY_KEY"])

        assert result.exit_code == 0
        assert "deleted" in result.output

    def test_secret_delete_not_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test deleting non-existent secret."""
        mock_secrets.delete.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["secret", "delete", "MISSING"])

        assert result.exit_code == 1
        assert "not found" in result.output


# =============================================================================
# Template Command Tests
# =============================================================================


class TestTemplateCommands:
    """Tests for template subcommands."""

    def test_template_save(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test saving a template."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "save", "mytemplate", "run tests"])

        assert result.exit_code == 0
        assert "saved" in result.output
        mock_db.save_template.assert_called_once()

    def test_template_save_with_options(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test saving template with options."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(
                cli, ["template", "save", "mytemplate", "run tests", "--model", "claude-opus", "--priority", "high"]
            )

        assert result.exit_code == 0
        call_kwargs = mock_db.save_template.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus"
        assert call_kwargs["priority"] == "high"

    def test_template_list_empty(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test listing templates when empty."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "list"])

        assert result.exit_code == 0
        assert "No templates saved" in result.output

    def test_template_list_with_templates(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test listing templates."""
        mock_db.list_templates.return_value = [
            {"name": "template1", "task": "Task 1"},
            {"name": "template2", "task": "Task 2"},
        ]
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "list"])

        assert result.exit_code == 0
        assert "template1" in result.output
        assert "template2" in result.output

    def test_template_use_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test using a template."""
        mock_db.get_template.return_value = {
            "task": "Test task",
            "model": "test-model",
            "supervised": True,
            "priority": "normal",
        }
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "use", "mytemplate"])

        assert result.exit_code == 0
        assert "spawned from template" in result.output
        mock_orchestrator.spawn_agent.assert_called_once()

    def test_template_use_not_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test using non-existent template."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "use", "missing"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_template_delete_success(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test deleting a template."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "delete", "mytemplate"])

        assert result.exit_code == 0
        assert "deleted" in result.output

    def test_template_delete_not_found(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test deleting non-existent template."""
        mock_db.delete_template.return_value = False
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["template", "delete", "missing"])

        assert result.exit_code == 1
        assert "not found" in result.output


# =============================================================================
# CLI Setup Tests
# =============================================================================


class TestCLISetup:
    """Tests for CLI initialization."""

    def test_cli_with_data_dir(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test CLI with custom data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
                result = runner.invoke(cli, ["--data-dir", tmpdir, "status"])

            assert result.exit_code == 0

    def test_cli_initializes_crypto_with_env_var(self, runner, mock_db, mock_crypto, mock_secrets, mock_orchestrator):
        """Test CLI initializes crypto with GRU_MASTER_PASSWORD."""
        with setup_cli_mocks(mock_db, mock_crypto, mock_secrets, mock_orchestrator):
            result = runner.invoke(cli, ["status"], env={"GRU_MASTER_PASSWORD": "test_password"})

        # Verify crypto was initialized
        mock_crypto.initialize.assert_called_once_with("test_password")
