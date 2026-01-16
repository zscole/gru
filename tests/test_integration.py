"""Integration tests for agent lifecycle."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gru.claude import Response, ToolUse
from gru.config import Config
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.orchestrator import Agent, Orchestrator


@pytest.fixture
async def setup():
    """Set up test environment with all components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        workspace = tmpdir_path / "workspace"
        workspace.mkdir()

        config = Config(
            data_dir=tmpdir_path,
            telegram_token="test_token",
            telegram_admin_ids=[123],
            anthropic_api_key="test_key",
            default_model="test-model",
            max_tokens=1000,
            default_workdir=workspace,
        )

        db = Database(config.db_path)
        await db.connect()

        crypto = CryptoManager(config.data_dir)
        crypto.initialize("test_password")
        secrets = SecretStore(db, crypto)

        orchestrator = Orchestrator(config, db, secrets)

        yield {
            "config": config,
            "db": db,
            "orchestrator": orchestrator,
            "tmpdir": tmpdir_path,
            "workspace": workspace,
        }

        await db.close()


def create_agent(orchestrator: Orchestrator, agent_data: dict, workdir: str) -> Agent:
    """Helper to create and register an Agent object."""
    agent = Agent(
        agent_id=agent_data["id"],
        task=agent_data["task"],
        model=agent_data["model"],
        supervised=bool(agent_data["supervised"]),
        timeout_mode=agent_data["timeout_mode"],
        workdir=workdir,
        orchestrator=orchestrator,
    )
    orchestrator._agents[agent_data["id"]] = agent
    return agent


@pytest.mark.asyncio
async def test_agent_spawn_and_complete(setup):
    """Test spawning an agent that completes immediately."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]

    mock_response = Response(
        content="Task completed successfully.",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    with patch.object(orchestrator.claude, "send_message", new_callable=AsyncMock, return_value=mock_response):
        agent_data = await orchestrator.spawn_agent(
            task="Test task",
            supervised=False,
        )
        agent = create_agent(orchestrator, agent_data, str(workspace))

        await orchestrator.run_agent(agent, "task123")

        result = await orchestrator.get_agent(agent_data["id"])
        assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_with_tool_execution(setup):
    """Test agent that uses a tool then completes."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]

    tool_response = Response(
        content="I'll write a test file.",
        tool_uses=[
            ToolUse(
                id="tool_1",
                name="write_file",
                input={"path": "test.txt", "content": "Hello, World!"},
            )
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    complete_response = Response(
        content="File written successfully.",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    call_count = 0

    async def mock_send(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return complete_response

    with patch.object(orchestrator.claude, "send_message", side_effect=mock_send):
        agent_data = await orchestrator.spawn_agent(
            task="Write a test file",
            supervised=False,
            workdir=str(workspace),
        )
        agent = create_agent(orchestrator, agent_data, str(workspace))

        await orchestrator.run_agent(agent, "task123")

        test_file = workspace / "test.txt"
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

        result = await orchestrator.get_agent(agent_data["id"])
        assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_with_bash_execution(setup):
    """Test agent that executes bash commands."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]

    tool_response = Response(
        content="Running command.",
        tool_uses=[
            ToolUse(
                id="tool_1",
                name="bash",
                input={"command": "echo 'test' > output.txt"},
            )
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    complete_response = Response(
        content="Command executed.",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    call_count = 0

    async def mock_send(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return complete_response

    with patch.object(orchestrator.claude, "send_message", side_effect=mock_send):
        agent_data = await orchestrator.spawn_agent(
            task="Run a command",
            supervised=False,
            workdir=str(workspace),
        )
        agent = create_agent(orchestrator, agent_data, str(workspace))

        await orchestrator.run_agent(agent, "task123")

        output_file = workspace / "output.txt"
        assert output_file.exists()
        assert "test" in output_file.read_text()


@pytest.mark.asyncio
async def test_agent_turn_limit(setup):
    """Test agent respects turn limit."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]
    orchestrator.config.max_agent_turns = 2

    tool_response = Response(
        content="Continuing...",
        tool_uses=[ToolUse(id="tool_1", name="bash", input={"command": "echo test"})],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    with patch.object(orchestrator.claude, "send_message", new_callable=AsyncMock, return_value=tool_response):
        agent_data = await orchestrator.spawn_agent(
            task="Infinite loop task",
            supervised=False,
        )
        agent = create_agent(orchestrator, agent_data, str(workspace))

        await orchestrator.run_agent(agent, "task123")

        result = await orchestrator.get_agent(agent_data["id"])
        assert result["status"] == "failed"
        assert "max turns" in result["error"].lower()


@pytest.mark.asyncio
async def test_secret_store_integration(setup):
    """Test secret storage and retrieval."""
    orchestrator = setup["orchestrator"]

    await orchestrator.secrets.set("api_key", "secret_value_123")

    value = await orchestrator.secrets.get("api_key")
    assert value == "secret_value_123"

    keys = await orchestrator.secrets.list_keys()
    assert "api_key" in keys

    await orchestrator.secrets.delete("api_key")
    value = await orchestrator.secrets.get("api_key")
    assert value is None


@pytest.mark.asyncio
async def test_agent_pause_resume(setup):
    """Test pausing and resuming an agent."""
    orchestrator = setup["orchestrator"]

    agent_data = await orchestrator.spawn_agent(
        task="Pausable task",
        supervised=False,
    )

    success = await orchestrator.pause_agent(agent_data["id"])
    assert success

    agent = await orchestrator.get_agent(agent_data["id"])
    assert agent["status"] == "paused"

    success = await orchestrator.resume_agent(agent_data["id"])
    assert success

    # Resume sets status back to idle (ready to run again)
    agent = await orchestrator.get_agent(agent_data["id"])
    # Status after resume depends on implementation - check it's not paused
    assert agent["status"] != "paused"


@pytest.mark.asyncio
async def test_agent_termination(setup):
    """Test agent termination."""
    orchestrator = setup["orchestrator"]

    agent_data = await orchestrator.spawn_agent(
        task="Terminatable task",
        supervised=False,
    )

    success = await orchestrator.terminate_agent(agent_data["id"])
    assert success

    agent = await orchestrator.get_agent(agent_data["id"])
    assert agent["status"] == "terminated"


@pytest.mark.asyncio
async def test_multiple_agents(setup):
    """Test running multiple agents."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]

    mock_response = Response(
        content="Done.",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    with patch.object(orchestrator.claude, "send_message", new_callable=AsyncMock, return_value=mock_response):
        agents = []
        for i in range(3):
            agent_data = await orchestrator.spawn_agent(
                task=f"Task {i}",
                supervised=False,
            )
            agents.append(agent_data)

        # Run all agents
        for agent_data in agents:
            agent = create_agent(orchestrator, agent_data, str(workspace))
            await orchestrator.run_agent(agent, f"task_{agent_data['id']}")

        # All should complete
        for agent_data in agents:
            result = await orchestrator.get_agent(agent_data["id"])
            assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_read_file_tool(setup):
    """Test read_file tool execution."""
    orchestrator = setup["orchestrator"]
    workspace = setup["workspace"]

    # Create a file to read
    test_file = workspace / "readable.txt"
    test_file.write_text("File contents here")

    tool_response = Response(
        content="Reading file.",
        tool_uses=[ToolUse(id="tool_1", name="read_file", input={"path": "readable.txt"})],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    complete_response = Response(
        content="File read.",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    call_count = 0

    async def mock_send(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return complete_response

    with patch.object(orchestrator.claude, "send_message", side_effect=mock_send):
        agent_data = await orchestrator.spawn_agent(
            task="Read a file",
            supervised=False,
            workdir=str(workspace),
        )
        agent = create_agent(orchestrator, agent_data, str(workspace))

        await orchestrator.run_agent(agent, "task123")

        result = await orchestrator.get_agent(agent_data["id"])
        assert result["status"] == "completed"
