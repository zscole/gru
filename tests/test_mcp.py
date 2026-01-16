"""Tests for MCP (Model Context Protocol) client."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gru.mcp import MCPClient, MCPServer


@pytest.fixture
def mcp_config_file():
    """Create a temporary MCP config file."""
    config = {
        "mcpServers": {
            "test_server": {
                "command": "echo",
                "args": ["hello"],
                "env": {"TEST_VAR": "test_value"},
            },
            "another_server": {
                "command": "cat",
                "args": [],
                "env": {},
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        return Path(f.name)


@pytest.fixture
def mock_process():
    """Create a mock subprocess."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.poll.return_value = None
    return process


class TestMCPServer:
    """Tests for MCPServer dataclass."""

    def test_mcp_server_creation(self):
        """Test creating an MCP server instance."""
        server = MCPServer(
            name="test",
            command="echo",
            args=["hello"],
            env={"KEY": "value"},
        )
        assert server.name == "test"
        assert server.command == "echo"
        assert server.args == ["hello"]
        assert server.env == {"KEY": "value"}
        assert server.process is None
        assert server.tools == []

    def test_mcp_server_defaults(self):
        """Test MCP server default values."""
        server = MCPServer(name="test", command="echo")
        assert server.args == []
        assert server.env == {}
        assert server.tools == []
        assert server._request_id == 0

    def test_next_id_increments(self):
        """Test that _next_id increments request counter."""
        server = MCPServer(name="test", command="echo")
        assert server._next_id() == 1
        assert server._next_id() == 2
        assert server._next_id() == 3
        assert server._request_id == 3


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_client_creation_no_config(self):
        """Test creating client without config."""
        client = MCPClient()
        assert client.config_path is None
        assert client.servers == {}
        assert client._all_tools == []
        assert client._tool_to_server == {}

    def test_client_creation_with_config_path(self):
        """Test creating client with config path."""
        path = Path("/some/config.json")
        client = MCPClient(config_path=path)
        assert client.config_path == path


class TestMCPClientConfig:
    """Tests for MCPClient configuration loading."""

    @pytest.mark.asyncio
    async def test_load_config_no_path(self):
        """Test loading config with no path set."""
        client = MCPClient()
        await client.load_config()
        assert client.servers == {}

    @pytest.mark.asyncio
    async def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file."""
        client = MCPClient(config_path=Path("/nonexistent/config.json"))
        await client.load_config()
        assert client.servers == {}

    @pytest.mark.asyncio
    async def test_load_config_success(self, mcp_config_file):
        """Test successfully loading config."""
        client = MCPClient(config_path=mcp_config_file)
        await client.load_config()

        assert "test_server" in client.servers
        assert "another_server" in client.servers

        test_server = client.servers["test_server"]
        assert test_server.command == "echo"
        assert test_server.args == ["hello"]
        assert test_server.env == {"TEST_VAR": "test_value"}

        another_server = client.servers["another_server"]
        assert another_server.command == "cat"
        assert another_server.args == []

    @pytest.mark.asyncio
    async def test_load_config_override_path(self, mcp_config_file):
        """Test loading config with override path."""
        client = MCPClient(config_path=Path("/nonexistent.json"))
        await client.load_config(config_path=mcp_config_file)
        assert "test_server" in client.servers

    @pytest.mark.asyncio
    async def test_load_config_invalid_json(self):
        """Test loading invalid JSON config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            config_path = Path(f.name)

        client = MCPClient(config_path=config_path)
        await client.load_config()
        assert client.servers == {}

    @pytest.mark.asyncio
    async def test_load_config_empty_command_skipped(self):
        """Test that servers with empty command are skipped."""
        config = {
            "mcpServers": {
                "no_command": {
                    "command": "",
                    "args": [],
                },
                "has_command": {
                    "command": "echo",
                    "args": [],
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)

        client = MCPClient(config_path=config_path)
        await client.load_config()
        assert "no_command" not in client.servers
        assert "has_command" in client.servers


class TestMCPClientServerLifecycle:
    """Tests for MCP server start/stop."""

    @pytest.mark.asyncio
    async def test_start_server_success(self, mock_process):
        """Test successfully starting a server."""
        server = MCPServer(name="test", command="echo", args=["hello"])

        init_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        tools_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            }
        )
        mock_process.stdout.readline.side_effect = [init_response, tools_response]

        client = MCPClient()
        client.servers["test"] = server

        with patch("subprocess.Popen", return_value=mock_process):
            result = await client.start_server(server)

        assert result is True
        assert server.process is not None
        assert len(server.tools) == 1
        assert server.tools[0].name == "test__test_tool"
        assert "test__test_tool" in client._tool_to_server
        assert client._tool_to_server["test__test_tool"] == "test"

    @pytest.mark.asyncio
    async def test_start_server_init_failure(self, mock_process):
        """Test server start failure during init."""
        server = MCPServer(name="test", command="echo")
        mock_process.stdout.readline.return_value = ""

        client = MCPClient()

        with patch("subprocess.Popen", return_value=mock_process):
            result = await client.start_server(server)

        assert result is False

    @pytest.mark.asyncio
    async def test_start_server_popen_failure(self):
        """Test server start failure when Popen raises."""
        server = MCPServer(name="test", command="nonexistent_command")
        client = MCPClient()

        with patch("subprocess.Popen", side_effect=FileNotFoundError("Command not found")):
            result = await client.start_server(server)

        assert result is False

    @pytest.mark.asyncio
    async def test_stop_server(self, mock_process):
        """Test stopping a server."""
        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        await client.stop_server(server)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert server.process is None

    @pytest.mark.asyncio
    async def test_stop_server_kill_on_timeout(self, mock_process):
        """Test server is killed if terminate times out."""
        server = MCPServer(name="test", command="echo")
        server.process = mock_process
        mock_process.wait.side_effect = subprocess.TimeoutExpired("echo", 5)

        client = MCPClient()
        await client.stop_server(server)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert server.process is None

    @pytest.mark.asyncio
    async def test_stop_server_no_process(self):
        """Test stopping server with no process."""
        server = MCPServer(name="test", command="echo")
        server.process = None

        client = MCPClient()
        await client.stop_server(server)  # Should not raise

    @pytest.mark.asyncio
    async def test_start_all(self, mock_process):
        """Test starting all servers."""
        init_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        tools_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
        )
        mock_process.stdout.readline.side_effect = [
            init_response,
            tools_response,
            init_response,
            tools_response,
        ]

        client = MCPClient()
        client.servers["server1"] = MCPServer(name="server1", command="echo")
        client.servers["server2"] = MCPServer(name="server2", command="cat")

        with patch("subprocess.Popen", return_value=mock_process):
            started = await client.start_all()

        assert started == 2

    @pytest.mark.asyncio
    async def test_stop_all(self, mock_process):
        """Test stopping all servers."""
        client = MCPClient()

        server1 = MCPServer(name="server1", command="echo")
        server1.process = mock_process
        server2 = MCPServer(name="server2", command="cat")
        server2.process = mock_process

        client.servers["server1"] = server1
        client.servers["server2"] = server2

        await client.stop_all()

        assert server1.process is None
        assert server2.process is None


class TestMCPClientToolCalls:
    """Tests for MCP tool calls."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Test calling an unknown tool."""
        client = MCPClient()
        result = await client.call_tool("unknown__tool", {})
        assert "Unknown MCP tool" in result

    @pytest.mark.asyncio
    async def test_call_tool_server_not_running(self):
        """Test calling tool when server is not running."""
        client = MCPClient()
        client._tool_to_server["test__tool"] = "test"
        client.servers["test"] = MCPServer(name="test", command="echo")
        # process is None

        result = await client.call_tool("test__tool", {})
        assert "MCP server not running" in result

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_process):
        """Test successfully calling a tool."""
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "Tool output"}]},
            }
        )
        mock_process.stdout.readline.return_value = response

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        client.servers["test"] = server
        client._tool_to_server["test__my_tool"] = "test"

        result = await client.call_tool("test__my_tool", {"arg": "value"})
        assert result == "Tool output"

    @pytest.mark.asyncio
    async def test_call_tool_multiple_text_blocks(self, mock_process):
        """Test tool response with multiple text blocks."""
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [
                        {"type": "text", "text": "Line 1"},
                        {"type": "text", "text": "Line 2"},
                    ]
                },
            }
        )
        mock_process.stdout.readline.return_value = response

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        client.servers["test"] = server
        client._tool_to_server["test__my_tool"] = "test"

        result = await client.call_tool("test__my_tool", {})
        assert result == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_call_tool_request_failure(self, mock_process):
        """Test tool call when request fails."""
        mock_process.stdout.readline.return_value = ""

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        client.servers["test"] = server
        client._tool_to_server["test__my_tool"] = "test"

        result = await client.call_tool("test__my_tool", {})
        assert result == "MCP tool call failed"


class TestMCPClientToolRegistry:
    """Tests for MCP tool registry methods."""

    def test_get_all_tools_empty(self):
        """Test getting tools when none registered."""
        client = MCPClient()
        tools = client.get_all_tools()
        assert tools == []

    def test_get_all_tools_returns_copy(self):
        """Test that get_all_tools returns a copy."""
        from gru.claude import ToolDefinition

        client = MCPClient()
        tool = ToolDefinition(
            name="test__tool",
            description="Test",
            input_schema={"type": "object", "properties": {}},
        )
        client._all_tools.append(tool)

        tools = client.get_all_tools()
        assert len(tools) == 1

        # Modifying returned list shouldn't affect internal list
        tools.clear()
        assert len(client._all_tools) == 1

    def test_is_mcp_tool_true(self):
        """Test is_mcp_tool returns true for registered tools."""
        client = MCPClient()
        client._tool_to_server["test__tool"] = "test"
        assert client.is_mcp_tool("test__tool") is True

    def test_is_mcp_tool_false(self):
        """Test is_mcp_tool returns false for unregistered tools."""
        client = MCPClient()
        assert client.is_mcp_tool("unknown_tool") is False

    def test_is_mcp_tool_builtin(self):
        """Test is_mcp_tool returns false for built-in tools."""
        client = MCPClient()
        assert client.is_mcp_tool("bash") is False
        assert client.is_mcp_tool("read_file") is False


class TestMCPClientRequests:
    """Tests for MCP JSON-RPC request handling."""

    @pytest.mark.asyncio
    async def test_send_request_no_process(self):
        """Test sending request with no process."""
        server = MCPServer(name="test", command="echo")
        server.process = None

        client = MCPClient()
        result = await client._send_request(server, "test/method", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_send_request_error_response(self, mock_process):
        """Test handling error response."""
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "Invalid request"},
            }
        )
        mock_process.stdout.readline.return_value = response

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        result = await client._send_request(server, "test/method", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_send_notification_no_process(self):
        """Test sending notification with no process."""
        server = MCPServer(name="test", command="echo")
        server.process = None

        client = MCPClient()
        await client._send_notification(server, "test/notify", {})  # Should not raise

    @pytest.mark.asyncio
    async def test_send_notification_no_stdin(self, mock_process):
        """Test sending notification with no stdin."""
        mock_process.stdin = None

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        await client._send_notification(server, "test/notify", {})  # Should not raise


class TestMCPHealthCheck:
    """Tests for MCP health check and recovery."""

    def test_is_server_healthy_no_process(self):
        """Test health check with no process."""
        server = MCPServer(name="test", command="echo")
        server.process = None

        client = MCPClient()
        assert client.is_server_healthy(server) is False

    def test_is_server_healthy_running(self, mock_process):
        """Test health check with running process."""
        mock_process.poll.return_value = None  # None means still running

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        assert client.is_server_healthy(server) is True

    def test_is_server_healthy_terminated(self, mock_process):
        """Test health check with terminated process."""
        mock_process.poll.return_value = 1  # Non-None means terminated

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        assert client.is_server_healthy(server) is False

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_process):
        """Test health check with all healthy servers."""
        mock_process.poll.return_value = None

        client = MCPClient()
        server1 = MCPServer(name="server1", command="echo")
        server1.process = mock_process
        server2 = MCPServer(name="server2", command="cat")
        server2.process = mock_process

        client.servers["server1"] = server1
        client.servers["server2"] = server2

        health = await client.health_check()
        assert health == {"server1": True, "server2": True}

    @pytest.mark.asyncio
    async def test_health_check_mixed(self, mock_process):
        """Test health check with mixed health status."""
        client = MCPClient()

        # Healthy server
        healthy_process = MagicMock()
        healthy_process.poll.return_value = None
        server1 = MCPServer(name="healthy", command="echo")
        server1.process = healthy_process

        # Unhealthy server (crashed)
        unhealthy_process = MagicMock()
        unhealthy_process.poll.return_value = 1
        server2 = MCPServer(name="unhealthy", command="cat")
        server2.process = unhealthy_process

        client.servers["healthy"] = server1
        client.servers["unhealthy"] = server2

        health = await client.health_check()
        assert health["healthy"] is True
        assert health["unhealthy"] is False

    @pytest.mark.asyncio
    async def test_restart_server_unknown(self):
        """Test restarting unknown server."""
        client = MCPClient()
        result = await client.restart_server("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_server_success(self, mock_process):
        """Test successful server restart."""
        from gru.claude import ToolDefinition

        # Set up initial response for new server start
        init_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        tools_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
        )
        mock_process.stdout.readline.side_effect = [init_response, tools_response]

        # Create server with existing tool
        server = MCPServer(name="test", command="echo")
        old_tool = ToolDefinition(
            name="test__old_tool",
            description="Old tool",
            input_schema={"type": "object", "properties": {}},
        )
        server.tools = [old_tool]
        server.process = mock_process

        client = MCPClient()
        client.servers["test"] = server
        client._all_tools = [old_tool]
        client._tool_to_server["test__old_tool"] = "test"

        with patch("subprocess.Popen", return_value=mock_process):
            result = await client.restart_server("test")

        assert result is True
        # Old tool should be removed from registry
        assert "test__old_tool" not in client._tool_to_server

    @pytest.mark.asyncio
    async def test_recover_unhealthy_none(self):
        """Test recovery when all servers healthy."""
        client = MCPClient()

        healthy_process = MagicMock()
        healthy_process.poll.return_value = None
        server = MCPServer(name="test", command="echo")
        server.process = healthy_process
        client.servers["test"] = server

        recovered = await client.recover_unhealthy()
        assert recovered == 0

    @pytest.mark.asyncio
    async def test_recover_unhealthy_restarts(self, mock_process):
        """Test recovery restarts unhealthy servers."""
        init_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        tools_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
        )
        mock_process.stdout.readline.side_effect = [init_response, tools_response]

        # Unhealthy server
        unhealthy_process = MagicMock()
        unhealthy_process.poll.return_value = 1  # Crashed

        server = MCPServer(name="crashed", command="echo")
        server.process = unhealthy_process

        client = MCPClient()
        client.servers["crashed"] = server

        with patch("subprocess.Popen", return_value=mock_process):
            recovered = await client.recover_unhealthy()

        assert recovered == 1

    @pytest.mark.asyncio
    async def test_ensure_healthy(self, mock_process):
        """Test ensure_healthy convenience method."""
        mock_process.poll.return_value = None  # All healthy

        server = MCPServer(name="test", command="echo")
        server.process = mock_process

        client = MCPClient()
        client.servers["test"] = server

        await client.ensure_healthy()  # Should not raise
