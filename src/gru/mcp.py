"""MCP (Model Context Protocol) client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from gru.claude import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Represents a connected MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    _request_id: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id


class MCPClient:
    """Client for managing MCP server connections."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path
        self.servers: dict[str, MCPServer] = {}
        self._all_tools: list[ToolDefinition] = []
        self._tool_to_server: dict[str, str] = {}

    async def load_config(self, config_path: Path | None = None) -> None:
        """Load MCP server configurations from JSON file."""
        path = config_path or self.config_path
        if not path or not path.exists():
            return

        try:
            config = json.loads(path.read_text())
            servers = config.get("mcpServers", {})

            for name, server_config in servers.items():
                command = server_config.get("command", "")
                args = server_config.get("args", [])
                env = server_config.get("env", {})

                if command:
                    self.servers[name] = MCPServer(
                        name=name,
                        command=command,
                        args=args,
                        env=env,
                    )
        except Exception as e:
            logger.error(f"Error loading MCP config: {e}")

    async def start_server(self, server: MCPServer) -> bool:
        """Start an MCP server process."""
        try:
            # Merge environment
            env = os.environ.copy()
            env.update(server.env)

            # Start process
            server.process = subprocess.Popen(
                [server.command] + server.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )

            # Initialize connection
            init_response = await self._send_request(
                server,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "gru", "version": "1.0.0"},
                },
            )

            if not init_response:
                return False

            # Send initialized notification
            await self._send_notification(server, "notifications/initialized", {})

            # Get available tools
            tools_response = await self._send_request(server, "tools/list", {})
            if tools_response and "tools" in tools_response:
                for tool in tools_response["tools"]:
                    tool_def = ToolDefinition(
                        name=f"{server.name}__{tool['name']}",
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    )
                    server.tools.append(tool_def)
                    self._all_tools.append(tool_def)
                    self._tool_to_server[tool_def.name] = server.name

            logger.info(f"MCP server '{server.name}' started with {len(server.tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Error starting MCP server '{server.name}': {e}")
            return False

    async def start_all(self) -> int:
        """Start all configured MCP servers."""
        started = 0
        for server in self.servers.values():
            if await self.start_server(server):
                started += 1
        return started

    async def stop_server(self, server: MCPServer) -> None:
        """Stop an MCP server."""
        if server.process:
            try:
                server.process.terminate()
                server.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                # Force kill if graceful shutdown fails
                server.process.kill()
            server.process = None

    async def stop_all(self) -> None:
        """Stop all MCP servers."""
        for server in self.servers.values():
            await self.stop_server(server)

    async def _send_request(self, server: MCPServer, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request to an MCP server."""
        if not server.process or not server.process.stdin or not server.process.stdout:
            return None

        async with server._lock:
            try:
                request = {
                    "jsonrpc": "2.0",
                    "id": server._next_id(),
                    "method": method,
                    "params": params,
                }

                # Send request
                request_str = json.dumps(request) + "\n"
                server.process.stdin.write(request_str)
                server.process.stdin.flush()

                # Read response (with timeout)
                loop = asyncio.get_running_loop()
                response_str = await asyncio.wait_for(
                    loop.run_in_executor(None, server.process.stdout.readline),
                    timeout=30,
                )

                if not response_str:
                    return None

                response = json.loads(response_str)
                if "error" in response:
                    logger.error(f"MCP error: {response['error']}")
                    return None

                return response.get("result", {})

            except asyncio.TimeoutError:
                logger.warning(f"MCP request timeout: {method}")
                return None
            except Exception as e:
                logger.error(f"MCP request error: {e}")
                return None

    async def _send_notification(self, server: MCPServer, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not server.process or not server.process.stdin:
            return

        try:
            notification = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }

            notification_str = json.dumps(notification) + "\n"
            server.process.stdin.write(notification_str)
            server.process.stdin.flush()

        except Exception as e:
            logger.error(f"MCP notification error: {e}")

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the appropriate MCP server."""
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"Unknown MCP tool: {tool_name}"

        server = self.servers.get(server_name)
        if not server or not server.process:
            return f"MCP server not running: {server_name}"

        # Extract original tool name (remove server prefix)
        original_name = tool_name.split("__", 1)[1] if "__" in tool_name else tool_name

        response = await self._send_request(
            server,
            "tools/call",
            {
                "name": original_name,
                "arguments": arguments,
            },
        )

        if not response:
            return "MCP tool call failed"

        # Extract content from response
        content = response.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else str(content)

        return str(content)

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all tools from all connected MCP servers."""
        return self._all_tools.copy()

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name is an MCP tool."""
        return tool_name in self._tool_to_server

    def is_server_healthy(self, server: MCPServer) -> bool:
        """Check if a server process is still running."""
        if not server.process:
            return False
        return server.process.poll() is None

    async def health_check(self) -> dict[str, bool]:
        """Check health of all MCP servers.

        Returns:
            Dict mapping server name to health status (True = healthy)
        """
        results = {}
        for name, server in self.servers.items():
            results[name] = self.is_server_healthy(server)
        return results

    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server.

        Args:
            server_name: Name of the server to restart

        Returns:
            True if restart successful, False otherwise
        """
        server = self.servers.get(server_name)
        if not server:
            logger.error(f"Cannot restart unknown server: {server_name}")
            return False

        logger.info(f"Restarting MCP server: {server_name}")

        # Remove old tools from registry
        old_tools = [t.name for t in server.tools]
        for tool_name in old_tools:
            self._tool_to_server.pop(tool_name, None)
            self._all_tools = [t for t in self._all_tools if t.name != tool_name]

        # Clear server tools
        server.tools = []

        # Stop if still running
        await self.stop_server(server)

        # Start fresh
        return await self.start_server(server)

    async def recover_unhealthy(self) -> int:
        """Restart any unhealthy MCP servers.

        Returns:
            Number of servers successfully recovered
        """
        recovered = 0
        health = await self.health_check()

        for server_name, is_healthy in health.items():
            if not is_healthy:
                logger.warning(f"MCP server '{server_name}' is unhealthy, attempting restart")
                if await self.restart_server(server_name):
                    recovered += 1
                    logger.info(f"Successfully recovered MCP server: {server_name}")
                else:
                    logger.error(f"Failed to recover MCP server: {server_name}")

        return recovered

    async def ensure_healthy(self) -> None:
        """Ensure all servers are healthy, restarting as needed.

        This is a convenience method that combines health check and recovery.
        """
        await self.recover_unhealthy()
