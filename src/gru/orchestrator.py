"""Core orchestrator for agent lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
import glob as glob_module
import json
import logging
import subprocess
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gru.claude import DEFAULT_TOOLS, ClaudeClient, Response, ToolResult
from gru.coordinator import Coordinator
from gru.mcp import MCPClient
from gru.scheduler import Scheduler
from gru.worktree import WorktreeInfo, cleanup_worktree, create_worktree, get_repo_root, is_git_repo

if TYPE_CHECKING:
    from gru.config import Config
    from gru.crypto import SecretStore
    from gru.db import Database

logger = logging.getLogger(__name__)


class Agent:
    """Represents a running agent."""

    def __init__(
        self,
        agent_id: str,
        task: str,
        model: str,
        supervised: bool,
        timeout_mode: str,
        workdir: str,
        orchestrator: Orchestrator,
        worktree_info: WorktreeInfo | None = None,
    ) -> None:
        self.id = agent_id
        self.task = task
        self.model = model
        self.supervised = supervised
        self.timeout_mode = timeout_mode
        self.workdir = workdir
        self.orchestrator = orchestrator
        self.worktree_info = worktree_info
        self.messages: list[dict[str, Any]] = []
        self._cancelled = False
        self._start_time: datetime | None = None
        self._turn_count: int = 0

    def cancel(self) -> None:
        """Mark agent as cancelled."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if agent is cancelled."""
        return self._cancelled

    def start(self) -> None:
        """Mark agent as started."""
        self._start_time = datetime.now()

    def increment_turn(self) -> None:
        """Increment turn count."""
        self._turn_count += 1

    @property
    def turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_count

    @property
    def runtime_seconds(self) -> float:
        """Get runtime in seconds."""
        if not self._start_time:
            return 0
        return (datetime.now() - self._start_time).total_seconds()


class Orchestrator:
    """Main orchestrator for managing agents."""

    def __init__(
        self,
        config: Config,
        db: Database,
        secrets: SecretStore,
        mcp_config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.secrets = secrets
        self.claude = ClaudeClient(config)
        self.scheduler = Scheduler(db, config.max_concurrent_agents)
        self.coordinator = Coordinator(db)
        self.mcp = MCPClient(mcp_config_path)
        self._agents: dict[str, Agent] = {}
        self._running = False
        self._notify_callback: Callable[[str, str], None] | None = None
        self._approval_callback: Callable[[str, dict], asyncio.Future] | None = None
        self._cancel_approval_callback: Callable[[str], Any] | None = None

    def set_notify_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for notifications."""
        self._notify_callback = callback

    def set_approval_callback(self, callback: Callable[[str, dict], asyncio.Future]) -> None:
        """Set callback for approval requests."""
        self._approval_callback = callback

    def set_cancel_approval_callback(self, callback: Callable[[str], Any]) -> None:
        """Set callback for cancelling approval requests on timeout."""
        self._cancel_approval_callback = callback

    def _truncate_conversation(self, messages: list[dict]) -> list[dict]:
        """Truncate conversation to max length, keeping first and recent messages."""
        max_messages = self.config.max_conversation_messages
        if len(messages) <= max_messages:
            return messages
        # Keep first message (task) and recent messages
        keep_recent = max_messages - 1
        truncated = [messages[0]]
        truncated.append(
            {
                "role": "user",
                "content": f"[Earlier conversation truncated. {len(messages) - max_messages} messages removed.]",
            }
        )
        truncated.extend(messages[-keep_recent:])
        return truncated

    async def notify(self, agent_id: str, message: str) -> None:
        """Send a notification."""
        if self._notify_callback:
            try:
                self._notify_callback(agent_id, message)
            except Exception as e:
                logger.error(f"Notify error: {e}")

    async def spawn_agent(
        self,
        task: str,
        name: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        supervised: bool = True,
        timeout_mode: str = "block",
        priority: str = "normal",
        deadline: str | None = None,
        workdir: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a new agent."""
        # Validate task length
        if len(task) > self.config.max_task_length:
            raise ValueError(f"Task too long: {len(task)} chars (max {self.config.max_task_length})")

        if not task.strip():
            raise ValueError("Task cannot be empty")

        agent_id = str(uuid.uuid4())[:8]
        model = model or self.config.default_model
        workdir = workdir or str(self.config.default_workdir)

        # Ensure workdir exists
        Path(workdir).mkdir(parents=True, exist_ok=True)

        # Check if we should use git worktrees for isolation
        worktree_info: WorktreeInfo | None = None
        worktree_path: str | None = None
        worktree_branch: str | None = None
        base_repo: str | None = None
        effective_workdir = workdir

        if self.config.enable_worktrees and is_git_repo(Path(workdir)):
            repo_root = get_repo_root(Path(workdir))
            if repo_root:
                # Create worktree for this agent
                branch_name = f"gru-agent-{agent_id}"
                wt_base = self.config.worktree_base_dir or repo_root.parent / ".gru-worktrees"
                wt_path = wt_base / branch_name

                try:
                    worktree_info = create_worktree(repo_root, wt_path, branch_name)
                    worktree_path = str(worktree_info.path)
                    worktree_branch = worktree_info.branch
                    base_repo = str(worktree_info.base_repo)
                    effective_workdir = worktree_path
                except RuntimeError:
                    # Fall back to shared workdir if worktree creation fails
                    pass

        # Create agent in database
        agent_data = await self.db.create_agent(
            agent_id=agent_id,
            task=task,
            model=model,
            name=name,
            system_prompt=system_prompt,
            supervised=supervised,
            timeout_mode=timeout_mode,
            priority=priority,
            workdir=workdir,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            base_repo=base_repo,
        )

        # Create task
        task_id = str(uuid.uuid4())[:8]
        await self.db.create_task(
            task_id=task_id,
            agent_id=agent_id,
            priority=priority,
            deadline=deadline,
        )

        # Create agent object with effective workdir (worktree path if available)
        agent = Agent(
            agent_id=agent_id,
            task=task,
            model=model,
            supervised=supervised,
            timeout_mode=timeout_mode,
            workdir=effective_workdir,
            orchestrator=self,
            worktree_info=worktree_info,
        )
        self._agents[agent_id] = agent

        # Queue for execution
        await self.scheduler.enqueue(task_id, agent_id, priority)

        return agent_data

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent by ID."""
        return await self.db.get_agent(agent_id)

    async def list_agents(self, status: str | None = None) -> list[dict[str, Any]]:
        """List agents."""
        return await self.db.get_agents(status)

    async def pause_agent(self, agent_id: str) -> bool:
        """Pause an agent."""
        agent = self._agents.get(agent_id)
        if agent:
            await self.db.update_agent(agent_id, status="paused")
            await self.notify(agent_id, f"Agent {agent_id} paused")
            return True
        return False

    async def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        agent_data = await self.db.get_agent(agent_id)
        if agent_data and agent_data["status"] == "paused":
            await self.db.update_agent(agent_id, status="running")
            await self.notify(agent_id, f"Agent {agent_id} resumed")
            return True
        return False

    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.cancel()
            # Clean up worktree if present
            self._cleanup_agent_worktree(agent)
            await self.db.update_agent(
                agent_id,
                status="terminated",
                completed_at=datetime.now().isoformat(),
            )
            self._agents.pop(agent_id, None)
            await self.notify(agent_id, f"Agent {agent_id} terminated")
            return True
        return False

    def _cleanup_agent_worktree(self, agent: Agent) -> None:
        """Clean up an agent's worktree if present."""
        if agent.worktree_info:
            try:
                cleanup_worktree(
                    repo_path=agent.worktree_info.base_repo,
                    worktree_path=agent.worktree_info.path,
                    branch_name=agent.worktree_info.branch,
                    delete_branch_after=self.config.delete_worktree_branch,
                )
            except Exception as e:
                logger.error(f"Failed to cleanup worktree for agent {agent.id}: {e}")

    async def nudge_agent(self, agent_id: str, message: str) -> bool:
        """Send a nudge message to an agent."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.messages.append({"role": "user", "content": message})
            await self.db.add_message(agent_id, "user", message)
            return True
        return False

    async def run_agent(self, agent: Agent, task_id: str) -> None:
        """Run an agent to completion."""
        agent.start()

        await self.db.update_agent(
            agent.id,
            status="running",
            started_at=datetime.now().isoformat(),
        )
        await self.db.update_task(task_id, status="running", started_at=datetime.now().isoformat())

        # Initialize messages with task
        agent_data = await self.db.get_agent(agent.id)
        system_prompt = agent_data.get("system_prompt") if agent_data else None

        agent.messages = [{"role": "user", "content": agent.task}]
        await self.db.add_message(agent.id, "user", agent.task)

        try:
            while not agent.is_cancelled:
                # Check runtime limit
                if agent.runtime_seconds > self.config.max_agent_runtime:
                    raise TimeoutError(f"Agent exceeded max runtime of {self.config.max_agent_runtime}s")

                # Check turn limit
                if agent.turn_count >= self.config.max_agent_turns:
                    raise RuntimeError(f"Agent exceeded max turns of {self.config.max_agent_turns}")

                agent.increment_turn()

                # Check for incoming messages from other agents
                incoming = await self.coordinator.get_messages(agent.id, unread_only=True)
                for msg in incoming:
                    agent.messages.append(
                        {
                            "role": "user",
                            "content": f"[Message from agent {msg['from_agent']}]: {msg['content']}",
                        }
                    )
                    await self.coordinator.mark_read(msg["id"])

                # Truncate conversation if needed to prevent memory issues
                truncated_messages = self._truncate_conversation(agent.messages)

                # Get response from Claude (include MCP tools)
                all_tools = DEFAULT_TOOLS + self.mcp.get_all_tools()
                response = await self.claude.send_message(
                    messages=truncated_messages,
                    system=system_prompt,
                    model=agent.model,
                    tools=all_tools,
                )

                # Store assistant response
                tool_use_data = None
                if response.tool_uses:
                    tool_use_data = [{"name": t.name, "input": t.input} for t in response.tool_uses]
                await self.db.add_message(
                    agent.id,
                    "assistant",
                    response.content,
                    tool_use=tool_use_data,
                )

                if response.stop_reason == "end_turn":
                    # Agent completed
                    agent.messages.append({"role": "assistant", "content": response.content})
                    break

                if response.tool_uses:
                    # Handle tool uses
                    tool_results = await self._handle_tool_uses(agent, response, task_id)

                    # Add assistant message and tool results to conversation
                    assistant_content: list[dict[str, Any]] = []
                    if response.content:
                        assistant_content.append({"type": "text", "text": response.content})
                    for tu in response.tool_uses:
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": tu.id,
                                "name": tu.name,
                                "input": tu.input,
                            }
                        )
                    agent.messages.append({"role": "assistant", "content": assistant_content})

                    # Add tool results
                    tool_result_content = []
                    for tr in tool_results:
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_use_id,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                        )
                    agent.messages.append({"role": "user", "content": tool_result_content})
                else:
                    agent.messages.append({"role": "assistant", "content": response.content})

            # Mark completed
            final_status = "completed" if not agent.is_cancelled else "terminated"
            await self.db.update_agent(
                agent.id,
                status=final_status,
                completed_at=datetime.now().isoformat(),
            )
            await self.db.update_task(
                task_id,
                status=final_status,
                completed_at=datetime.now().isoformat(),
                result=response.content if not agent.is_cancelled else None,
            )

            output_preview = response.content[:200] if response.content else "No output"
            await self.notify(agent.id, f"Agent {agent.id} {final_status}: {output_preview}")

        except Exception as e:
            await self.db.update_agent(
                agent.id,
                status="failed",
                error=str(e),
                completed_at=datetime.now().isoformat(),
            )
            await self.db.update_task(
                task_id,
                status="failed",
                error=str(e),
                completed_at=datetime.now().isoformat(),
            )
            await self.notify(agent.id, f"Agent {agent.id} failed: {e}")

        finally:
            # Clean up worktree if present
            self._cleanup_agent_worktree(agent)
            self._agents.pop(agent.id, None)
            self.scheduler.unregister_running(task_id)

    async def _handle_tool_uses(self, agent: Agent, response: Response, task_id: str) -> list[ToolResult]:
        """Handle tool use requests from Claude."""
        results = []

        for tool_use in response.tool_uses:
            # Check if approval needed
            if agent.supervised and tool_use.name in ("bash", "write_file"):
                approved = await self._request_approval(agent, tool_use.name, tool_use.input, task_id)
                if not approved:
                    results.append(
                        ToolResult(
                            tool_use_id=tool_use.id,
                            content="Action rejected by user",
                            is_error=True,
                        )
                    )
                    continue

            # Execute tool
            try:
                result = await self._execute_tool(agent, tool_use.name, tool_use.input, task_id)
                results.append(
                    ToolResult(
                        tool_use_id=tool_use.id,
                        content=result,
                    )
                )
            except Exception as e:
                results.append(
                    ToolResult(
                        tool_use_id=tool_use.id,
                        content=str(e),
                        is_error=True,
                    )
                )

        return results

    async def _request_approval(self, agent: Agent, action: str, details: dict, task_id: str) -> bool:
        """Request approval for an action."""
        if not self._approval_callback:
            return True  # Auto-approve if no callback

        approval_id = str(uuid.uuid4())[:8]

        # Calculate timeout
        timeout_seconds = self.config.default_timeout
        timeout_at = (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat()

        await self.db.create_approval(
            approval_id=approval_id,
            agent_id=agent.id,
            action_type=action,
            action_details=details,
            timeout_at=timeout_at,
            task_id=task_id,
        )

        # Notify and wait for approval
        await self.notify(
            agent.id,
            f"Agent {agent.id} requests approval for {action}: {json.dumps(details)[:200]}",
        )

        try:
            future = self._approval_callback(approval_id, {"action": action, "details": details})
            approved = await asyncio.wait_for(future, timeout=timeout_seconds)

            status = "approved" if approved else "rejected"
            await self.db.resolve_approval(approval_id, status, "user")
            return approved

        except asyncio.TimeoutError:
            await self.db.resolve_approval(approval_id, "timeout", "system")

            if agent.timeout_mode == "auto":
                return True
            elif agent.timeout_mode == "pause":
                await self.pause_agent(agent.id)
                return False
            else:  # block
                return False

    async def _execute_tool(self, agent: Agent, tool_name: str, tool_input: dict, task_id: str) -> str:
        """Execute a tool and return the result."""
        # Check if it's an MCP tool first
        if self.mcp.is_mcp_tool(tool_name):
            return await self.mcp.call_tool(tool_name, tool_input)

        # Built-in tool dispatch
        handlers = {
            "bash": lambda: self._execute_bash(tool_input.get("command", ""), agent.workdir),
            "read_file": lambda: self._read_file(tool_input.get("path", ""), agent.workdir),
            "write_file": lambda: self._write_file(
                tool_input.get("path", ""),
                tool_input.get("content", ""),
                agent.workdir,
            ),
            "search_files": lambda: self._search_files(
                tool_input.get("pattern", ""),
                tool_input.get("directory", "."),
                agent.workdir,
            ),
            "request_human_input": lambda: self._request_human_input(
                agent,
                tool_input.get("question", ""),
                tool_input.get("options"),
            ),
            "send_message_to_agent": lambda: self._send_to_agent(
                agent,
                tool_input.get("to_agent", ""),
                tool_input.get("message", ""),
                tool_input.get("message_type", "info"),
                task_id,
            ),
            "get_shared_context": lambda: self._get_context(task_id, tool_input.get("key")),
            "set_shared_context": lambda: self._set_context(
                agent,
                task_id,
                tool_input.get("key", ""),
                tool_input.get("value", ""),
            ),
        }

        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await handler()

    async def _execute_bash(self, command: str, workdir: str) -> str:
        """Execute a bash command in the agent's working directory."""
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=workdir,
                ),
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output or "Command completed with no output"
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds"
        except Exception as e:
            return f"Error executing command: {e}"

    async def _read_file(self, path: str, workdir: str) -> str:
        """Read a file, relative paths resolved from workdir."""
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = Path(workdir) / p
            if not p.exists():
                return f"File not found: {p}"
            content = p.read_text()
            if len(content) > 100000:
                return content[:100000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    async def _write_file(self, path: str, content: str, workdir: str) -> str:
        """Write to a file, relative paths resolved from workdir."""
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = Path(workdir) / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Successfully wrote {len(content)} bytes to {p}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _search_files(self, pattern: str, directory: str, workdir: str) -> str:
        """Search for files matching a pattern."""
        try:
            d = Path(directory).expanduser()
            if not d.is_absolute():
                d = Path(workdir) / d
            matches = list(glob_module.glob(str(d / pattern), recursive=True))
            if not matches:
                return f"No files found matching {pattern} in {d}"
            return "\n".join(matches[:100])
        except Exception as e:
            return f"Error searching files: {e}"

    async def _request_human_input(self, agent: Agent, question: str, options: list[str] | None) -> str:
        """Request human input via approval callback."""
        if not self._approval_callback:
            return "No human input mechanism available. Proceeding without input."

        approval_id = str(uuid.uuid4())[:8]
        timeout_seconds = self.config.default_timeout
        timeout_at = (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat()

        details: dict[str, str | list[str]] = {"question": question}
        if options:
            details["options"] = options

        await self.db.create_approval(
            approval_id=approval_id,
            agent_id=agent.id,
            action_type="human_input",
            action_details=details,
            timeout_at=timeout_at,
        )

        await self.notify(
            agent.id,
            f"Agent {agent.id} asks: {question}" + (f"\nOptions: {', '.join(options)}" if options else ""),
        )

        try:
            future = self._approval_callback(approval_id, details)
            response = await asyncio.wait_for(future, timeout=timeout_seconds)

            if response is None:
                await self.db.resolve_approval(approval_id, "rejected", "user")
                return "User declined to answer."
            else:
                await self.db.resolve_approval(approval_id, "answered", "user")
                return f"User response: {response}"

        except asyncio.TimeoutError:
            await self.db.resolve_approval(approval_id, "timeout", "system")
            # Clean up UI and pending state
            if self._cancel_approval_callback:
                with contextlib.suppress(Exception):
                    await self._cancel_approval_callback(approval_id)
            return "No response from user within timeout. Proceeding."

    async def _send_to_agent(
        self,
        agent: Agent,
        to_agent: str,
        message: str,
        message_type: str,
        task_id: str,
    ) -> str:
        """Send a message to another agent."""
        resolved = await self.coordinator.resolve_agent(to_agent)
        if not resolved:
            return f"Agent not found: {to_agent}"

        msg_id = await self.coordinator.send_message(
            from_agent=agent.id,
            to_agent=resolved,
            content=message,
            message_type=message_type,
            task_id=task_id,
        )
        return f"Message sent (id: {msg_id})"

    async def _get_context(self, task_id: str, key: str | None) -> str:
        """Get shared context."""
        context = await self.coordinator.get_context(task_id, key)
        return json.dumps(context)

    async def _set_context(self, agent: Agent, task_id: str, key: str, value: str) -> str:
        """Set shared context."""
        await self.coordinator.set_context(task_id, key, value, agent.id)
        return f"Context set: {key}"

    async def start(self) -> None:
        """Start the orchestrator main loop."""
        self._running = True

        while self._running:
            try:
                # Check for starvation
                await self.scheduler.prevent_starvation()

                # Process queued tasks
                while self.scheduler.can_run_more():
                    queued = await self.scheduler.dequeue()
                    if not queued:
                        break

                    agent = self._agents.get(queued.agent_id)
                    if not agent:
                        # Agent was removed, skip
                        continue

                    # Start agent task
                    task = asyncio.create_task(self.run_agent(agent, queued.task_id))
                    self.scheduler.register_running(queued.task_id, task)

                # Small delay
                await asyncio.sleep(self.config.scheduler_interval)

            except Exception as e:
                # Log error but keep running
                logger.error(f"Orchestrator error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        # Cancel all running agents
        for agent_id in list(self._agents.keys()):
            await self.terminate_agent(agent_id)

    async def approve(self, approval_id: str, approved: bool = True) -> bool:
        """Approve or reject a pending action."""
        pending = await self.db.get_approval(approval_id)
        if not pending or pending["status"] != "pending":
            return False

        status = "approved" if approved else "rejected"
        await self.db.resolve_approval(approval_id, status, "user")
        return True

    async def get_pending_approvals(self) -> list[dict[str, Any]]:
        """Get all pending approvals."""
        return await self.db.get_pending_approvals()

    async def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        scheduler_status = await self.scheduler.get_status()
        agents = await self.list_agents()

        return {
            "running": self._running,
            "agents": {
                "total": len(agents),
                "running": len([a for a in agents if a["status"] == "running"]),
                "paused": len([a for a in agents if a["status"] == "paused"]),
                "completed": len([a for a in agents if a["status"] == "completed"]),
                "failed": len([a for a in agents if a["status"] == "failed"]),
            },
            "scheduler": scheduler_status,
        }
