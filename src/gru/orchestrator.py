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

import anthropic

from gru.claude import DEFAULT_TOOLS, ClaudeClient, Response, ToolResult
from gru.coordinator import Coordinator
from gru.mcp import MCPClient
from gru.scheduler import Scheduler
from gru.worktree import (
    WorktreeInfo,
    cleanup_worktree,
    commit_and_push,
    create_worktree,
    get_repo_root,
    is_git_repo,
)

if TYPE_CHECKING:
    from gru.config import Config
    from gru.crypto import SecretStore
    from gru.db import Database

logger = logging.getLogger(__name__)

DEFAULT_AGENT_SYSTEM = """You are an AI agent that completes tasks by using tools.

IMPORTANT: You must USE the available tools to complete tasks. Do not just explain what you would do - actually do it.

Working directory: {workdir}

Available tools:
- write_file: Create or overwrite files
- read_file: Read file contents
- bash: Execute shell commands
- search_files: Find files by pattern

When given a task:
1. Use write_file to create the necessary files
2. Use bash to run commands (install dependencies, test code, etc.)
3. Report the result with any relevant output (URLs, file paths, etc.)

Always create files in the working directory unless specified otherwise.
Do not ask for confirmation - just execute the task."""


def friendly_error(error: Exception) -> str:
    """Convert technical errors to plain English."""
    error_str = str(error).lower()

    if "rate limit" in error_str or "429" in error_str:
        return "Hit rate limit - waiting a moment before trying again"
    elif "api key" in error_str or "authentication" in error_str or "401" in error_str:
        return "API key issue - check ANTHROPIC_API_KEY is set correctly"
    elif "connection" in error_str or "network" in error_str:
        return "Connection issue - check your internet connection"
    elif "timeout" in error_str:
        return "Request timed out - the task might be too complex, trying again"
    elif "quota" in error_str or "billing" in error_str:
        return "API quota exceeded - check your Anthropic account billing"
    elif "model" in error_str and "not found" in error_str:
        return "Model not available - try a different model"
    elif "permission" in error_str or "access denied" in error_str:
        return "Permission denied - check file/directory permissions"
    elif "not found" in error_str or "404" in error_str:
        return "Resource not found - the file or endpoint doesn't exist"
    elif "git" in error_str:
        return f"Git issue: {error_str[:100]}"
    elif "memory" in error_str or "oom" in error_str:
        return "Out of memory - try a simpler task or smaller files"
    else:
        # Keep original but clean it up
        return str(error)[:200]


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
        self._last_report_time: datetime | None = None
        self._recent_tools: list[str] = []  # Track recent tool calls for progress reports
        self._turns_since_tool: int = 0  # Track turns without tool calls for stuck detection
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._token_alert_sent: bool = False
        self._stuck_alert_sent: bool = False
        self.live_output: bool = False  # Stream output to chat in real-time

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

    def should_send_progress_report(self, interval_minutes: int) -> bool:
        """Check if it's time to send a progress report."""
        if interval_minutes <= 0:
            return False
        if self._last_report_time is None:
            # First report after interval from start
            if self._start_time is None:
                return False
            elapsed = (datetime.now() - self._start_time).total_seconds()
            return elapsed >= interval_minutes * 60
        elapsed = (datetime.now() - self._last_report_time).total_seconds()
        return elapsed >= interval_minutes * 60

    def mark_report_sent(self) -> None:
        """Mark that a progress report was sent."""
        self._last_report_time = datetime.now()
        self._recent_tools = []

    def add_tool_call(self, tool_name: str, summary: str) -> None:
        """Track a tool call for progress reporting."""
        self._recent_tools.append(f"{tool_name}: {summary}")
        self._turns_since_tool = 0  # Reset stuck counter

    def increment_turns_since_tool(self) -> None:
        """Increment turns without tool calls."""
        self._turns_since_tool += 1

    def is_stuck(self, threshold: int) -> bool:
        """Check if agent appears stuck (no tool calls for threshold turns)."""
        return threshold > 0 and self._turns_since_tool >= threshold

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self._total_input_tokens + self._total_output_tokens

    def should_alert_token_burn(self, threshold: int) -> bool:
        """Check if token usage exceeds threshold and alert not yet sent."""
        if self._token_alert_sent or threshold <= 0:
            return False
        if self.total_tokens >= threshold:
            self._token_alert_sent = True
            return True
        return False

    def should_alert_stuck(self, threshold: int) -> bool:
        """Check if stuck and alert not yet sent."""
        if self._stuck_alert_sent or not self.is_stuck(threshold):
            return False
        self._stuck_alert_sent = True
        return True

    def get_progress_summary(self) -> str:
        """Generate a progress summary with visual progress bar."""
        runtime_mins = int(self.runtime_seconds / 60)

        # Estimate progress based on turns and token usage
        # Assume typical task takes ~50 turns or 100k tokens
        turn_progress = min(100, int(self._turn_count / 50 * 100))
        token_progress = min(100, int(self.total_tokens / 100000 * 100))
        estimated_progress = max(turn_progress, token_progress)

        # Visual progress bar
        bar_width = 15
        filled = int(bar_width * estimated_progress / 100)
        bar = "=" * filled + "-" * (bar_width - filled)

        summary = f"[{bar}] ~{estimated_progress}%\n"
        summary += f"Turn {self._turn_count} | {runtime_mins}m | {self.total_tokens:,} tokens"

        if self._recent_tools:
            recent = self._recent_tools[-3:]  # Last 3 tool calls
            summary += "\nRecent: " + ", ".join(recent)
        return summary


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
        self._ralph_loops: dict[str, dict[str, Any]] = {}  # Track Ralph loop metadata
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
        """Truncate conversation to max length, preserving tool_use/tool_result pairs."""
        max_messages = self.config.max_conversation_messages
        if len(messages) <= max_messages:
            return messages

        # Find a safe truncation point that doesn't break tool pairs
        # We need to keep messages from a point where there are no orphaned tool_results
        keep_recent = max_messages - 2  # Reserve space for first msg + truncation notice

        # Start from the candidate truncation point and scan forward
        # to find a safe boundary (no tool_result without its tool_use)
        candidate_start = len(messages) - keep_recent

        # Collect tool_use_ids in the kept portion
        def get_tool_use_ids(msg: dict) -> set[str]:
            """Extract tool_use_ids from an assistant message."""
            ids = set()
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        ids.add(block.get("id", ""))
            return ids

        def get_tool_result_ids(msg: dict) -> set[str]:
            """Extract tool_use_ids referenced in tool_results."""
            ids = set()
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        ids.add(block.get("tool_use_id", ""))
            return ids

        # Move start forward until we have a clean boundary
        while candidate_start < len(messages) - 1:
            kept_messages = messages[candidate_start:]
            # Collect all tool_use_ids in kept assistant messages
            tool_use_ids = set()
            for msg in kept_messages:
                if msg.get("role") == "assistant":
                    tool_use_ids.update(get_tool_use_ids(msg))

            # Check all tool_results reference a kept tool_use
            valid = True
            for msg in kept_messages:
                if msg.get("role") == "user":
                    result_ids = get_tool_result_ids(msg)
                    if result_ids and not result_ids.issubset(tool_use_ids):
                        valid = False
                        break

            if valid:
                break
            candidate_start += 1

        truncated = [messages[0]]
        removed_count = candidate_start - 1
        if removed_count > 0:
            truncated.append(
                {
                    "role": "user",
                    "content": f"[Earlier conversation truncated. {removed_count} messages removed.]",
                }
            )
        truncated.extend(messages[candidate_start:])
        return truncated

    async def notify(self, agent_id: str, message: str) -> None:
        """Send a notification."""
        if self._notify_callback:
            try:
                self._notify_callback(agent_id, message)
            except Exception as e:
                logger.error(f"Notify error: {e}")

    def _estimate_cost(self, agent: Agent) -> str:
        """Estimate cost for an agent based on token usage and model."""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
            "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
            "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        }
        rates = pricing.get(agent.model, {"input": 3.0, "output": 15.0})
        input_cost = (agent._total_input_tokens / 1_000_000) * rates["input"]
        output_cost = (agent._total_output_tokens / 1_000_000) * rates["output"]
        return f"{input_cost + output_cost:.4f}"

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
        live_output: bool = False,
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
        agent.live_output = live_output
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
            # Auto-push changes before pausing
            task = agent.messages[0]["content"] if agent.messages else "Agent work"
            success, status = self._auto_push_agent(agent, f"WIP: {task[:50]}")
            await self.db.update_agent(agent_id, status="paused")
            msg = f"Agent {agent_id} paused"
            if success and "Pushed" in status:
                msg += f" ({status})"
            await self.notify(agent_id, msg)
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

    def _auto_push_agent(self, agent: Agent, message: str) -> tuple[bool, str]:
        """Auto commit and push agent's worktree if enabled."""
        if not self.config.auto_push:
            return True, "Auto-push disabled"
        if not agent.worktree_info:
            return True, "No worktree"
        try:
            success, status = commit_and_push(agent.worktree_info.path, message)
            if success:
                logger.info(f"Agent {agent.id}: {status}")
            else:
                logger.error(f"Agent {agent.id} auto-push failed: {status}")
            return success, status
        except Exception as e:
            logger.error(f"Agent {agent.id} auto-push error: {e}")
            return False, str(e)

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

        # Use default agent system prompt if none provided
        if not system_prompt:
            system_prompt = DEFAULT_AGENT_SYSTEM.format(workdir=agent.workdir)

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

                # Send periodic progress report if enabled
                if agent.should_send_progress_report(self.config.progress_report_interval):
                    summary = agent.get_progress_summary()
                    await self.notify(agent.id, f"Progress: {summary}")
                    agent.mark_report_sent()

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
                try:
                    response = await self.claude.send_message(
                        messages=truncated_messages,
                        system=system_prompt,
                        model=agent.model,
                        tools=all_tools,
                    )
                except anthropic.RateLimitError as e:
                    await self.notify(agent.id, f"Rate limited: {e}. Retrying...")
                    raise

                # Track token usage
                agent.add_tokens(response.usage["input_tokens"], response.usage["output_tokens"])
                await self.db.add_tokens(agent.id, response.usage["input_tokens"], response.usage["output_tokens"])

                # Check for token burn alert
                if agent.should_alert_token_burn(self.config.token_burn_alert):
                    await self.notify(
                        agent.id,
                        f"High token usage: {agent.total_tokens:,} tokens (~${self._estimate_cost(agent)})",
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

                    # Live output: send tool calls to chat
                    if agent.live_output:
                        for tu in response.tool_uses:
                            summary = self._summarize_tool_input(tu.name, tu.input)
                            await self.notify(agent.id, f"[{tu.name}] {summary}")

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
                    # No tool uses - track for stuck detection
                    agent.increment_turns_since_tool()
                    if agent.should_alert_stuck(self.config.stuck_threshold_turns):
                        await self.notify(
                            agent.id,
                            f"Agent may be stuck: {agent._turns_since_tool} turns without tool calls",
                        )
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

            output_preview = response.content[:1000] if response.content else "No output"
            await self.notify(agent.id, f"Agent {agent.id} {final_status}: {output_preview}")

        except Exception as e:
            error_msg = friendly_error(e)
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
            await self.notify(agent.id, f"Agent {agent.id} failed: {error_msg}")

        finally:
            # Auto-push changes before cleanup
            task = agent.messages[0]["content"] if agent.messages else "Agent work"
            self._auto_push_agent(agent, task[:100])
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
                # Track tool call for progress reports
                tool_summary = self._summarize_tool_input(tool_use.name, tool_use.input)
                agent.add_tool_call(tool_use.name, tool_summary)
                # Truncate large outputs to prevent context overflow
                if len(result) > self.config.max_tool_output:
                    result = result[: self.config.max_tool_output] + (
                        f"\n\n[truncated - output exceeded {self.config.max_tool_output} chars]"
                    )
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
            f"Agent {agent.id} requests approval for {action}: {json.dumps(details)[:1000]}",
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

    def _summarize_tool_input(self, tool_name: str, tool_input: dict) -> str:
        """Generate a brief summary of tool input for progress reports."""
        if tool_name == "bash":
            cmd = tool_input.get("command", "")
            return cmd[:40] + "..." if len(cmd) > 40 else cmd
        elif tool_name == "read_file":
            return tool_input.get("path", "")[:40]
        elif tool_name == "write_file":
            path = tool_input.get("path", "")
            return path[:40]
        elif tool_name == "search_files":
            return tool_input.get("pattern", "")[:30]
        else:
            return tool_name

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

    async def search_agents(self, query: str) -> list[dict[str, Any]]:
        """Search agents by task, name, or id."""
        return await self.db.search_agents(query)

    def get_agent_cost(self, agent_id: str) -> tuple[int, int, str] | None:
        """Get token usage and cost for an agent."""
        agent = self._agents.get(agent_id)
        if agent:
            cost = self._estimate_cost(agent)
            return agent._total_input_tokens, agent._total_output_tokens, cost
        return None

    async def get_agent_cost_from_db(self, agent_id: str) -> tuple[int, int, str] | None:
        """Get token usage and cost for an agent from database."""
        agent_data = await self.db.get_agent(agent_id)
        if agent_data:
            input_tokens = agent_data.get("input_tokens", 0) or 0
            output_tokens = agent_data.get("output_tokens", 0) or 0
            # Create temporary agent for cost calculation
            pricing = {
                "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
                "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
                "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
                "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
                "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            }
            model = agent_data.get("model", "claude-sonnet-4-20250514")
            rates = pricing.get(model, {"input": 3.0, "output": 15.0})
            input_cost = (input_tokens / 1_000_000) * rates["input"]
            output_cost = (output_tokens / 1_000_000) * rates["output"]
            cost = f"{input_cost + output_cost:.4f}"
            return input_tokens, output_tokens, cost
        return None

    async def spawn_ralph_loop(
        self,
        task: str,
        max_iterations: int = 20,
        completion_promise: str | None = None,
        name: str | None = None,
        model: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Start a Ralph Wiggum iterative development loop.

        Ralph is an AI development methodology that creates self-referential
        feedback loops where an agent iteratively improves work through
        continuous iterations.

        Args:
            task: The task to iteratively work on
            max_iterations: Maximum number of iterations (default 20)
            completion_promise: String to detect completion
            name: Optional agent name
            model: Model to use
            priority: Task priority

        Returns:
            Agent data dictionary
        """
        # Create the initial agent with Ralph loop metadata
        agent_name = name or f"ralph-{str(uuid.uuid4())[:8]}"

        # Add Ralph loop instructions to the task
        ralph_task = f"""RALPH LOOP TASK:
{task}

INSTRUCTIONS:
- This is an iterative Ralph loop. You will work on this task multiple times.
- Each iteration, review your previous work and improve it.
- Continue refining until the task is complete or you reach the iteration limit.
- Current iteration: 1/{max_iterations}
"""
        if completion_promise:
            ralph_task += f"\n- When complete, output exactly: {completion_promise}"

        # Store Ralph metadata
        ralph_metadata = {
            "is_ralph_loop": True,
            "max_iterations": max_iterations,
            "current_iteration": 1,
            "completion_promise": completion_promise,
            "original_task": task,
        }

        # Spawn the initial agent
        agent = await self.spawn_agent(
            task=ralph_task,
            name=agent_name,
            model=model,
            supervised=False,  # Ralph loops run autonomously
            priority=priority,
        )

        # Store Ralph metadata in agent
        agent_id = agent["id"]
        self._ralph_loops[agent_id] = ralph_metadata

        # Start the Ralph loop monitor
        asyncio.create_task(self._monitor_ralph_loop(agent_id))

        return agent

    async def _monitor_ralph_loop(self, agent_id: str) -> None:
        """Monitor a Ralph loop and re-spawn iterations as needed."""
        while agent_id in self._ralph_loops:
            await asyncio.sleep(5)  # Check every 5 seconds

            agent = await self.get_agent(agent_id)
            if not agent:
                break

            ralph_meta = self._ralph_loops.get(agent_id)
            if not ralph_meta:
                break

            # Check if agent completed
            if agent["status"] in ["completed", "failed"]:
                current_iter = ralph_meta["current_iteration"]
                max_iter = ralph_meta["max_iterations"]

                # Check for completion promise in conversation
                if ralph_meta["completion_promise"]:
                    conversation = await self.db.get_conversation(agent_id)
                    if conversation:
                        last_msg = conversation[-1] if conversation else None
                        if last_msg and ralph_meta["completion_promise"] in str(last_msg.get("content", "")):
                            # Completion promise found
                            await self.notify(
                                agent_id, f"Ralph loop completed: {ralph_meta['completion_promise']} detected"
                            )
                            del self._ralph_loops[agent_id]
                            break

                # Check iteration limit
                if current_iter >= max_iter:
                    await self.notify(agent_id, f"Ralph loop completed: max iterations ({max_iter}) reached")
                    del self._ralph_loops[agent_id]
                    break

                # Continue with next iteration
                ralph_meta["current_iteration"] += 1

                # Get the conversation history for context
                conversation = await self.db.get_conversation(agent_id)

                # Create task for next iteration
                next_task = f"""RALPH LOOP CONTINUATION:
{ralph_meta["original_task"]}

INSTRUCTIONS:
- This is iteration {ralph_meta["current_iteration"]}/{ralph_meta["max_iterations"]} of the Ralph loop.
- Review your previous work and continue improving it.
- Build on what you've done, fix any issues, and enhance the solution.
"""
                if ralph_meta["completion_promise"]:
                    next_task += f"\n- When complete, output exactly: {ralph_meta['completion_promise']}"

                # Add summary of previous work
                if conversation and len(conversation) > 1:
                    # Get last assistant message
                    for msg in reversed(conversation):
                        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                            preview = msg["content"][:500]
                            next_task += f"\n\nPREVIOUS WORK SUMMARY:\n{preview}..."
                            break

                # Spawn next iteration
                await self.spawn_agent(
                    task=next_task,
                    name=f"{agent['name'] or agent_id}-iter{ralph_meta['current_iteration']}",
                    model=agent.get("model"),
                    supervised=False,
                    priority=agent.get("priority", "normal"),
                )

                await self.notify(
                    agent_id, f"Ralph loop iteration {ralph_meta['current_iteration']}/{max_iter} started"
                )

    async def cancel_ralph_loop(self, agent_id: str) -> bool:
        """Cancel an active Ralph loop."""
        if agent_id in self._ralph_loops:
            del self._ralph_loops[agent_id]
            # Also terminate the current agent
            success = await self.terminate_agent(agent_id)
            if success:
                await self.notify(agent_id, f"Ralph loop {agent_id} cancelled")
            return success
        return False
