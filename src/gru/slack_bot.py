"""Slack bot interface for Gru."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

if TYPE_CHECKING:
    from gru.config import Config
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using sliding window."""

    CLEANUP_THRESHOLD = 100

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed and record it."""
        now = time.time()
        cutoff = now - self.window_seconds

        if user_id in self._requests:
            self._requests[user_id] = [t for t in self._requests[user_id] if t > cutoff]
        else:
            self._requests[user_id] = []

        if len(self._requests[user_id]) >= self.max_requests:
            return False

        self._requests[user_id].append(now)

        if len(self._requests) > self.CLEANUP_THRESHOLD:
            self._cleanup()

        return True

    def _cleanup(self) -> None:
        """Remove stale entries to prevent memory growth."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests = {
            uid: [t for t in times if t > cutoff]
            for uid, times in self._requests.items()
            if any(t > cutoff for t in times)
        }


class SlackBot:
    """Slack bot interface for Gru orchestrator."""

    MAX_MESSAGE_LENGTH = 3000  # Slack's limit is 4000, leave buffer

    def __init__(self, config: Config, orchestrator: Orchestrator) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._pending_options: dict[str, list[str]] = {}
        self._pending_messages: dict[str, list[tuple[str, str]]] = {}  # approval_id -> [(channel_id, ts)]
        self._rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        self._spawn_limiter = RateLimiter(max_requests=5, window_seconds=60)
        self._agent_numbers: dict[str, int] = {}  # agent_id -> number
        self._number_to_agent: dict[int, str] = {}  # number -> agent_id
        self._next_agent_number: int = 1
        self._agent_nicknames: dict[str, str] = {}  # agent_id -> nickname
        self._nickname_to_agent: dict[str, str] = {}  # nickname -> agent_id

        # Create Slack Bolt app
        self._app = AsyncApp(token=config.slack_bot_token)
        self._client: AsyncWebClient = self._app.client
        self._handler: AsyncSocketModeHandler | None = None

        # Register handlers
        self._register_commands()
        self._register_actions()
        self._register_messages()

    def _is_admin(self, user_id: str) -> bool:
        """Check if user is an admin."""
        return user_id in self.config.slack_admin_ids

    def _assign_agent_number(self, agent_id: str) -> int:
        """Assign a short number to an agent and return it."""
        if agent_id in self._agent_numbers:
            return self._agent_numbers[agent_id]
        num = self._next_agent_number
        self._next_agent_number += 1
        self._agent_numbers[agent_id] = num
        self._number_to_agent[num] = agent_id
        return num

    def _resolve_agent_ref(self, ref: str) -> str | None:
        """Resolve a nickname, number, or agent ID to the actual agent ID."""
        # Check nickname first
        if ref in self._nickname_to_agent:
            return self._nickname_to_agent[ref]
        # Check number
        if ref.isdigit():
            num = int(ref)
            return self._number_to_agent.get(num)
        # Assume it's a full agent ID
        return ref

    def _set_agent_nickname(self, agent_id: str, nickname: str) -> bool:
        """Set a nickname for an agent. Returns False if nickname is taken."""
        # Remove old nickname if exists
        old_nick = self._agent_nicknames.get(agent_id)
        if old_nick:
            del self._nickname_to_agent[old_nick]
        # Check if new nickname is taken by another agent
        if nickname in self._nickname_to_agent and self._nickname_to_agent[nickname] != agent_id:
            return False
        self._agent_nicknames[agent_id] = nickname
        self._nickname_to_agent[nickname] = agent_id
        return True

    def _get_agent_display(self, agent_id: str) -> str:
        """Get display string for agent with its number and optional nickname."""
        num = self._agent_numbers.get(agent_id)
        nick = self._agent_nicknames.get(agent_id)
        if num and nick:
            return f"[{num}:{nick}]"
        elif num:
            return f"[{num}]"
        return agent_id

    async def _respond_ephemeral(self, respond: Any, text: str) -> None:
        """Send an ephemeral response."""
        await respond(text=text, response_type="ephemeral")

    async def _respond(self, respond: Any, text: str, blocks: list | None = None) -> None:
        """Send a response."""
        if blocks:
            await respond(text=text, blocks=blocks)
        else:
            await respond(text=text)

    def _split_message(self, text: str) -> list[str]:
        """Split text into chunks at line boundaries."""
        chunks = []
        current = ""

        for line in text.split("\n"):
            if len(current) + len(line) + 1 > self.MAX_MESSAGE_LENGTH:
                if current:
                    chunks.append(current)
                if len(line) > self.MAX_MESSAGE_LENGTH:
                    for i in range(0, len(line), self.MAX_MESSAGE_LENGTH):
                        chunks.append(line[i : i + self.MAX_MESSAGE_LENGTH])
                    current = ""
                else:
                    current = line
            else:
                current = current + "\n" + line if current else line

        if current:
            chunks.append(current)

        return chunks

    def _register_commands(self) -> None:
        """Register slash command handler."""

        @self._app.command("/gru")
        async def handle_gru_command(ack: Any, command: dict, respond: Any) -> None:
            await ack()

            user_id = command.get("user_id", "")
            if not self._is_admin(user_id):
                await self._respond_ephemeral(respond, "Unauthorized")
                return

            if not self._rate_limiter.is_allowed(user_id):
                await self._respond_ephemeral(respond, "Rate limit exceeded. Please wait.")
                return

            text = command.get("text", "").strip()
            parts = text.split() if text else []

            if not parts:
                await self._respond(respond, "Usage: /gru <command> [args]")
                return

            cmd = parts[0].lower()
            args = parts[1:]

            handlers = {
                "help": self._cmd_help,
                "spawn": self._cmd_spawn,
                "status": self._cmd_status,
                "list": self._cmd_list,
                "pause": self._cmd_pause,
                "resume": self._cmd_resume,
                "terminate": self._cmd_terminate,
                "nudge": self._cmd_nudge,
                "approve": self._cmd_approve,
                "reject": self._cmd_reject,
                "pending": self._cmd_pending,
                "logs": self._cmd_logs,
                "secret": self._cmd_secret,
                "template": self._cmd_template,
            }

            handler = handlers.get(cmd)
            if handler:
                await handler(respond, args, user_id)
            else:
                await self._respond(respond, f"Unknown command: {cmd}")

    async def _cmd_help(self, respond: Any, args: list[str], user_id: str) -> None:
        """Show help."""
        help_text = f"""*Gru Commands:*

*Agent Management:*
`/gru spawn <task>` - Spawn a new agent
  Options: `--oneshot`, `--supervised`, `--unsupervised`, `--priority high|normal|low`, `--workdir /path`
`/gru status [agent_id]` - Show status
`/gru list [running|paused|completed|failed]` - List agents
`/gru pause <agent_id>` - Pause agent
`/gru resume <agent_id>` - Resume agent
`/gru terminate <agent_id>` - Terminate agent
`/gru nudge <agent_id> <message>` - Send message to agent
`/gru logs <agent_id>` - Get conversation logs

*Approvals:*
`/gru pending` - Show pending approvals
`/gru approve <approval_id>` - Approve action
`/gru reject <approval_id>` - Reject action

*Secrets:*
`/gru secret set <key> <value>` - Set secret
`/gru secret get <key>` - Get secret
`/gru secret list` - List secrets
`/gru secret delete <key>` - Delete secret

*Templates:*
`/gru template save <name> <task>` - Save template
`/gru template list` - List templates
`/gru template use <name>` - Use template
`/gru template delete <name>` - Delete template

Default workdir: `{self.config.default_workdir}`"""
        await self._respond(respond, help_text)

    async def _cmd_spawn(self, respond: Any, args: list[str], user_id: str) -> None:
        """Spawn a new agent."""
        if not self._spawn_limiter.is_allowed(user_id):
            await self._respond_ephemeral(respond, "Spawn rate limit exceeded. Please wait.")
            return

        if not args:
            await self._respond(
                respond,
                "Usage: /gru spawn <task> [--oneshot] [--supervised] [--priority X] [--workdir /path]",
            )
            return

        task_parts = []
        supervised = True
        priority = "normal"
        workdir = None
        timeout_mode = "block"
        oneshot = False

        i = 0
        while i < len(args):
            if args[i] == "--supervised":
                supervised = True
                i += 1
            elif args[i] == "--unsupervised":
                supervised = False
                i += 1
            elif args[i] == "--oneshot":
                oneshot = True
                supervised = False
                timeout_mode = "auto"
                i += 1
            elif args[i] == "--priority" and i + 1 < len(args):
                priority = args[i + 1]
                i += 2
            elif args[i] == "--workdir" and i + 1 < len(args):
                workdir = args[i + 1]
                i += 2
            else:
                task_parts.append(args[i])
                i += 1

        task = " ".join(task_parts)
        if not task:
            await self._respond(respond, "Task description required")
            return

        try:
            agent = await self.orchestrator.spawn_agent(
                task=task,
                supervised=supervised,
                priority=priority,
                workdir=workdir,
                timeout_mode=timeout_mode,
            )
        except ValueError as e:
            await self._respond(respond, f"Error: {e}")
            return

        mode_str = "oneshot (fully autonomous)" if oneshot else ("supervised" if supervised else "unsupervised")
        await self._respond(
            respond,
            f"Agent spawned: `{agent['id']}`\n"
            f"Task: {task}\n"
            f"Mode: {mode_str}\n"
            f"Priority: {priority}\n"
            f"Workdir: {agent.get('workdir', self.config.default_workdir)}",
        )

    async def _cmd_status(self, respond: Any, args: list[str], user_id: str) -> None:
        """Show status."""
        if args:
            agent = await self.orchestrator.get_agent(args[0])
            if not agent:
                await self._respond(respond, f"Agent not found: {args[0]}")
                return
            msg = (
                f"*Agent {agent['id']}*\n"
                f"Status: {agent['status']}\n"
                f"Task: {agent['task'][:100]}\n"
                f"Model: {agent['model']}\n"
                f"Supervised: {bool(agent['supervised'])}\n"
                f"Workdir: {agent.get('workdir', 'N/A')}\n"
                f"Created: {agent['created_at']}"
            )
            if agent.get("error"):
                msg += f"\nError: {agent['error']}"
            await self._respond(respond, msg)
        else:
            status = await self.orchestrator.get_status()
            await self._respond(
                respond,
                f"*Orchestrator Status*\n"
                f"Running: {status['running']}\n"
                f"Agents: {status['agents']['total']} total, "
                f"{status['agents']['running']} running, "
                f"{status['agents']['paused']} paused\n"
                f"Queue: {status['scheduler']['queued']} queued\n"
                f"Default workdir: {self.config.default_workdir}",
            )

    async def _cmd_list(self, respond: Any, args: list[str], user_id: str) -> None:
        """List agents."""
        status_filter = args[0] if args else None
        agents = await self.orchestrator.list_agents(status_filter)

        if not agents:
            await self._respond(respond, "No agents found")
            return

        lines = []
        for agent in agents[:20]:
            lines.append(f"`{agent['id']}` [{agent['status']}] {agent['task'][:40]}...")

        await self._respond(respond, "\n".join(lines))

    async def _cmd_pause(self, respond: Any, args: list[str], user_id: str) -> None:
        """Pause an agent."""
        if not args:
            await self._respond(respond, "Usage: /gru pause <agent_id>")
            return

        success = await self.orchestrator.pause_agent(args[0])
        if success:
            await self._respond(respond, f"Agent {args[0]} paused")
        else:
            await self._respond(respond, f"Could not pause agent {args[0]}")

    async def _cmd_resume(self, respond: Any, args: list[str], user_id: str) -> None:
        """Resume an agent."""
        if not args:
            await self._respond(respond, "Usage: /gru resume <agent_id>")
            return

        success = await self.orchestrator.resume_agent(args[0])
        if success:
            await self._respond(respond, f"Agent {args[0]} resumed")
        else:
            await self._respond(respond, f"Could not resume agent {args[0]}")

    async def _cmd_terminate(self, respond: Any, args: list[str], user_id: str) -> None:
        """Terminate an agent."""
        if not args:
            await self._respond(respond, "Usage: /gru terminate <agent_id>")
            return

        success = await self.orchestrator.terminate_agent(args[0])
        if success:
            await self._respond(respond, f"Agent {args[0]} terminated")
        else:
            await self._respond(respond, f"Could not terminate agent {args[0]}")

    async def _cmd_nudge(self, respond: Any, args: list[str], user_id: str) -> None:
        """Nudge an agent with a message."""
        if len(args) < 2:
            await self._respond(respond, "Usage: /gru nudge <agent_id> <message>")
            return

        agent_id = args[0]
        message = " ".join(args[1:])

        success = await self.orchestrator.nudge_agent(agent_id, message)
        if success:
            await self._respond(respond, f"Nudge sent to agent {agent_id}")
        else:
            await self._respond(respond, f"Could not nudge agent {agent_id}")

    async def _cmd_approve(self, respond: Any, args: list[str], user_id: str) -> None:
        """Approve a pending action."""
        if not args:
            await self._respond(respond, "Usage: /gru approve <approval_id>")
            return

        approval_id = args[0]

        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].set_result("Confirmed")
            del self._pending_approvals[approval_id]

        success = await self.orchestrator.approve(approval_id, approved=True)
        if success:
            await self._respond(respond, f"Approved: {approval_id}")
        else:
            await self._respond(respond, f"Approval not found or already resolved: {approval_id}")

    async def _cmd_reject(self, respond: Any, args: list[str], user_id: str) -> None:
        """Reject a pending action."""
        if not args:
            await self._respond(respond, "Usage: /gru reject <approval_id>")
            return

        approval_id = args[0]

        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].set_result(None)
            del self._pending_approvals[approval_id]

        success = await self.orchestrator.approve(approval_id, approved=False)
        if success:
            await self._respond(respond, f"Rejected: {approval_id}")
        else:
            await self._respond(respond, f"Approval not found or already resolved: {approval_id}")

    async def _cmd_pending(self, respond: Any, args: list[str], user_id: str) -> None:
        """Show pending approvals."""
        pending = await self.orchestrator.get_pending_approvals()

        if not pending:
            await self._respond(respond, "No pending approvals")
            return

        lines = []
        for p in pending:
            details = json.dumps(p["action_details"])[:50]
            lines.append(f"`{p['id']}`: {p['action_type']} - {details}...")

        await self._respond(respond, "\n".join(lines))

    async def _cmd_logs(self, respond: Any, args: list[str], user_id: str) -> None:
        """Show agent conversation logs."""
        if not args:
            await self._respond(respond, "Usage: /gru logs <agent_id>")
            return

        agent_id = args[0]
        conversation = await self.orchestrator.db.get_conversation(agent_id)

        if not conversation:
            await self._respond(respond, f"No logs found for agent {agent_id}")
            return

        lines = []
        for msg in conversation[-20:]:
            content = msg["content"]
            if isinstance(content, list):
                content = str(content)
            lines.append(f"[{msg['role']}] {content[:200]}")

        output = "\n\n".join(lines)
        chunks = self._split_message(output)
        for chunk in chunks:
            await self._respond(respond, f"```\n{chunk}\n```")

    async def _cmd_secret(self, respond: Any, args: list[str], user_id: str) -> None:
        """Manage secrets."""
        if not args:
            await self._respond(respond, "Usage: /gru secret <set|get|list|delete> ...")
            return

        subcmd = args[0].lower()

        if subcmd == "set" and len(args) >= 3:
            key = args[1]
            value = " ".join(args[2:])
            await self.orchestrator.secrets.set(key, value)
            await self._respond(respond, f"Secret '{key}' set")

        elif subcmd == "get" and len(args) >= 2:
            key = args[1]
            secret_value = await self.orchestrator.secrets.get(key)
            if secret_value:
                await self._respond(respond, f"Secret '{key}' exists ({len(secret_value)} chars)")
            else:
                await self._respond(respond, f"Secret '{key}' not found")

        elif subcmd == "list":
            keys = await self.orchestrator.secrets.list_keys()
            if keys:
                await self._respond(respond, "Secrets:\n" + "\n".join(keys))
            else:
                await self._respond(respond, "No secrets stored")

        elif subcmd == "delete" and len(args) >= 2:
            key = args[1]
            success = await self.orchestrator.secrets.delete(key)
            if success:
                await self._respond(respond, f"Secret '{key}' deleted")
            else:
                await self._respond(respond, f"Secret '{key}' not found")

        else:
            await self._respond(respond, "Usage: /gru secret <set|get|list|delete> ...")

    async def _cmd_template(self, respond: Any, args: list[str], user_id: str) -> None:
        """Manage templates."""
        if not args:
            await self._respond(respond, "Usage: /gru template <save|list|use|delete> ...")
            return

        subcmd = args[0].lower()

        if subcmd == "save" and len(args) >= 3:
            name = args[1]
            task = " ".join(args[2:])
            await self.orchestrator.db.save_template(name, task)
            await self._respond(respond, f"Template '{name}' saved")

        elif subcmd == "list":
            templates = await self.orchestrator.db.list_templates()
            if templates:
                lines = [f"`{t['name']}`: {t['task'][:50]}..." for t in templates]
                await self._respond(respond, "Templates:\n" + "\n".join(lines))
            else:
                await self._respond(respond, "No templates saved")

        elif subcmd == "use" and len(args) >= 2:
            name = args[1]
            template = await self.orchestrator.db.get_template(name)
            if template:
                agent = await self.orchestrator.spawn_agent(
                    task=template["task"],
                    model=template.get("model"),
                    system_prompt=template.get("system_prompt"),
                    supervised=bool(template.get("supervised", 1)),
                    priority=template.get("priority", "normal"),
                )
                await self._respond(respond, f"Agent spawned from template: `{agent['id']}`")
            else:
                await self._respond(respond, f"Template '{name}' not found")

        elif subcmd == "delete" and len(args) >= 2:
            name = args[1]
            success = await self.orchestrator.db.delete_template(name)
            if success:
                await self._respond(respond, f"Template '{name}' deleted")
            else:
                await self._respond(respond, f"Template '{name}' not found")

        else:
            await self._respond(respond, "Usage: /gru template <save|list|use|delete> ...")

    def _register_actions(self) -> None:
        """Register interactive component handlers."""

        @self._app.action("gru_approve")
        async def handle_approve(ack: Any, body: dict, client: AsyncWebClient) -> None:
            await ack()

            user_id = body.get("user", {}).get("id", "")
            if not self._is_admin(user_id):
                return

            action = body.get("actions", [{}])[0]
            approval_id = action.get("value", "")

            if approval_id not in self._pending_approvals:
                await client.chat_update(
                    channel=body["channel"]["id"],
                    ts=body["message"]["ts"],
                    text=f"Expired: {approval_id}",
                    blocks=[],
                )
                return

            self._pending_approvals[approval_id].set_result("Confirmed")
            del self._pending_approvals[approval_id]
            self._pending_options.pop(approval_id, None)
            self._pending_messages.pop(approval_id, None)

            await self.orchestrator.approve(approval_id, approved=True)
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f"Approved: {approval_id}",
                blocks=[],
            )

        @self._app.action("gru_reject")
        async def handle_reject(ack: Any, body: dict, client: AsyncWebClient) -> None:
            await ack()

            user_id = body.get("user", {}).get("id", "")
            if not self._is_admin(user_id):
                return

            action = body.get("actions", [{}])[0]
            approval_id = action.get("value", "")

            if approval_id not in self._pending_approvals:
                await client.chat_update(
                    channel=body["channel"]["id"],
                    ts=body["message"]["ts"],
                    text=f"Expired: {approval_id}",
                    blocks=[],
                )
                return

            self._pending_approvals[approval_id].set_result(None)
            del self._pending_approvals[approval_id]
            self._pending_options.pop(approval_id, None)
            self._pending_messages.pop(approval_id, None)

            await self.orchestrator.approve(approval_id, approved=False)
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f"Rejected: {approval_id}",
                blocks=[],
            )

        @self._app.action("gru_option")
        async def handle_option(ack: Any, body: dict, client: AsyncWebClient) -> None:
            await ack()

            user_id = body.get("user", {}).get("id", "")
            if not self._is_admin(user_id):
                return

            action = body.get("actions", [{}])[0]
            value = action.get("value", "")

            # Parse value: "approval_id:option_index"
            if ":" not in value:
                return

            approval_id, idx_str = value.rsplit(":", 1)
            try:
                option_idx = int(idx_str)
            except ValueError:
                return

            if approval_id not in self._pending_approvals:
                await client.chat_update(
                    channel=body["channel"]["id"],
                    ts=body["message"]["ts"],
                    text=f"Expired: {approval_id}",
                    blocks=[],
                )
                return

            options = self._pending_options.get(approval_id, [])
            if option_idx < 0 or option_idx >= len(options):
                return

            option_text = options[option_idx]
            self._pending_approvals[approval_id].set_result(option_text)
            del self._pending_approvals[approval_id]
            self._pending_options.pop(approval_id, None)
            self._pending_messages.pop(approval_id, None)

            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f"Selected: {option_text}",
                blocks=[],
            )

    def _get_chat_tools(self) -> list:
        """Get tool definitions for natural language chat."""
        from gru.claude import ToolDefinition

        return [
            ToolDefinition(
                name="spawn_agent",
                description="Spawn a new AI agent to perform a task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task description for the agent"},
                        "workdir": {"type": "string", "description": "Working directory path (optional)"},
                        "oneshot": {
                            "type": "boolean",
                            "description": "If true, run fully autonomous (no approvals)",
                            "default": False,
                        },
                        "supervised": {
                            "type": "boolean",
                            "description": "If true, require approval for file writes and bash",
                            "default": True,
                        },
                        "priority": {"type": "string", "enum": ["high", "normal", "low"], "default": "normal"},
                    },
                    "required": ["task"],
                },
            ),
            ToolDefinition(
                name="terminate_agent",
                description="Terminate/kill a running or failed agent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}
                    },
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="pause_agent",
                description="Pause a running agent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}
                    },
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="resume_agent",
                description="Resume a paused agent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}
                    },
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="get_status",
                description="Get status of a specific agent or overall system status",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID (optional)"}
                    },
                },
            ),
            ToolDefinition(
                name="list_agents",
                description="List all agents, optionally filtered by status",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["running", "paused", "completed", "failed"],
                            "description": "Filter by status (optional)",
                        }
                    },
                },
            ),
            ToolDefinition(
                name="get_pending_approvals",
                description="Get list of pending approval requests",
                input_schema={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="approve_action",
                description="Approve a pending approval request",
                input_schema={
                    "type": "object",
                    "properties": {"approval_id": {"type": "string", "description": "The approval ID to approve"}},
                    "required": ["approval_id"],
                },
            ),
            ToolDefinition(
                name="reject_action",
                description="Reject a pending approval request",
                input_schema={
                    "type": "object",
                    "properties": {"approval_id": {"type": "string", "description": "The approval ID to reject"}},
                    "required": ["approval_id"],
                },
            ),
            ToolDefinition(
                name="nudge_agent",
                description="Send a message to a running agent to ask for status or give instructions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1'), nickname, or full ID"},
                        "message": {
                            "type": "string",
                            "description": "Message to send (default: ask for status)",
                            "default": "Briefly report your current progress and what you're working on.",
                        },
                    },
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="nickname_agent",
                description="Assign a nickname to an agent for easier reference",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"},
                        "nickname": {"type": "string", "description": "Nickname to assign (e.g. 'linter', 'deploy')"},
                    },
                    "required": ["agent_ref", "nickname"],
                },
            ),
        ]

    async def _handle_tool_use(self, tool_name: str, params: dict) -> str:
        """Handle a tool use and return the result message."""
        if tool_name == "spawn_agent":
            task = params.get("task", "")
            workdir = params.get("workdir")
            oneshot = params.get("oneshot", False)
            supervised = not oneshot and params.get("supervised", True)
            timeout_mode = "auto" if oneshot else "block"
            priority = params.get("priority", "normal")

            agent = await self.orchestrator.spawn_agent(
                task=task,
                supervised=supervised,
                priority=priority,
                workdir=workdir,
                timeout_mode=timeout_mode,
            )

            num = self._assign_agent_number(agent["id"])
            mode_str = "oneshot (fully autonomous)" if oneshot else ("supervised" if supervised else "unsupervised")
            return (
                f"Agent spawned: `[{num}]` {agent['id']}\n"
                f"Task: {task}\n"
                f"Mode: {mode_str}\n"
                f"Priority: {priority}\n"
                f"Workdir: {agent.get('workdir', self.config.default_workdir)}"
            )

        elif tool_name == "terminate_agent":
            ref = params.get("agent_ref", "")
            agent_id = self._resolve_agent_ref(ref)
            if not agent_id:
                return f"Unknown agent: {ref}"
            success = await self.orchestrator.terminate_agent(agent_id)
            display = self._get_agent_display(agent_id)
            return f"Agent {display} terminated" if success else f"Could not terminate agent {display}"

        elif tool_name == "pause_agent":
            ref = params.get("agent_ref", "")
            agent_id = self._resolve_agent_ref(ref)
            if not agent_id:
                return f"Unknown agent: {ref}"
            success = await self.orchestrator.pause_agent(agent_id)
            display = self._get_agent_display(agent_id)
            return f"Agent {display} paused" if success else f"Could not pause agent {display}"

        elif tool_name == "resume_agent":
            ref = params.get("agent_ref", "")
            agent_id = self._resolve_agent_ref(ref)
            if not agent_id:
                return f"Unknown agent: {ref}"
            success = await self.orchestrator.resume_agent(agent_id)
            display = self._get_agent_display(agent_id)
            return f"Agent {display} resumed" if success else f"Could not resume agent {display}"

        elif tool_name == "get_status":
            ref = params.get("agent_ref")
            if ref:
                agent_id = self._resolve_agent_ref(ref)
                if not agent_id:
                    return f"Unknown agent: {ref}"
                agent_info = await self.orchestrator.get_agent(agent_id)
                if not agent_info:
                    return f"Agent not found: {ref}"
                display = self._get_agent_display(agent_id)
                msg = (
                    f"*Agent {display}*\n"
                    f"Status: {agent_info['status']}\n"
                    f"Task: {agent_info['task'][:100]}\n"
                    f"Model: {agent_info['model']}\n"
                    f"Supervised: {bool(agent_info['supervised'])}\n"
                    f"Workdir: {agent_info.get('workdir', 'N/A')}\n"
                    f"Created: {agent_info['created_at']}"
                )
                if agent_info.get("error"):
                    msg += f"\nError: {agent_info['error']}"
                return msg
            else:
                status = await self.orchestrator.get_status()
                return (
                    f"*Orchestrator Status*\n"
                    f"Running: {status['running']}\n"
                    f"Agents: {status['agents']['total']} total, "
                    f"{status['agents']['running']} running, "
                    f"{status['agents']['paused']} paused\n"
                    f"Queue: {status['scheduler']['queued']} queued"
                )

        elif tool_name == "list_agents":
            status_filter = params.get("status")
            agents = await self.orchestrator.list_agents(status_filter)
            if not agents:
                return "No agents found"
            lines = []
            for a in agents[:20]:
                num = self._assign_agent_number(a["id"])
                nick = self._agent_nicknames.get(a["id"])
                prefix = f"`[{num}:{nick}]`" if nick else f"`[{num}]`"
                lines.append(f"{prefix} [{a['status']}] {a['task']}")
            return "\n".join(lines)

        elif tool_name == "get_pending_approvals":
            pending = await self.orchestrator.get_pending_approvals()
            if not pending:
                return "No pending approvals"
            lines = [f"`{p['id']}`: {p['action_type']} for agent {p['agent_id']}" for p in pending]
            return "\n".join(lines)

        elif tool_name == "approve_action":
            approval_id = params.get("approval_id", "")
            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result("Confirmed")
                del self._pending_approvals[approval_id]
            success = await self.orchestrator.approve(approval_id, approved=True)
            return f"Approved: {approval_id}" if success else f"Approval not found: {approval_id}"

        elif tool_name == "reject_action":
            approval_id = params.get("approval_id", "")
            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result(None)
                del self._pending_approvals[approval_id]
            success = await self.orchestrator.approve(approval_id, approved=False)
            return f"Rejected: {approval_id}" if success else f"Approval not found: {approval_id}"

        elif tool_name == "nudge_agent":
            ref = params.get("agent_ref", "")
            agent_id = self._resolve_agent_ref(ref)
            if not agent_id:
                return f"Unknown agent: {ref}"
            message = params.get("message", "Briefly report your current progress and what you're working on.")
            success = await self.orchestrator.nudge_agent(agent_id, message)
            display = self._get_agent_display(agent_id)
            if success:
                return f"Nudge sent to agent {display}. Response will appear when agent processes it."
            return f"Could not nudge agent {display} (not running or not found)"

        elif tool_name == "nickname_agent":
            ref = params.get("agent_ref", "")
            agent_id = self._resolve_agent_ref(ref)
            if not agent_id:
                return f"Unknown agent: {ref}"
            nickname = params.get("nickname", "").strip()
            if not nickname:
                return "Nickname cannot be empty"
            agent_num = self._agent_numbers.get(agent_id, 0)
            if self._set_agent_nickname(agent_id, nickname):
                return f"Agent [{agent_num}] is now nicknamed `{nickname}`"
            return f"Nickname `{nickname}` is already taken"

        return f"Unknown tool: {tool_name}"

    def _register_messages(self) -> None:
        """Register message handlers for natural language."""

        @self._app.event("message")
        async def handle_message(event: dict, say: Any) -> None:
            if event.get("channel_type") != "im":
                return

            user_id = event.get("user", "")
            if not self._is_admin(user_id):
                return

            if not self._rate_limiter.is_allowed(user_id):
                await say("Rate limit exceeded. Please wait.")
                return

            text = event.get("text", "").strip()
            if not text:
                return

            if event.get("bot_id"):
                return

            system_prompt, _ = await self._build_chat_context()
            tools = self._get_chat_tools()

            try:
                response = await self.orchestrator.claude.send_message(
                    messages=[{"role": "user", "content": text}],
                    system=system_prompt,
                    max_tokens=1000,
                    tools=tools,
                )

                if response.tool_uses:
                    for tool_use in response.tool_uses:
                        result = await self._handle_tool_use(tool_use.name, tool_use.input)
                        await say(result)
                    return

                await say(response.content or "I couldn't generate a response.")
            except Exception as e:
                await say(f"Error processing message: {e}")

    async def _build_chat_context(self) -> tuple[str, str]:
        """Build system prompt and context for natural language chat."""
        status = await self.orchestrator.get_status()
        pending = await self.orchestrator.get_pending_approvals()
        recent_agents = await self.orchestrator.list_agents()

        lines = [
            "Current Gru State:",
            f"- Running agents: {status['agents']['running']}",
            f"- Queued tasks: {status['scheduler']['queued']}",
            f"- Pending approvals: {len(pending)}",
            f"- Default workdir: {self.config.default_workdir}",
            f"- Total agents (all time): {status['agents']['total']}",
            "",
            "Recent agents:",
        ]

        for a in recent_agents[:5]:
            lines.append(f"  - {a['id']} [{a['status']}]: {a['task'][:50]}...")
            if a.get("workdir"):
                lines.append(f"    workdir: {a['workdir']}")
            if a.get("error"):
                lines.append(f"    error: {a['error'][:100]}")

        if pending:
            lines.append("")
            lines.append("Pending approvals:")
            for p in pending[:3]:
                lines.append(f"  - {p['id']}: {p['action_type']} for agent {p['agent_id']}")

        context_info = "\n".join(lines)

        system_prompt = f"""You are Gru, an AI agent orchestrator running on the user's machine.
You help manage AI agents that perform tasks.

{context_info}

You have access to a spawn_agent tool. Use it when the user wants to start an agent.
Parse their natural language request into the appropriate parameters.

Modes:
- supervised (default): agent asks for approval before file writes and bash commands
- unsupervised: no approvals needed
- oneshot: fully autonomous, fire and forget (unsupervised + auto timeout)

If the user mentions a directory path, use it as the workdir.
If they say "oneshot" or imply they want to fire and forget, set oneshot=true.

For questions about status, agents, or other info, just respond with text.
Only use the spawn tool when they want to start new work.

Be concise and helpful."""

        return system_prompt, context_info

    def notify_callback(self, agent_id: str, message: str) -> None:
        """Callback for orchestrator notifications."""

        async def send() -> None:
            for admin_id in self.config.slack_admin_ids:
                try:
                    await self._client.chat_postMessage(
                        channel=admin_id,
                        text=f"[{agent_id}] {message}",
                    )
                except Exception as e:
                    logger.error(f"Failed to send Slack notification: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(send())
        except RuntimeError:
            pass

    def approval_callback(self, approval_id: str, details: dict) -> asyncio.Future:
        """Callback for orchestrator approval requests."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_approvals[approval_id] = future

        async def send() -> None:
            options = details.get("options")

            blocks: list[dict[str, Any]] = []

            if options and isinstance(options, list):
                self._pending_options[approval_id] = list(options)
                question = details.get("question", "Please select an option:")

                # Build option buttons
                option_blocks: list[dict[str, Any]] = []
                for idx, opt in enumerate(options):
                    display_text = str(opt)[:75] if len(str(opt)) > 75 else str(opt)
                    option_blocks.append(
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": display_text},
                            "action_id": "gru_option",
                            "value": f"{approval_id}:{idx}",
                        }
                    )
                # Add decline button
                option_blocks.append(
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Decline"},
                        "style": "danger",
                        "action_id": "gru_reject",
                        "value": approval_id,
                    }
                )

                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Input requested:* `{approval_id}`\n\n{question}"},
                    }
                )
                blocks.append({"type": "actions", "elements": option_blocks[:5]})  # Slack limits to 5 buttons

                # If more than 5 options, add additional action blocks
                if len(option_blocks) > 5:
                    for i in range(5, len(option_blocks), 5):
                        blocks.append({"type": "actions", "elements": option_blocks[i : i + 5]})

                text = f"Input requested: {approval_id}"
            else:
                details_str = json.dumps(details, indent=2)[:500]
                msg_text = f"*Approval requested:* `{approval_id}`\n```{details_str}```"
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": msg_text}})
                blocks.append(
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Approve"},
                                "style": "primary",
                                "action_id": "gru_approve",
                                "value": approval_id,
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Reject"},
                                "style": "danger",
                                "action_id": "gru_reject",
                                "value": approval_id,
                            },
                        ],
                    }
                )
                text = f"Approval requested: {approval_id}"

            sent_messages: list[tuple[str, str]] = []
            for admin_id in self.config.slack_admin_ids:
                try:
                    result = await self._client.chat_postMessage(
                        channel=admin_id,
                        text=text,
                        blocks=blocks,
                    )
                    if result.get("ok"):
                        sent_messages.append((result["channel"], result["ts"]))
                except Exception as e:
                    logger.error(f"Failed to send Slack approval request: {e}")

            if sent_messages:
                self._pending_messages[approval_id] = sent_messages

        loop.create_task(send())
        return future

    async def cancel_approval(self, approval_id: str) -> None:
        """Cancel a pending approval and clean up resources."""
        self._pending_approvals.pop(approval_id, None)
        self._pending_options.pop(approval_id, None)

        messages = self._pending_messages.pop(approval_id, [])
        for channel_id, ts in messages:
            try:
                await self._client.chat_update(
                    channel=channel_id,
                    ts=ts,
                    text=f"Expired: {approval_id}",
                    blocks=[],
                )
            except Exception as e:
                logger.error(f"Failed to update expired Slack message: {e}")

    async def start(self) -> None:
        """Start the Slack bot."""
        # Set up orchestrator callbacks
        self.orchestrator.set_notify_callback(self.notify_callback)
        self.orchestrator.set_approval_callback(self.approval_callback)
        self.orchestrator.set_cancel_approval_callback(self.cancel_approval)

        # Start Socket Mode handler
        self._handler = AsyncSocketModeHandler(self._app, self.config.slack_app_token)
        await self._handler.start_async()
        logger.info("Slack bot started via Socket Mode")

    async def stop(self) -> None:
        """Stop the Slack bot."""
        if self._handler:
            await self._handler.close_async()
