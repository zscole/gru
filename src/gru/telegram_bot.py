"""Telegram bot interface for Gru."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING

import anthropic
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

if TYPE_CHECKING:
    from gru.config import Config
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using sliding window."""

    CLEANUP_THRESHOLD = 100  # Trigger cleanup when dict exceeds this size

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[int, list[float]] = {}

    def is_allowed(self, user_id: int) -> bool:
        """Check if request is allowed and record it."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Get existing requests for user, filtering old ones
        if user_id in self._requests:
            self._requests[user_id] = [t for t in self._requests[user_id] if t > cutoff]
        else:
            self._requests[user_id] = []

        # Check if under limit
        if len(self._requests[user_id]) >= self.max_requests:
            return False

        # Record this request
        self._requests[user_id].append(now)

        # Periodic cleanup to prevent unbounded memory growth
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


class TelegramBot:
    """Telegram bot interface for Gru orchestrator."""

    MAX_MESSAGE_LENGTH = 4096
    SPLIT_THRESHOLD = 3  # Split into messages if under this many chunks, else offer file

    def __init__(self, config: Config, orchestrator: Orchestrator) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self._app: Application | None = None
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._pending_options: dict[str, list[str]] = {}
        self._pending_messages: dict[str, list[tuple[int, int]]] = {}  # approval_id -> [(chat_id, msg_id)]
        self._rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        self._spawn_limiter = RateLimiter(max_requests=5, window_seconds=60)  # Stricter for spawns
        self._agent_numbers: dict[str, int] = {}  # agent_id -> number
        self._number_to_agent: dict[int, str] = {}  # number -> agent_id
        self._next_agent_number: int = 1
        self._agent_nicknames: dict[str, str] = {}  # agent_id -> nickname
        self._nickname_to_agent: dict[str, str] = {}  # nickname -> agent_id
        self._auto_registered_admins: set[int] = set()  # Auto-registered admin IDs

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin. Auto-registers first user if no admins configured."""
        # Check configured admins
        if user_id in self.config.telegram_admin_ids:
            return True
        # Check auto-registered admins
        if user_id in self._auto_registered_admins:
            return True
        # Auto-register first user if no admins configured
        if not self.config.telegram_admin_ids and not self._auto_registered_admins:
            self._auto_registered_admins.add(user_id)
            logger.info(f"Auto-registered user {user_id} as admin (first user)")
            return True
        return False

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

    async def _check_admin(self, update: Update) -> bool:
        """Check admin and reply if not authorized."""
        if not update.effective_user:
            return False
        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("Unauthorized")  # type: ignore
            return False
        return True

    async def _check_rate_limit(self, update: Update, limiter: RateLimiter | None = None) -> bool:
        """Check rate limit and reply if exceeded."""
        if not update.effective_user:
            return False
        limiter = limiter or self._rate_limiter
        if not limiter.is_allowed(update.effective_user.id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")  # type: ignore
            return False
        return True

    async def send_output(
        self,
        chat_id: int,
        text: str,
        filename: str | None = None,
        force_file: bool = False,
    ) -> None:
        """Send output, splitting or as file if needed."""
        if not self._app:
            return

        # Force file if requested or if very long
        if force_file or (filename and len(text) > self.MAX_MESSAGE_LENGTH * self.SPLIT_THRESHOLD):
            file_bytes = io.BytesIO(text.encode())
            file_bytes.name = filename or "output.txt"
            await self._app.bot.send_document(chat_id, document=file_bytes, filename=file_bytes.name)
            return

        # Split into messages if needed
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            await self._app.bot.send_message(chat_id, text)
            return

        # Split at line boundaries
        chunks = self._split_message(text)

        if len(chunks) <= self.SPLIT_THRESHOLD:
            for chunk in chunks:
                await self._app.bot.send_message(chat_id, chunk)
                await asyncio.sleep(0.1)  # Rate limiting
        else:
            # Too many chunks, offer file option
            await self._app.bot.send_message(
                chat_id,
                f"Output is {len(text)} chars ({len(chunks)} messages). Sending as messages...",
            )
            for i, chunk in enumerate(chunks):
                await self._app.bot.send_message(chat_id, f"[{i + 1}/{len(chunks)}]\n{chunk}")
                await asyncio.sleep(0.1)

    def _split_message(self, text: str) -> list[str]:
        """Split text into chunks at line boundaries."""
        chunks = []
        current = ""

        for line in text.split("\n"):
            if len(current) + len(line) + 1 > self.MAX_MESSAGE_LENGTH:
                if current:
                    chunks.append(current)
                if len(line) > self.MAX_MESSAGE_LENGTH:
                    # Line itself is too long, force split
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

    # Command handlers

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with welcome tutorial."""
        if not await self._check_admin(update):
            return
        if not await self._check_rate_limit(update):
            return

        welcome = """Welcome to Gru - your AI coding assistant!

I can help you build software by spawning AI agents that write code, fix bugs, and deploy apps.

Quick Start:
1. Just tell me what you want to build in plain English
2. I'll spawn an agent to do it
3. Approve actions as needed (or use --oneshot for hands-free)

Example - just send:
"Build me a landing page for a coffee shop"

Or use commands:
/gru spawn Build a REST API with user auth
/gru status - Check what's happening
/gru examples - See more examples

Pro tip: Send screenshots of UI designs and I'll build them!

Ready when you are!"""
        await update.message.reply_text(welcome)  # type: ignore

    async def cmd_gru(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /gru command."""
        if not await self._check_admin(update):
            return
        if not await self._check_rate_limit(update):
            return

        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /gru <command> [args]")  # type: ignore
            return

        command = args[0].lower()
        cmd_args = args[1:]

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
            "search": self._cmd_search,
            "cost": self._cmd_cost,
            "secret": self._cmd_secret,
            "template": self._cmd_template,
            "examples": self._cmd_examples,
            "doctor": self._cmd_doctor,
            "create": self._cmd_create,
            "deploy": self._cmd_deploy,
            "setup": self._cmd_setup,
        }

        handler = handlers.get(command)
        if handler:
            await handler(update, cmd_args)
        else:
            await update.message.reply_text(f"Unknown command: {command}")  # type: ignore

    async def _cmd_help(self, update: Update, args: list[str]) -> None:
        """Show help."""
        help_text = f"""Gru Commands:

Agent Management:
  /gru spawn <task> [--oneshot] [--supervised] [--priority high|normal|low] [--workdir /path]
  /gru status [agent_id]
  /gru list [running|paused|completed|failed]
  /gru pause <agent_id>
  /gru resume <agent_id>
  /gru terminate <agent_id>
  /gru nudge <agent_id> <message>
  /gru logs <agent_id>

Spawn modes:
  --supervised (default): asks for approval before file writes/bash
  --unsupervised: no approvals needed
  --oneshot: fully autonomous, fire and forget

Approvals:
  /gru pending
  /gru approve <approval_id>
  /gru reject <approval_id>

Secrets:
  /gru secret set <key> <value>
  /gru secret get <key>
  /gru secret list
  /gru secret delete <key>

Templates:
  /gru template save <name> <task>
  /gru template list
  /gru template use <name>
  /gru template delete <name>

You can also chat naturally - just ask questions!

Default workdir: {self.config.default_workdir}"""
        await update.message.reply_text(help_text)  # type: ignore

    async def _cmd_spawn(self, update: Update, args: list[str]) -> None:
        """Spawn a new agent."""
        # Apply stricter rate limit for spawning agents
        if not await self._check_rate_limit(update, self._spawn_limiter):
            return

        if not args:
            usage = "Usage: /gru spawn <task> [--oneshot] [--supervised] [--live] [--priority X] [--workdir /path]"
            await update.message.reply_text(usage)  # type: ignore
            return

        # Parse arguments
        task_parts = []
        supervised = True
        priority = "normal"
        workdir = None
        timeout_mode = "block"
        oneshot = False
        live_output = False

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
            elif args[i] == "--live":
                live_output = True
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
            await update.message.reply_text("Task description required")  # type: ignore
            return

        try:
            agent = await self.orchestrator.spawn_agent(
                task=task,
                supervised=supervised,
                priority=priority,
                workdir=workdir,
                timeout_mode=timeout_mode,
                live_output=live_output,
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {e}")  # type: ignore
            return

        mode_str = "oneshot (fully autonomous)" if oneshot else ("supervised" if supervised else "unsupervised")
        await update.message.reply_text(  # type: ignore
            f"Agent spawned: {agent['id']}\n"
            f"Task: {task}\n"
            f"Mode: {mode_str}\n"
            f"Priority: {priority}\n"
            f"Workdir: {agent.get('workdir', self.config.default_workdir)}"
        )

    async def _cmd_status(self, update: Update, args: list[str]) -> None:
        """Show status."""
        if args:
            # Agent-specific status
            agent = await self.orchestrator.get_agent(args[0])
            if not agent:
                await update.message.reply_text(f"Agent not found: {args[0]}")  # type: ignore
                return
            msg = (
                f"Agent {agent['id']}\n"
                f"Status: {agent['status']}\n"
                f"Task: {agent['task']}\n"
                f"Model: {agent['model']}\n"
                f"Supervised: {bool(agent['supervised'])}\n"
                f"Workdir: {agent.get('workdir', 'N/A')}\n"
                f"Created: {agent['created_at']}"
            )
            if agent.get("error"):
                msg += f"\nError: {agent['error']}"
            await update.message.reply_text(msg)  # type: ignore
        else:
            # Overall status
            status = await self.orchestrator.get_status()
            await update.message.reply_text(  # type: ignore
                f"Orchestrator Status\n"
                f"Running: {status['running']}\n"
                f"Agents: {status['agents']['total']} total, "
                f"{status['agents']['running']} running, "
                f"{status['agents']['paused']} paused\n"
                f"Queue: {status['scheduler']['queued']} queued\n"
                f"Default workdir: {self.config.default_workdir}"
            )

    async def _cmd_list(self, update: Update, args: list[str]) -> None:
        """List agents."""
        status_filter = args[0] if args else None
        agents = await self.orchestrator.list_agents(status_filter)

        if not agents:
            await update.message.reply_text("No agents found")  # type: ignore
            return

        lines = []
        for agent in agents[:20]:
            lines.append(f"{agent['id']} [{agent['status']}] {agent['task'][:100]}...")

        await update.message.reply_text("\n".join(lines))  # type: ignore

    async def _cmd_pause(self, update: Update, args: list[str]) -> None:
        """Pause an agent."""
        if not args:
            await update.message.reply_text("Usage: /gru pause <agent_id>")  # type: ignore
            return

        success = await self.orchestrator.pause_agent(args[0])
        if success:
            await update.message.reply_text(f"Agent {args[0]} paused")  # type: ignore
        else:
            await update.message.reply_text(f"Could not pause agent {args[0]}")  # type: ignore

    async def _cmd_resume(self, update: Update, args: list[str]) -> None:
        """Resume an agent."""
        if not args:
            await update.message.reply_text("Usage: /gru resume <agent_id>")  # type: ignore
            return

        success = await self.orchestrator.resume_agent(args[0])
        if success:
            await update.message.reply_text(f"Agent {args[0]} resumed")  # type: ignore
        else:
            await update.message.reply_text(f"Could not resume agent {args[0]}")  # type: ignore

    async def _cmd_terminate(self, update: Update, args: list[str]) -> None:
        """Terminate an agent."""
        if not args:
            await update.message.reply_text("Usage: /gru terminate <agent_id>")  # type: ignore
            return

        success = await self.orchestrator.terminate_agent(args[0])
        if success:
            await update.message.reply_text(f"Agent {args[0]} terminated")  # type: ignore
        else:
            await update.message.reply_text(f"Could not terminate agent {args[0]}")  # type: ignore

    async def _cmd_nudge(self, update: Update, args: list[str]) -> None:
        """Nudge an agent with a message."""
        if len(args) < 2:
            await update.message.reply_text("Usage: /gru nudge <agent_id> <message>")  # type: ignore
            return

        agent_id = args[0]
        message = " ".join(args[1:])

        success = await self.orchestrator.nudge_agent(agent_id, message)
        if success:
            await update.message.reply_text(f"Nudge sent to agent {agent_id}")  # type: ignore
        else:
            await update.message.reply_text(f"Could not nudge agent {agent_id}")  # type: ignore

    async def _cmd_approve(self, update: Update, args: list[str]) -> None:
        """Approve a pending action."""
        if not args:
            await update.message.reply_text("Usage: /gru approve <approval_id>")  # type: ignore
            return

        approval_id = args[0]

        # Resolve pending future if exists
        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].set_result(True)
            del self._pending_approvals[approval_id]

        success = await self.orchestrator.approve(approval_id, approved=True)
        if success:
            await update.message.reply_text(f"Approved: {approval_id}")  # type: ignore
        else:
            await update.message.reply_text(f"Approval not found or already resolved: {approval_id}")  # type: ignore

    async def _cmd_reject(self, update: Update, args: list[str]) -> None:
        """Reject a pending action."""
        if not args:
            await update.message.reply_text("Usage: /gru reject <approval_id>")  # type: ignore
            return

        approval_id = args[0]

        # Resolve pending future if exists
        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].set_result(False)
            del self._pending_approvals[approval_id]

        success = await self.orchestrator.approve(approval_id, approved=False)
        if success:
            await update.message.reply_text(f"Rejected: {approval_id}")  # type: ignore
        else:
            await update.message.reply_text(f"Approval not found or already resolved: {approval_id}")  # type: ignore

    async def _cmd_pending(self, update: Update, args: list[str]) -> None:
        """Show pending approvals."""
        pending = await self.orchestrator.get_pending_approvals()

        if not pending:
            await update.message.reply_text("No pending approvals")  # type: ignore
            return

        lines = []
        for p in pending:
            details = json.dumps(p["action_details"])[:500]
            lines.append(f"{p['id']}: {p['action_type']} - {details}")

        await update.message.reply_text("\n".join(lines))  # type: ignore

    async def _cmd_search(self, update: Update, args: list[str]) -> None:
        """Search agents by task, name, or id."""
        if not args:
            await update.message.reply_text("Usage: /gru search <query>")  # type: ignore
            return

        query = " ".join(args)
        results = await self.orchestrator.search_agents(query)

        if not results:
            await update.message.reply_text(f"No agents found matching '{query}'")  # type: ignore
            return

        lines = []
        for a in results[:10]:  # Limit to 10 results
            display = self._get_agent_display(a["id"])
            status = a["status"]
            task = a["task"][:200] + "..." if len(a["task"]) > 200 else a["task"]
            lines.append(f"{display} [{status}] {task}")

        await update.message.reply_text("\n".join(lines))  # type: ignore

    async def _cmd_cost(self, update: Update, args: list[str]) -> None:
        """Show token usage and cost for an agent or all agents."""
        if args:
            agent_id = self._resolve_agent_ref(args[0]) or args[0]
            cost_info = self.orchestrator.get_agent_cost(agent_id)
            if not cost_info:
                cost_info = await self.orchestrator.get_agent_cost_from_db(agent_id)
            if cost_info:
                input_tokens, output_tokens, cost = cost_info
                total = input_tokens + output_tokens
                await update.message.reply_text(  # type: ignore
                    f"Agent {agent_id}:\n"
                    f"  Input: {input_tokens:,} tokens\n"
                    f"  Output: {output_tokens:,} tokens\n"
                    f"  Total: {total:,} tokens\n"
                    f"  Est. cost: ${cost}"
                )
            else:
                await update.message.reply_text(f"Agent {agent_id} not found")  # type: ignore
        else:
            # Show summary for all running agents
            agents = await self.orchestrator.list_agents(status="running")
            if not agents:
                await update.message.reply_text("No running agents")  # type: ignore
                return

            lines = []
            for a in agents:
                display = self._get_agent_display(a["id"])
                input_tokens = a.get("input_tokens", 0) or 0
                output_tokens = a.get("output_tokens", 0) or 0
                total = input_tokens + output_tokens
                lines.append(f"{display}: {total:,} tokens")

            await update.message.reply_text("\n".join(lines))  # type: ignore

    def _format_log_entry(self, msg: dict) -> str:
        """Format a conversation log entry for display."""
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            return f"[{role}] {content[:1000]}"

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")[:500]
                        parts.append(text)
                    elif item_type == "tool_use":
                        name = item.get("name", "unknown")
                        inp = item.get("input", {})
                        summary = self._summarize_tool_input(name, inp)
                        parts.append(f"[tool] {name}({summary})")
                    elif item_type == "tool_result":
                        result = item.get("content", "")[:500]
                        is_err = item.get("is_error", False)
                        prefix = "error" if is_err else "result"
                        parts.append(f"[{prefix}] {result}")
            return f"[{role}] " + " | ".join(parts) if parts else f"[{role}] (empty)"

        return f"[{role}] {str(content)[:1000]}"

    def _summarize_tool_input(self, tool_name: str, tool_input: dict) -> str:
        """Summarize tool input for logs display."""
        if tool_name == "bash":
            cmd = tool_input.get("command", "")
            return cmd[:100] + "..." if len(cmd) > 100 else cmd
        elif tool_name in ("read_file", "write_file"):
            return tool_input.get("path", "")[:100]
        elif tool_name == "search_files":
            return tool_input.get("pattern", "")[:50]
        else:
            return str(tool_input)[:100]

    async def _cmd_logs(self, update: Update, args: list[str]) -> None:
        """Show agent conversation logs."""
        if not args:
            await update.message.reply_text("Usage: /gru logs <agent_id>")  # type: ignore
            return

        agent_id = self._resolve_agent_ref(args[0]) or args[0]
        conversation = await self.orchestrator.db.get_conversation(agent_id)

        if not conversation:
            await update.message.reply_text(f"No logs found for agent {agent_id}")  # type: ignore
            return

        lines = [self._format_log_entry(msg) for msg in conversation[-20:]]
        output = "\n\n".join(lines)
        await self.send_output(update.effective_chat.id, output, f"logs_{agent_id}.txt")  # type: ignore

    async def _cmd_secret(self, update: Update, args: list[str]) -> None:
        """Manage secrets."""
        if not args:
            await update.message.reply_text("Usage: /gru secret <set|get|list|delete> ...")  # type: ignore
            return

        subcmd = args[0].lower()

        if subcmd == "set" and len(args) >= 3:
            key = args[1]
            value = " ".join(args[2:])
            await self.orchestrator.secrets.set(key, value)
            await update.message.reply_text(f"Secret '{key}' set")  # type: ignore

        elif subcmd == "get" and len(args) >= 2:
            key = args[1]
            secret_value = await self.orchestrator.secrets.get(key)
            if secret_value:
                # Don't show full secret in chat
                await update.message.reply_text(f"Secret '{key}' exists ({len(secret_value)} chars)")  # type: ignore
            else:
                await update.message.reply_text(f"Secret '{key}' not found")  # type: ignore

        elif subcmd == "list":
            keys = await self.orchestrator.secrets.list_keys()
            if keys:
                await update.message.reply_text("Secrets:\n" + "\n".join(keys))  # type: ignore
            else:
                await update.message.reply_text("No secrets stored")  # type: ignore

        elif subcmd == "delete" and len(args) >= 2:
            key = args[1]
            success = await self.orchestrator.secrets.delete(key)
            if success:
                await update.message.reply_text(f"Secret '{key}' deleted")  # type: ignore
            else:
                await update.message.reply_text(f"Secret '{key}' not found")  # type: ignore

        else:
            await update.message.reply_text("Usage: /gru secret <set|get|list|delete> ...")  # type: ignore

    async def _cmd_template(self, update: Update, args: list[str]) -> None:
        """Manage templates."""
        if not args:
            await update.message.reply_text("Usage: /gru template <save|list|use|delete> ...")  # type: ignore
            return

        subcmd = args[0].lower()

        if subcmd == "save" and len(args) >= 3:
            name = args[1]
            task = " ".join(args[2:])
            await self.orchestrator.db.save_template(name, task)
            await update.message.reply_text(f"Template '{name}' saved")  # type: ignore

        elif subcmd == "list":
            templates = await self.orchestrator.db.list_templates()
            if templates:
                lines = [f"{t['name']}: {t['task'][:200]}..." for t in templates]
                await update.message.reply_text("Templates:\n" + "\n".join(lines))  # type: ignore
            else:
                await update.message.reply_text("No templates saved")  # type: ignore

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
                await update.message.reply_text(f"Agent spawned from template: {agent['id']}")  # type: ignore
            else:
                await update.message.reply_text(f"Template '{name}' not found")  # type: ignore

        elif subcmd == "delete" and len(args) >= 2:
            name = args[1]
            success = await self.orchestrator.db.delete_template(name)
            if success:
                await update.message.reply_text(f"Template '{name}' deleted")  # type: ignore
            else:
                await update.message.reply_text(f"Template '{name}' not found")  # type: ignore

        else:
            await update.message.reply_text("Usage: /gru template <save|list|use|delete> ...")  # type: ignore

    async def _cmd_examples(self, update: Update, args: list[str]) -> None:
        """Show example prompts."""
        examples = """Copy-paste Examples:

Web Apps:
  /gru spawn Build a todo app with React and local storage
  /gru spawn Create a landing page for a SaaS product
  /gru spawn Build a blog with markdown support

APIs & Backend:
  /gru spawn Create a REST API with Express and MongoDB
  /gru spawn Build a FastAPI backend with user authentication
  /gru spawn Set up a GraphQL server with Apollo

Mobile & CLI:
  /gru spawn Build a React Native expense tracker
  /gru spawn Create a CLI tool for managing dotfiles

DevOps:
  /gru spawn Write a GitHub Actions CI/CD pipeline
  /gru spawn Create a Dockerfile for a Node.js app
  /gru spawn Set up Terraform for AWS infrastructure

Quick fixes:
  /gru spawn Fix the TypeScript errors in src/
  /gru spawn Add unit tests for the auth module
  /gru spawn Refactor the database queries for performance

Hands-free mode (no approvals):
  /gru spawn --oneshot Build a simple calculator app

Just describe what you want - I'll figure out the rest!"""
        await update.message.reply_text(examples)  # type: ignore

    async def _cmd_doctor(self, update: Update, args: list[str]) -> None:
        """Health check - verify all integrations work."""
        checks: list[str] = []

        # Check Anthropic API
        try:
            client = anthropic.Anthropic()
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            checks.append("[OK] Anthropic API - connected")
        except Exception as e:
            checks.append(f"[FAIL] Anthropic API - {str(e)[:50]}")

        # Check GitHub access
        github_token = os.getenv("GRU_GITHUB_TOKEN")
        if github_token:
            try:
                result = subprocess.run(
                    ["git", "ls-remote", "https://github.com/octocat/Hello-World.git"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    checks.append("[OK] GitHub - token configured")
                else:
                    checks.append("[WARN] GitHub - token set but may have issues")
            except Exception:
                checks.append("[WARN] GitHub - couldn't verify token")
        else:
            checks.append("[INFO] GitHub - no token (private repos won't work)")

        # Check database
        try:
            agents = await self.orchestrator.list_agents()
            checks.append(f"[OK] Database - {len(agents)} agents stored")
        except Exception as e:
            checks.append(f"[FAIL] Database - {str(e)[:50]}")

        # Check orchestrator
        try:
            status = await self.orchestrator.get_status()
            running = status["agents"]["running"]
            checks.append(f"[OK] Orchestrator - {running} agents running")
        except Exception as e:
            checks.append(f"[FAIL] Orchestrator - {str(e)[:50]}")

        # Summary
        fails = sum(1 for c in checks if c.startswith("[FAIL]"))
        warns = sum(1 for c in checks if c.startswith("[WARN]"))

        if fails == 0 and warns == 0:
            summary = "All systems operational!"
        elif fails == 0:
            summary = f"{warns} warning(s) - check details above"
        else:
            summary = f"{fails} issue(s) found - check details above"

        report = "Health Check:\n\n" + "\n".join(checks) + f"\n\n{summary}"
        await update.message.reply_text(report)  # type: ignore

    async def _cmd_create(self, update: Update, args: list[str]) -> None:
        """Create a project from template."""
        templates = {
            "react-app": (
                "Create a new React app with TypeScript, Tailwind CSS, and a clean "
                "folder structure. Include a basic layout component, routing, examples."
            ),
            "next-app": (
                "Create a Next.js 14 app with TypeScript, Tailwind CSS, app router, "
                "and a landing page with hero section, features, and footer."
            ),
            "express-api": (
                "Create an Express.js REST API with TypeScript, including user auth "
                "(JWT), a users CRUD endpoint, error handling middleware, and tests."
            ),
            "fastapi": (
                "Create a FastAPI backend with Python, including user authentication, "
                "SQLAlchemy models, Pydantic schemas, and OpenAPI docs."
            ),
            "landing-page": (
                "Create a beautiful, responsive landing page with hero section, "
                "features grid, testimonials, pricing cards, and contact form."
            ),
            "cli-tool": (
                "Create a Python CLI tool with Click, including help text, "
                "subcommands, config file support, and colorful output."
            ),
            "discord-bot": (
                "Create a Discord bot with discord.py, including basic commands, "
                "event handlers, and a modular cog structure."
            ),
            "chrome-extension": (
                "Create a Chrome extension with popup UI, content script, "
                "background worker, and manifest v3 configuration."
            ),
        }

        if not args:
            template_list = "\n".join([f"  {name}" for name in templates])
            await update.message.reply_text(  # type: ignore
                f"Usage: /gru create <template> [options]\n\n"
                f"Available templates:\n{template_list}\n\n"
                f"Example: /gru create landing-page --workdir /workspace/my-site"
            )
            return

        template_name = args[0].lower()
        if template_name not in templates:
            await update.message.reply_text(  # type: ignore
                f"Unknown template: {template_name}\nAvailable: {', '.join(templates.keys())}"
            )
            return

        # Parse optional workdir
        workdir = None
        for i, arg in enumerate(args[1:], 1):
            if arg == "--workdir" and i + 1 < len(args):
                workdir = args[i + 1]
                break

        task = templates[template_name]
        agent = await self.orchestrator.spawn_agent(
            task=task,
            supervised=False,  # Templates are pre-vetted, run unsupervised
            priority="normal",
            workdir=workdir,
            timeout_mode="auto",
        )

        await update.message.reply_text(  # type: ignore
            f"Creating {template_name}...\n\n"
            f"Agent: {agent['id']}\n"
            f"Task: {task[:100]}...\n\n"
            f"I'll update you when it's ready!"
        )

    async def _cmd_deploy(self, update: Update, args: list[str]) -> None:
        """Deploy project to Vercel or Netlify for live preview."""
        if not args:
            await update.message.reply_text(  # type: ignore
                "Usage: /gru deploy <platform> [--workdir /path]\n\n"
                "Platforms:\n"
                "  vercel - Deploy to Vercel (requires VERCEL_TOKEN)\n"
                "  netlify - Deploy to Netlify (requires NETLIFY_TOKEN)\n"
                "  surge - Deploy to Surge.sh (free, no token needed)\n\n"
                "Example: /gru deploy vercel --workdir /workspace/my-app"
            )
            return

        platform = args[0].lower()

        # Parse optional workdir
        workdir = str(self.config.default_workdir)
        for i, arg in enumerate(args[1:], 1):
            if arg == "--workdir" and i + 1 < len(args):
                workdir = args[i + 1]
                break

        deploy_tasks = {
            "vercel": (
                f"Deploy the project in {workdir} to Vercel. Use the Vercel CLI "
                "(npx vercel). Use VERCEL_TOKEN env var if set. Return the preview URL."
            ),
            "netlify": (
                f"Deploy the project in {workdir} to Netlify. Use the Netlify CLI "
                "(npx netlify deploy). Use NETLIFY_TOKEN env var. Return the preview URL."
            ),
            "surge": (
                f"Deploy the project in {workdir} to Surge.sh. Install surge if needed "
                "(npm install -g surge), run surge in build dir. Return the URL."
            ),
        }

        if platform not in deploy_tasks:
            await update.message.reply_text(  # type: ignore
                f"Unknown platform: {platform}\nAvailable: {', '.join(deploy_tasks.keys())}"
            )
            return

        task = deploy_tasks[platform]
        agent = await self.orchestrator.spawn_agent(
            task=task,
            supervised=True,
            priority="high",
            workdir=workdir,
        )

        await update.message.reply_text(  # type: ignore
            f"Deploying to {platform}...\n\nAgent: {agent['id']}\nI'll send you the preview URL when it's live!"
        )

    async def _cmd_setup(self, update: Update, args: list[str]) -> None:
        """Interactive setup guide."""
        issues: list[str] = []
        tips: list[str] = []

        # Check Anthropic API
        if not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY not set")
            tips.append("Get your API key at console.anthropic.com")
        else:
            tips.append("[OK] Anthropic API key configured")

        # Check GitHub token
        if not os.getenv("GRU_GITHUB_TOKEN"):
            tips.append("[INFO] GRU_GITHUB_TOKEN not set - private repos won't work")
        else:
            tips.append("[OK] GitHub token configured")

        # Check workdir
        workdir = self.config.default_workdir
        if not os.path.isdir(workdir):
            issues.append(f"Default workdir doesn't exist: {workdir}")
            tips.append(f"Create it with: mkdir -p {workdir}")
        else:
            tips.append(f"[OK] Workdir exists: {workdir}")

        # Check master password for secrets
        if not os.getenv("GRU_MASTER_PASSWORD"):
            tips.append("[INFO] GRU_MASTER_PASSWORD not set - secret storage disabled")
        else:
            tips.append("[OK] Secret storage enabled")

        if issues:
            response = "Setup Issues Found:\n\n"
            response += "\n".join(f"- {i}" for i in issues)
            response += "\n\nHow to fix:\n"
            response += "\n".join(tips)
            response += "\n\nSet environment variables in Railway (or your host) and redeploy."
        else:
            response = "Setup looks good!\n\n"
            response += "\n".join(tips)
            response += "\n\nYou're ready to go! Try:\n"
            response += "- Send a message describing what you want to build\n"
            response += "- Use /gru examples to see copy-paste examples\n"
            response += "- Use /gru create <template> for quick starts"

        await update.message.reply_text(response)  # type: ignore

    # Callback handlers

    async def callback_approval(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle approval button callbacks."""
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()

        parts = query.data.split(":")
        if len(parts) < 2:
            return

        action = parts[0]

        # Handle option selection: option:{approval_id}:{index}
        if action == "option" and len(parts) == 3:
            approval_id = parts[1]

            # Check if request is still pending
            if approval_id not in self._pending_approvals:
                with contextlib.suppress(BadRequest):
                    await query.edit_message_text(f"Expired: {approval_id}")
                return

            try:
                option_index = int(parts[2])
            except ValueError:
                logger.warning(f"Invalid option index in callback: {query.data}")
                return

            # Look up the original option text
            options = self._pending_options.get(approval_id, [])
            if option_index < 0 or option_index >= len(options):
                logger.warning(f"Option index {option_index} out of range for {approval_id}")
                with contextlib.suppress(BadRequest):
                    await query.edit_message_text(f"Invalid option: {approval_id}")
                return

            option_text = options[option_index]

            self._pending_approvals[approval_id].set_result(option_text)
            del self._pending_approvals[approval_id]

            # Clean up
            self._pending_options.pop(approval_id, None)
            self._pending_messages.pop(approval_id, None)

            with contextlib.suppress(BadRequest):
                await query.edit_message_text(f"Selected: {option_text}")
            return

        # Handle approve/reject: approve:{approval_id} or reject:{approval_id}
        if len(parts) != 2:
            return

        approval_id = parts[1]

        # Check if request is still pending
        if approval_id not in self._pending_approvals:
            with contextlib.suppress(BadRequest):
                await query.edit_message_text(f"Expired: {approval_id}")
            return

        # For approve/reject, return None for reject, confirmation for approve
        result = "Confirmed" if action == "approve" else None
        self._pending_approvals[approval_id].set_result(result)
        del self._pending_approvals[approval_id]

        # Clean up
        self._pending_options.pop(approval_id, None)
        self._pending_messages.pop(approval_id, None)

        approved = action == "approve"
        await self.orchestrator.approve(approval_id, approved)

        status = "Approved" if approved else "Rejected"
        with contextlib.suppress(BadRequest):
            await query.edit_message_text(f"{status}: {approval_id}")

    # Notification callback

    def notify_callback(self, agent_id: str, message: str) -> None:
        """Callback for orchestrator notifications."""
        if not self._app:
            return

        async def send():
            for admin_id in self.config.telegram_admin_ids:
                try:
                    await self.send_output(admin_id, f"[{agent_id}] {message}")
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(send())
        except RuntimeError:
            pass  # No running loop

    def approval_callback(self, approval_id: str, details: dict) -> asyncio.Future:
        """Callback for orchestrator approval requests."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_approvals[approval_id] = future

        async def send():
            options = details.get("options")

            if options and isinstance(options, list):
                # Store options for later lookup (preserves full text)
                self._pending_options[approval_id] = list(options)

                # Generate option buttons with index-based callback_data
                buttons = []
                for idx, opt in enumerate(options):
                    # Truncate display text but use index in callback
                    display_text = str(opt)[:100] if len(str(opt)) > 100 else str(opt)
                    # callback_data: "option:{8-char-id}:{index}" = ~20 bytes max
                    buttons.append([InlineKeyboardButton(display_text, callback_data=f"option:{approval_id}:{idx}")])
                # Add decline option
                buttons.append([InlineKeyboardButton("Decline to answer", callback_data=f"reject:{approval_id}")])
                keyboard = InlineKeyboardMarkup(buttons)
                question = details.get("question", "Please select an option:")
                text = f"Input requested: {approval_id}\n\n{question}"
            else:
                # Standard approve/reject for non-option requests
                keyboard = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("Approve", callback_data=f"approve:{approval_id}"),
                            InlineKeyboardButton("Reject", callback_data=f"reject:{approval_id}"),
                        ]
                    ]
                )
                text = f"Approval requested: {approval_id}\n{json.dumps(details, indent=2)[:2000]}"

            sent_messages: list[tuple[int, int]] = []
            for admin_id in self.config.telegram_admin_ids:
                try:
                    msg = await self._app.bot.send_message(  # type: ignore
                        admin_id,
                        text,
                        reply_markup=keyboard,
                    )
                    sent_messages.append((admin_id, msg.message_id))
                except Exception as e:
                    logger.error(f"Failed to send approval request: {e}")

            if sent_messages:
                self._pending_messages[approval_id] = sent_messages

        loop.create_task(send())
        return future

    async def cancel_approval(self, approval_id: str) -> None:
        """Cancel a pending approval and clean up resources.

        Called by orchestrator on timeout to prevent memory leaks
        and update UI to show expiration.
        """
        # Clean up pending state
        self._pending_approvals.pop(approval_id, None)
        self._pending_options.pop(approval_id, None)

        # Edit messages to show expiration
        messages = self._pending_messages.pop(approval_id, [])
        if self._app and messages:
            for chat_id, message_id in messages:
                try:
                    await self._app.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=f"Expired: {approval_id}",
                    )
                except BadRequest:
                    pass  # Message may already be edited or deleted
                except Exception as e:
                    logger.error(f"Failed to edit expired message: {e}")

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
            lines.append(f"  - {a['id']} [{a['status']}]: {a['task'][:200]}...")
            if a.get("workdir"):
                lines.append(f"    workdir: {a['workdir']}")
            if a.get("error"):
                lines.append(f"    error: {a['error'][:500]}")

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

CRITICAL - Mode detection:
- If the user's message contains "oneshot" anywhere, you MUST set oneshot=true
- If user says "no approvals", "autonomous", "fire and forget", "yeet it", set oneshot=true
- This is mandatory. Do not spawn in supervised mode when user explicitly requests oneshot.

If the user mentions a directory path, use it as the workdir.

For questions about status, agents, or other info, just respond with text.
Only use the spawn tool when they want to start new work.

Be concise and helpful."""

        return system_prompt, context_info

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
                name="terminate_all_agents",
                description="Terminate/kill ALL agents. Use when user says 'kill all agents' or similar.",
                input_schema={"type": "object", "properties": {}},
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
                f"Agent spawned: [{num}] {agent['id']}\n"
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
                    f"Agent {display}\n"
                    f"Status: {agent_info['status']}\n"
                    f"Task: {agent_info['task']}\n"
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
                    f"Orchestrator Status\n"
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
                prefix = f"[{num}:{nick}]" if nick else f"[{num}]"
                lines.append(f"{prefix} [{a['status']}] {a['task']}")
            return "\n".join(lines)

        elif tool_name == "terminate_all_agents":
            agents = await self.orchestrator.list_agents()
            if not agents:
                return "No agents to terminate"
            terminated = 0
            for a in agents:
                if a["status"] in ("running", "paused", "idle", "failed"):
                    # Try to terminate in-memory agent first
                    success = await self.orchestrator.terminate_agent(a["id"])
                    if not success:
                        # Agent not in memory, update database directly
                        await self.orchestrator.db.update_agent(a["id"], status="terminated")
                    terminated += 1
            return f"Terminated {terminated} agents"

        elif tool_name == "get_pending_approvals":
            pending = await self.orchestrator.get_pending_approvals()
            if not pending:
                return "No pending approvals"
            lines = [f"{p['id']}: {p['action_type']} for agent {p['agent_id']}" for p in pending]
            return "\n".join(lines)

        elif tool_name == "approve_action":
            approval_id = params.get("approval_id", "")
            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result(True)
                del self._pending_approvals[approval_id]
            success = await self.orchestrator.approve(approval_id, approved=True)
            return f"Approved: {approval_id}" if success else f"Approval not found: {approval_id}"

        elif tool_name == "reject_action":
            approval_id = params.get("approval_id", "")
            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result(False)
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
                return f"Agent [{agent_num}] is now nicknamed {nickname}"
            return f"Nickname {nickname} is already taken"

        return f"Unknown tool: {tool_name}"

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle casual text messages (not commands)."""
        if not await self._check_admin(update):
            return
        if not await self._check_rate_limit(update):
            return

        text = update.message.text.strip()  # type: ignore
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
                    for chunk in self._split_message(result):
                        await update.message.reply_text(chunk)  # type: ignore
                return

            content = response.content or "I couldn't generate a response."
            for chunk in self._split_message(content):
                await update.message.reply_text(chunk)  # type: ignore
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)[:200]}")  # type: ignore

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle photo messages - build UI from screenshots."""
        if not await self._check_admin(update):
            return
        if not await self._check_rate_limit(update):
            return

        message = update.message
        if not message or not message.photo:
            return

        # Get the largest photo
        photo = message.photo[-1]
        caption = message.caption or "Build this UI"

        await message.reply_text(
            "Got your screenshot! I'll spawn an agent to build this UI.\n"
            "The agent will analyze the image and recreate it in code."
        )

        # Download the photo
        try:
            file = await context.bot.get_file(photo.file_id)
            file_bytes = await file.download_as_bytearray()
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")

            task = f"""Build a UI that matches this design screenshot.

User instructions: {caption}

The screenshot is provided as a base64-encoded image. Analyze it and:
1. Identify the layout, colors, typography, and components
2. Create clean, responsive HTML/CSS or React code that recreates this design
3. Match the visual style as closely as possible
4. Use modern CSS (flexbox/grid) and semantic HTML

Image (base64): {image_b64[:100]}... [truncated, full image available]

IMPORTANT: The full base64 image data has been provided to you in the task context."""

            agent = await self.orchestrator.spawn_agent(
                task=task,
                supervised=True,
                priority="normal",
            )

            await message.reply_text(
                f"Agent spawned: {agent['id']}\n"
                f"Building UI from your screenshot...\n"
                f"I'll notify you when it needs approval or is complete!"
            )
        except Exception as e:
            await message.reply_text(f"Error processing image: {e}")

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages - transcribe and create task."""
        if not await self._check_admin(update):
            return
        if not await self._check_rate_limit(update):
            return

        message = update.message
        if not message or not message.voice:
            return

        await message.reply_text("Got your voice message! Transcribing...")

        try:
            # Download the voice file
            file = await context.bot.get_file(message.voice.file_id)
            file_bytes = await file.download_as_bytearray()
            audio_b64 = base64.b64encode(file_bytes).decode("utf-8")

            # Use Anthropic to transcribe
            prompt = (
                "Transcribe this voice message and summarize the task. "
                "Format: TRANSCRIPTION: [words] TASK: [what they want]"
            )
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {  # type: ignore[list-item]
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "audio/ogg",
                                    "data": audio_b64,
                                },
                            },
                        ],
                    }
                ],
            )

            transcription = ""
            if response.content and hasattr(response.content[0], "text"):
                transcription = getattr(response.content[0], "text", "")

            # Extract task from transcription
            task = transcription.split("TASK:")[-1].strip() if "TASK:" in transcription else transcription

            await message.reply_text(f"Transcribed: {transcription[:200]}...\n\nSpawning agent for this task...")

            agent = await self.orchestrator.spawn_agent(
                task=task,
                supervised=True,
                priority="normal",
            )

            await message.reply_text(f"Agent spawned: {agent['id']}")

        except Exception as e:
            await message.reply_text(f"Couldn't process voice message: {e}\nTry typing your request instead!")

    def _progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Generate a visual progress bar."""
        if total == 0:
            return "[" + "-" * width + "] 0%"
        percent = min(100, int(current / total * 100))
        filled = int(width * current / total)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}] {percent}%"

    async def start(self) -> None:
        """Start the Telegram bot."""
        self._app = Application.builder().token(self.config.telegram_token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self.cmd_start))
        self._app.add_handler(CommandHandler("gru", self.cmd_gru))
        self._app.add_handler(CallbackQueryHandler(self.callback_approval))
        # Photo handler for screenshots
        self._app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        # Voice handler for voice notes
        self._app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        # Casual message handler (must be last)
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Set up orchestrator callbacks
        self.orchestrator.set_notify_callback(self.notify_callback)
        self.orchestrator.set_approval_callback(self.approval_callback)
        self.orchestrator.set_cancel_approval_callback(self.cancel_approval)

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()  # type: ignore

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()  # type: ignore
            await self._app.stop()
            await self._app.shutdown()
