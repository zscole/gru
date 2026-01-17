"""Discord bot interface for Gru."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands
from discord.ext import commands

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
        self._requests: dict[int, list[float]] = {}

    def is_allowed(self, user_id: int) -> bool:
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


class ApprovalView(discord.ui.View):
    """View for approval buttons."""

    def __init__(self, approval_id: str, bot: DiscordBot, options: list[str] | None = None) -> None:
        super().__init__(timeout=300)
        self.approval_id = approval_id
        self.bot = bot
        self.options = options
        self._setup_buttons()

    def _setup_buttons(self) -> None:
        """Set up the buttons for this view."""
        if self.options:
            # Add option buttons
            for idx, opt in enumerate(self.options):
                display_text = str(opt)[:80] if len(str(opt)) > 80 else str(opt)
                self.add_item(OptionButton(self, idx, display_text))
            # Add decline button
            self.add_item(RejectButton(self, label="Decline"))
        else:
            # Standard approve/reject buttons
            self.add_item(ApproveButton(self))
            self.add_item(RejectButton(self))


class ApproveButton(discord.ui.Button["ApprovalView"]):
    """Approve button for approval view."""

    def __init__(self, parent: ApprovalView) -> None:
        super().__init__(label="Approve", style=discord.ButtonStyle.green)
        self.parent_view = parent

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.parent_view.approval_id not in self.parent_view.bot._pending_approvals:
            await interaction.response.edit_message(content=f"Expired: {self.parent_view.approval_id}", view=None)
            return

        self.parent_view.bot._pending_approvals[self.parent_view.approval_id].set_result("Confirmed")
        del self.parent_view.bot._pending_approvals[self.parent_view.approval_id]
        self.parent_view.bot._pending_options.pop(self.parent_view.approval_id, None)
        self.parent_view.bot._pending_messages.pop(self.parent_view.approval_id, None)

        await self.parent_view.bot.orchestrator.approve(self.parent_view.approval_id, approved=True)
        await interaction.response.edit_message(content=f"Approved: {self.parent_view.approval_id}", view=None)


class RejectButton(discord.ui.Button["ApprovalView"]):
    """Reject button for approval view."""

    def __init__(self, parent: ApprovalView, label: str = "Reject") -> None:
        super().__init__(
            label=label, style=discord.ButtonStyle.red if label == "Reject" else discord.ButtonStyle.secondary
        )
        self.parent_view = parent

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.parent_view.approval_id not in self.parent_view.bot._pending_approvals:
            await interaction.response.edit_message(content=f"Expired: {self.parent_view.approval_id}", view=None)
            return

        self.parent_view.bot._pending_approvals[self.parent_view.approval_id].set_result(None)
        del self.parent_view.bot._pending_approvals[self.parent_view.approval_id]
        self.parent_view.bot._pending_options.pop(self.parent_view.approval_id, None)
        self.parent_view.bot._pending_messages.pop(self.parent_view.approval_id, None)

        await self.parent_view.bot.orchestrator.approve(self.parent_view.approval_id, approved=False)
        await interaction.response.edit_message(content=f"Rejected: {self.parent_view.approval_id}", view=None)


class OptionButton(discord.ui.Button["ApprovalView"]):
    """Option button for approval view."""

    def __init__(self, parent: ApprovalView, idx: int, label: str) -> None:
        super().__init__(label=label, style=discord.ButtonStyle.primary)
        self.parent_view = parent
        self.idx = idx

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.parent_view.approval_id not in self.parent_view.bot._pending_approvals:
            await interaction.response.edit_message(content=f"Expired: {self.parent_view.approval_id}", view=None)
            return

        option_text = self.parent_view.options[self.idx] if self.parent_view.options else ""
        self.parent_view.bot._pending_approvals[self.parent_view.approval_id].set_result(option_text)
        del self.parent_view.bot._pending_approvals[self.parent_view.approval_id]
        self.parent_view.bot._pending_options.pop(self.parent_view.approval_id, None)
        self.parent_view.bot._pending_messages.pop(self.parent_view.approval_id, None)

        await interaction.response.edit_message(content=f"Selected: {option_text}", view=None)


class DiscordBot:
    """Discord bot interface for Gru orchestrator."""

    MAX_MESSAGE_LENGTH = 2000
    SPLIT_THRESHOLD = 3

    def __init__(self, config: Config, orchestrator: Orchestrator) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._pending_options: dict[str, list[str]] = {}
        self._pending_messages: dict[str, list[tuple[int, int]]] = {}  # approval_id -> [(channel_id, msg_id)]
        self._rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        self._spawn_limiter = RateLimiter(max_requests=5, window_seconds=60)
        self._agent_numbers: dict[str, int] = {}  # agent_id -> number
        self._number_to_agent: dict[int, str] = {}  # number -> agent_id
        self._next_agent_number: int = 1
        self._agent_nicknames: dict[str, str] = {}  # agent_id -> nickname
        self._nickname_to_agent: dict[str, str] = {}  # nickname -> agent_id

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True

        # Create bot with command prefix (for fallback text commands)
        self._bot = commands.Bot(command_prefix="!", intents=intents)
        self._tree = self._bot.tree

        # Register event handlers
        self._bot.event(self.on_ready)
        self._bot.event(self.on_message)

        # Register slash commands
        self._register_commands()

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        return user_id in self.config.discord_admin_ids

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

    async def _check_admin(self, interaction: discord.Interaction) -> bool:
        """Check admin and reply if not authorized."""
        if not self._is_admin(interaction.user.id):
            await interaction.response.send_message("Unauthorized", ephemeral=True)
            return False
        return True

    async def _check_rate_limit(self, interaction: discord.Interaction, limiter: RateLimiter | None = None) -> bool:
        """Check rate limit and reply if exceeded."""
        limiter = limiter or self._rate_limiter
        if not limiter.is_allowed(interaction.user.id):
            await interaction.response.send_message("Rate limit exceeded. Please wait.", ephemeral=True)
            return False
        return True

    async def send_output(
        self,
        channel: discord.abc.Messageable,
        text: str,
        filename: str | None = None,
        force_file: bool = False,
    ) -> None:
        """Send output, splitting or as file if needed."""
        if force_file or (filename and len(text) > self.MAX_MESSAGE_LENGTH * self.SPLIT_THRESHOLD):
            file_bytes = io.BytesIO(text.encode())
            await channel.send(file=discord.File(file_bytes, filename=filename or "output.txt"))
            return

        if len(text) <= self.MAX_MESSAGE_LENGTH:
            await channel.send(text)
            return

        chunks = self._split_message(text)

        if len(chunks) <= self.SPLIT_THRESHOLD:
            for chunk in chunks:
                await channel.send(chunk)
                await asyncio.sleep(0.1)
        else:
            await channel.send(f"Output is {len(text)} chars ({len(chunks)} messages). Sending as messages...")
            for i, chunk in enumerate(chunks):
                await channel.send(f"[{i + 1}/{len(chunks)}]\n{chunk}")
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
        """Register slash commands."""
        gru_group = app_commands.Group(name="gru", description="Gru orchestrator commands")

        @gru_group.command(name="help", description="Show help")
        async def cmd_help(interaction: discord.Interaction):
            if not await self._check_admin(interaction):
                return
            help_text = f"""**Gru Commands:**

**Agent Management:**
`/gru spawn <task>` - Spawn a new agent
`/gru status [agent_id]` - Show status
`/gru list [status]` - List agents
`/gru pause <agent_id>` - Pause agent
`/gru resume <agent_id>` - Resume agent
`/gru terminate <agent_id>` - Terminate agent
`/gru nudge <agent_id> <message>` - Send message to agent
`/gru logs <agent_id>` - Get conversation logs

**Approvals:**
`/gru pending` - Show pending approvals
`/gru approve <approval_id>` - Approve action
`/gru reject <approval_id>` - Reject action

**Secrets:**
`/gru secret_set <key> <value>` - Set secret
`/gru secret_get <key>` - Get secret
`/gru secret_list` - List secrets
`/gru secret_delete <key>` - Delete secret

**Templates:**
`/gru template_save <name> <task>` - Save template
`/gru template_list` - List templates
`/gru template_use <name>` - Use template
`/gru template_delete <name>` - Delete template

Default workdir: {self.config.default_workdir}"""
            await interaction.response.send_message(help_text)

        @gru_group.command(name="spawn", description="Spawn a new agent")
        @app_commands.describe(
            task="Task description for the agent",
            supervised="Require approval for file writes and bash (default: true)",
            oneshot="Fully autonomous mode (default: false)",
            priority="Task priority",
            workdir="Working directory path",
        )
        @app_commands.choices(
            priority=[
                app_commands.Choice(name="high", value="high"),
                app_commands.Choice(name="normal", value="normal"),
                app_commands.Choice(name="low", value="low"),
            ]
        )
        async def cmd_spawn(
            interaction: discord.Interaction,
            task: str,
            supervised: bool = True,
            oneshot: bool = False,
            priority: str = "normal",
            workdir: Optional[str] = None,  # noqa: UP045
        ):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction, self._spawn_limiter):
                return

            if oneshot:
                supervised = False
                timeout_mode = "auto"
            else:
                timeout_mode = "block"

            try:
                agent = await self.orchestrator.spawn_agent(
                    task=task,
                    supervised=supervised,
                    priority=priority,
                    workdir=workdir,
                    timeout_mode=timeout_mode,
                )
            except ValueError as e:
                await interaction.response.send_message(f"Error: {e}", ephemeral=True)
                return

            mode_str = "oneshot (fully autonomous)" if oneshot else ("supervised" if supervised else "unsupervised")
            await interaction.response.send_message(
                f"Agent spawned: `{agent['id']}`\n"
                f"Task: {task}\n"
                f"Mode: {mode_str}\n"
                f"Priority: {priority}\n"
                f"Workdir: {agent.get('workdir', self.config.default_workdir)}"
            )

        @gru_group.command(name="status", description="Show status")
        @app_commands.describe(agent_id="Optional agent ID for specific status")
        async def cmd_status(interaction: discord.Interaction, agent_id: Optional[str] = None):  # noqa: UP045
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            if agent_id:
                agent = await self.orchestrator.get_agent(agent_id)
                if not agent:
                    await interaction.response.send_message(f"Agent not found: {agent_id}", ephemeral=True)
                    return
                msg = (
                    f"**Agent {agent['id']}**\n"
                    f"Status: {agent['status']}\n"
                    f"Task: {agent['task'][:100]}\n"
                    f"Model: {agent['model']}\n"
                    f"Supervised: {bool(agent['supervised'])}\n"
                    f"Workdir: {agent.get('workdir', 'N/A')}\n"
                    f"Created: {agent['created_at']}"
                )
                if agent.get("error"):
                    msg += f"\nError: {agent['error']}"
                await interaction.response.send_message(msg)
            else:
                status = await self.orchestrator.get_status()
                await interaction.response.send_message(
                    f"**Orchestrator Status**\n"
                    f"Running: {status['running']}\n"
                    f"Agents: {status['agents']['total']} total, "
                    f"{status['agents']['running']} running, "
                    f"{status['agents']['paused']} paused\n"
                    f"Queue: {status['scheduler']['queued']} queued\n"
                    f"Default workdir: {self.config.default_workdir}"
                )

        @gru_group.command(name="list", description="List agents")
        @app_commands.describe(status="Filter by status")
        @app_commands.choices(
            status=[
                app_commands.Choice(name="running", value="running"),
                app_commands.Choice(name="paused", value="paused"),
                app_commands.Choice(name="completed", value="completed"),
                app_commands.Choice(name="failed", value="failed"),
            ]
        )
        async def cmd_list(interaction: discord.Interaction, status: Optional[str] = None):  # noqa: UP045
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            agents = await self.orchestrator.list_agents(status)

            if not agents:
                await interaction.response.send_message("No agents found")
                return

            lines = []
            for agent in agents[:20]:
                lines.append(f"`{agent['id']}` [{agent['status']}] {agent['task'][:40]}...")

            await interaction.response.send_message("\n".join(lines))

        @gru_group.command(name="pause", description="Pause an agent")
        @app_commands.describe(agent_id="Agent ID to pause")
        async def cmd_pause(interaction: discord.Interaction, agent_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            success = await self.orchestrator.pause_agent(agent_id)
            if success:
                await interaction.response.send_message(f"Agent {agent_id} paused")
            else:
                await interaction.response.send_message(f"Could not pause agent {agent_id}", ephemeral=True)

        @gru_group.command(name="resume", description="Resume an agent")
        @app_commands.describe(agent_id="Agent ID to resume")
        async def cmd_resume(interaction: discord.Interaction, agent_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            success = await self.orchestrator.resume_agent(agent_id)
            if success:
                await interaction.response.send_message(f"Agent {agent_id} resumed")
            else:
                await interaction.response.send_message(f"Could not resume agent {agent_id}", ephemeral=True)

        @gru_group.command(name="terminate", description="Terminate an agent")
        @app_commands.describe(agent_id="Agent ID to terminate")
        async def cmd_terminate(interaction: discord.Interaction, agent_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            success = await self.orchestrator.terminate_agent(agent_id)
            if success:
                await interaction.response.send_message(f"Agent {agent_id} terminated")
            else:
                await interaction.response.send_message(f"Could not terminate agent {agent_id}", ephemeral=True)

        @gru_group.command(name="nudge", description="Send a message to an agent")
        @app_commands.describe(agent_id="Agent ID", message="Message to send")
        async def cmd_nudge(interaction: discord.Interaction, agent_id: str, message: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            success = await self.orchestrator.nudge_agent(agent_id, message)
            if success:
                await interaction.response.send_message(f"Nudge sent to agent {agent_id}")
            else:
                await interaction.response.send_message(f"Could not nudge agent {agent_id}", ephemeral=True)

        @gru_group.command(name="approve", description="Approve a pending action")
        @app_commands.describe(approval_id="Approval ID")
        async def cmd_approve(interaction: discord.Interaction, approval_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result(True)
                del self._pending_approvals[approval_id]

            success = await self.orchestrator.approve(approval_id, approved=True)
            if success:
                await interaction.response.send_message(f"Approved: {approval_id}")
            else:
                await interaction.response.send_message(
                    f"Approval not found or already resolved: {approval_id}", ephemeral=True
                )

        @gru_group.command(name="reject", description="Reject a pending action")
        @app_commands.describe(approval_id="Approval ID")
        async def cmd_reject(interaction: discord.Interaction, approval_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            if approval_id in self._pending_approvals:
                self._pending_approvals[approval_id].set_result(False)
                del self._pending_approvals[approval_id]

            success = await self.orchestrator.approve(approval_id, approved=False)
            if success:
                await interaction.response.send_message(f"Rejected: {approval_id}")
            else:
                await interaction.response.send_message(
                    f"Approval not found or already resolved: {approval_id}", ephemeral=True
                )

        @gru_group.command(name="pending", description="Show pending approvals")
        async def cmd_pending(interaction: discord.Interaction):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            pending = await self.orchestrator.get_pending_approvals()

            if not pending:
                await interaction.response.send_message("No pending approvals")
                return

            lines = []
            for p in pending:
                details = json.dumps(p["action_details"])[:50]
                lines.append(f"`{p['id']}`: {p['action_type']} - {details}...")

            await interaction.response.send_message("\n".join(lines))

        @gru_group.command(name="logs", description="Get agent conversation logs")
        @app_commands.describe(agent_id="Agent ID")
        async def cmd_logs(interaction: discord.Interaction, agent_id: str):
            if not await self._check_admin(interaction):
                return
            if not await self._check_rate_limit(interaction):
                return

            conversation = await self.orchestrator.db.get_conversation(agent_id)

            if not conversation:
                await interaction.response.send_message(f"No logs found for agent {agent_id}", ephemeral=True)
                return

            lines = []
            for msg in conversation[-20:]:
                content = msg["content"]
                if isinstance(content, list):
                    content = str(content)
                lines.append(f"[{msg['role']}] {content[:200]}")

            output = "\n\n".join(lines)
            await interaction.response.defer()
            if interaction.channel:
                await self.send_output(interaction.channel, output, f"logs_{agent_id}.txt")  # type: ignore[arg-type]

        @gru_group.command(name="secret_set", description="Set a secret")
        @app_commands.describe(key="Secret key", value="Secret value")
        async def cmd_secret_set(interaction: discord.Interaction, key: str, value: str):
            if not await self._check_admin(interaction):
                return
            await self.orchestrator.secrets.set(key, value)
            await interaction.response.send_message(f"Secret '{key}' set", ephemeral=True)

        @gru_group.command(name="secret_get", description="Get a secret")
        @app_commands.describe(key="Secret key")
        async def cmd_secret_get(interaction: discord.Interaction, key: str):
            if not await self._check_admin(interaction):
                return
            secret_value = await self.orchestrator.secrets.get(key)
            if secret_value:
                await interaction.response.send_message(
                    f"Secret '{key}' exists ({len(secret_value)} chars)", ephemeral=True
                )
            else:
                await interaction.response.send_message(f"Secret '{key}' not found", ephemeral=True)

        @gru_group.command(name="secret_list", description="List secrets")
        async def cmd_secret_list(interaction: discord.Interaction):
            if not await self._check_admin(interaction):
                return
            keys = await self.orchestrator.secrets.list_keys()
            if keys:
                await interaction.response.send_message("Secrets:\n" + "\n".join(keys), ephemeral=True)
            else:
                await interaction.response.send_message("No secrets stored", ephemeral=True)

        @gru_group.command(name="secret_delete", description="Delete a secret")
        @app_commands.describe(key="Secret key")
        async def cmd_secret_delete(interaction: discord.Interaction, key: str):
            if not await self._check_admin(interaction):
                return
            success = await self.orchestrator.secrets.delete(key)
            if success:
                await interaction.response.send_message(f"Secret '{key}' deleted", ephemeral=True)
            else:
                await interaction.response.send_message(f"Secret '{key}' not found", ephemeral=True)

        @gru_group.command(name="template_save", description="Save a template")
        @app_commands.describe(name="Template name", task="Task description")
        async def cmd_template_save(interaction: discord.Interaction, name: str, task: str):
            if not await self._check_admin(interaction):
                return
            await self.orchestrator.db.save_template(name, task)
            await interaction.response.send_message(f"Template '{name}' saved")

        @gru_group.command(name="template_list", description="List templates")
        async def cmd_template_list(interaction: discord.Interaction):
            if not await self._check_admin(interaction):
                return
            templates = await self.orchestrator.db.list_templates()
            if templates:
                lines = [f"`{t['name']}`: {t['task'][:50]}..." for t in templates]
                await interaction.response.send_message("Templates:\n" + "\n".join(lines))
            else:
                await interaction.response.send_message("No templates saved")

        @gru_group.command(name="template_use", description="Use a template")
        @app_commands.describe(name="Template name")
        async def cmd_template_use(interaction: discord.Interaction, name: str):
            if not await self._check_admin(interaction):
                return
            template = await self.orchestrator.db.get_template(name)
            if template:
                agent = await self.orchestrator.spawn_agent(
                    task=template["task"],
                    model=template.get("model"),
                    system_prompt=template.get("system_prompt"),
                    supervised=bool(template.get("supervised", 1)),
                    priority=template.get("priority", "normal"),
                )
                await interaction.response.send_message(f"Agent spawned from template: `{agent['id']}`")
            else:
                await interaction.response.send_message(f"Template '{name}' not found", ephemeral=True)

        @gru_group.command(name="template_delete", description="Delete a template")
        @app_commands.describe(name="Template name")
        async def cmd_template_delete(interaction: discord.Interaction, name: str):
            if not await self._check_admin(interaction):
                return
            success = await self.orchestrator.db.delete_template(name)
            if success:
                await interaction.response.send_message(f"Template '{name}' deleted")
            else:
                await interaction.response.send_message(f"Template '{name}' not found", ephemeral=True)

        self._tree.add_command(gru_group)

    async def on_ready(self) -> None:
        """Handle bot ready event."""
        logger.info(f"Discord bot logged in as {self._bot.user}")

        # Sync commands to guild or globally
        if self.config.discord_guild_id:
            guild = discord.Object(id=self.config.discord_guild_id)
            self._tree.copy_global_to(guild=guild)
            await self._tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {self.config.discord_guild_id}")
        else:
            await self._tree.sync()
            logger.info("Synced commands globally")

        # Set up orchestrator callbacks
        self.orchestrator.set_notify_callback(self.notify_callback)
        self.orchestrator.set_approval_callback(self.approval_callback)
        self.orchestrator.set_cancel_approval_callback(self.cancel_approval)

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
                    "properties": {"agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}},
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="pause_agent",
                description="Pause a running agent",
                input_schema={
                    "type": "object",
                    "properties": {"agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}},
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="resume_agent",
                description="Resume a paused agent",
                input_schema={
                    "type": "object",
                    "properties": {"agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID"}},
                    "required": ["agent_ref"],
                },
            ),
            ToolDefinition(
                name="get_status",
                description="Get status of a specific agent or overall system status",
                input_schema={
                    "type": "object",
                    "properties": {"agent_ref": {"type": "string", "description": "Agent number (e.g. '1') or full ID (optional)"}},
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
                    f"**Agent {display}**\n"
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
                    f"**Orchestrator Status**\n"
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
                if nick:
                    prefix = f"`[{num}:{nick}]`"
                else:
                    prefix = f"`[{num}]`"
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

    async def on_message(self, message: discord.Message) -> None:
        """Handle natural language messages."""
        if message.author.bot:
            return

        if not self._is_admin(message.author.id):
            return

        if message.content.startswith(("!", "/")):
            return

        if not self._rate_limiter.is_allowed(message.author.id):
            await message.reply("Rate limit exceeded. Please wait.")
            return

        text = message.content.strip()
        if not text:
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
                    await message.reply(result)
                return

            await message.reply(response.content or "I couldn't generate a response.")
        except Exception as e:
            await message.reply(f"Error processing message: {e}")

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

        async def send():
            for admin_id in self.config.discord_admin_ids:
                try:
                    user = await self._bot.fetch_user(admin_id)
                    if user:
                        await user.send(f"[{agent_id}] {message}")
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")

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

        async def send():
            options = details.get("options")

            if options and isinstance(options, list):
                self._pending_options[approval_id] = list(options)
                question = details.get("question", "Please select an option:")
                text = f"**Input requested:** `{approval_id}`\n\n{question}"
                view = ApprovalView(approval_id, self, options)
            else:
                text = f"**Approval requested:** `{approval_id}`\n```json\n{json.dumps(details, indent=2)[:500]}\n```"
                view = ApprovalView(approval_id, self)

            sent_messages: list[tuple[int, int]] = []
            for admin_id in self.config.discord_admin_ids:
                try:
                    user = await self._bot.fetch_user(admin_id)
                    if user:
                        msg = await user.send(text, view=view)
                        sent_messages.append((msg.channel.id, msg.id))
                except Exception as e:
                    logger.error(f"Failed to send approval request: {e}")

            if sent_messages:
                self._pending_messages[approval_id] = sent_messages

        loop.create_task(send())
        return future

    async def cancel_approval(self, approval_id: str) -> None:
        """Cancel a pending approval and clean up resources."""
        self._pending_approvals.pop(approval_id, None)
        self._pending_options.pop(approval_id, None)

        messages = self._pending_messages.pop(approval_id, [])
        for channel_id, message_id in messages:
            try:
                channel = self._bot.get_channel(channel_id)
                if channel and hasattr(channel, "fetch_message"):
                    msg = await channel.fetch_message(message_id)
                    await msg.edit(content=f"Expired: {approval_id}", view=None)
            except Exception as e:
                logger.error(f"Failed to edit expired message: {e}")

    async def start(self) -> None:
        """Start the Discord bot."""
        await self._bot.start(self.config.discord_token)

    async def stop(self) -> None:
        """Stop the Discord bot."""
        await self._bot.close()
