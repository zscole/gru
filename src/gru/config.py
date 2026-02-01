"""Configuration management for Gru."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path.home() / ".gru")
    db_path: Path = field(init=False)

    # Telegram
    telegram_token: str = ""
    telegram_admin_ids: list[int] = field(default_factory=list)

    # Discord
    discord_token: str = ""
    discord_admin_ids: list[int] = field(default_factory=list)
    discord_guild_id: int | None = None  # Optional: restrict to specific server

    # Slack
    slack_bot_token: str = ""  # xoxb-... Bot User OAuth Token
    slack_app_token: str = ""  # xapp-... App-Level Token for Socket Mode
    slack_admin_ids: list[str] = field(default_factory=list)  # Slack user IDs (strings)

    # Anthropic
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192

    # Agent defaults
    default_timeout: int = 300  # seconds per approval
    max_concurrent_agents: int = 10
    default_priority: str = "normal"
    default_workdir: Path = field(default_factory=lambda: Path.home() / "gru-workspace")

    # Limits
    max_task_length: int = 10000  # characters
    max_agent_runtime: int = 3600  # seconds (1 hour)
    max_agent_turns: int = 100  # max Claude API calls per agent
    max_conversation_messages: int = 50  # max messages before truncation
    max_tool_output: int = 50000  # max chars per tool output (~12k tokens)

    # Scheduler
    scheduler_interval: float = 0.1  # seconds
    starvation_threshold: int = 10  # promotions before boost

    # Resource limits (cgroups)
    enable_cgroups: bool = False
    default_memory_limit: str = "512M"
    default_cpu_quota: int = 50  # percent

    # Git worktrees (agent isolation)
    enable_worktrees: bool = True  # Auto-create worktrees when workdir is a git repo
    worktree_base_dir: Path | None = None  # Where to create worktrees (default: workdir/../.gru-worktrees)
    delete_worktree_branch: bool = False  # Delete branch when agent completes
    auto_push: bool = True  # Auto commit and push on agent pause/complete

    # Progress reports
    progress_report_interval: int = 0  # minutes between progress reports (0 = disabled)

    # Smart notifications
    stuck_threshold_turns: int = 5  # Alert if no tool calls for X turns
    token_burn_alert: int = 50000  # Alert if agent uses more than X tokens

    # Webhook server
    webhook_enabled: bool = False
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8080
    webhook_secret: str = ""  # Optional secret for signature verification

    # Encryption
    master_key_iterations: int = 480000

    # Memory system
    memory_enabled: bool = True
    memory_extraction_enabled: bool = True  # Auto-extract facts from conversations
    memory_context_limit: int = 10  # Max facts to inject into context

    # Browser/Actions
    browser_mode: str = "headless"  # headless or headed
    browser_type: str = "chromium"  # chromium, firefox, webkit
    browser_timeout: int = 30000  # ms
    action_confirm_payments: bool = True  # Require confirmation for purchases
    
    # Voice Messages
    voice_enabled: bool = True  # Enable voice message support
    voice_tts_provider: str = "edge"  # TTS provider: eleven_labs, openai, edge
    voice_stt_provider: str = "openai"  # STT provider: openai, whisper (claude not supported)
    voice_default_speed: float = 1.0  # Default speech speed
    voice_max_text_length: int = 4000  # Max characters for TTS
    voice_auto_reply: bool = False  # Auto-reply to voice messages with voice

    def __post_init__(self) -> None:
        self.db_path = self.data_dir / "gru.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.default_workdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Config:
        """Load configuration from environment variables."""
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()

        admin_ids_raw = os.getenv("GRU_ADMIN_IDS", "")
        admin_ids = [int(x.strip()) for x in admin_ids_raw.split(",") if x.strip()]

        discord_admin_ids_raw = os.getenv("GRU_DISCORD_ADMIN_IDS", "")
        discord_admin_ids = [int(x.strip()) for x in discord_admin_ids_raw.split(",") if x.strip()]

        discord_guild_id = None
        if guild_id_str := os.getenv("GRU_DISCORD_GUILD_ID"):
            discord_guild_id = int(guild_id_str)

        slack_admin_ids_raw = os.getenv("GRU_SLACK_ADMIN_IDS", "")
        slack_admin_ids = [x.strip() for x in slack_admin_ids_raw.split(",") if x.strip()]

        data_dir = Path(os.getenv("GRU_DATA_DIR", str(Path.home() / ".gru")))
        workdir = Path(os.getenv("GRU_WORKDIR", str(Path.home() / "gru-workspace")))

        return cls(
            data_dir=data_dir,
            telegram_token=os.getenv("GRU_TELEGRAM_TOKEN", ""),
            telegram_admin_ids=admin_ids,
            discord_token=os.getenv("GRU_DISCORD_TOKEN", ""),
            discord_admin_ids=discord_admin_ids,
            discord_guild_id=discord_guild_id,
            slack_bot_token=os.getenv("GRU_SLACK_BOT_TOKEN", ""),
            slack_app_token=os.getenv("GRU_SLACK_APP_TOKEN", ""),
            slack_admin_ids=slack_admin_ids,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            default_model=os.getenv("GRU_DEFAULT_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.getenv("GRU_MAX_TOKENS", "8192")),
            default_timeout=int(os.getenv("GRU_DEFAULT_TIMEOUT", "300")),
            max_concurrent_agents=int(os.getenv("GRU_MAX_AGENTS", "10")),
            default_workdir=workdir,
            enable_cgroups=os.getenv("GRU_ENABLE_CGROUPS", "false").lower() == "true",
            default_memory_limit=os.getenv("GRU_MEMORY_LIMIT", "512M"),
            default_cpu_quota=int(os.getenv("GRU_CPU_QUOTA", "50")),
            enable_worktrees=os.getenv("GRU_ENABLE_WORKTREES", "true").lower() == "true",
            worktree_base_dir=Path(wt_dir) if (wt_dir := os.getenv("GRU_WORKTREE_DIR")) else None,
            delete_worktree_branch=os.getenv("GRU_DELETE_WORKTREE_BRANCH", "false").lower() == "true",
            webhook_enabled=os.getenv("GRU_WEBHOOK_ENABLED", "false").lower() == "true",
            webhook_host=os.getenv("GRU_WEBHOOK_HOST", "0.0.0.0"),
            webhook_port=int(os.getenv("GRU_WEBHOOK_PORT", "8080")),
            webhook_secret=os.getenv("GRU_WEBHOOK_SECRET", ""),
            progress_report_interval=int(os.getenv("GRU_PROGRESS_REPORT_INTERVAL", "0")),
            memory_enabled=os.getenv("GRU_MEMORY_ENABLED", "true").lower() == "true",
            memory_extraction_enabled=os.getenv("GRU_MEMORY_EXTRACTION", "true").lower() == "true",
            memory_context_limit=int(os.getenv("GRU_MEMORY_CONTEXT_LIMIT", "10")),
            browser_mode=os.getenv("GRU_BROWSER_MODE", "headless"),
            browser_type=os.getenv("GRU_BROWSER_TYPE", "chromium"),
            browser_timeout=int(os.getenv("GRU_BROWSER_TIMEOUT", "30000")),
            action_confirm_payments=os.getenv("GRU_ACTION_CONFIRM_PAYMENTS", "true").lower() == "true",
            voice_enabled=os.getenv("GRU_VOICE_ENABLED", "true").lower() == "true",
            voice_tts_provider=os.getenv("GRU_TTS_PROVIDER", "edge"),
            voice_stt_provider=os.getenv("GRU_STT_PROVIDER", "openai"),
            voice_default_speed=float(os.getenv("GRU_VOICE_SPEED", "1.0")),
            voice_max_text_length=int(os.getenv("GRU_VOICE_MAX_LENGTH", "4000")),
            voice_auto_reply=os.getenv("GRU_VOICE_AUTO_REPLY", "false").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        # At least one bot interface required
        has_telegram = self.telegram_token and self.telegram_admin_ids
        has_discord = self.discord_token and self.discord_admin_ids
        has_slack = self.slack_bot_token and self.slack_app_token and self.slack_admin_ids

        if not has_telegram and not has_discord and not has_slack:
            errors.append(
                "At least one bot interface required: "
                "set GRU_TELEGRAM_TOKEN + GRU_ADMIN_IDS or "
                "GRU_DISCORD_TOKEN + GRU_DISCORD_ADMIN_IDS or "
                "GRU_SLACK_BOT_TOKEN + GRU_SLACK_APP_TOKEN + GRU_SLACK_ADMIN_IDS"
            )

        # Validate partial configs
        if self.telegram_token and not self.telegram_admin_ids:
            errors.append("GRU_ADMIN_IDS required when using Telegram")
        if self.telegram_admin_ids and not self.telegram_token:
            errors.append("GRU_TELEGRAM_TOKEN required when using Telegram")
        if self.discord_token and not self.discord_admin_ids:
            errors.append("GRU_DISCORD_ADMIN_IDS required when using Discord")
        if self.discord_admin_ids and not self.discord_token:
            errors.append("GRU_DISCORD_TOKEN required when using Discord")

        # Slack requires both tokens
        slack_partial = self.slack_bot_token or self.slack_app_token or self.slack_admin_ids
        if slack_partial and not has_slack:
            missing = []
            if not self.slack_bot_token:
                missing.append("GRU_SLACK_BOT_TOKEN")
            if not self.slack_app_token:
                missing.append("GRU_SLACK_APP_TOKEN")
            if not self.slack_admin_ids:
                missing.append("GRU_SLACK_ADMIN_IDS")
            errors.append(f"Slack requires all of: {', '.join(missing)}")

        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")

        return errors
