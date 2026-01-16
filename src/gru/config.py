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

    # Encryption
    master_key_iterations: int = 480000

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

        data_dir = Path(os.getenv("GRU_DATA_DIR", str(Path.home() / ".gru")))
        workdir = Path(os.getenv("GRU_WORKDIR", str(Path.home() / "gru-workspace")))

        return cls(
            data_dir=data_dir,
            telegram_token=os.getenv("GRU_TELEGRAM_TOKEN", ""),
            telegram_admin_ids=admin_ids,
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
            worktree_base_dir=Path(os.getenv("GRU_WORKTREE_DIR")) if os.getenv("GRU_WORKTREE_DIR") else None,
            delete_worktree_branch=os.getenv("GRU_DELETE_WORKTREE_BRANCH", "false").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.telegram_token:
            errors.append("GRU_TELEGRAM_TOKEN is required")
        if not self.telegram_admin_ids:
            errors.append("GRU_ADMIN_IDS is required (comma-separated Telegram user IDs)")
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        return errors
