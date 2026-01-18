"""Main entry point for Gru server."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from gru.config import Config
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.discord_bot import DiscordBot
from gru.orchestrator import Orchestrator
from gru.slack_bot import SlackBot
from gru.telegram_bot import TelegramBot
from gru.webhook import WebhookServer

logger = logging.getLogger(__name__)


def setup_git_credentials() -> None:
    """Configure git to use GRU_GITHUB_TOKEN for authentication."""
    token = os.getenv("GRU_GITHUB_TOKEN")
    if not token:
        return

    try:
        # Configure git credential helper to use the token
        subprocess.run(
            ["git", "config", "--global", "credential.helper", "store"],
            check=True,
            capture_output=True,
        )

        # Write credentials to git credential store
        credentials_path = Path.home() / ".git-credentials"
        credentials_path.write_text(f"https://x-access-token:{token}@github.com\n")
        credentials_path.chmod(0o600)

        logger.info("Git credentials configured for GitHub access")
    except Exception as e:
        logger.warning("Failed to configure git credentials: %s", e)


async def run_server(config: Config) -> None:
    """Run the Gru server."""
    # Initialize database
    db = Database(config.db_path)
    await db.connect()

    # Initialize crypto
    crypto = CryptoManager(config.data_dir)
    master_pass = os.getenv("GRU_MASTER_PASSWORD")
    if master_pass:
        crypto.initialize(master_pass)
    else:
        logger.warning("GRU_MASTER_PASSWORD not set. Secret storage disabled.")

    # Initialize components
    secrets = SecretStore(db, crypto)

    # Find MCP config file
    mcp_config_path = config.data_dir.parent / "mcp_servers.json"
    if not mcp_config_path.exists():
        mcp_config_path = Path(__file__).parent.parent.parent / "mcp_servers.json"

    orchestrator = Orchestrator(config, db, secrets, mcp_config_path if mcp_config_path.exists() else None)

    # Initialize bots based on configuration
    telegram_bot: TelegramBot | None = None
    discord_bot: DiscordBot | None = None
    slack_bot: SlackBot | None = None

    if config.telegram_token and config.telegram_admin_ids:
        telegram_bot = TelegramBot(config, orchestrator)

    if config.discord_token and config.discord_admin_ids:
        discord_bot = DiscordBot(config, orchestrator)

    if config.slack_bot_token and config.slack_app_token and config.slack_admin_ids:
        slack_bot = SlackBot(config, orchestrator)

    # Initialize webhook server
    webhook_server = WebhookServer(config, orchestrator)

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutdown requested")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    try:
        # Start MCP servers
        await orchestrator.mcp.load_config()
        mcp_count = await orchestrator.mcp.start_all()
        if mcp_count > 0:
            logger.info("Started %d MCP server(s)", mcp_count)

        # Start Telegram bot
        if telegram_bot:
            await telegram_bot.start()
            logger.info("Telegram bot started")
            logger.info("Telegram admin IDs: %s", config.telegram_admin_ids)

        # Start Discord bot (runs in background task since it's blocking)
        if discord_bot:
            asyncio.create_task(discord_bot.start())
            logger.info("Discord bot starting...")
            logger.info("Discord admin IDs: %s", config.discord_admin_ids)

        # Start Slack bot (runs in background task via Socket Mode)
        if slack_bot:
            asyncio.create_task(slack_bot.start())
            logger.info("Slack bot starting...")
            logger.info("Slack admin IDs: %s", config.slack_admin_ids)

        # Start webhook server
        await webhook_server.start()

        logger.info("Gru server started")
        logger.info("Data directory: %s", config.data_dir)

        # Start orchestrator in background
        asyncio.create_task(orchestrator.start())

        # Wait for shutdown signal
        await shutdown_event.wait()

    finally:
        logger.info("Shutting down...")
        await webhook_server.stop()
        await orchestrator.mcp.stop_all()
        await orchestrator.stop()
        if telegram_bot:
            await telegram_bot.stop()
        if discord_bot:
            await discord_bot.stop()
        if slack_bot:
            await slack_bot.stop()
        await db.close()
        logger.info("Shutdown complete")


def main() -> None:
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load and validate config
    config = Config.from_env()
    errors = config.validate()

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nAt least one bot interface required:")
        print("  Telegram: GRU_TELEGRAM_TOKEN + GRU_ADMIN_IDS")
        print("  Discord:  GRU_DISCORD_TOKEN + GRU_DISCORD_ADMIN_IDS")
        print("  Slack:    GRU_SLACK_BOT_TOKEN + GRU_SLACK_APP_TOKEN + GRU_SLACK_ADMIN_IDS")
        print("\nRequired:")
        print("  ANTHROPIC_API_KEY - Anthropic API key")
        print("\nTelegram:")
        print("  GRU_TELEGRAM_TOKEN - Telegram bot token")
        print("  GRU_ADMIN_IDS - Comma-separated admin Telegram user IDs")
        print("\nDiscord:")
        print("  GRU_DISCORD_TOKEN - Discord bot token")
        print("  GRU_DISCORD_ADMIN_IDS - Comma-separated admin Discord user IDs")
        print("  GRU_DISCORD_GUILD_ID - (optional) Restrict to specific server")
        print("\nSlack:")
        print("  GRU_SLACK_BOT_TOKEN - Slack Bot User OAuth Token (xoxb-...)")
        print("  GRU_SLACK_APP_TOKEN - Slack App-Level Token for Socket Mode (xapp-...)")
        print("  GRU_SLACK_ADMIN_IDS - Comma-separated admin Slack user IDs")
        print("\nOptional:")
        print("  GRU_DATA_DIR - Data directory (default: ~/.gru)")
        print("  GRU_MASTER_PASSWORD - Master password for secret encryption")
        print("  GRU_DEFAULT_MODEL - Default Claude model")
        print("  GRU_MAX_AGENTS - Max concurrent agents (default: 10)")
        sys.exit(1)

    # Set up git credentials if token is available
    setup_git_credentials()

    # Run server
    asyncio.run(run_server(config))


if __name__ == "__main__":
    main()
