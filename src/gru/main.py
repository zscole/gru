"""Main entry point for Gru server."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from gru.config import Config
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.discord_bot import DiscordBot
from gru.orchestrator import Orchestrator
from gru.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


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

    if config.telegram_token and config.telegram_admin_ids:
        telegram_bot = TelegramBot(config, orchestrator)

    if config.discord_token and config.discord_admin_ids:
        discord_bot = DiscordBot(config, orchestrator)

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

        logger.info("Gru server started")
        logger.info("Data directory: %s", config.data_dir)

        # Start orchestrator in background
        asyncio.create_task(orchestrator.start())

        # Wait for shutdown signal
        await shutdown_event.wait()

    finally:
        logger.info("Shutting down...")
        await orchestrator.mcp.stop_all()
        await orchestrator.stop()
        if telegram_bot:
            await telegram_bot.stop()
        if discord_bot:
            await discord_bot.stop()
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
        print("\nRequired:")
        print("  ANTHROPIC_API_KEY - Anthropic API key")
        print("\nTelegram:")
        print("  GRU_TELEGRAM_TOKEN - Telegram bot token")
        print("  GRU_ADMIN_IDS - Comma-separated admin Telegram user IDs")
        print("\nDiscord:")
        print("  GRU_DISCORD_TOKEN - Discord bot token")
        print("  GRU_DISCORD_ADMIN_IDS - Comma-separated admin Discord user IDs")
        print("  GRU_DISCORD_GUILD_ID - (optional) Restrict to specific server")
        print("\nOptional:")
        print("  GRU_DATA_DIR - Data directory (default: ~/.gru)")
        print("  GRU_MASTER_PASSWORD - Master password for secret encryption")
        print("  GRU_DEFAULT_MODEL - Default Claude model")
        print("  GRU_MAX_AGENTS - Max concurrent agents (default: 10)")
        sys.exit(1)

    # Run server
    asyncio.run(run_server(config))


if __name__ == "__main__":
    main()
