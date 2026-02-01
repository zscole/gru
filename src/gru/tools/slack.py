"""Slack tools - reading messages, DMs, mentions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.connectors.slack import SlackConnector

logger = logging.getLogger(__name__)

# Connector set by orchestrator
_connector: SlackConnector | None = None


def set_slack_connector(connector: SlackConnector) -> None:
    """Set the Slack connector (called by orchestrator)."""
    global _connector
    _connector = connector
    logger.debug("Slack connector set for tools")


def _get_connector():
    """Get the Slack connector."""
    global _connector
    if _connector is not None:
        return _connector

    # Fallback: try to load from disk
    from gru.connectors.slack import SlackConnector

    data_dir = Path.home() / ".gru"
    fallback = SlackConnector(data_dir)
    if fallback.load_token():
        _connector = fallback
        return _connector

    return None


async def get_slack_dms(limit: int = 10) -> dict:
    """Get recent direct messages from Slack."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Slack not configured. Run 'gru slack setup' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Slack. Run 'gru slack login' first."}

    try:
        dms = await connector.get_unread_dms(limit=limit)

        results = []
        for dm in dms:
            sender = await connector.get_user_name(dm.get("from_user", ""))
            results.append(
                {
                    "from": sender,
                    "text": dm.get("text", ""),
                    "timestamp": dm.get("ts"),
                }
            )

        return {
            "messages": results,
            "count": len(results),
            "type": "direct_messages",
        }

    except Exception as e:
        logger.error(f"Slack DMs fetch failed: {e}")
        return {"error": str(e)}


async def get_slack_mentions(hours_back: int = 24, limit: int = 10) -> dict:
    """Get recent @mentions from Slack."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Slack not configured. Run 'gru slack setup' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Slack. Run 'gru slack login' first."}

    try:
        mentions = await connector.get_mentions(hours_back=hours_back, limit=limit)

        results = []
        for m in mentions:
            sender = await connector.get_user_name(m.get("from_user", ""))
            results.append(
                {
                    "from": sender,
                    "text": m.get("text", ""),
                    "channel": m.get("channel"),
                    "timestamp": m.get("ts"),
                }
            )

        return {
            "messages": results,
            "count": len(results),
            "type": "mentions",
            "hours_back": hours_back,
        }

    except Exception as e:
        logger.error(f"Slack mentions fetch failed: {e}")
        return {"error": str(e)}


async def get_slack_channel_messages(channel_name: str, hours_back: int = 24, limit: int = 20) -> dict:
    """Get recent messages from a specific Slack channel."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Slack not configured. Run 'gru slack setup' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Slack. Run 'gru slack login' first."}

    try:
        messages = await connector.get_channel_activity(
            channel_names=[channel_name],
            hours_back=hours_back,
            limit=limit,
        )

        results = []
        for msg in messages:
            sender = await connector.get_user_name(msg.get("from_user", ""))
            results.append(
                {
                    "from": sender,
                    "text": msg.get("text", ""),
                    "timestamp": msg.get("ts"),
                }
            )

        return {
            "messages": results,
            "count": len(results),
            "channel": channel_name,
            "hours_back": hours_back,
        }

    except Exception as e:
        logger.error(f"Slack channel fetch failed: {e}")
        return {"error": str(e)}


async def setup_slack(user_token: str) -> dict:
    """Set up Slack integration with a user token."""
    from gru.connectors.slack import SlackConnector

    data_dir = Path.home() / ".gru"
    data_dir.mkdir(exist_ok=True)

    # Also save to gru/data for consistency
    gru_data_dir = Path.home() / "gru" / "data"
    if gru_data_dir.exists():
        data_dir = gru_data_dir

    connector = SlackConnector(data_dir)

    try:
        success = await connector.setup_token(user_token)
        if success:
            # Update the global connector
            global _connector
            _connector = connector
            return {
                "status": "success",
                "message": "Slack connected! I can now read your DMs and mentions.",
            }
        else:
            return {
                "status": "error",
                "error": "Invalid token. Make sure it starts with 'xoxp-' and has the right scopes.",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_slack_status() -> dict:
    """Check if Slack is configured and working."""
    connector = _get_connector()

    if connector is None:
        return {
            "configured": False,
            "message": "Slack not set up. Use auto_setup_slack to configure automatically.",
        }

    if not connector.is_authenticated():
        return {
            "configured": True,
            "authenticated": False,
            "message": "Slack token exists but isn't working. May need to re-authenticate.",
        }

    return {
        "configured": True,
        "authenticated": True,
        "status": connector.get_status(),
        "message": "Slack is connected and working.",
    }


async def auto_setup_slack() -> dict:
    """Automatically set up Slack integration using browser automation."""
    from gru.tools.slack_setup_agent import run_slack_setup_agent

    logger.info("Starting automated Slack setup...")
    result = await run_slack_setup_agent()

    if result.get("status") == "success":
        # Reload the connector with new token
        global _connector
        from gru.connectors.slack import SlackConnector

        data_dir = Path.home() / "gru" / "data"
        if not data_dir.exists():
            data_dir = Path.home() / ".gru"
        _connector = SlackConnector(data_dir)
        _connector.load_token()

    return result


def register_slack_tools() -> None:
    """Register Slack tools."""
    register_tool(
        name="auto_setup_slack",
        description="Automatically set up Slack integration using browser automation. Creates a Slack app, configures permissions, and extracts the token. Use when user wants to connect Slack.",
        parameters={},
        handler=auto_setup_slack,
    )

    register_tool(
        name="setup_slack",
        description="Set up Slack integration with a provided token. Use when user provides a Slack user token (xoxp-...).",
        parameters={
            "user_token": {
                "type": "string",
                "description": "Slack User OAuth Token (starts with xoxp-)",
            },
        },
        handler=setup_slack,
    )

    register_tool(
        name="check_slack_status",
        description="Check if Slack is configured. Use when Slack tools fail or user asks about Slack status.",
        parameters={},
        handler=check_slack_status,
    )

    register_tool(
        name="get_slack_dms",
        description="Get recent direct messages from Slack. Use when user asks about their Slack DMs or messages.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum messages to return (default 10)",
                "optional": True,
            },
        },
        handler=get_slack_dms,
    )

    register_tool(
        name="get_slack_mentions",
        description="Get recent @mentions from Slack. Use when user asks about Slack mentions or notifications.",
        parameters={
            "hours_back": {
                "type": "integer",
                "description": "How many hours back to look (default 24)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum mentions to return (default 10)",
                "optional": True,
            },
        },
        handler=get_slack_mentions,
    )

    register_tool(
        name="get_slack_channel",
        description="Get recent messages from a specific Slack channel.",
        parameters={
            "channel_name": {
                "type": "string",
                "description": "Name of the channel (without #)",
            },
            "hours_back": {
                "type": "integer",
                "description": "How many hours back to look (default 24)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum messages to return (default 20)",
                "optional": True,
            },
        },
        handler=get_slack_channel_messages,
    )
