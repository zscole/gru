"""Slack connector for reading user's workspace messages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from gru.proactive import ProactiveEngine

logger = logging.getLogger(__name__)


class SlackConnector:
    """Connector for reading Slack workspace messages.

    This uses a user token (xoxp-...) to read the user's messages,
    unlike the bot token which only sees messages sent to the bot.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.token_path = data_dir / "slack_user_token.json"
        self._token: str | None = None
        self._user_id: str | None = None
        self._last_sync: datetime | None = None
        self._seen_message_ids: set[str] = set()
        self._watched_channels: list[str] = []  # Channel IDs to watch

    def is_configured(self) -> bool:
        """Check if Slack user token is configured."""
        return self.token_path.exists()

    def is_authenticated(self) -> bool:
        """Check if we have a valid token."""
        return self._token is not None

    async def setup_token(self, user_token: str) -> bool:
        """Set up the user OAuth token.

        To get a user token:
        1. Create a Slack app at api.slack.com/apps
        2. Add User Token Scopes: channels:history, channels:read, groups:history,
           groups:read, im:history, im:read, mpim:history, mpim:read, users:read
        3. Install to workspace and copy the User OAuth Token (xoxp-...)
        """
        if not user_token.startswith("xoxp-"):
            logger.error("Invalid Slack user token. Must start with 'xoxp-'")
            return False

        # Verify token works
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": f"Bearer {user_token}"},
                )
                data = resp.json()

                if not data.get("ok"):
                    logger.error(f"Slack auth failed: {data.get('error')}")
                    return False

                self._user_id = data.get("user_id")
                self._token = user_token

                # Save token
                self.token_path.write_text(
                    json.dumps(
                        {
                            "token": user_token,
                            "user_id": self._user_id,
                            "team": data.get("team"),
                        }
                    )
                )

                logger.info(f"Slack user token saved for {data.get('user')}")
                return True

        except Exception as e:
            logger.error(f"Slack setup failed: {e}")
            return False

    def load_token(self) -> bool:
        """Load saved token."""
        if not self.token_path.exists():
            return False

        try:
            data = json.loads(self.token_path.read_text())
            self._token = data.get("token")
            self._user_id = data.get("user_id")
            return self._token is not None
        except Exception as e:
            logger.warning(f"Failed to load Slack token: {e}")
            return False

    async def set_watched_channels(self, channel_names: list[str]) -> list[str]:
        """Set channels to watch by name. Returns list of channel IDs found."""
        if not self._token:
            return []

        found_ids = []
        try:
            async with httpx.AsyncClient() as client:
                # Get all channels
                resp = await client.get(
                    "https://slack.com/api/conversations.list",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params={"types": "public_channel,private_channel"},
                )
                data = resp.json()

                if data.get("ok"):
                    channels = data.get("channels", [])
                    for ch in channels:
                        if ch.get("name") in channel_names:
                            found_ids.append(ch["id"])

        except Exception as e:
            logger.error(f"Failed to list Slack channels: {e}")

        self._watched_channels = found_ids
        return found_ids

    async def get_unread_dms(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent DMs (direct messages to user)."""
        if not self._token:
            return []

        messages = []
        try:
            async with httpx.AsyncClient() as client:
                # Get DM channels
                resp = await client.get(
                    "https://slack.com/api/conversations.list",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params={"types": "im"},
                )
                data = resp.json()

                if not data.get("ok"):
                    logger.error(f"Failed to list DMs: {data.get('error')}")
                    return []

                dm_channels = data.get("channels", [])

                # Get messages from each DM
                for dm in dm_channels[:10]:  # Limit channels to check
                    channel_id = dm["id"]
                    dm.get("user")

                    resp = await client.get(
                        "https://slack.com/api/conversations.history",
                        headers={"Authorization": f"Bearer {self._token}"},
                        params={
                            "channel": channel_id,
                            "limit": 5,
                        },
                    )
                    history = resp.json()

                    if history.get("ok"):
                        for msg in history.get("messages", []):
                            msg_id = f"{channel_id}:{msg.get('ts')}"
                            if msg_id in self._seen_message_ids:
                                continue

                            # Skip own messages
                            if msg.get("user") == self._user_id:
                                continue

                            self._seen_message_ids.add(msg_id)
                            messages.append(
                                {
                                    "type": "dm",
                                    "from_user": msg.get("user"),
                                    "text": msg.get("text", "")[:200],
                                    "ts": msg.get("ts"),
                                    "channel_id": channel_id,
                                }
                            )

        except Exception as e:
            logger.error(f"Failed to get Slack DMs: {e}")

        return messages[:limit]

    async def get_mentions(self, hours_back: int = 24, limit: int = 20) -> list[dict[str, Any]]:
        """Get messages that mention the user."""
        if not self._token or not self._user_id:
            return []

        messages = []
        try:
            async with httpx.AsyncClient() as client:
                # Search for mentions
                resp = await client.get(
                    "https://slack.com/api/search.messages",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params={
                        "query": f"<@{self._user_id}>",
                        "count": limit,
                        "sort": "timestamp",
                        "sort_dir": "desc",
                    },
                )
                data = resp.json()

                if not data.get("ok"):
                    # Search API requires paid Slack - fall back to checking channels
                    logger.debug("Slack search API not available, checking channels directly")
                    return await self._get_mentions_from_channels(hours_back, limit)

                for match in data.get("messages", {}).get("matches", []):
                    msg_id = f"{match.get('channel', {}).get('id')}:{match.get('ts')}"
                    if msg_id in self._seen_message_ids:
                        continue

                    self._seen_message_ids.add(msg_id)
                    messages.append(
                        {
                            "type": "mention",
                            "from_user": match.get("user"),
                            "text": match.get("text", "")[:200],
                            "ts": match.get("ts"),
                            "channel": match.get("channel", {}).get("name"),
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to get Slack mentions: {e}")

        return messages[:limit]

    async def _get_mentions_from_channels(self, hours_back: int, limit: int) -> list[dict[str, Any]]:
        """Fallback: scan watched channels for mentions."""
        if not self._token or not self._user_id:
            return []

        messages = []
        oldest = (datetime.now() - timedelta(hours=hours_back)).timestamp()

        try:
            async with httpx.AsyncClient() as client:
                for channel_id in self._watched_channels:
                    resp = await client.get(
                        "https://slack.com/api/conversations.history",
                        headers={"Authorization": f"Bearer {self._token}"},
                        params={
                            "channel": channel_id,
                            "oldest": str(oldest),
                            "limit": 50,
                        },
                    )
                    history = resp.json()

                    if not history.get("ok"):
                        continue

                    for msg in history.get("messages", []):
                        text = msg.get("text", "")
                        if f"<@{self._user_id}>" not in text:
                            continue

                        msg_id = f"{channel_id}:{msg.get('ts')}"
                        if msg_id in self._seen_message_ids:
                            continue

                        self._seen_message_ids.add(msg_id)
                        messages.append(
                            {
                                "type": "mention",
                                "from_user": msg.get("user"),
                                "text": text[:200],
                                "ts": msg.get("ts"),
                                "channel_id": channel_id,
                            }
                        )

        except Exception as e:
            logger.error(f"Failed to scan Slack channels: {e}")

        return messages[:limit]

    async def get_channel_activity(
        self, channel_names: list[str] | None = None, hours_back: int = 24, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get recent messages from specified channels."""
        if not self._token:
            return []

        messages = []
        oldest = (datetime.now() - timedelta(hours=hours_back)).timestamp()

        try:
            async with httpx.AsyncClient() as client:
                # Get channels if names specified
                if channel_names:
                    resp = await client.get(
                        "https://slack.com/api/conversations.list",
                        headers={"Authorization": f"Bearer {self._token}"},
                        params={"types": "public_channel,private_channel"},
                    )
                    data = resp.json()

                    if data.get("ok"):
                        channel_ids = [ch["id"] for ch in data.get("channels", []) if ch.get("name") in channel_names]
                    else:
                        channel_ids = []
                else:
                    channel_ids = self._watched_channels

                for channel_id in channel_ids:
                    resp = await client.get(
                        "https://slack.com/api/conversations.history",
                        headers={"Authorization": f"Bearer {self._token}"},
                        params={
                            "channel": channel_id,
                            "oldest": str(oldest),
                            "limit": 20,
                        },
                    )
                    history = resp.json()

                    if not history.get("ok"):
                        continue

                    for msg in history.get("messages", []):
                        msg_id = f"{channel_id}:{msg.get('ts')}"
                        if msg_id in self._seen_message_ids:
                            continue

                        # Skip own messages
                        if msg.get("user") == self._user_id:
                            continue

                        self._seen_message_ids.add(msg_id)
                        messages.append(
                            {
                                "type": "channel",
                                "from_user": msg.get("user"),
                                "text": msg.get("text", "")[:200],
                                "ts": msg.get("ts"),
                                "channel_id": channel_id,
                            }
                        )

        except Exception as e:
            logger.error(f"Failed to get Slack channel activity: {e}")

        return messages[:limit]

    async def get_user_name(self, user_id: str) -> str:
        """Resolve a user ID to display name."""
        if not self._token:
            return user_id

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://slack.com/api/users.info",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params={"user": user_id},
                )
                data = resp.json()

                if data.get("ok"):
                    user = data.get("user", {})
                    return (
                        user.get("profile", {}).get("display_name")
                        or user.get("real_name")
                        or user.get("name")
                        or user_id
                    )

        except Exception as e:
            logger.debug(f"Failed to get user name: {e}")

        return user_id

    async def sync_messages(
        self,
        proactive: ProactiveEngine,
        include_dms: bool = True,
        include_mentions: bool = True,
        include_channels: bool = False,
    ) -> list[dict[str, Any]]:
        """Sync Slack messages and create observations.

        Args:
            proactive: ProactiveEngine to add observations to
            include_dms: Include direct messages
            include_mentions: Include @mentions
            include_channels: Include watched channel activity

        Returns:
            List of new messages found
        """
        all_messages = []

        if include_dms:
            dms = await self.get_unread_dms(limit=10)
            all_messages.extend(dms)

        if include_mentions:
            mentions = await self.get_mentions(hours_back=24, limit=10)
            all_messages.extend(mentions)

        if include_channels and self._watched_channels:
            channel_msgs = await self.get_channel_activity(hours_back=12, limit=10)
            all_messages.extend(channel_msgs)

        # Create observations for important messages
        for msg in all_messages:
            sender = await self.get_user_name(msg.get("from_user", "Unknown"))
            text = msg.get("text", "")[:100]
            msg_type = msg.get("type", "message")

            if msg_type == "dm":
                importance = 0.8
                content = f"Slack DM from {sender}: {text}"
            elif msg_type == "mention":
                importance = 0.85
                content = f"Slack mention from {sender}: {text}"
            else:
                importance = 0.5
                content = f"Slack ({sender}): {text}"

            await proactive.add_observation(
                content=content,
                category="follow_up",
                importance=importance,
                source="slack",
                expires_in_hours=48,
                metadata={
                    "ts": msg.get("ts"),
                    "from_user": msg.get("from_user"),
                    "channel_id": msg.get("channel_id"),
                },
            )

        self._last_sync = datetime.now()
        logger.info(f"Slack sync: {len(all_messages)} new messages")
        return all_messages

    def get_status(self) -> dict[str, Any]:
        """Get connector status."""
        return {
            "configured": self.is_configured(),
            "authenticated": self.is_authenticated(),
            "user_id": self._user_id,
            "watched_channels": len(self._watched_channels),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "seen_messages": len(self._seen_message_ids),
        }

    def clear_seen_cache(self) -> None:
        """Clear the seen messages cache."""
        self._seen_message_ids.clear()
        logger.info("Cleared Slack seen messages cache")
