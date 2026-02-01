"""Setup and configuration management for Gru."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of API keys/tokens Gru can detect."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE_CLIENT_ID = "google_client_id"
    GOOGLE_CLIENT_SECRET = "google_client_secret"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK_BOT = "slack_bot"
    SLACK_APP = "slack_app"
    UNKNOWN = "unknown"


# Patterns for detecting key types
KEY_PATTERNS = {
    KeyType.ANTHROPIC: [
        r"^sk-ant-api\d{2}-[\w-]+$",
        r"^sk-ant-[\w-]+$",
    ],
    KeyType.OPENAI: [
        r"^sk-[a-zA-Z0-9]{20,}$",
        r"^sk-proj-[\w-]+$",
    ],
    KeyType.GOOGLE_CLIENT_ID: [
        r"^\d+-[\w]+\.apps\.googleusercontent\.com$",
    ],
    KeyType.GOOGLE_CLIENT_SECRET: [
        r"^GOCSPX-[\w-]+$",
    ],
    KeyType.TELEGRAM: [
        r"^\d{8,10}:[A-Za-z0-9_-]{30,}$",
    ],
    KeyType.DISCORD: [
        r"^[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}$",
        r"^[MN][A-Za-z0-9_-]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}$",
    ],
    KeyType.SLACK_BOT: [
        r"^xoxb-[\d]+-[\d]+-[\w]+$",
    ],
    KeyType.SLACK_APP: [
        r"^xapp-[\d]+-[\w]+-[\d]+-[\w]+$",
    ],
}

# Friendly names for key types
KEY_TYPE_NAMES = {
    KeyType.ANTHROPIC: "Anthropic API Key",
    KeyType.OPENAI: "OpenAI API Key",
    KeyType.GOOGLE_CLIENT_ID: "Google OAuth Client ID",
    KeyType.GOOGLE_CLIENT_SECRET: "Google OAuth Client Secret",
    KeyType.TELEGRAM: "Telegram Bot Token",
    KeyType.DISCORD: "Discord Bot Token",
    KeyType.SLACK_BOT: "Slack Bot Token",
    KeyType.SLACK_APP: "Slack App Token",
    KeyType.UNKNOWN: "Unknown Key",
}

# Environment variable mappings
KEY_ENV_VARS = {
    KeyType.ANTHROPIC: "ANTHROPIC_API_KEY",
    KeyType.OPENAI: "OPENAI_API_KEY",
    KeyType.TELEGRAM: "GRU_TELEGRAM_TOKEN",
    KeyType.DISCORD: "GRU_DISCORD_TOKEN",
    KeyType.SLACK_BOT: "GRU_SLACK_BOT_TOKEN",
    KeyType.SLACK_APP: "GRU_SLACK_APP_TOKEN",
}


def detect_key_type(value: str) -> KeyType:
    """Detect what type of API key or token this is.

    Args:
        value: The key/token string to analyze

    Returns:
        The detected KeyType
    """
    value = value.strip()

    for key_type, patterns in KEY_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, value):
                return key_type

    return KeyType.UNKNOWN


def detect_multiple_keys(text: str) -> list[tuple[KeyType, str]]:
    """Detect multiple keys in a block of text.

    Useful when user pastes multiple keys at once.

    Args:
        text: Text that may contain multiple keys

    Returns:
        List of (KeyType, value) tuples
    """
    results = []
    seen = set()

    # First, try to find keys using patterns directly
    for key_type, patterns in KEY_PATTERNS.items():
        for pattern in patterns:
            # Make pattern non-anchored for searching
            search_pattern = pattern.lstrip("^").rstrip("$")
            for match in re.finditer(search_pattern, text):
                value = match.group(0)
                if value not in seen:
                    seen.add(value)
                    results.append((key_type, value))

    # If no patterns found, try splitting (but preserve colons for tokens)
    if not results:
        # Split by whitespace and newlines only
        tokens = re.split(r"[\s\n]+", text)

        for token in tokens:
            # Clean up punctuation at start/end but preserve internal structure
            token = token.strip().strip(",;")
            if len(token) < 10:
                continue

            key_type = detect_key_type(token)
            if key_type != KeyType.UNKNOWN and token not in seen:
                seen.add(token)
                results.append((key_type, token))

    return results


@dataclass
class ConfigValue:
    """A configuration value with metadata."""

    key: str
    value: str
    key_type: KeyType | None = None
    source: str = "user"  # user, env, default
    is_secret: bool = False


class ConfigManager:
    """Manages Gru configuration with persistence."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.config_file = data_dir / "config.json"
        self.env_file = data_dir / ".env"
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                self._config = json.loads(self.config_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self._config = {}

    def _save(self) -> None:
        """Save configuration to file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self._config, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        # Check environment first
        env_key = f"GRU_{key.upper().replace('-', '_')}"
        if env_val := os.getenv(env_key):
            return env_val

        # Special mappings
        if key == "anthropic-key" and (env_val := os.getenv("ANTHROPIC_API_KEY")):
            return env_val

        return self._config.get(key, default)

    def set(self, key: str, value: str, update_env: bool = True) -> ConfigValue:
        """Set a config value.

        Args:
            key: Configuration key
            value: Value to set
            update_env: If True, also update .env file

        Returns:
            ConfigValue with details about what was set
        """
        # Detect key type
        key_type = detect_key_type(value)

        # Determine if this is a secret
        is_secret = (
            key_type != KeyType.UNKNOWN or "key" in key.lower() or "token" in key.lower() or "secret" in key.lower()
        )

        # Store in config
        self._config[key] = value
        self._save()

        # Update .env file for secrets
        if update_env and is_secret:
            self._update_env_file(key, value, key_type)

        logger.info(f"Config set: {key} = {'<secret>' if is_secret else value}")

        return ConfigValue(
            key=key,
            value=value,
            key_type=key_type if key_type != KeyType.UNKNOWN else None,
            source="user",
            is_secret=is_secret,
        )

    def set_from_detection(self, value: str) -> ConfigValue | None:
        """Auto-detect and set a key based on its format.

        Args:
            value: The key/token value

        Returns:
            ConfigValue if detected and set, None if unknown
        """
        key_type = detect_key_type(value)

        if key_type == KeyType.UNKNOWN:
            return None

        # Map to config key
        key_map = {
            KeyType.ANTHROPIC: "anthropic-key",
            KeyType.OPENAI: "openai-key",
            KeyType.GOOGLE_CLIENT_ID: "google-client-id",
            KeyType.GOOGLE_CLIENT_SECRET: "google-client-secret",
            KeyType.TELEGRAM: "telegram-token",
            KeyType.DISCORD: "discord-token",
            KeyType.SLACK_BOT: "slack-bot-token",
            KeyType.SLACK_APP: "slack-app-token",
        }

        key = key_map.get(key_type)
        if not key:
            return None

        return self.set(key, value)

    def _update_env_file(self, key: str, value: str, key_type: KeyType) -> None:
        """Update the .env file with a new value."""
        # Determine env var name
        if key_type != KeyType.UNKNOWN and key_type in KEY_ENV_VARS:
            env_var = KEY_ENV_VARS[key_type]
        else:
            env_var = f"GRU_{key.upper().replace('-', '_')}"

        # Read existing .env
        env_content = ""
        if self.env_file.exists():
            env_content = self.env_file.read_text()

        # Check if variable already exists
        pattern = rf"^{re.escape(env_var)}=.*$"
        if re.search(pattern, env_content, re.MULTILINE):
            # Update existing
            env_content = re.sub(pattern, f"{env_var}={value}", env_content, flags=re.MULTILINE)
        else:
            # Add new
            if env_content and not env_content.endswith("\n"):
                env_content += "\n"
            env_content += f"{env_var}={value}\n"

        self.env_file.write_text(env_content)
        logger.info(f"Updated .env: {env_var}")

    def list_keys(self) -> list[str]:
        """List all config keys."""
        return list(self._config.keys())

    def get_status(self) -> dict[str, Any]:
        """Get configuration status."""
        status = {
            "configured": {},
            "missing": [],
        }

        # Check required configs
        required = [
            ("anthropic-key", "ANTHROPIC_API_KEY", "Anthropic API (required)"),
        ]

        optional = [
            ("telegram-token", "GRU_TELEGRAM_TOKEN", "Telegram Bot"),
            ("discord-token", "GRU_DISCORD_TOKEN", "Discord Bot"),
            ("slack-bot-token", "GRU_SLACK_BOT_TOKEN", "Slack Bot"),
            ("google-client-id", None, "Google OAuth"),
        ]

        for key, env_var, name in required + optional:
            value = self.get(key)
            if not value and env_var:
                value = os.getenv(env_var)

            if value:
                # Mask the value
                masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
                status["configured"][name] = masked
            elif (key, env_var, name) in required:
                status["missing"].append(name)

        return status

    def delete(self, key: str) -> bool:
        """Delete a config key."""
        if key in self._config:
            del self._config[key]
            self._save()
            return True
        return False


class SetupWizard:
    """Interactive setup wizard for Gru."""

    def __init__(self, config_manager: ConfigManager, data_dir: Path) -> None:
        self.config = config_manager
        self.data_dir = data_dir

    def get_setup_status(self) -> dict[str, Any]:
        """Get current setup status."""
        status = {
            "complete": False,
            "steps": {},
        }

        # Check Anthropic API - check config first, then env
        anthropic_key = self.config._config.get("anthropic-key") or os.getenv("ANTHROPIC_API_KEY")
        status["steps"]["anthropic"] = {
            "name": "Anthropic API",
            "required": True,
            "configured": bool(anthropic_key),
            "help": "Get your API key from https://console.anthropic.com/",
        }

        # Check messaging platform
        has_telegram = bool(self.config.get("telegram-token") or os.getenv("GRU_TELEGRAM_TOKEN"))
        has_discord = bool(self.config.get("discord-token") or os.getenv("GRU_DISCORD_TOKEN"))
        has_slack = bool(
            (self.config.get("slack-bot-token") or os.getenv("GRU_SLACK_BOT_TOKEN"))
            and (self.config.get("slack-app-token") or os.getenv("GRU_SLACK_APP_TOKEN"))
        )

        status["steps"]["messaging"] = {
            "name": "Messaging Platform",
            "required": True,
            "configured": has_telegram or has_discord or has_slack,
            "platforms": {
                "telegram": has_telegram,
                "discord": has_discord,
                "slack": has_slack,
            },
            "help": "Set up at least one: Telegram, Discord, or Slack",
        }

        # Check Google
        google_creds = (self.data_dir / "google_credentials.json").exists()
        google_token = (self.data_dir / "google_token.json").exists()
        status["steps"]["google"] = {
            "name": "Google Integration",
            "required": False,
            "configured": google_creds and google_token,
            "credentials_set": google_creds,
            "authenticated": google_token,
            "help": "For Calendar, Gmail, and Docs integration",
        }

        # Check admin ID
        has_admin = bool(
            os.getenv("GRU_ADMIN_IDS")
            or os.getenv("GRU_DISCORD_ADMIN_IDS")
            or os.getenv("GRU_SLACK_ADMIN_IDS")
            or self.config.get("admin-id")
        )
        status["steps"]["admin"] = {
            "name": "Admin User",
            "required": True,
            "configured": has_admin,
            "help": "Your user ID on the messaging platform",
        }

        # Check if complete
        required_complete = all(step["configured"] for step in status["steps"].values() if step["required"])
        status["complete"] = required_complete

        return status

    def get_next_step(self) -> dict[str, Any] | None:
        """Get the next setup step needed."""
        status = self.get_setup_status()

        for step_id, step in status["steps"].items():
            if step["required"] and not step["configured"]:
                return {"id": step_id, **step}

        return None

    def get_setup_instructions(self) -> str:
        """Get human-readable setup instructions."""
        status = self.get_setup_status()
        lines = ["Gru Setup Status", "=" * 40, ""]

        for _step_id, step in status["steps"].items():
            icon = "[x]" if step["configured"] else "[ ]"
            required = "(required)" if step["required"] else "(optional)"
            lines.append(f"{icon} {step['name']} {required}")

            if not step["configured"]:
                lines.append(f"    {step['help']}")

        lines.append("")
        if status["complete"]:
            lines.append("Setup complete! Gru is ready to run.")
        else:
            lines.append("To complete setup, provide the missing items above.")
            lines.append("You can paste API keys directly in chat or use:")
            lines.append("  gru config set <key> <value>")

        return "\n".join(lines)


def parse_config_from_message(message: str) -> list[ConfigValue]:
    """Parse configuration values from a chat message.

    Handles formats like:
    - Just a key pasted directly
    - "my anthropic key is sk-ant-..."
    - "telegram: 123456:ABC..."
    - Multiple keys pasted together

    Args:
        message: User message that may contain config values

    Returns:
        List of ConfigValue objects detected
    """
    results = []

    # Try to detect keys in the message
    detected = detect_multiple_keys(message)

    for key_type, value in detected:
        results.append(
            ConfigValue(
                key=key_type.value,
                value=value,
                key_type=key_type,
                source="chat",
                is_secret=True,
            )
        )

    return results


def get_config_manager(data_dir: Path) -> ConfigManager:
    """Get a ConfigManager instance."""
    return ConfigManager(data_dir)


def get_setup_wizard(data_dir: Path) -> SetupWizard:
    """Get a SetupWizard instance."""
    config = get_config_manager(data_dir)
    return SetupWizard(config, data_dir)
