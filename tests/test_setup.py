"""Tests for the setup and configuration management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from gru.setup import (
    detect_key_type,
    detect_multiple_keys,
    ConfigManager,
    SetupWizard,
    KeyType,
    KEY_TYPE_NAMES,
    get_config_manager,
    get_setup_wizard,
    parse_config_from_message,
)


class TestKeyDetection:
    """Tests for API key detection."""

    def test_detect_anthropic_key(self):
        """Test detecting Anthropic API key."""
        key = "sk-ant-api03-abc123def456-xyz789"
        assert detect_key_type(key) == KeyType.ANTHROPIC

    def test_detect_anthropic_key_variant(self):
        """Test detecting Anthropic API key variant."""
        key = "sk-ant-abcdef123456"
        assert detect_key_type(key) == KeyType.ANTHROPIC

    def test_detect_openai_key(self):
        """Test detecting OpenAI API key."""
        key = "sk-proj-abc123def456ghi789jkl"
        assert detect_key_type(key) == KeyType.OPENAI

    def test_detect_google_client_id(self):
        """Test detecting Google OAuth Client ID."""
        key = "123456789012-abcdefghijk.apps.googleusercontent.com"
        assert detect_key_type(key) == KeyType.GOOGLE_CLIENT_ID

    def test_detect_google_client_secret(self):
        """Test detecting Google OAuth Client Secret."""
        key = "GOCSPX-abcdef123456"
        assert detect_key_type(key) == KeyType.GOOGLE_CLIENT_SECRET

    def test_detect_telegram_token(self):
        """Test detecting Telegram bot token."""
        key = "123456789:AABBccDDeeFfGgHhIiJjKkLlMmNnOoPpQqRr"
        assert detect_key_type(key) == KeyType.TELEGRAM

    def test_detect_discord_token(self):
        """Test detecting Discord bot token."""
        # Fake token that matches format but uses invalid base64 user ID
        key = "AAAAAAAAAAAAAAAAAAAAAA.AAAAAA.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        assert detect_key_type(key) == KeyType.DISCORD

    def test_detect_slack_bot_token(self):
        """Test detecting Slack bot token."""
        key = "xoxb-123456789012-123456789012-abcdefGHIJKL"
        assert detect_key_type(key) == KeyType.SLACK_BOT

    def test_detect_slack_app_token(self):
        """Test detecting Slack app token."""
        key = "xapp-1-A012345-123456789012-abcdef"
        assert detect_key_type(key) == KeyType.SLACK_APP

    def test_detect_unknown_key(self):
        """Test unknown key detection."""
        key = "some-random-string"
        assert detect_key_type(key) == KeyType.UNKNOWN

    def test_detect_multiple_keys(self):
        """Test detecting multiple keys in text."""
        text = """
        Here are my keys:
        Anthropic: sk-ant-api03-abc123def456-xyz789
        Telegram: 123456789:AABBccDDeeFfGgHhIiJjKkLlMmNnOoPpQqRr
        """
        detected = detect_multiple_keys(text)

        assert len(detected) == 2
        assert detected[0][0] == KeyType.ANTHROPIC
        assert detected[1][0] == KeyType.TELEGRAM


class TestConfigManager:
    """Tests for ConfigManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for config."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def config_manager(self, temp_dir):
        """Create ConfigManager with temp directory."""
        return ConfigManager(temp_dir)

    def test_set_and_get(self, config_manager):
        """Test setting and getting config values."""
        config_manager.set("test-key", "test-value", update_env=False)
        assert config_manager.get("test-key") == "test-value"

    def test_set_creates_env_for_secret(self, config_manager, temp_dir):
        """Test that secrets are written to .env file."""
        config_manager.set("api-key", "secret123")

        env_file = temp_dir / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "secret123" in content

    def test_set_from_detection(self, config_manager):
        """Test auto-detection and setting."""
        key = "sk-ant-api03-abc123def456-xyz789"
        result = config_manager.set_from_detection(key)

        assert result is not None
        assert result.key == "anthropic-key"
        assert result.key_type == KeyType.ANTHROPIC
        assert config_manager.get("anthropic-key") == key

    def test_set_from_detection_unknown(self, config_manager):
        """Test auto-detection returns None for unknown keys."""
        result = config_manager.set_from_detection("random-string")
        assert result is None

    def test_list_keys(self, config_manager):
        """Test listing config keys."""
        config_manager.set("key1", "value1", update_env=False)
        config_manager.set("key2", "value2", update_env=False)

        keys = config_manager.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_delete(self, config_manager):
        """Test deleting config keys."""
        config_manager.set("to-delete", "value", update_env=False)
        assert config_manager.get("to-delete") == "value"

        config_manager.delete("to-delete")
        assert config_manager.get("to-delete") is None

    def test_get_status(self, config_manager):
        """Test getting config status."""
        config_manager.set("anthropic-key", "sk-ant-api03-test", update_env=False)
        status = config_manager.get_status()

        assert "configured" in status
        assert "missing" in status


class TestSetupWizard:
    """Tests for SetupWizard."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def wizard(self, temp_dir):
        """Create SetupWizard."""
        config = ConfigManager(temp_dir)
        return SetupWizard(config, temp_dir)

    def test_get_setup_status_empty(self, wizard):
        """Test setup status with no configuration."""
        status = wizard.get_setup_status()

        assert status["complete"] is False
        assert not status["steps"]["anthropic"]["configured"]
        assert not status["steps"]["messaging"]["configured"]

    def test_get_setup_status_partial(self, temp_dir):
        """Test setup status with partial configuration."""
        # Need to use the same ConfigManager instance
        config = ConfigManager(temp_dir)
        config.set("anthropic-key", "sk-ant-api03-test123", update_env=False)
        wizard = SetupWizard(config, temp_dir)

        status = wizard.get_setup_status()

        assert status["steps"]["anthropic"]["configured"]
        assert not status["steps"]["messaging"]["configured"]
        assert status["complete"] is False

    def test_get_next_step(self, wizard):
        """Test getting next setup step."""
        step = wizard.get_next_step()

        assert step is not None
        assert step["id"] == "anthropic"
        assert step["required"] is True

    def test_get_setup_instructions(self, wizard):
        """Test getting setup instructions."""
        instructions = wizard.get_setup_instructions()

        assert "Setup Status" in instructions or "Gru Setup" in instructions
        assert "Anthropic" in instructions


class TestParseConfigFromMessage:
    """Tests for parsing config from chat messages."""

    def test_parse_direct_key(self):
        """Test parsing a directly pasted key."""
        message = "sk-ant-api03-abc123def456-xyz789"
        results = parse_config_from_message(message)

        assert len(results) == 1
        assert results[0].key_type == KeyType.ANTHROPIC

    def test_parse_key_in_sentence(self):
        """Test parsing a key within a sentence."""
        message = "Here is my key: sk-ant-api03-abc123def456-xyz789"
        results = parse_config_from_message(message)

        assert len(results) == 1
        assert results[0].key_type == KeyType.ANTHROPIC

    def test_parse_multiple_keys(self):
        """Test parsing multiple keys in one message."""
        message = """
        Anthropic: sk-ant-api03-abc123def456-xyz789
        Telegram: 123456789:AABBccDDeeFfGgHhIiJjKkLlMmNnOoPpQqRr
        """
        results = parse_config_from_message(message)

        assert len(results) == 2

    def test_parse_no_keys(self):
        """Test parsing message with no keys."""
        message = "Hello, how are you?"
        results = parse_config_from_message(message)

        assert len(results) == 0


class TestChatConfigIntegration:
    """Tests for chat-based configuration."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.fetchone = AsyncMock(return_value=None)
        db.fetchall = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def mock_claude(self):
        """Create mock Claude client."""
        claude = MagicMock()
        claude.send_message = AsyncMock()
        return claude

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    async def session_manager(self, mock_db, mock_claude, temp_dir):
        """Create SessionManager with mocks."""
        from gru.session import SessionManager

        manager = SessionManager(
            db=mock_db,
            claude=mock_claude,
            data_dir=temp_dir,
        )
        await manager.initialize()
        return manager

    async def test_chat_detects_api_key(self, session_manager, temp_dir):
        """Test that chat detects and configures API keys."""
        # Send a message containing an API key
        result = await session_manager.chat(
            user_id="user1",
            channel="telegram",
            message="sk-ant-api03-abc123def456-xyz789",
        )

        assert "Anthropic" in result["response"]
        assert result["quick_action"]["type"] == "config_keys"

        # Verify key was saved
        config = ConfigManager(temp_dir)
        assert config.get("anthropic-key") == "sk-ant-api03-abc123def456-xyz789"

    async def test_chat_shows_setup_status(self, session_manager):
        """Test that 'setup status' shows configuration."""
        result = await session_manager.chat(
            user_id="user1",
            channel="telegram",
            message="setup status",
        )

        assert "Setup" in result["response"] or "config" in result["response"].lower()
        assert result["quick_action"]["type"] == "config_status"

    async def test_chat_explicit_config_set(self, session_manager, temp_dir):
        """Test explicit 'config set' command."""
        result = await session_manager.chat(
            user_id="user1",
            channel="telegram",
            message="config set location San Francisco",
        )

        # Should have stored it (but as config, not memory since we didn't set up memory)
        assert result["quick_action"]["type"] == "config_set"
