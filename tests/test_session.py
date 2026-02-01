"""Tests for the session management module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gru.session import (
    PERSONAS,
    Persona,
    Session,
    SessionManager,
    get_available_personas,
)


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.fetchone = AsyncMock(return_value=None)
    db.fetchall = AsyncMock(return_value=[])
    return db


@pytest.fixture
def mock_claude():
    """Create a mock Claude client."""
    claude = MagicMock()
    claude.send_message = AsyncMock(return_value=MagicMock(content="Hello! How can I help you today?"))
    return claude


@pytest.fixture
def mock_memory():
    """Create a mock memory store."""
    memory = MagicMock()
    memory.get_user_profile = AsyncMock(return_value={"preferences": {}})
    memory.get_personalized_context = AsyncMock(return_value="")
    return memory


@pytest.fixture
def mock_proactive():
    """Create a mock proactive engine."""
    proactive = MagicMock()
    proactive.get_observation_summary = AsyncMock(return_value="")
    proactive.add_observation = AsyncMock()
    return proactive


@pytest.fixture
def session_manager(mock_db, mock_claude, mock_memory, mock_proactive):
    """Create a session manager with mocks."""
    return SessionManager(
        db=mock_db,
        claude=mock_claude,
        memory=mock_memory,
        proactive=mock_proactive,
    )


class TestPersona:
    """Tests for the Persona dataclass."""

    def test_default_values(self):
        """Test persona default values."""
        persona = Persona(
            name="test",
            description="Test persona",
            system_prompt="You are a test assistant.",
        )
        assert persona.escalation_threshold == 0.7
        assert persona.voice_style == "concise"

    def test_to_dict(self):
        """Test persona serialization."""
        persona = Persona(
            name="test",
            description="Test persona",
            system_prompt="You are a test assistant.",
            escalation_threshold=0.5,
            voice_style="casual",
        )
        d = persona.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test persona"
        assert d["escalation_threshold"] == 0.5
        assert d["voice_style"] == "casual"


class TestBuiltInPersonas:
    """Tests for built-in personas."""

    def test_all_personas_exist(self):
        """Test that all expected personas are defined."""
        assert "general" in PERSONAS
        assert "dev" in PERSONAS
        assert "exec" in PERSONAS
        assert "casual" in PERSONAS

    def test_general_persona(self):
        """Test general persona settings."""
        p = PERSONAS["general"]
        assert p.name == "general"
        assert "helpful" in p.system_prompt.lower()
        assert p.voice_style == "concise"

    def test_dev_persona(self):
        """Test dev persona settings."""
        p = PERSONAS["dev"]
        assert p.name == "dev"
        assert "developer" in p.system_prompt.lower() or "code" in p.system_prompt.lower()
        assert p.escalation_threshold == 0.5  # More readily escalates

    def test_exec_persona(self):
        """Test exec persona settings."""
        p = PERSONAS["exec"]
        assert p.name == "exec"
        assert "executive" in p.system_prompt.lower() or "assistant" in p.system_prompt.lower()
        assert p.escalation_threshold == 0.8  # Less likely to escalate

    def test_casual_persona(self):
        """Test casual persona settings."""
        p = PERSONAS["casual"]
        assert p.name == "casual"
        assert p.escalation_threshold == 0.9  # Rarely escalates

    def test_get_available_personas(self):
        """Test getting list of available personas."""
        personas = get_available_personas()
        assert len(personas) == len(PERSONAS)
        names = [p["name"] for p in personas]
        assert "general" in names
        assert "dev" in names


class TestSession:
    """Tests for the Session dataclass."""

    def test_create_session(self):
        """Test session creation."""
        session = Session(
            id="test123",
            user_id="user1",
            channel="cli",
        )
        assert session.id == "test123"
        assert session.user_id == "user1"
        assert session.channel == "cli"
        assert session.persona == "general"
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_active, datetime)

    def test_add_message(self):
        """Test adding messages to session."""
        session = Session(id="test", user_id="user", channel="cli")
        initial_time = session.last_active

        session.add_message("user", "Hello")

        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"
        assert "timestamp" in session.messages[0]
        assert session.last_active >= initial_time

    def test_get_recent_messages(self):
        """Test getting recent messages for Claude."""
        session = Session(id="test", user_id="user", channel="cli")

        for i in range(30):
            session.add_message("user" if i % 2 == 0 else "assistant", f"Message {i}")

        recent = session.get_recent_messages(limit=10)
        assert len(recent) == 10
        assert recent[0]["content"] == "Message 20"  # Last 10 messages
        assert "timestamp" not in recent[0]  # Should be stripped for Claude

    def test_clear_messages(self):
        """Test clearing session messages."""
        session = Session(id="test", user_id="user", channel="cli")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        session.clear_messages()

        assert session.messages == []


class TestSessionManager:
    """Tests for the SessionManager class."""

    async def test_initialize(self, session_manager, mock_db):
        """Test session manager initialization."""
        await session_manager.initialize()

        # Should create tables
        assert mock_db.execute.call_count >= 3
        mock_db.commit.assert_called()

    async def test_get_or_create_session_new(self, session_manager, mock_db):
        """Test creating a new session."""
        mock_db.fetchone.return_value = None

        session = await session_manager.get_or_create_session("user1", "cli")

        assert session.user_id == "user1"
        assert session.channel == "cli"
        assert session.persona == "general"

    async def test_get_or_create_session_existing(self, session_manager, mock_db):
        """Test retrieving an existing session."""
        mock_db.fetchone.return_value = {
            "id": "existing123",
            "user_id": "user1",
            "channel": "cli",
            "persona": "dev",
            "messages": "[]",
            "created_at": "2024-01-01T12:00:00",
            "last_active": "2024-01-01T13:00:00",
            "metadata": "{}",
        }

        session = await session_manager.get_or_create_session("user1", "cli")

        assert session.id == "existing123"
        assert session.persona == "dev"

    async def test_get_or_create_session_cached(self, session_manager, mock_db):
        """Test session caching."""
        mock_db.fetchone.return_value = None

        session1 = await session_manager.get_or_create_session("user1", "cli")
        session2 = await session_manager.get_or_create_session("user1", "cli")

        assert session1 is session2  # Same object

    async def test_set_user_persona(self, session_manager, mock_db):
        """Test setting user persona."""
        await session_manager.set_user_persona("user1", "dev")

        mock_db.execute.assert_called()
        mock_db.commit.assert_called()
        assert session_manager._user_personas["user1"] == "dev"

    async def test_set_user_persona_invalid(self, session_manager):
        """Test setting invalid persona."""
        with pytest.raises(ValueError):
            await session_manager.set_user_persona("user1", "invalid_persona")


class TestEscalationPatterns:
    """Tests for escalation pattern detection."""

    def test_go_build_pattern(self, session_manager):
        """Test 'go build' pattern."""
        persona = PERSONAS["general"]
        escalate, task = session_manager._check_escalation("go build me a website", persona)
        assert escalate is True
        assert task == "go build me a website"

    def test_spawn_agent_pattern(self, session_manager):
        """Test 'spawn agent' pattern."""
        persona = PERSONAS["general"]
        escalate, task = session_manager._check_escalation("spawn an agent to handle this", persona)
        assert escalate is True

    def test_autonomously_pattern(self, session_manager):
        """Test 'do it autonomously' pattern."""
        persona = PERSONAS["general"]
        escalate, task = session_manager._check_escalation("run this autonomously and get back to me", persona)
        assert escalate is True

    def test_no_escalation_simple(self, session_manager):
        """Test that simple queries don't escalate."""
        persona = PERSONAS["general"]
        escalate, task = session_manager._check_escalation("What's the weather?", persona)
        assert escalate is False
        assert task is None


class TestQuickActions:
    """Tests for quick action detection."""

    async def test_reminder_detection(self, session_manager, mock_proactive):
        """Test reminder pattern detection."""
        session = Session(id="test", user_id="user", channel="cli")

        action = await session_manager._check_quick_action("remind me to call Mom at 5pm", session)

        assert action is not None
        assert action["type"] == "reminder"
        assert "call mom" in action["content"].lower()
        mock_proactive.add_observation.assert_called()

    async def test_todo_detection(self, session_manager, mock_proactive):
        """Test todo pattern detection."""
        session = Session(id="test", user_id="user", channel="cli")

        action = await session_manager._check_quick_action("add a reminder to buy groceries", session)

        assert action is not None
        assert action["type"] == "reminder"

    async def test_calendar_query_detection(self, session_manager):
        """Test calendar query detection."""
        session = Session(id="test", user_id="user", channel="cli")

        action = await session_manager._check_quick_action("what's on my calendar today?", session)

        assert action is not None
        assert action["type"] == "calendar_query"

    async def test_no_quick_action(self, session_manager):
        """Test that regular messages don't trigger quick actions."""
        session = Session(id="test", user_id="user", channel="cli")

        action = await session_manager._check_quick_action("How are you doing?", session)

        assert action is None


class TestChat:
    """Tests for the chat method."""

    async def test_basic_chat(self, session_manager, mock_db, mock_claude):
        """Test basic chat interaction."""
        mock_db.fetchone.return_value = None

        result = await session_manager.chat("user1", "cli", "Hello there!")

        assert "response" in result
        assert result["escalate"] is False
        mock_claude.send_message.assert_called()

    async def test_chat_with_escalation(self, session_manager, mock_db):
        """Test chat that triggers escalation."""
        mock_db.fetchone.return_value = None

        result = await session_manager.chat("user1", "cli", "go build me a REST API for user management")

        assert result["escalate"] is True
        assert result["escalate_task"] is not None
        assert "I'll work on that" in result["response"]

    async def test_chat_with_quick_action(self, session_manager, mock_db, mock_proactive):
        """Test chat that triggers quick action."""
        mock_db.fetchone.return_value = None

        result = await session_manager.chat("user1", "cli", "remind me to submit the report")

        assert result["quick_action"] is not None
        assert result["escalate"] is False

    async def test_chat_saves_messages(self, session_manager, mock_db, mock_claude):
        """Test that chat saves messages to session."""
        mock_db.fetchone.return_value = None

        await session_manager.chat("user1", "cli", "Hello!")

        # Get the session
        session = await session_manager.get_or_create_session("user1", "cli")

        # Should have user message and assistant response
        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"
        assert session.messages[1]["role"] == "assistant"


class TestResponseEscalation:
    """Tests for response-based escalation detection."""

    def test_escalation_phrases(self, session_manager):
        """Test detection of escalation phrases in responses."""
        response = "I'll work on that for you and let you know when it's done."
        result = session_manager._check_response_escalation(response)
        assert result is not None

    def test_no_escalation_in_response(self, session_manager):
        """Test that normal responses don't trigger escalation."""
        response = "The weather in San Francisco is currently sunny."
        result = session_manager._check_response_escalation(response)
        assert result is None


class TestSystemPromptBuilding:
    """Tests for system prompt construction."""

    async def test_basic_system_prompt(self, session_manager, mock_memory, mock_proactive):
        """Test basic system prompt building."""
        session = Session(id="test", user_id="user", channel="cli")
        persona = PERSONAS["general"]

        prompt = await session_manager._build_system_prompt(session, persona)

        assert persona.system_prompt in prompt
        assert "Current time:" in prompt

    async def test_system_prompt_with_memory(self, session_manager, mock_memory, mock_proactive):
        """Test system prompt with memory context."""
        mock_memory.get_user_profile.return_value = {"preferences": {"language": "Python", "editor": "vim"}}
        mock_memory.get_personalized_context.return_value = "User prefers Python for new projects."

        session = Session(id="test", user_id="user", channel="cli")
        session.add_message("user", "What should I use?")
        persona = PERSONAS["general"]

        prompt = await session_manager._build_system_prompt(session, persona)

        assert "Python" in prompt or "preferences" in prompt.lower()

    async def test_system_prompt_with_observations(self, session_manager, mock_memory, mock_proactive):
        """Test system prompt with observation summary."""
        mock_proactive.get_observation_summary.return_value = "Pending: Meeting with team at 3pm"

        session = Session(id="test", user_id="user", channel="cli")
        persona = PERSONAS["general"]

        prompt = await session_manager._build_system_prompt(session, persona)

        assert "Meeting with team" in prompt


class TestSessionReset:
    """Tests for session reset functionality."""

    async def test_reset_session(self, session_manager, mock_db):
        """Test resetting a session."""
        mock_db.fetchone.return_value = None

        # Create session with messages
        session = await session_manager.get_or_create_session("user1", "cli")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")

        # Reset
        await session_manager.reset_session("user1", "cli")

        # Get session again
        session = await session_manager.get_or_create_session("user1", "cli")
        assert session.messages == []


class TestSessionStats:
    """Tests for session statistics."""

    async def test_get_session_stats(self, session_manager, mock_db):
        """Test getting session statistics."""
        mock_db.fetchall.return_value = [
            {
                "channel": "cli",
                "persona": "general",
                "created_at": "2024-01-01T12:00:00",
                "last_active": "2024-01-01T13:00:00",
                "msg_count": 10,
            },
            {
                "channel": "telegram",
                "persona": "dev",
                "created_at": "2024-01-02T12:00:00",
                "last_active": "2024-01-02T14:00:00",
                "msg_count": 25,
            },
        ]

        stats = await session_manager.get_session_stats("user1")

        assert stats["total_sessions"] == 2
        assert len(stats["sessions"]) == 2
