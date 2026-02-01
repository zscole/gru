"""Tests for the proactive engine."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gru.db import Database
from gru.proactive import (
    Observation,
    ProactiveEngine,
    Trigger,
    TriggerType,
    setup_default_triggers,
)


@pytest.fixture
async def db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        await db.connect()
        yield db
        await db.close()


@pytest.fixture
def mock_config():
    """Create a mock config."""
    class MockConfig:
        data_dir = Path(tempfile.mkdtemp())
        memory_enabled = True

    return MockConfig()


@pytest.fixture
async def proactive_engine(db, mock_config):
    """Create a proactive engine with a temporary database."""
    engine = ProactiveEngine(mock_config, db, memory=None)
    await engine.initialize()
    yield engine
    await engine.stop()


class TestTrigger:
    """Tests for the Trigger class."""

    def test_interval_trigger_should_fire_first_time(self):
        """Test that interval trigger fires on first check."""
        trigger = Trigger(
            id="test1",
            name="test",
            trigger_type=TriggerType.INTERVAL,
            config={},
            action="notify:test",
            interval_minutes=5,
        )
        assert trigger.should_fire(datetime.now(), {}) is True

    def test_interval_trigger_respects_interval(self):
        """Test that interval trigger respects the interval."""
        trigger = Trigger(
            id="test1",
            name="test",
            trigger_type=TriggerType.INTERVAL,
            config={},
            action="notify:test",
            interval_minutes=5,
            last_fired=datetime.now(),
        )
        assert trigger.should_fire(datetime.now(), {}) is False

    def test_interval_trigger_fires_after_interval(self):
        """Test that interval trigger fires after interval passes."""
        trigger = Trigger(
            id="test1",
            name="test",
            trigger_type=TriggerType.INTERVAL,
            config={},
            action="notify:test",
            interval_minutes=5,
            last_fired=datetime.now() - timedelta(minutes=6),
        )
        assert trigger.should_fire(datetime.now(), {}) is True

    def test_disabled_trigger_does_not_fire(self):
        """Test that disabled triggers don't fire."""
        trigger = Trigger(
            id="test1",
            name="test",
            trigger_type=TriggerType.INTERVAL,
            config={},
            action="notify:test",
            interval_minutes=5,
            enabled=False,
        )
        assert trigger.should_fire(datetime.now(), {}) is False

    def test_condition_trigger_evaluates(self):
        """Test that condition triggers evaluate their condition."""
        trigger = Trigger(
            id="test1",
            name="test",
            trigger_type=TriggerType.CONDITION,
            config={},
            action="notify:test",
            condition="hour >= 9",
        )
        context = {"hour": 10}
        assert trigger.should_fire(datetime.now(), context) is True

        context = {"hour": 8}
        assert trigger.should_fire(datetime.now(), context) is False


class TestObservation:
    """Tests for the Observation class."""

    def test_observation_not_expired(self):
        """Test observation that hasn't expired."""
        obs = Observation(
            id="test1",
            content="Test observation",
            category="reminder",
            importance=0.8,
            source="test",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert obs.is_expired() is False

    def test_observation_expired(self):
        """Test observation that has expired."""
        obs = Observation(
            id="test1",
            content="Test observation",
            category="reminder",
            importance=0.8,
            source="test",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert obs.is_expired() is True

    def test_observation_no_expiry(self):
        """Test observation with no expiry."""
        obs = Observation(
            id="test1",
            content="Test observation",
            category="reminder",
            importance=0.8,
            source="test",
            created_at=datetime.now(),
            expires_at=None,
        )
        assert obs.is_expired() is False

    def test_observation_to_message(self):
        """Test observation message formatting."""
        obs = Observation(
            id="test1",
            content="Check the build",
            category="follow_up",
            importance=0.8,
            source="test",
            created_at=datetime.now(),
        )
        assert "Follow-up needed" in obs.to_message()
        assert "Check the build" in obs.to_message()


class TestProactiveEngine:
    """Tests for the ProactiveEngine class."""

    async def test_add_trigger(self, proactive_engine):
        """Test adding a trigger."""
        trigger_id = await proactive_engine.add_trigger(
            name="test_trigger",
            trigger_type=TriggerType.INTERVAL,
            action="notify:Test message",
            interval_minutes=30,
        )
        assert trigger_id is not None

        triggers = await proactive_engine.list_triggers()
        assert len(triggers) == 1
        assert triggers[0]["name"] == "test_trigger"

    async def test_remove_trigger(self, proactive_engine):
        """Test removing a trigger."""
        trigger_id = await proactive_engine.add_trigger(
            name="test_trigger",
            trigger_type=TriggerType.INTERVAL,
            action="notify:Test",
            interval_minutes=30,
        )

        success = await proactive_engine.remove_trigger(trigger_id)
        assert success is True

        triggers = await proactive_engine.list_triggers()
        assert len(triggers) == 0

    async def test_remove_nonexistent_trigger(self, proactive_engine):
        """Test removing a trigger that doesn't exist."""
        success = await proactive_engine.remove_trigger("nonexistent")
        assert success is False

    async def test_add_observation(self, proactive_engine):
        """Test adding an observation."""
        obs_id = await proactive_engine.add_observation(
            content="Remember to follow up",
            category="follow_up",
            importance=0.7,
            source="test",
        )
        assert obs_id is not None

        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 1
        assert pending[0].content == "Remember to follow up"

    async def test_mark_observation_handled(self, proactive_engine):
        """Test marking an observation as handled."""
        obs_id = await proactive_engine.add_observation(
            content="Test observation",
            category="note",
            importance=0.5,
            source="test",
        )

        success = await proactive_engine.mark_observation_handled(obs_id)
        assert success is True

        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 0

    async def test_observation_summary(self, proactive_engine):
        """Test getting observation summary."""
        await proactive_engine.add_observation(
            content="High priority item",
            category="deadline",
            importance=0.9,
            source="test",
        )
        await proactive_engine.add_observation(
            content="Low priority item",
            category="note",
            importance=0.3,
            source="test",
        )

        summary = await proactive_engine.get_observation_summary()
        assert "PENDING OBSERVATIONS" in summary
        assert "High priority item" in summary

    async def test_observation_sorted_by_importance(self, proactive_engine):
        """Test that observations are sorted by importance."""
        await proactive_engine.add_observation(
            content="Low priority",
            category="note",
            importance=0.3,
            source="test",
        )
        await proactive_engine.add_observation(
            content="High priority",
            category="deadline",
            importance=0.9,
            source="test",
        )

        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 2
        assert pending[0].content == "High priority"
        assert pending[1].content == "Low priority"


class TestSetupDefaultTriggers:
    """Tests for default trigger setup."""

    async def test_setup_creates_triggers(self, proactive_engine):
        """Test that setup creates default triggers."""
        await setup_default_triggers(proactive_engine)

        triggers = await proactive_engine.list_triggers()
        assert len(triggers) >= 1

        names = [t["name"] for t in triggers]
        assert "morning_summary" in names or "check_pending" in names

    async def test_setup_is_idempotent(self, proactive_engine):
        """Test that setup doesn't duplicate triggers."""
        await setup_default_triggers(proactive_engine)
        count1 = len(await proactive_engine.list_triggers())

        await setup_default_triggers(proactive_engine)
        count2 = len(await proactive_engine.list_triggers())

        assert count1 == count2
