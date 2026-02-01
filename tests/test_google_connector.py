"""Tests for the Google connector."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gru.connectors.google import GoogleConnector, setup_google_triggers
from gru.db import Database
from gru.proactive import ProactiveEngine


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def google_connector(temp_dir):
    """Create a Google connector with temp directory."""
    return GoogleConnector(temp_dir)


@pytest.fixture
async def db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        await db.connect()
        yield db
        await db.close()


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock config."""

    class MockConfig:
        data_dir = temp_dir
        memory_enabled = True

    return MockConfig()


@pytest.fixture
async def proactive_engine(db, mock_config):
    """Create a proactive engine."""
    engine = ProactiveEngine(mock_config, db, memory=None)
    await engine.initialize()
    yield engine
    await engine.stop()


class TestGoogleConnector:
    """Tests for the GoogleConnector class."""

    def test_not_configured_by_default(self, google_connector):
        """Test that connector is not configured by default."""
        assert google_connector.is_configured() is False

    def test_not_authenticated_by_default(self, google_connector):
        """Test that connector is not authenticated by default."""
        assert google_connector.is_authenticated() is False

    async def test_setup_credentials(self, google_connector):
        """Test setting up credentials."""
        await google_connector.setup_credentials("test-client-id", "test-client-secret")

        assert google_connector.is_configured() is True
        assert google_connector.credentials_path.exists()

        # Verify file contents
        creds = json.loads(google_connector.credentials_path.read_text())
        assert creds["installed"]["client_id"] == "test-client-id"
        assert creds["installed"]["client_secret"] == "test-client-secret"

    def test_get_status_unconfigured(self, google_connector):
        """Test status when not configured."""
        status = google_connector.get_status()

        assert status["configured"] is False
        assert status["authenticated"] is False
        assert status["last_calendar_sync"] is None
        assert status["last_email_sync"] is None

    def test_clear_seen_cache(self, google_connector):
        """Test clearing seen items cache."""
        # Add some items to the cache
        google_connector._seen_email_ids.add("email1")
        google_connector._seen_event_ids.add("event1")

        google_connector.clear_seen_cache()

        assert len(google_connector._seen_email_ids) == 0
        assert len(google_connector._seen_event_ids) == 0


class TestGoogleCalendarSync:
    """Tests for calendar sync functionality."""

    async def test_sync_calendar_without_auth(self, google_connector, proactive_engine):
        """Test that sync fails gracefully without auth."""
        events = await google_connector.sync_calendar(proactive_engine)
        assert events == []

    async def test_sync_calendar_creates_observations(self, google_connector, proactive_engine):
        """Test that syncing calendar creates observations."""
        # Mock the calendar service
        mock_service = MagicMock()
        mock_events = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Team Meeting",
                    "start": {"dateTime": (datetime.utcnow() + timedelta(hours=2)).isoformat() + "Z"},
                    "attendees": [
                        {"email": "alice@example.com"},
                        {"email": "bob@example.com"},
                    ],
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events
        google_connector._calendar_service = mock_service

        events = await google_connector.sync_calendar(proactive_engine)

        assert len(events) == 1
        assert events[0]["summary"] == "Team Meeting"

        # Check observation was created
        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 1
        assert "Team Meeting" in pending[0].content
        assert pending[0].source == "google_calendar"

    async def test_sync_calendar_skips_seen_events(self, google_connector, proactive_engine):
        """Test that already-seen events are skipped."""
        mock_service = MagicMock()
        mock_events = {
            "items": [
                {"id": "event1", "summary": "Meeting", "start": {"date": "2024-01-15"}},
            ]
        }
        mock_service.events().list().execute.return_value = mock_events
        google_connector._calendar_service = mock_service

        # First sync
        events1 = await google_connector.sync_calendar(proactive_engine)
        assert len(events1) == 1

        # Second sync - should skip the same event
        events2 = await google_connector.sync_calendar(proactive_engine)
        assert len(events2) == 0


class TestGmailSync:
    """Tests for email sync functionality."""

    async def test_sync_email_without_auth(self, google_connector, proactive_engine):
        """Test that sync fails gracefully without auth."""
        emails = await google_connector.sync_email(proactive_engine)
        assert emails == []

    async def test_sync_email_creates_observations(self, google_connector, proactive_engine):
        """Test that syncing email creates observations for important emails."""
        # Mock the Gmail service
        mock_service = MagicMock()

        # Mock list response
        mock_service.users().messages().list().execute.return_value = {"messages": [{"id": "msg1"}]}

        # Mock get response for the message
        mock_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "labelIds": ["IMPORTANT", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "Subject", "value": "Urgent: Project Update"},
                ]
            },
        }
        google_connector._gmail_service = mock_service

        emails = await google_connector.sync_email(proactive_engine)

        assert len(emails) == 1

        # Check observation was created
        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 1
        assert "Alice" in pending[0].content
        assert "Project Update" in pending[0].content
        assert pending[0].source == "gmail"

    async def test_sync_email_skips_low_importance(self, google_connector, proactive_engine):
        """Test that low-importance emails don't create observations."""
        mock_service = MagicMock()

        mock_service.users().messages().list().execute.return_value = {"messages": [{"id": "msg1"}]}

        # No IMPORTANT or STARRED label
        mock_service.users().messages().get().execute.return_value = {
            "id": "msg1",
            "labelIds": ["UNREAD"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "newsletter@spam.com"},
                    {"name": "Subject", "value": "Weekly Newsletter"},
                ]
            },
        }
        google_connector._gmail_service = mock_service

        emails = await google_connector.sync_email(proactive_engine)

        # Email was fetched but no observation created due to low importance
        assert len(emails) == 1
        pending = await proactive_engine.get_pending_observations()
        assert len(pending) == 0


class TestSetupGoogleTriggers:
    """Tests for setting up Google sync triggers."""

    async def test_creates_triggers(self, proactive_engine, google_connector):
        """Test that setup creates the expected triggers."""
        await setup_google_triggers(proactive_engine, google_connector)

        triggers = await proactive_engine.list_triggers()
        names = [t["name"] for t in triggers]

        assert "google_calendar_sync" in names
        assert "google_email_sync" in names

    async def test_triggers_are_idempotent(self, proactive_engine, google_connector):
        """Test that running setup twice doesn't duplicate triggers."""
        await setup_google_triggers(proactive_engine, google_connector)
        len(await proactive_engine.list_triggers())

        await setup_google_triggers(proactive_engine, google_connector)
        len(await proactive_engine.list_triggers())

        # Should have same number (no duplicates)
        # Note: may include default triggers too
        google_triggers = [t for t in await proactive_engine.list_triggers() if t["name"].startswith("google_")]
        assert len(google_triggers) == 2
