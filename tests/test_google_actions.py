"""Tests for Google Docs and Gmail actions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gru.actions.base import ActionContext, ActionStatus
from gru.actions.services.google import (
    CompileDocumentAction,
    CreateDocumentAction,
    SendEmailAction,
    WriteDocumentAction,
    set_google_connector,
)


@pytest.fixture
def mock_connector():
    """Create a mock Google connector."""
    connector = MagicMock()
    connector.is_authenticated.return_value = True
    connector.create_document = AsyncMock(
        return_value={
            "document_id": "doc123",
            "url": "https://docs.google.com/document/d/doc123/edit",
            "title": "Test Document",
        }
    )
    connector.write_to_document = AsyncMock()
    connector.send_email = AsyncMock(
        return_value={
            "message_id": "msg123",
        }
    )
    return connector


@pytest.fixture
def mock_browser():
    """Create a mock browser."""
    return MagicMock()


@pytest.fixture
def action_context(mock_browser):
    """Create a test action context."""
    return ActionContext(
        browser=mock_browser,
        user_id="test_user",
    )


@pytest.fixture(autouse=True)
def setup_connector(mock_connector):
    """Set up the mock connector for all tests."""
    set_google_connector(mock_connector)
    yield
    set_google_connector(None)


class TestCreateDocumentAction:
    """Tests for CreateDocumentAction."""

    async def test_create_document_success(self, action_context, mock_connector):
        """Test successful document creation."""
        action = CreateDocumentAction()
        result = await action.execute(
            action_context,
            title="Test Document",
            content="Hello, world!",
        )

        assert result.status == ActionStatus.COMPLETED
        assert "doc123" in result.data["document_id"]
        mock_connector.create_document.assert_called_once_with("Test Document", "Hello, world!")

    async def test_create_document_missing_title(self, action_context):
        """Test validation fails without title."""
        action = CreateDocumentAction()
        valid, error = await action.validate_params(content="some content")

        assert valid is False
        assert "title" in error.lower()

    async def test_create_document_no_connector(self, action_context):
        """Test error when connector not configured."""
        set_google_connector(None)
        action = CreateDocumentAction()
        result = await action.execute(action_context, title="Test")

        assert result.status == ActionStatus.FAILED
        assert "not configured" in result.message.lower()

    async def test_create_document_not_authenticated(self, action_context, mock_connector):
        """Test auth required when not authenticated."""
        mock_connector.is_authenticated.return_value = False
        action = CreateDocumentAction()
        result = await action.execute(action_context, title="Test")

        assert result.status == ActionStatus.NEEDS_AUTH


class TestWriteDocumentAction:
    """Tests for WriteDocumentAction."""

    async def test_write_document_success(self, action_context, mock_connector):
        """Test successful document write."""
        action = WriteDocumentAction()
        result = await action.execute(
            action_context,
            document_id="doc123",
            content="New content",
        )

        assert result.status == ActionStatus.COMPLETED
        assert result.data["chars_written"] == len("New content")
        mock_connector.write_to_document.assert_called_once()

    async def test_write_document_missing_params(self, action_context):
        """Test validation fails without required params."""
        action = WriteDocumentAction()

        valid, error = await action.validate_params(document_id="doc123")
        assert valid is False
        assert "content" in error.lower()

        valid, error = await action.validate_params(content="test")
        assert valid is False
        assert "document_id" in error.lower()


class TestSendEmailAction:
    """Tests for SendEmailAction."""

    async def test_send_email_success(self, action_context, mock_connector):
        """Test successful email send."""
        action = SendEmailAction()
        result = await action.execute(
            action_context,
            to="test@example.com",
            subject="Test Subject",
            body="Test body",
        )

        assert result.status == ActionStatus.COMPLETED
        assert result.data["to"] == "test@example.com"
        mock_connector.send_email.assert_called_once_with("test@example.com", "Test Subject", "Test body", False)

    async def test_send_email_requires_confirmation(self):
        """Test that send email requires confirmation."""
        action = SendEmailAction()
        assert action.requires_confirmation is True

    async def test_send_email_missing_params(self, action_context):
        """Test validation fails without required params."""
        action = SendEmailAction()

        valid, error = await action.validate_params(subject="Test", body="Body")
        assert valid is False
        assert "to" in error.lower()

        valid, error = await action.validate_params(to="test@example.com", body="Body")
        assert valid is False
        assert "subject" in error.lower()

        valid, error = await action.validate_params(to="test@example.com", subject="Test")
        assert valid is False
        assert "body" in error.lower()


class TestCompileDocumentAction:
    """Tests for CompileDocumentAction."""

    async def test_compile_document_with_content(self, action_context, mock_connector):
        """Test compiling document with direct content."""
        action = CompileDocumentAction()
        result = await action.execute(
            action_context,
            title="PRD: New Feature",
            content="# Requirements\n\n1. Feature A\n2. Feature B",
        )

        assert result.status == ActionStatus.COMPLETED
        assert "doc123" in result.data["document_id"]

    async def test_compile_document_with_conversation(self, action_context, mock_connector):
        """Test compiling document from conversation."""
        action = CompileDocumentAction()
        conversation = [
            {"role": "user", "content": "We need a login feature"},
            {"role": "assistant", "content": "I'll add that to the requirements"},
        ]

        result = await action.execute(
            action_context,
            title="Meeting Notes",
            conversation=conversation,
            doc_type="meeting_notes",
        )

        assert result.status == ActionStatus.COMPLETED

    async def test_compile_document_with_email(self, action_context, mock_connector):
        """Test compiling document and sending email."""
        action = CompileDocumentAction()
        result = await action.execute(
            action_context,
            title="PRD: New Feature",
            content="# Requirements",
            email_to="partner@example.com",
        )

        assert result.status == ActionStatus.COMPLETED
        assert result.data["email_sent_to"] == "partner@example.com"
        mock_connector.send_email.assert_called_once()

    async def test_compile_document_missing_title(self, action_context):
        """Test validation fails without title."""
        action = CompileDocumentAction()
        valid, error = await action.validate_params(content="test")

        assert valid is False
        assert "title" in error.lower()

    async def test_compile_document_missing_content(self, action_context):
        """Test validation fails without content or conversation."""
        action = CompileDocumentAction()
        valid, error = await action.validate_params(title="Test")

        assert valid is False
        assert "conversation" in error.lower() or "content" in error.lower()

    def test_format_conversation_prd(self):
        """Test PRD formatting."""
        action = CompileDocumentAction()
        conversation = [
            {"role": "user", "content": "We need user auth"},
        ]

        result = action._format_conversation(conversation, "prd")

        assert "PRODUCT REQUIREMENTS DOCUMENT" in result
        assert "We need user auth" in result

    def test_format_conversation_meeting_notes(self):
        """Test meeting notes formatting."""
        action = CompileDocumentAction()
        conversation = [
            {"role": "user", "content": "Action item: review PR"},
        ]

        result = action._format_conversation(conversation, "meeting_notes")

        assert "MEETING NOTES" in result
        assert "Action item: review PR" in result


class TestIntentPatterns:
    """Test intent patterns for document/email actions."""

    @pytest.fixture
    def mock_claude(self):
        """Create mock Claude client."""
        claude = MagicMock()
        claude.send_message = AsyncMock()
        return claude

    @pytest.fixture
    def classifier(self, mock_claude):
        """Create classifier with mocks."""
        from gru.intent import IntentClassifier

        return IntentClassifier(claude=mock_claude)

    async def test_classify_create_prd(self, classifier):
        """Test classifying PRD creation intent."""
        intent = await classifier.classify("write a PRD")

        assert intent.category == "document"
        assert intent.action == "compile_document"
        assert intent.requires_action is True

    async def test_classify_create_doc(self, classifier):
        """Test classifying document creation intent."""
        intent = await classifier.classify("write this to a google doc")

        assert intent.category == "document"
        assert intent.action == "compile_document"

    async def test_classify_email_link(self, classifier):
        """Test classifying email intent."""
        intent = await classifier.classify("email the link to partner@example.com")

        assert intent.category == "email"
        assert intent.action == "send_email"
        assert "partner@example.com" in intent.parameters.get("to", "")

    async def test_classify_meeting_notes(self, classifier):
        """Test classifying meeting notes intent."""
        intent = await classifier.classify("create meeting notes")

        assert intent.category == "document"
        assert intent.action == "compile_document"
