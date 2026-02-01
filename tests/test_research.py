"""Tests for research action and intent patterns."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gru.intent import IntentClassifier, parse_time_expression


class TestResearchIntentPatterns:
    """Tests for research intent detection."""

    @pytest.fixture
    def mock_claude(self):
        """Create mock Claude client."""
        claude = MagicMock()
        claude.send_message = AsyncMock()
        return claude

    @pytest.fixture
    def classifier(self, mock_claude):
        """Create classifier with mocks."""
        return IntentClassifier(claude=mock_claude)

    async def test_classify_research_topic(self, classifier):
        """Test classifying research request."""
        intent = await classifier.classify("research the current state of AI assistants")

        assert intent.category == "research"
        assert intent.action == "research"
        assert intent.requires_action is True

    async def test_classify_investigate(self, classifier):
        """Test 'investigate' keyword."""
        intent = await classifier.classify("investigate consensus algorithm designs")

        assert intent.category == "research"
        assert intent.action == "research"

    async def test_classify_analyze(self, classifier):
        """Test 'analyze' keyword."""
        intent = await classifier.classify("analyze the smart home market")

        assert intent.category == "research"
        assert intent.action == "research"

    async def test_classify_report_request(self, classifier):
        """Test explicit report request."""
        intent = await classifier.classify("write a thorough report on electric vehicles")

        assert intent.category == "research"
        assert intent.action == "research"

    async def test_classify_do_research(self, classifier):
        """Test 'do research' phrasing."""
        intent = await classifier.classify("do some research on power generators")

        assert intent.category == "research"
        assert intent.action == "research"

    async def test_classify_what_is_best(self, classifier):
        """Test 'what is the best' pattern."""
        intent = await classifier.classify("what is the best whole home generator for a 5000sqft house")

        assert intent.category == "research"
        assert intent.action == "research"

    async def test_classify_scheduled_research(self, classifier):
        """Test 'by morning' scheduled research."""
        intent = await classifier.classify("by morning, i'd like a thorough report on consumer AI products")

        assert intent.category == "research_scheduled"
        assert intent.action == "research"
        assert intent.schedule_for is not None

    async def test_classify_quick_question(self, classifier):
        """Test quick question detection."""
        intent = await classifier.classify("what is the capital of France?")

        assert intent.category == "question"
        assert intent.action == "quick_answer"


class TestTimeOfDayParsing:
    """Tests for time of day parsing."""

    def test_parse_morning(self):
        """Test parsing 'morning'."""
        ref = datetime(2024, 1, 15, 22, 0, 0)  # 10pm
        result = parse_time_expression("morning", ref)

        assert result is not None
        assert result.hour == 7 or result.hour == 8
        assert result.day == 16  # Next day

    def test_parse_tonight(self):
        """Test parsing 'tonight'."""
        ref = datetime(2024, 1, 15, 14, 0, 0)  # 2pm
        result = parse_time_expression("tonight", ref)

        assert result is not None
        assert result.hour >= 20

    def test_parse_tomorrow(self):
        """Test parsing 'tomorrow'."""
        ref = datetime(2024, 1, 15, 22, 0, 0)
        result = parse_time_expression("tomorrow", ref)

        assert result is not None
        assert result.day == 16

    def test_parse_by_morning(self):
        """Test parsing 'by morning'."""
        ref = datetime(2024, 1, 15, 23, 0, 0)  # 11pm
        result = parse_time_expression("by morning", ref)

        assert result is not None
        assert result.day == 16
        assert result.hour == 7


class TestResearchAction:
    """Tests for ResearchAction execution."""

    @pytest.fixture
    def mock_context(self):
        """Create mock action context."""
        context = MagicMock()
        context.user_id = "test_user"
        context.notify_callback = AsyncMock()
        return context

    @pytest.fixture
    def mock_claude(self):
        """Create mock Claude for research."""
        claude = MagicMock()
        claude.send_message = AsyncMock(return_value=MagicMock(content="Query 1\nQuery 2\nQuery 3"))
        return claude

    async def test_research_validates_topic(self):
        """Test that research requires a topic."""
        from gru.actions.services.research import ResearchAction

        action = ResearchAction()
        valid, error = await action.validate_params()

        assert valid is False
        assert "topic" in error.lower() or "query" in error.lower()

    async def test_research_accepts_topic(self):
        """Test that research accepts topic parameter."""
        from gru.actions.services.research import ResearchAction

        action = ResearchAction()
        valid, error = await action.validate_params(topic="AI assistants")

        assert valid is True

    async def test_quick_answer_validates_question(self):
        """Test that quick_answer requires a question."""
        from gru.actions.services.research import QuickAnswerAction

        action = QuickAnswerAction()
        valid, error = await action.validate_params()

        assert valid is False
        assert "question" in error.lower()

    async def test_quick_answer_accepts_question(self):
        """Test that quick_answer accepts question parameter."""
        from gru.actions.services.research import QuickAnswerAction

        action = QuickAnswerAction()
        valid, error = await action.validate_params(question="What is Python?")

        assert valid is True


class TestResearchIntegration:
    """Integration tests for research flow."""

    @pytest.fixture
    def mock_claude(self):
        """Create mock Claude client."""
        claude = MagicMock()
        # Mock query generation
        claude.send_message = AsyncMock(
            side_effect=[
                MagicMock(content="AI assistants 2024\nBest AI tools\nAI comparison"),
                MagicMock(content="# Research Report\n\nThis is the report content."),
            ]
        )
        return claude

    async def test_research_generates_queries(self, mock_claude):
        """Test that research generates search queries."""
        from gru.actions.services.research import ResearchAction, set_research_claude

        set_research_claude(mock_claude)
        action = ResearchAction()

        queries = await action._generate_search_queries(mock_claude, "AI assistants", "moderate")

        assert len(queries) > 0
        assert "AI assistants" in queries[0] or any("AI" in q for q in queries)

    async def test_research_synthesizes_report(self, mock_claude):
        """Test that research synthesizes sources into report."""
        from gru.actions.services.research import ResearchAction, set_research_claude

        set_research_claude(mock_claude)
        action = ResearchAction()

        sources = [
            {"title": "AI Guide", "url": "https://example.com/ai", "snippet": "AI is amazing"},
            {"title": "Top Tools", "url": "https://example.com/tools", "snippet": "Best tools list"},
        ]

        report = await action._synthesize_report(mock_claude, "AI assistants", sources, "report", "moderate")

        assert "Research Report" in report
        assert len(report) > 100
