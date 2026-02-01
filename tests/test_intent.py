"""Tests for the intent classification system."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gru.intent import (
    Intent,
    IntentClassifier,
    parse_time_expression,
)


class TestParseTimeExpression:
    """Tests for time expression parsing."""

    def test_in_minutes(self):
        """Test 'in X minutes' parsing."""
        ref = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_time_expression("in 30 minutes", ref)
        assert result == datetime(2024, 1, 15, 12, 30, 0)

    def test_in_hours(self):
        """Test 'in X hours' parsing."""
        ref = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_time_expression("in 2 hours", ref)
        assert result == datetime(2024, 1, 15, 14, 0, 0)

    def test_at_time_pm(self):
        """Test 'at X pm' parsing."""
        ref = datetime(2024, 1, 15, 10, 0, 0)
        result = parse_time_expression("at 2 pm", ref)
        assert result.hour == 14
        assert result.minute == 0

    def test_at_time_am(self):
        """Test 'at X am' parsing."""
        ref = datetime(2024, 1, 15, 6, 0, 0)
        result = parse_time_expression("at 8 am", ref)
        assert result.hour == 8

    def test_at_time_with_minutes(self):
        """Test 'at X:XX' parsing."""
        ref = datetime(2024, 1, 15, 10, 0, 0)
        result = parse_time_expression("at 2:30 pm", ref)
        assert result.hour == 14
        assert result.minute == 30

    def test_lunch_time(self):
        """Test 'lunch' parsing."""
        ref = datetime(2024, 1, 15, 9, 0, 0)
        result = parse_time_expression("for lunch", ref)
        # Should be middle of lunch window (11-13, so 12)
        assert result.hour == 12

    def test_dinner_time(self):
        """Test 'dinner' parsing."""
        ref = datetime(2024, 1, 15, 12, 0, 0)
        result = parse_time_expression("dinner", ref)
        # Should be middle of dinner window (18-20, so 19)
        assert result.hour == 19

    def test_time_wraps_to_next_day(self):
        """Test that past times wrap to next day."""
        ref = datetime(2024, 1, 15, 15, 0, 0)  # 3pm
        result = parse_time_expression("at 10 am", ref)
        assert result.day == 16  # Next day

    def test_empty_expression(self):
        """Test empty expression returns None."""
        assert parse_time_expression("") is None
        assert parse_time_expression(None) is None


class TestIntent:
    """Tests for Intent dataclass."""

    def test_intent_creation(self):
        """Test creating an intent."""
        intent = Intent(
            category="food_order",
            action="ubereats_order",
            parameters={"item": "burger"},
            confidence=0.9,
            requires_action=True,
            original_text="order me a burger",
        )
        assert intent.category == "food_order"
        assert intent.action == "ubereats_order"
        assert intent.parameters["item"] == "burger"
        assert intent.requires_action is True


class TestIntentClassifier:
    """Tests for IntentClassifier."""

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

    async def test_classify_order_for_lunch(self, classifier):
        """Test classifying food order intent."""
        intent = await classifier.classify("order me a burger for lunch")

        assert intent.category == "food_order"
        assert intent.action == "ubereats_order"
        assert "burger" in intent.parameters.get("item", "").lower()
        assert intent.requires_action is True
        assert intent.schedule_for is not None

    async def test_classify_order_from_restaurant(self, classifier):
        """Test classifying order from specific restaurant."""
        intent = await classifier.classify("order me a pizza from Dominos")

        assert intent.category == "food_order"
        assert "pizza" in intent.parameters.get("item", "").lower()
        assert "dominos" in intent.parameters.get("restaurant", "").lower()

    async def test_classify_food_search(self, classifier):
        """Test classifying food search intent."""
        intent = await classifier.classify("find me the best sushi nearby")

        assert intent.category == "food_search"
        assert intent.action == "restaurant_search"
        assert "sushi" in intent.parameters.get("query", "").lower()

    async def test_classify_food_search_with_location(self, classifier):
        """Test classifying food search with location."""
        intent = await classifier.classify("find me tacos near downtown Austin")

        assert intent.category == "food_search"
        assert "tacos" in intent.parameters.get("query", "").lower()
        assert "austin" in intent.parameters.get("location", "").lower()

    async def test_classify_calendar_query(self, classifier):
        """Test classifying calendar intent."""
        intent = await classifier.classify("what's on my calendar today")

        assert intent.category == "calendar"
        assert intent.action == "check_calendar"

    async def test_classify_reminder(self, classifier):
        """Test classifying reminder intent."""
        intent = await classifier.classify("remind me to call mom at 5pm")

        assert intent.category == "reminder"
        assert intent.action == "add_reminder"
        assert "call mom" in intent.parameters.get("content", "").lower()

    async def test_classify_i_would_like(self, classifier):
        """Test 'I would like' pattern."""
        intent = await classifier.classify("I'd like a cheeseburger for lunch")

        assert intent.category == "food_order"
        assert "cheeseburger" in intent.parameters.get("item", "").lower()

    async def test_classify_general_chat(self, classifier, mock_claude):
        """Test that general chat falls back to LLM classification."""
        mock_claude.send_message.return_value = MagicMock(
            content='{"category": "general", "action": null, "parameters": {}, "requires_action": false}'
        )

        intent = await classifier.classify("Hello there, nice to chat with you")

        assert intent.category == "general"
        assert intent.requires_action is False


class TestIntentEnrichment:
    """Tests for intent enrichment with preferences and calendar."""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory store."""
        memory = MagicMock()
        memory.get_user_profile = AsyncMock(
            return_value={
                "preferences": {
                    "location": "San Francisco, CA",
                    "food": "spicy",
                }
            }
        )
        return memory

    @pytest.fixture
    def classifier_with_memory(self, mock_memory):
        """Create classifier with memory."""
        claude = MagicMock()
        return IntentClassifier(claude=claude, memory=mock_memory)

    async def test_enrich_adds_location(self, classifier_with_memory):
        """Test that enrichment adds location from preferences."""
        intent = Intent(
            category="food_order",
            action="ubereats_order",
            parameters={"item": "burger"},
            requires_action=True,
            original_text="order me a burger",
        )

        enriched = await classifier_with_memory.enrich_intent(intent, "user1")

        assert enriched.parameters.get("location") == "San Francisco, CA"

    async def test_enrich_adds_food_preferences(self, classifier_with_memory):
        """Test that enrichment adds food preferences."""
        intent = Intent(
            category="food_search",
            action="restaurant_search",
            parameters={"query": "tacos"},
            requires_action=True,
            original_text="find me tacos",
        )

        enriched = await classifier_with_memory.enrich_intent(intent, "user1")

        assert enriched.parameters.get("preferences") == "spicy"

    async def test_calendar_conflict_rescheduling(self, classifier_with_memory):
        """Test that calendar conflicts cause rescheduling."""
        target_time = datetime(2024, 1, 15, 12, 30, 0)

        intent = Intent(
            category="food_order",
            action="ubereats_order",
            parameters={"item": "burger"},
            requires_action=True,
            schedule_for=target_time,
            original_text="order burger for lunch",
        )

        # Calendar event from 12:00-13:00
        calendar_events = [
            {
                "summary": "Team Meeting",
                "start_time": "2024-01-15T12:00:00",
                "end_time": "2024-01-15T13:00:00",
            }
        ]

        enriched = await classifier_with_memory.enrich_intent(intent, "user1", calendar_events)

        # Should be rescheduled to after the meeting
        assert enriched.schedule_for > target_time
        assert "rescheduled_reason" in enriched.parameters

    async def test_no_enrichment_for_non_action(self, classifier_with_memory):
        """Test that non-action intents aren't enriched."""
        intent = Intent(
            category="general",
            action=None,
            requires_action=False,
            original_text="hello",
        )

        enriched = await classifier_with_memory.enrich_intent(intent, "user1")

        assert "location" not in enriched.parameters
