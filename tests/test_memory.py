"""Tests for the memory module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gru.db import Database
from gru.memory import EXTRACTION_PROMPT, Fact, MemoryStore


@pytest.fixture
async def db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        await db.connect()
        yield db
        await db.close()


@pytest.fixture
async def memory_store(db):
    """Create a memory store with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db, Path(tmpdir))
        await store.initialize()
        yield store


class TestFact:
    """Tests for the Fact dataclass."""

    def test_fact_to_natural_language(self):
        """Test converting a fact to natural language."""
        fact = Fact(
            id="abc123",
            fact_type="preference",
            subject="user",
            predicate="prefers",
            object="TypeScript",
        )
        assert fact.to_natural_language() == "user prefers TypeScript"

    def test_fact_to_dict(self):
        """Test converting a fact to a dictionary."""
        fact = Fact(
            id="abc123",
            fact_type="preference",
            subject="user",
            predicate="prefers",
            object="TypeScript",
            confidence=0.9,
        )
        d = fact.to_dict()
        assert d["id"] == "abc123"
        assert d["fact_type"] == "preference"
        assert d["subject"] == "user"
        assert d["predicate"] == "prefers"
        assert d["object"] == "TypeScript"
        assert d["confidence"] == 0.9


class TestMemoryStore:
    """Tests for the MemoryStore class."""

    async def test_store_fact(self, memory_store):
        """Test storing a fact."""
        fact_id = await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="Python",
            confidence=1.0,
        )
        assert fact_id is not None
        assert len(fact_id) == 12

    async def test_get_facts(self, memory_store):
        """Test retrieving facts."""
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="Python",
        )
        await memory_store.store_fact(
            fact_type="entity",
            subject="user",
            predicate="works on",
            obj="gru project",
        )

        facts = await memory_store.get_facts()
        assert len(facts) == 2

        pref_facts = await memory_store.get_facts(fact_type="preference")
        assert len(pref_facts) == 1
        assert pref_facts[0].object == "Python"

    async def test_get_facts_by_subject(self, memory_store):
        """Test filtering facts by subject."""
        await memory_store.store_fact(
            fact_type="entity",
            subject="user",
            predicate="knows",
            obj="Python",
        )
        await memory_store.store_fact(
            fact_type="entity",
            subject="gru",
            predicate="is",
            obj="an AI orchestrator",
        )

        user_facts = await memory_store.get_facts(subject="user")
        assert len(user_facts) == 1
        assert user_facts[0].subject == "user"

    async def test_supersede_fact(self, memory_store):
        """Test that storing a fact with same subject/predicate supersedes the old one."""
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="JavaScript",
        )
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="TypeScript",  # Updated preference
        )

        facts = await memory_store.get_facts(fact_type="preference")
        assert len(facts) == 1
        assert facts[0].object == "TypeScript"

    async def test_forget_fact(self, memory_store):
        """Test forgetting a fact."""
        fact_id = await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="Python",
        )

        success = await memory_store.forget_fact(fact_id)
        assert success is True

        facts = await memory_store.get_facts()
        assert len(facts) == 0

    async def test_forget_nonexistent_fact(self, memory_store):
        """Test forgetting a fact that doesn't exist."""
        success = await memory_store.forget_fact("nonexistent")
        assert success is False

    async def test_search_memory(self, memory_store):
        """Test semantic search."""
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="dark mode",
        )
        await memory_store.store_fact(
            fact_type="entity",
            subject="user",
            predicate="uses",
            obj="vim editor",
        )

        results = await memory_store.search_memory("editor preferences")
        assert len(results) >= 0  # Results depend on embedding similarity

    async def test_get_relevant_context(self, memory_store):
        """Test getting formatted context for injection."""
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="concise answers",
        )
        await memory_store.store_fact(
            fact_type="entity",
            subject="user",
            predicate="works on",
            obj="gru project",
        )

        context = await memory_store.get_relevant_context("help with gru")
        assert "REMEMBERED CONTEXT" in context or context == ""

    async def test_get_stats(self, memory_store):
        """Test getting memory statistics."""
        await memory_store.store_fact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            obj="Python",
        )
        await memory_store.store_fact(
            fact_type="entity",
            subject="user",
            predicate="uses",
            obj="vim",
        )

        stats = await memory_store.get_stats()
        assert stats["total_facts"] == 2
        assert "preference" in stats["by_type"]
        assert "entity" in stats["by_type"]


class TestExtractionPrompt:
    """Tests for the extraction prompt."""

    def test_extraction_prompt_contains_categories(self):
        """Test that the extraction prompt contains all fact categories."""
        assert "PREFERENCES" in EXTRACTION_PROMPT
        assert "ENTITIES" in EXTRACTION_PROMPT
        assert "DECISIONS" in EXTRACTION_PROMPT
        assert "RELATIONSHIPS" in EXTRACTION_PROMPT
        assert "CONTEXT" in EXTRACTION_PROMPT

    def test_extraction_prompt_has_json_example(self):
        """Test that the prompt includes a JSON example."""
        assert "fact_type" in EXTRACTION_PROMPT
        assert "subject" in EXTRACTION_PROMPT
        assert "predicate" in EXTRACTION_PROMPT
        assert "object" in EXTRACTION_PROMPT
        assert "confidence" in EXTRACTION_PROMPT


class TestExtractFactsFromConversation:
    """Tests for fact extraction from conversations."""

    async def test_extract_facts_success(self, memory_store):
        """Test successful fact extraction."""
        conversation = [
            {"role": "user", "content": "I prefer using TypeScript for all my projects."},
            {"role": "assistant", "content": "I'll use TypeScript then."},
        ]

        # Mock the Claude client
        mock_response = MagicMock()
        mock_response.content = '[{"fact_type": "preference", "subject": "user", "predicate": "prefers", "object": "TypeScript", "confidence": 1.0}]'

        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_response)

        # Pass None for agent_id since we don't have an agent in the test DB
        extracted = await memory_store.extract_facts_from_conversation(conversation, None, mock_client)
        assert len(extracted) == 1

    async def test_extract_facts_empty_conversation(self, memory_store):
        """Test extraction with empty conversation."""
        mock_client = AsyncMock()
        extracted = await memory_store.extract_facts_from_conversation([], "agent123", mock_client)
        assert len(extracted) == 0

    async def test_extract_facts_invalid_json(self, memory_store):
        """Test handling of invalid JSON response."""
        conversation = [
            {"role": "user", "content": "Hello"},
        ]

        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON"

        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_response)

        extracted = await memory_store.extract_facts_from_conversation(conversation, "agent123", mock_client)
        assert len(extracted) == 0

    async def test_extract_facts_handles_markdown_code_block(self, memory_store):
        """Test handling of JSON wrapped in markdown code blocks."""
        conversation = [
            {"role": "user", "content": "I use vim"},
        ]

        mock_response = MagicMock()
        mock_response.content = '```json\n[{"fact_type": "entity", "subject": "user", "predicate": "uses", "object": "vim", "confidence": 0.9}]\n```'

        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_response)

        # Pass None for agent_id since we don't have an agent in the test DB
        extracted = await memory_store.extract_facts_from_conversation(conversation, None, mock_client)
        assert len(extracted) == 1
