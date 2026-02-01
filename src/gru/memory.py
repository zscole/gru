"""Memory system for persistent user knowledge."""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

if TYPE_CHECKING:
    from gru.db import Database

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A structured fact about the user."""

    id: str
    fact_type: str  # preference, entity, decision, relationship, context
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_agent_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_natural_language(self) -> str:
        """Convert fact to natural language."""
        return f"{self.subject} {self.predicate} {self.object}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fact_type": self.fact_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_agent_id": self.source_agent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


EXTRACTION_PROMPT = """Analyze this conversation and extract structured facts about the user.

CONVERSATION:
{conversation}

Extract facts in these categories:
1. PREFERENCES - What the user likes, dislikes, prefers (e.g., "user prefers concise answers")
2. ENTITIES - People, projects, companies, tools the user mentions (e.g., "user works on ProjectX")
3. DECISIONS - Choices the user made (e.g., "user chose PostgreSQL for the database")
4. RELATIONSHIPS - Connections between entities (e.g., "ProjectX uses React")
5. CONTEXT - Background information (e.g., "user is building a SaaS product")

Return a JSON array of facts. Each fact should have:
- fact_type: one of "preference", "entity", "decision", "relationship", "context"
- subject: the subject of the fact (often "user" for preferences)
- predicate: the relationship or action
- object: the object of the fact
- confidence: 0.0-1.0 how confident you are (1.0 = explicitly stated, 0.7 = implied)

Only extract facts that would be useful to remember for future conversations.
Skip trivial or task-specific details that won't matter later.

Return ONLY valid JSON array, no other text. Example:
[
  {{"fact_type": "preference", "subject": "user", "predicate": "prefers", "object": "TypeScript over JavaScript", "confidence": 1.0}},
  {{"fact_type": "entity", "subject": "user", "predicate": "works on", "object": "gru project", "confidence": 0.9}}
]

If no meaningful facts to extract, return: []"""


class MemoryStore:
    """Persistent memory store for user knowledge."""

    def __init__(self, db: Database, data_dir: Path) -> None:
        self.db = db
        self.data_dir = data_dir
        self._chroma_client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    async def initialize(self) -> None:
        """Initialize the memory store."""
        chroma_dir = self.data_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="gru_memory",
            metadata={"description": "User memory and conversation embeddings"},
        )
        logger.info(f"Memory store initialized at {chroma_dir}")

    async def store_fact(
        self,
        fact_type: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source_agent_id: str | None = None,
        source_conversation: str | None = None,
    ) -> str:
        """Store a fact in the database."""
        fact_id = str(uuid.uuid4())[:12]

        # Check for existing similar facts to supersede
        existing = await self._find_similar_fact(subject, predicate)

        # Insert the new fact first
        await self.db.execute(
            """
            INSERT INTO memory_facts (id, fact_type, subject, predicate, object, confidence,
                                      source_agent_id, source_conversation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fact_id, fact_type, subject, predicate, obj, confidence, source_agent_id, source_conversation),
        )
        await self.db.commit()

        # Now supersede the old fact (after new one exists)
        if existing:
            await self.db.execute(
                "UPDATE memory_facts SET active = 0, superseded_by = ? WHERE id = ?",
                (fact_id, existing["id"]),
            )
            await self.db.commit()

        # Also store in vector DB for semantic search
        if self._collection is not None:
            fact_text = f"{subject} {predicate} {obj}"
            self._collection.add(
                ids=[fact_id],
                documents=[fact_text],
                metadatas=[{"type": "fact", "fact_type": fact_type, "subject": subject}],
            )

        logger.debug(f"Stored fact: {subject} {predicate} {obj}")
        return fact_id

    async def _find_similar_fact(self, subject: str, predicate: str) -> dict[str, Any] | None:
        """Find an existing fact with the same subject and predicate."""
        return await self.db.fetchone(
            "SELECT * FROM memory_facts WHERE subject = ? AND predicate = ? AND active = 1",
            (subject, predicate),
        )

    async def get_facts(
        self,
        fact_type: str | None = None,
        subject: str | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """Retrieve facts from the database."""
        query = "SELECT * FROM memory_facts WHERE active = 1"
        params: list[Any] = []

        if fact_type:
            query += " AND fact_type = ?"
            params.append(fact_type)
        if subject:
            query += " AND subject = ?"
            params.append(subject)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.fetchall(query, tuple(params))

        # Update access counts
        fact_ids = [row["id"] for row in rows]
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            await self.db.execute(
                f"""
                UPDATE memory_facts
                SET access_count = access_count + 1, last_accessed_at = datetime('now')
                WHERE id IN ({placeholders})
                """,
                tuple(fact_ids),
            )
            await self.db.commit()

        return [
            Fact(
                id=row["id"],
                fact_type=row["fact_type"],
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                source_agent_id=row.get("source_agent_id"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
            for row in rows
        ]

    async def search_memory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memory using semantic similarity."""
        if self._collection is None:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Fetch full facts from database
        fact_ids = results["ids"][0]
        facts = []
        for fact_id in fact_ids:
            fact = await self.db.fetchone(
                "SELECT * FROM memory_facts WHERE id = ? AND active = 1",
                (fact_id,),
            )
            if fact:
                facts.append(fact)

        return facts

    async def store_conversation_embedding(
        self,
        conversation_summary: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a conversation summary embedding for later retrieval."""
        embed_id = str(uuid.uuid4())[:12]

        await self.db.execute(
            """
            INSERT INTO memory_embeddings (id, content_type, content_preview, source_agent_id, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                embed_id,
                "conversation",
                conversation_summary[:500],
                agent_id,
                json.dumps(metadata) if metadata else None,
            ),
        )
        await self.db.commit()

        if self._collection is not None:
            self._collection.add(
                ids=[embed_id],
                documents=[conversation_summary],
                metadatas=[{"type": "conversation", "agent_id": agent_id, **(metadata or {})}],
            )

        return embed_id

    async def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Get relevant context for a query, formatted for injection into system prompt."""
        # Get relevant facts via semantic search
        semantic_results = await self.search_memory(query, limit=limit)

        # Also get recent high-confidence facts
        recent_facts = await self.get_facts(limit=10)

        # Deduplicate
        seen_ids = set()
        all_facts = []
        for fact in semantic_results:
            if fact["id"] not in seen_ids:
                seen_ids.add(fact["id"])
                all_facts.append(fact)
        for fact in recent_facts:
            if fact.id not in seen_ids:
                seen_ids.add(fact.id)
                all_facts.append(fact.to_dict())

        if not all_facts:
            return ""

        # Format for system prompt
        lines = ["REMEMBERED CONTEXT (from previous conversations):"]

        # Group by type
        preferences = [f for f in all_facts if f.get("fact_type") == "preference"]
        entities = [f for f in all_facts if f.get("fact_type") == "entity"]
        decisions = [f for f in all_facts if f.get("fact_type") == "decision"]
        other = [f for f in all_facts if f.get("fact_type") not in ("preference", "entity", "decision")]

        if preferences:
            lines.append("\nUser preferences:")
            for f in preferences[:5]:
                lines.append(f"  - {f['subject']} {f['predicate']} {f['object']}")

        if entities:
            lines.append("\nKnown entities:")
            for f in entities[:5]:
                lines.append(f"  - {f['subject']} {f['predicate']} {f['object']}")

        if decisions:
            lines.append("\nPast decisions:")
            for f in decisions[:3]:
                lines.append(f"  - {f['subject']} {f['predicate']} {f['object']}")

        if other:
            lines.append("\nOther context:")
            for f in other[:3]:
                lines.append(f"  - {f['subject']} {f['predicate']} {f['object']}")

        return "\n".join(lines)

    async def extract_facts_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        agent_id: str | None,
        claude_client: Any,
    ) -> list[str]:
        """Extract and store facts from a conversation using Claude."""
        # Format conversation for extraction
        conv_text = []
        for msg in conversation[-20:]:  # Last 20 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle tool use messages
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(text_parts)
            if content:
                conv_text.append(f"{role.upper()}: {content[:500]}")

        if not conv_text:
            return []

        conversation_str = "\n".join(conv_text)
        prompt = EXTRACTION_PROMPT.format(conversation=conversation_str)

        try:
            response = await claude_client.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You extract structured facts from conversations. Return only valid JSON.",
                model="claude-3-haiku-20240307",  # Use fast model for extraction
                max_tokens=2000,
            )

            # Parse the response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            facts_data = json.loads(content)
            if not isinstance(facts_data, list):
                return []

            stored_ids = []
            for fact in facts_data:
                if not all(k in fact for k in ("fact_type", "subject", "predicate", "object")):
                    continue

                fact_id = await self.store_fact(
                    fact_type=fact["fact_type"],
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    obj=fact["object"],
                    confidence=fact.get("confidence", 0.8),
                    source_agent_id=agent_id,
                    source_conversation=conversation_str[:1000],
                )
                stored_ids.append(fact_id)

            logger.info(f"Extracted {len(stored_ids)} facts from conversation")
            return stored_ids

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    async def forget_fact(self, fact_id: str) -> bool:
        """Mark a fact as inactive (soft delete)."""
        cursor = await self.db.execute(
            "UPDATE memory_facts SET active = 0 WHERE id = ?",
            (fact_id,),
        )
        await self.db.commit()

        # Remove from vector store
        if self._collection is not None:
            with contextlib.suppress(Exception):
                self._collection.delete(ids=[fact_id])

        return cursor.rowcount > 0

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        total = await self.db.fetchone("SELECT COUNT(*) as count FROM memory_facts WHERE active = 1")
        by_type = await self.db.fetchall(
            "SELECT fact_type, COUNT(*) as count FROM memory_facts WHERE active = 1 GROUP BY fact_type"
        )
        embeddings = await self.db.fetchone("SELECT COUNT(*) as count FROM memory_embeddings")
        most_accessed = await self.db.fetchall(
            "SELECT * FROM memory_facts WHERE active = 1 ORDER BY access_count DESC LIMIT 5"
        )

        return {
            "total_facts": total["count"] if total else 0,
            "by_type": {row["fact_type"]: row["count"] for row in by_type},
            "total_embeddings": embeddings["count"] if embeddings else 0,
            "most_accessed": [f"{row['subject']} {row['predicate']} {row['object']}" for row in most_accessed],
        }

    async def process_feedback(self, feedback: str, claude_client: Any) -> list[str]:
        """Process user feedback to update memory.

        Handles corrections like:
        - "I actually prefer X" -> updates preference
        - "That's not right, I use Y" -> corrects entity
        - "Remember that I..." -> adds new fact
        """
        prompt = f"""The user provided feedback that may contain corrections or new information to remember.

FEEDBACK: {feedback}

Analyze this feedback and extract:
1. Corrections to existing knowledge (things that should replace old facts)
2. New facts to remember
3. Things to forget/remove

Return a JSON object with:
{{
  "updates": [
    {{"fact_type": "preference", "subject": "user", "predicate": "prefers", "object": "X", "confidence": 1.0}}
  ],
  "forget_patterns": ["old pattern to forget"],
  "observations": ["things noticed but not facts"]
}}

Return ONLY valid JSON."""

        try:
            response = await claude_client.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You process user feedback to update a memory system. Return only valid JSON.",
                model="claude-3-haiku-20240307",
                max_tokens=1000,
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            stored_ids = []

            # Process updates
            for fact in data.get("updates", []):
                if all(k in fact for k in ("fact_type", "subject", "predicate", "object")):
                    fact_id = await self.store_fact(
                        fact_type=fact["fact_type"],
                        subject=fact["subject"],
                        predicate=fact["predicate"],
                        obj=fact["object"],
                        confidence=fact.get("confidence", 1.0),
                    )
                    stored_ids.append(fact_id)

            # Process forget patterns
            for pattern in data.get("forget_patterns", []):
                await self._forget_by_pattern(pattern)

            logger.info(f"Processed feedback: {len(stored_ids)} updates")
            return stored_ids

        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return []

    async def _forget_by_pattern(self, pattern: str) -> int:
        """Forget facts matching a pattern."""
        pattern_like = f"%{pattern}%"
        cursor = await self.db.execute(
            """
            UPDATE memory_facts SET active = 0
            WHERE active = 1 AND (
                object LIKE ? OR predicate LIKE ? OR subject LIKE ?
            )
            """,
            (pattern_like, pattern_like, pattern_like),
        )
        await self.db.commit()
        return cursor.rowcount

    async def set_preference(self, key: str, value: str, confidence: float = 1.0) -> str:
        """Set a user preference.

        This is a convenience method for storing common preferences like:
        - location: "San Francisco, CA"
        - food: "vegetarian"
        - default_restaurant: "Joyland"

        Args:
            key: Preference key (e.g., "location", "food", "budget")
            value: Preference value
            confidence: Confidence score

        Returns:
            Fact ID
        """
        return await self.store_fact(
            fact_type="preference",
            subject="user",
            predicate=key,
            obj=value,
            confidence=confidence,
        )

    async def get_preference(self, key: str) -> str | None:
        """Get a specific preference value.

        Args:
            key: Preference key

        Returns:
            Preference value or None
        """
        fact = await self._find_similar_fact("user", key)
        return fact["object"] if fact else None

    async def get_user_profile(self) -> dict[str, Any]:
        """Build a user profile from stored facts."""
        preferences = await self.get_facts(fact_type="preference", limit=20)
        entities = await self.get_facts(fact_type="entity", limit=20)
        decisions = await self.get_facts(fact_type="decision", limit=10)

        # Build structured profile
        profile: dict[str, Any] = {
            "preferences": {},
            "tools": [],
            "projects": [],
            "people": [],
            "decisions": [],
        }

        for pref in preferences:
            key = pref.predicate.replace(" ", "_")
            profile["preferences"][key] = pref.object

        for entity in entities:
            if "tool" in entity.predicate.lower() or "uses" in entity.predicate.lower():
                profile["tools"].append(entity.object)
            elif "project" in entity.predicate.lower() or "works on" in entity.predicate.lower():
                profile["projects"].append(entity.object)
            elif "knows" in entity.predicate.lower() or "works with" in entity.predicate.lower():
                profile["people"].append(entity.object)

        for decision in decisions:
            profile["decisions"].append(f"{decision.predicate} {decision.object}")

        return profile

    async def get_personalized_context(self, task: str, limit: int = 10) -> str:
        """Get highly personalized context for a specific task.

        This goes beyond basic context injection to provide:
        - Task-relevant preferences
        - Related past decisions
        - Known entities that might be relevant
        - Suggested approaches based on history
        """
        # Semantic search for task-relevant facts
        relevant = await self.search_memory(task, limit=limit)

        # Get user preferences that might affect this task
        preferences = await self.get_facts(fact_type="preference", limit=10)

        # Get frequently accessed facts (likely important)
        important = await self.db.fetchall(
            """
            SELECT * FROM memory_facts
            WHERE active = 1 AND access_count > 0
            ORDER BY access_count DESC, confidence DESC
            LIMIT 5
            """
        )

        if not relevant and not preferences and not important:
            return ""

        lines = ["PERSONALIZED CONTEXT:"]

        # Add relevant facts from semantic search
        if relevant:
            lines.append("\nRelevant to this task:")
            for fact in relevant[:5]:
                conf = fact.get("confidence", 1.0)
                marker = "*" if conf >= 0.9 else ""
                lines.append(f"  {marker}{fact['subject']} {fact['predicate']} {fact['object']}")

        # Add preferences that might influence approach
        if preferences:
            lines.append("\nYour preferences to keep in mind:")
            for pref in preferences[:5]:
                lines.append(f"  - {pref.object}")

        # Add important facts (frequently accessed)
        important_new = [f for f in important if f["id"] not in {r.get("id") for r in relevant}]
        if important_new:
            lines.append("\nImportant context:")
            for fact in important_new[:3]:
                lines.append(f"  - {fact['subject']} {fact['predicate']} {fact['object']}")

        return "\n".join(lines)

    async def detect_observations(
        self,
        conversation: list[dict[str, Any]],
        claude_client: Any,
    ) -> list[dict[str, Any]]:
        """Detect things worth observing/following up on from a conversation.

        Returns observations like:
        - Deadlines mentioned
        - Follow-ups needed
        - Questions left unanswered
        - Commitments made
        """
        conv_text = []
        for msg in conversation[-10:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(text_parts)
            if content:
                conv_text.append(f"{role.upper()}: {content[:300]}")

        if not conv_text:
            return []

        conversation_str = "\n".join(conv_text)

        prompt = f"""Analyze this conversation for things that might need follow-up or attention.

CONVERSATION:
{conversation_str}

Look for:
1. DEADLINES - Any dates, times, or "by X" mentioned
2. FOLLOW_UPS - Things the user said they'd do or check on
3. COMMITMENTS - Things the assistant promised to do
4. QUESTIONS - Unanswered questions or uncertainties
5. OPPORTUNITIES - Suggestions that weren't acted on

Return a JSON array of observations:
[
  {{
    "category": "deadline",
    "content": "Project demo is due Friday",
    "importance": 0.9,
    "expires_hours": 72
  }}
]

Return [] if nothing notable. Return ONLY valid JSON."""

        try:
            response = await claude_client.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You detect follow-up items from conversations. Return only valid JSON array.",
                model="claude-3-haiku-20240307",
                max_tokens=1000,
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            observations = json.loads(content)
            if not isinstance(observations, list):
                return []

            logger.info(f"Detected {len(observations)} observations")
            return observations

        except Exception as e:
            logger.warning(f"Observation detection failed: {e}")
            return []

    async def get_conversation_summary(self, conversation: list[dict[str, Any]]) -> str:
        """Generate a brief summary of a conversation for storage."""
        if not conversation:
            return ""

        # Extract key points
        user_messages = [
            msg.get("content", "")[:200]
            for msg in conversation
            if msg.get("role") == "user" and isinstance(msg.get("content"), str)
        ]

        if not user_messages:
            return ""

        # Simple summary: first and last user messages
        if len(user_messages) == 1:
            return user_messages[0]

        return f"Started with: {user_messages[0][:100]}... Ended with: {user_messages[-1][:100]}"

    async def boost_fact_importance(self, fact_id: str, amount: float = 0.1) -> bool:
        """Boost a fact's importance/confidence based on positive feedback."""
        cursor = await self.db.execute(
            """
            UPDATE memory_facts
            SET confidence = MIN(1.0, confidence + ?),
                access_count = access_count + 1
            WHERE id = ? AND active = 1
            """,
            (amount, fact_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def decay_unused_facts(self, days_threshold: int = 30, decay_amount: float = 0.1) -> int:
        """Decay confidence of facts that haven't been accessed recently."""
        cursor = await self.db.execute(
            """
            UPDATE memory_facts
            SET confidence = MAX(0.1, confidence - ?)
            WHERE active = 1
              AND last_accessed_at IS NOT NULL
              AND julianday('now') - julianday(last_accessed_at) > ?
              AND confidence > 0.1
            """,
            (decay_amount, days_threshold),
        )
        await self.db.commit()
        return cursor.rowcount
