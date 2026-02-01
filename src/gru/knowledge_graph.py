"""Personal Knowledge Graph for understanding relationships, history, and context."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.db import Database
    from gru.memory import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A named entity in the knowledge graph."""

    id: str
    entity_type: str  # person, project, company, topic, tool, place
    name: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    first_seen: str | None = None
    last_seen: str | None = None
    mention_count: int = 0

    def matches(self, text: str) -> bool:
        """Check if text matches this entity."""
        text_lower = text.lower()
        if self.name.lower() in text_lower:
            return True
        return any(alias.lower() in text_lower for alias in self.aliases)


@dataclass
class Relationship:
    """A relationship between two entities."""

    id: str
    from_entity_id: str
    to_entity_id: str
    relationship_type: str  # knows, works_with, discussed, mentioned_with, etc.
    context: str | None = None
    strength: float = 1.0  # How strong/frequent the relationship is
    first_seen: str | None = None
    last_seen: str | None = None
    occurrence_count: int = 1


@dataclass
class Interaction:
    """A recorded interaction/discussion about entities."""

    id: str
    entity_ids: list[str]
    source: str  # conversation, email, slack, calendar
    summary: str
    timestamp: str
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """Extract named entities and their relationships from this text.

TEXT:
{text}

Extract:
1. PEOPLE - Names of individuals mentioned
2. PROJECTS - Project names, codenames, products
3. COMPANIES - Company or organization names
4. TOPICS - Technical topics, concepts, areas of discussion
5. TOOLS - Software, frameworks, technologies
6. PLACES - Locations, offices, cities

Also extract RELATIONSHIPS between entities (who knows whom, what's related to what).

Return JSON:
{{
  "entities": [
    {{"type": "person", "name": "John Smith", "aliases": ["John", "JS"], "context": "colleague"}}
  ],
  "relationships": [
    {{"from": "John Smith", "to": "Project Alpha", "type": "works_on", "context": "lead developer"}}
  ],
  "topics_discussed": ["database migration", "API design"]
}}

Return ONLY valid JSON. If nothing to extract, return {{"entities": [], "relationships": [], "topics_discussed": []}}"""


class KnowledgeGraph:
    """Personal knowledge graph for tracking relationships and history."""

    def __init__(self, db: Database, memory: MemoryStore) -> None:
        self.db = db
        self.memory = memory
        self._entity_cache: dict[str, Entity] = {}

    async def initialize(self) -> None:
        """Initialize knowledge graph tables."""
        # Create entities table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                aliases JSON,
                metadata JSON,
                first_seen TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen TEXT NOT NULL DEFAULT (datetime('now')),
                mention_count INTEGER DEFAULT 1,
                active INTEGER DEFAULT 1
            )
        """)

        # Create relationships table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS kg_relationships (
                id TEXT PRIMARY KEY,
                from_entity_id TEXT NOT NULL REFERENCES kg_entities(id),
                to_entity_id TEXT NOT NULL REFERENCES kg_entities(id),
                relationship_type TEXT NOT NULL,
                context TEXT,
                strength REAL DEFAULT 1.0,
                first_seen TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen TEXT NOT NULL DEFAULT (datetime('now')),
                occurrence_count INTEGER DEFAULT 1,
                UNIQUE(from_entity_id, to_entity_id, relationship_type)
            )
        """)

        # Create interactions table (time-indexed discussions)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS kg_interactions (
                id TEXT PRIMARY KEY,
                entity_ids JSON NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT,
                summary TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                metadata JSON
            )
        """)

        # Create indexes
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type, active)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_interactions_time ON kg_interactions(timestamp)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_interactions_source ON kg_interactions(source)"
        )

        await self.db.commit()
        logger.info("Knowledge graph initialized")

    async def add_entity(
        self,
        entity_type: str,
        name: str,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add or update an entity in the graph."""
        # Check if entity already exists
        existing = await self._find_entity(name, entity_type)
        if existing:
            # Update last_seen and mention_count
            await self.db.execute(
                """
                UPDATE kg_entities
                SET last_seen = datetime('now'),
                    mention_count = mention_count + 1,
                    aliases = COALESCE(?, aliases),
                    metadata = COALESCE(?, metadata)
                WHERE id = ?
                """,
                (
                    json.dumps(aliases) if aliases else None,
                    json.dumps(metadata) if metadata else None,
                    existing["id"],
                ),
            )
            await self.db.commit()
            return existing["id"]

        # Create new entity
        entity_id = str(uuid.uuid4())[:12]
        await self.db.execute(
            """
            INSERT INTO kg_entities (id, entity_type, name, aliases, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                entity_type,
                name,
                json.dumps(aliases or []),
                json.dumps(metadata or {}),
            ),
        )
        await self.db.commit()

        logger.debug(f"Added entity: {entity_type}/{name}")
        return entity_id

    async def _find_entity(self, name: str, entity_type: str | None = None) -> dict | None:
        """Find an entity by name or alias."""
        # Exact name match first
        if entity_type:
            row = await self.db.fetchone(
                "SELECT * FROM kg_entities WHERE name = ? AND entity_type = ? AND active = 1",
                (name, entity_type),
            )
        else:
            row = await self.db.fetchone(
                "SELECT * FROM kg_entities WHERE name = ? AND active = 1",
                (name,),
            )

        if row:
            return dict(row)

        # Search in aliases
        all_entities = await self.db.fetchall(
            "SELECT * FROM kg_entities WHERE active = 1"
        )
        name_lower = name.lower()
        for entity in all_entities:
            aliases = json.loads(entity.get("aliases") or "[]")
            if any(alias.lower() == name_lower for alias in aliases):
                if entity_type is None or entity["entity_type"] == entity_type:
                    return dict(entity)

        return None

    async def add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        context: str | None = None,
    ) -> str:
        """Add or strengthen a relationship between entities."""
        # Find or create entities
        from_id = await self._get_or_create_entity_id(from_entity)
        to_id = await self._get_or_create_entity_id(to_entity)

        # Check if relationship exists
        existing = await self.db.fetchone(
            """
            SELECT * FROM kg_relationships
            WHERE from_entity_id = ? AND to_entity_id = ? AND relationship_type = ?
            """,
            (from_id, to_id, relationship_type),
        )

        if existing:
            # Strengthen existing relationship
            await self.db.execute(
                """
                UPDATE kg_relationships
                SET strength = MIN(10.0, strength + 0.5),
                    occurrence_count = occurrence_count + 1,
                    last_seen = datetime('now'),
                    context = COALESCE(?, context)
                WHERE id = ?
                """,
                (context, existing["id"]),
            )
            await self.db.commit()
            return existing["id"]

        # Create new relationship
        rel_id = str(uuid.uuid4())[:12]
        await self.db.execute(
            """
            INSERT INTO kg_relationships
            (id, from_entity_id, to_entity_id, relationship_type, context)
            VALUES (?, ?, ?, ?, ?)
            """,
            (rel_id, from_id, to_id, relationship_type, context),
        )
        await self.db.commit()

        logger.debug(f"Added relationship: {from_entity} -{relationship_type}-> {to_entity}")
        return rel_id

    async def _get_or_create_entity_id(self, name: str) -> str:
        """Get entity ID, creating if needed."""
        existing = await self._find_entity(name)
        if existing:
            return existing["id"]
        # Create as unknown type, will be refined later
        return await self.add_entity("unknown", name)

    async def record_interaction(
        self,
        entities: list[str],
        source: str,
        summary: str,
        source_id: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an interaction involving entities."""
        # Ensure all entities exist
        entity_ids = []
        for entity_name in entities:
            entity_id = await self._get_or_create_entity_id(entity_name)
            entity_ids.append(entity_id)

        interaction_id = str(uuid.uuid4())[:12]
        ts = timestamp or datetime.now().isoformat()

        await self.db.execute(
            """
            INSERT INTO kg_interactions
            (id, entity_ids, source, source_id, summary, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                json.dumps(entity_ids),
                source,
                source_id,
                summary,
                ts,
                json.dumps(metadata or {}),
            ),
        )
        await self.db.commit()

        # Create co-occurrence relationships
        for i, eid1 in enumerate(entity_ids):
            for eid2 in entity_ids[i + 1 :]:
                await self.db.execute(
                    """
                    INSERT INTO kg_relationships
                    (id, from_entity_id, to_entity_id, relationship_type, context)
                    VALUES (?, ?, ?, 'discussed_with', ?)
                    ON CONFLICT(from_entity_id, to_entity_id, relationship_type)
                    DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        last_seen = datetime('now')
                    """,
                    (str(uuid.uuid4())[:12], eid1, eid2, summary[:100]),
                )
        await self.db.commit()

        return interaction_id

    async def query_entity(self, name: str) -> dict[str, Any] | None:
        """Get full information about an entity including relationships."""
        entity = await self._find_entity(name)
        if not entity:
            return None

        # Get relationships where this entity is involved
        outgoing = await self.db.fetchall(
            """
            SELECT r.*, e.name as to_name, e.entity_type as to_type
            FROM kg_relationships r
            JOIN kg_entities e ON r.to_entity_id = e.id
            WHERE r.from_entity_id = ?
            ORDER BY r.strength DESC
            """,
            (entity["id"],),
        )

        incoming = await self.db.fetchall(
            """
            SELECT r.*, e.name as from_name, e.entity_type as from_type
            FROM kg_relationships r
            JOIN kg_entities e ON r.from_entity_id = e.id
            WHERE r.to_entity_id = ?
            ORDER BY r.strength DESC
            """,
            (entity["id"],),
        )

        # Get recent interactions
        interactions = await self.db.fetchall(
            """
            SELECT * FROM kg_interactions
            WHERE entity_ids LIKE ?
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            (f'%"{entity["id"]}"%',),
        )

        return {
            "entity": entity,
            "relationships": {
                "outgoing": [dict(r) for r in outgoing],
                "incoming": [dict(r) for r in incoming],
            },
            "recent_interactions": [dict(i) for i in interactions],
        }

    async def query_interactions(
        self,
        entity_name: str | None = None,
        source: str | None = None,
        days_back: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query interactions with flexible filters."""
        query = "SELECT * FROM kg_interactions WHERE 1=1"
        params: list[Any] = []

        if entity_name:
            entity = await self._find_entity(entity_name)
            if entity:
                query += " AND entity_ids LIKE ?"
                params.append(f'%"{entity["id"]}"%')

        if source:
            query += " AND source = ?"
            params.append(source)

        if days_back:
            cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        interactions = await self.db.fetchall(query, tuple(params))

        # Resolve entity names
        results = []
        for interaction in interactions:
            entity_ids = json.loads(interaction.get("entity_ids") or "[]")
            entity_names = []
            for eid in entity_ids:
                entity = await self.db.fetchone(
                    "SELECT name FROM kg_entities WHERE id = ?", (eid,)
                )
                if entity:
                    entity_names.append(entity["name"])

            results.append({
                **dict(interaction),
                "entity_names": entity_names,
            })

        return results

    async def answer_query(self, question: str, claude_client: Any) -> str:
        """Answer a natural language query about the knowledge graph.

        Examples:
        - "What did I discuss with John last month?"
        - "Who do I work with on Project X?"
        - "What topics have I talked about recently?"
        """
        # Extract entities and time references from question
        extraction = await self._extract_query_parts(question, claude_client)

        entity_name = extraction.get("entity")
        time_range = extraction.get("time_range")
        query_type = extraction.get("query_type", "general")

        # Convert time range to days
        days_back = None
        if time_range:
            time_map = {
                "today": 1,
                "yesterday": 2,
                "this week": 7,
                "last week": 14,
                "this month": 30,
                "last month": 60,
                "recent": 14,
            }
            days_back = time_map.get(time_range.lower(), 30)

        results = []

        if query_type == "entity_info" and entity_name:
            # Get entity information
            entity_info = await self.query_entity(entity_name)
            if entity_info:
                results.append(f"About {entity_name}:")
                for rel in entity_info["relationships"]["outgoing"][:5]:
                    results.append(f"  - {rel['relationship_type']} {rel['to_name']}")
                for rel in entity_info["relationships"]["incoming"][:5]:
                    results.append(f"  - {rel['from_name']} {rel['relationship_type']} them")

        elif query_type == "interactions" or entity_name:
            # Get interactions
            interactions = await self.query_interactions(
                entity_name=entity_name,
                days_back=days_back,
                limit=10,
            )

            if interactions:
                results.append(f"Found {len(interactions)} interactions:")
                for i in interactions:
                    ts = i.get("timestamp", "")[:10]
                    source = i.get("source", "unknown")
                    summary = i.get("summary", "")[:100]
                    names = ", ".join(i.get("entity_names", []))
                    results.append(f"  [{ts}] ({source}) {names}: {summary}")
            else:
                results.append("No matching interactions found.")

        elif query_type == "relationships":
            # Query relationships
            if entity_name:
                entity_info = await self.query_entity(entity_name)
                if entity_info:
                    all_rels = (
                        entity_info["relationships"]["outgoing"]
                        + entity_info["relationships"]["incoming"]
                    )
                    results.append(f"Relationships for {entity_name}:")
                    for rel in all_rels[:10]:
                        other = rel.get("to_name") or rel.get("from_name")
                        results.append(f"  - {rel['relationship_type']} {other}")

        else:
            # General query - get recent activity
            interactions = await self.query_interactions(days_back=days_back or 7, limit=10)
            if interactions:
                results.append("Recent activity:")
                for i in interactions:
                    ts = i.get("timestamp", "")[:10]
                    names = ", ".join(i.get("entity_names", []))
                    results.append(f"  [{ts}] {names}: {i.get('summary', '')[:80]}")

        if not results:
            return "I don't have enough information to answer that question."

        return "\n".join(results)

    async def _extract_query_parts(self, question: str, claude_client: Any) -> dict:
        """Extract entities and time references from a question."""
        prompt = f"""Parse this question about personal history/relationships:

QUESTION: {question}

Extract:
1. entity: Main person/project/topic being asked about (or null)
2. time_range: Time reference like "last month", "this week", "yesterday" (or null)
3. query_type: One of "entity_info", "interactions", "relationships", "topics", "general"

Return ONLY valid JSON:
{{"entity": "John", "time_range": "last month", "query_type": "interactions"}}"""

        try:
            response = await claude_client.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You parse questions about personal knowledge graphs. Return only valid JSON.",
                model="claude-3-haiku-20240307",
                max_tokens=200,
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}")
            # Fallback: simple extraction
            return self._simple_extract(question)

    def _simple_extract(self, question: str) -> dict:
        """Simple regex-based extraction as fallback."""
        result: dict[str, Any] = {"entity": None, "time_range": None, "query_type": "general"}

        # Time patterns
        time_patterns = [
            (r"last\s+month", "last month"),
            (r"this\s+month", "this month"),
            (r"last\s+week", "last week"),
            (r"this\s+week", "this week"),
            (r"yesterday", "yesterday"),
            (r"today", "today"),
            (r"recently", "recent"),
        ]
        for pattern, value in time_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                result["time_range"] = value
                break

        # Look for "with X" or "about X" patterns
        with_match = re.search(r"with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", question)
        if with_match:
            result["entity"] = with_match.group(1)
            result["query_type"] = "interactions"

        about_match = re.search(r"about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", question)
        if about_match:
            result["entity"] = about_match.group(1)
            result["query_type"] = "entity_info"

        return result

    async def extract_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        agent_id: str,
        claude_client: Any,
    ) -> dict[str, int]:
        """Extract entities and relationships from a conversation."""
        # Format conversation
        conv_text = []
        for msg in conversation[-15:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = " ".join(text_parts)
            if content:
                conv_text.append(f"{role.upper()}: {content[:400]}")

        if not conv_text:
            return {"entities": 0, "relationships": 0, "interactions": 0}

        conversation_str = "\n".join(conv_text)
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=conversation_str)

        try:
            response = await claude_client.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You extract entities and relationships from conversations. Return only valid JSON.",
                model="claude-3-haiku-20240307",
                max_tokens=2000,
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            counts = {"entities": 0, "relationships": 0, "interactions": 0}

            # Add entities
            for entity_data in data.get("entities", []):
                await self.add_entity(
                    entity_type=entity_data.get("type", "unknown"),
                    name=entity_data.get("name", ""),
                    aliases=entity_data.get("aliases", []),
                    metadata={"context": entity_data.get("context")},
                )
                counts["entities"] += 1

            # Add relationships
            for rel_data in data.get("relationships", []):
                await self.add_relationship(
                    from_entity=rel_data.get("from", ""),
                    to_entity=rel_data.get("to", ""),
                    relationship_type=rel_data.get("type", "related_to"),
                    context=rel_data.get("context"),
                )
                counts["relationships"] += 1

            # Record interaction with topics discussed
            topics = data.get("topics_discussed", [])
            all_entities = [e.get("name") for e in data.get("entities", [])]
            all_entities.extend(topics)

            if all_entities:
                summary = f"Discussion about: {', '.join(topics[:3]) if topics else 'various topics'}"
                await self.record_interaction(
                    entities=all_entities,
                    source="conversation",
                    source_id=agent_id,
                    summary=summary,
                )
                counts["interactions"] += 1

            logger.info(
                f"Extracted {counts['entities']} entities, "
                f"{counts['relationships']} relationships from conversation"
            )
            return counts

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return {"entities": 0, "relationships": 0, "interactions": 0}
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": 0, "relationships": 0, "interactions": 0}

    async def extract_from_email(
        self,
        sender: str,
        subject: str,
        body: str,
        timestamp: str,
    ) -> None:
        """Extract entities from an email and record the interaction."""
        # Add sender as entity
        await self.add_entity("person", sender, metadata={"source": "email"})

        # Simple extraction of other entities mentioned
        # In production, would use Claude for better extraction
        await self.record_interaction(
            entities=[sender],
            source="email",
            summary=f"Email: {subject[:100]}",
            timestamp=timestamp,
            metadata={"subject": subject},
        )

    async def extract_from_slack(
        self,
        sender: str,
        channel: str | None,
        message: str,
        timestamp: str,
    ) -> None:
        """Extract entities from a Slack message."""
        await self.add_entity("person", sender, metadata={"source": "slack"})

        entities = [sender]
        if channel:
            entities.append(channel)

        await self.record_interaction(
            entities=entities,
            source="slack",
            summary=message[:200],
            timestamp=timestamp,
            metadata={"channel": channel},
        )

    async def extract_from_calendar(
        self,
        title: str,
        attendees: list[str],
        timestamp: str,
    ) -> None:
        """Extract entities from a calendar event."""
        for attendee in attendees:
            await self.add_entity("person", attendee, metadata={"source": "calendar"})

        await self.record_interaction(
            entities=attendees,
            source="calendar",
            summary=f"Meeting: {title}",
            timestamp=timestamp,
            metadata={"title": title},
        )

    async def get_entity_summary(self, name: str) -> str:
        """Get a natural language summary of an entity."""
        info = await self.query_entity(name)
        if not info:
            return f"I don't have information about {name}."

        entity = info["entity"]
        lines = [f"{name} ({entity['entity_type']})"]

        # Mention count
        count = entity.get("mention_count", 0)
        if count > 1:
            lines.append(f"Mentioned {count} times")

        # Relationships
        outgoing = info["relationships"]["outgoing"]
        incoming = info["relationships"]["incoming"]

        if outgoing:
            lines.append("Relationships:")
            for rel in outgoing[:5]:
                lines.append(f"  - {rel['relationship_type']} {rel['to_name']}")

        if incoming:
            for rel in incoming[:5]:
                lines.append(f"  - {rel['from_name']} {rel['relationship_type']} them")

        # Recent interactions
        interactions = info["recent_interactions"]
        if interactions:
            lines.append(f"Recent activity ({len(interactions)} interactions)")

        return "\n".join(lines)

    async def get_people(self, limit: int = 20) -> list[dict]:
        """Get all known people entities."""
        return await self.db.fetchall(
            """
            SELECT * FROM kg_entities
            WHERE entity_type = 'person' AND active = 1
            ORDER BY mention_count DESC, last_seen DESC
            LIMIT ?
            """,
            (limit,),
        )

    async def get_topics(self, days_back: int = 30, limit: int = 20) -> list[dict]:
        """Get recently discussed topics."""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        return await self.db.fetchall(
            """
            SELECT * FROM kg_entities
            WHERE entity_type = 'topic' AND active = 1 AND last_seen >= ?
            ORDER BY mention_count DESC
            LIMIT ?
            """,
            (cutoff, limit),
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        entity_counts = await self.db.fetchall(
            """
            SELECT entity_type, COUNT(*) as count
            FROM kg_entities WHERE active = 1
            GROUP BY entity_type
            """
        )

        total_relationships = await self.db.fetchone(
            "SELECT COUNT(*) as count FROM kg_relationships"
        )

        total_interactions = await self.db.fetchone(
            "SELECT COUNT(*) as count FROM kg_interactions"
        )

        recent_interactions = await self.db.fetchone(
            """
            SELECT COUNT(*) as count FROM kg_interactions
            WHERE timestamp >= datetime('now', '-7 days')
            """
        )

        return {
            "entities_by_type": {row["entity_type"]: row["count"] for row in entity_counts},
            "total_relationships": total_relationships["count"] if total_relationships else 0,
            "total_interactions": total_interactions["count"] if total_interactions else 0,
            "interactions_last_7_days": recent_interactions["count"] if recent_interactions else 0,
        }
