"""Knowledge graph tools for querying relationships and history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_knowledge_graph: KnowledgeGraph | None = None
_claude_client: Any = None


def set_knowledge_dependencies(kg: KnowledgeGraph, claude: Any) -> None:
    """Set dependencies for knowledge tools."""
    global _knowledge_graph, _claude_client
    _knowledge_graph = kg
    _claude_client = claude
    logger.debug("Knowledge graph tools initialized")


async def ask_knowledge(question: str) -> dict:
    """Answer a question using the knowledge graph.

    Examples:
    - "What did I discuss with John last month?"
    - "Who have I been meeting with recently?"
    - "What do I know about Project Alpha?"
    """
    if not _knowledge_graph or not _claude_client:
        return {"error": "Knowledge graph not initialized"}

    try:
        answer = await _knowledge_graph.answer_query(question, _claude_client)
        return {
            "question": question,
            "answer": answer,
        }
    except Exception as e:
        logger.error(f"Knowledge query failed: {e}")
        return {"error": str(e)}


async def get_person_info(name: str) -> dict:
    """Get information about a person from the knowledge graph."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        info = await _knowledge_graph.query_entity(name)
        if not info:
            return {"error": f"No information found for {name}"}

        # Format for readability
        entity = info["entity"]
        relationships = []

        for rel in info["relationships"]["outgoing"][:10]:
            relationships.append(
                {
                    "type": rel["relationship_type"],
                    "to": rel["to_name"],
                    "context": rel.get("context"),
                }
            )

        for rel in info["relationships"]["incoming"][:10]:
            relationships.append(
                {
                    "type": f"{rel['from_name']} {rel['relationship_type']}",
                    "to": name,
                    "context": rel.get("context"),
                }
            )

        return {
            "name": entity["name"],
            "type": entity["entity_type"],
            "first_seen": entity.get("first_seen"),
            "last_seen": entity.get("last_seen"),
            "mention_count": entity.get("mention_count", 0),
            "relationships": relationships,
            "recent_interactions": len(info["recent_interactions"]),
        }
    except Exception as e:
        logger.error(f"Person info query failed: {e}")
        return {"error": str(e)}


async def get_recent_interactions(
    entity: str | None = None,
    source: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> dict:
    """Get recent interactions, optionally filtered by entity or source."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        interactions = await _knowledge_graph.query_interactions(
            entity_name=entity,
            source=source,
            days_back=days,
            limit=limit,
        )

        return {
            "count": len(interactions),
            "filters": {
                "entity": entity,
                "source": source,
                "days": days,
            },
            "interactions": [
                {
                    "timestamp": i.get("timestamp"),
                    "source": i.get("source"),
                    "entities": i.get("entity_names", []),
                    "summary": i.get("summary"),
                }
                for i in interactions
            ],
        }
    except Exception as e:
        logger.error(f"Interactions query failed: {e}")
        return {"error": str(e)}


async def get_known_people(limit: int = 20) -> dict:
    """Get list of known people from the knowledge graph."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        people = await _knowledge_graph.get_people(limit=limit)
        return {
            "count": len(people),
            "people": [
                {
                    "name": p["name"],
                    "mention_count": p.get("mention_count", 0),
                    "last_seen": p.get("last_seen"),
                }
                for p in people
            ],
        }
    except Exception as e:
        logger.error(f"People query failed: {e}")
        return {"error": str(e)}


async def get_recent_topics(days: int = 30, limit: int = 20) -> dict:
    """Get recently discussed topics."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        topics = await _knowledge_graph.get_topics(days_back=days, limit=limit)
        return {
            "count": len(topics),
            "days": days,
            "topics": [
                {
                    "name": t["name"],
                    "mention_count": t.get("mention_count", 0),
                    "last_seen": t.get("last_seen"),
                }
                for t in topics
            ],
        }
    except Exception as e:
        logger.error(f"Topics query failed: {e}")
        return {"error": str(e)}


async def add_person(
    name: str,
    context: str | None = None,
    aliases: list[str] | None = None,
) -> dict:
    """Add a person to the knowledge graph."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        entity_id = await _knowledge_graph.add_entity(
            entity_type="person",
            name=name,
            aliases=aliases,
            metadata={"context": context} if context else None,
        )
        return {
            "status": "success",
            "message": f"Added {name} to knowledge graph",
            "entity_id": entity_id,
        }
    except Exception as e:
        logger.error(f"Add person failed: {e}")
        return {"error": str(e)}


async def add_relationship(
    from_entity: str,
    relationship: str,
    to_entity: str,
    context: str | None = None,
) -> dict:
    """Add a relationship between two entities."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        rel_id = await _knowledge_graph.add_relationship(
            from_entity=from_entity,
            to_entity=to_entity,
            relationship_type=relationship,
            context=context,
        )
        return {
            "status": "success",
            "message": f"Added relationship: {from_entity} {relationship} {to_entity}",
            "relationship_id": rel_id,
        }
    except Exception as e:
        logger.error(f"Add relationship failed: {e}")
        return {"error": str(e)}


async def knowledge_stats() -> dict:
    """Get statistics about the knowledge graph."""
    if not _knowledge_graph:
        return {"error": "Knowledge graph not initialized"}

    try:
        stats = await _knowledge_graph.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        return {"error": str(e)}


def register_knowledge_tools() -> None:
    """Register knowledge graph tools."""
    register_tool(
        name="ask_knowledge",
        description=(
            "Answer questions about your personal history, relationships, and past discussions. "
            "Use for questions like 'What did I discuss with John last month?' or "
            "'Who have I been working with on Project X?'"
        ),
        parameters={
            "question": {
                "type": "string",
                "description": "Natural language question about history, relationships, or past interactions",
            },
        },
        handler=ask_knowledge,
    )

    register_tool(
        name="get_person_info",
        description="Get detailed information about a person including relationships and interaction history.",
        parameters={
            "name": {
                "type": "string",
                "description": "Name of the person to look up",
            },
        },
        handler=get_person_info,
    )

    register_tool(
        name="get_recent_interactions",
        description="Get recent interactions, optionally filtered by person, source (email, slack, calendar), or time.",
        parameters={
            "entity": {
                "type": "string",
                "description": "Name of person or topic to filter by",
                "optional": True,
            },
            "source": {
                "type": "string",
                "description": "Source to filter by (email, slack, calendar, conversation)",
                "optional": True,
            },
            "days": {
                "type": "integer",
                "description": "Number of days to look back (default 7)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum interactions to return (default 20)",
                "optional": True,
            },
        },
        handler=get_recent_interactions,
    )

    register_tool(
        name="get_known_people",
        description="Get list of people in the knowledge graph, sorted by interaction frequency.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum people to return (default 20)",
                "optional": True,
            },
        },
        handler=get_known_people,
    )

    register_tool(
        name="get_recent_topics",
        description="Get topics that have been discussed recently.",
        parameters={
            "days": {
                "type": "integer",
                "description": "Number of days to look back (default 30)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum topics to return (default 20)",
                "optional": True,
            },
        },
        handler=get_recent_topics,
    )

    register_tool(
        name="add_person",
        description="Add a person to the knowledge graph manually.",
        parameters={
            "name": {
                "type": "string",
                "description": "Person's name",
            },
            "context": {
                "type": "string",
                "description": "Context about this person (e.g., 'colleague at Acme')",
                "optional": True,
            },
            "aliases": {
                "type": "array",
                "description": "Alternative names or nicknames",
                "optional": True,
            },
        },
        handler=add_person,
    )

    register_tool(
        name="add_relationship",
        description="Add a relationship between two entities in the knowledge graph.",
        parameters={
            "from_entity": {
                "type": "string",
                "description": "First entity (e.g., person name)",
            },
            "relationship": {
                "type": "string",
                "description": "Relationship type (e.g., 'works_with', 'manages', 'knows')",
            },
            "to_entity": {
                "type": "string",
                "description": "Second entity (e.g., another person or project)",
            },
            "context": {
                "type": "string",
                "description": "Additional context about the relationship",
                "optional": True,
            },
        },
        handler=add_relationship,
    )

    register_tool(
        name="knowledge_stats",
        description="Get statistics about the knowledge graph (entity counts, relationships, etc.).",
        parameters={},
        handler=knowledge_stats,
    )
