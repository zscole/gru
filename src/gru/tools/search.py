"""Web search tools."""

from __future__ import annotations

import logging

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)


async def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web using available search providers."""
    from gru.actions.services.search_providers import get_search_chain

    try:
        chain = get_search_chain()
        results = await chain.search(query, num_results)

        return {
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"error": str(e)}


async def quick_answer(question: str) -> dict:
    """Get a quick answer to a factual question using web search."""
    # Search for the question
    search_results = await web_search(question, num_results=5)

    if "error" in search_results:
        return search_results

    if not search_results.get("results"):
        return {"answer": "Could not find relevant information.", "sources": []}

    # Return search results for Claude to synthesize
    return {
        "results": search_results["results"],
        "instruction": "Synthesize an answer from these search results",
    }


def register_search_tools() -> None:
    """Register all search tools."""
    register_tool(
        name="web_search",
        description="Search the web for information. Use this for general knowledge questions, current events, research, etc.",
        parameters={
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
                "optional": True,
            },
        },
        handler=web_search,
    )

    register_tool(
        name="quick_answer",
        description="Get a quick answer to a factual question by searching the web.",
        parameters={
            "question": {
                "type": "string",
                "description": "The question to answer",
            },
        },
        handler=quick_answer,
    )
