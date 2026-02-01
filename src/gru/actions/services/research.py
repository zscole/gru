"""Research action for autonomous information gathering and report generation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from gru.actions.base import Action, ActionContext, ActionResult

if TYPE_CHECKING:
    from gru.claude import ClaudeClient

logger = logging.getLogger(__name__)

# Research Claude client (set by orchestrator)
_research_claude: ClaudeClient | None = None


def set_research_claude(claude: ClaudeClient) -> None:
    """Set the Claude client for research."""
    global _research_claude
    _research_claude = claude


def get_research_claude() -> ClaudeClient | None:
    """Get the Claude client."""
    return _research_claude


class ResearchAction(Action):
    """Perform thorough research on a topic and generate a report."""

    name = "research"
    description = "Research a topic thoroughly and generate a detailed report"
    category = "research"
    requires_auth = False
    requires_confirmation = False

    # Research configuration
    max_searches = 10
    max_sources = 20

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("topic") and not params.get("query"):
            return False, "topic or query is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        topic = params.get("topic") or params.get("query")
        depth = params.get("depth", "thorough")  # quick, moderate, thorough
        output_format = params.get("format", "report")  # report, bullets, summary
        write_to_doc = params.get("write_to_doc", True)
        doc_title = params.get("doc_title") or f"Research: {topic[:50]}"
        notify_user = params.get("notify", True)

        claude = get_research_claude()
        if not claude:
            return ActionResult.error_result("Research requires Claude client. Not configured.")

        try:
            # Phase 1: Generate search queries
            queries = await self._generate_search_queries(claude, topic, depth)
            logger.info(f"Generated {len(queries)} search queries for: {topic}")

            # Phase 2: Execute searches and gather sources
            sources = await self._gather_sources(context, queries)
            logger.info(f"Gathered {len(sources)} sources")

            if not sources:
                return ActionResult.error_result(f"Could not find relevant information about: {topic}")

            # Phase 3: Synthesize research into report
            report = await self._synthesize_report(claude, topic, sources, output_format, depth)
            logger.info(f"Generated report: {len(report)} chars")

            result_data = {
                "topic": topic,
                "sources_count": len(sources),
                "report_length": len(report),
                "queries_used": queries,
            }

            # Phase 4: Write to Google Doc if requested
            if write_to_doc:
                doc_result = await self._write_to_doc(doc_title, report)
                if doc_result:
                    result_data["document_id"] = doc_result["document_id"]
                    result_data["document_url"] = doc_result["url"]

                    # Notify user
                    if notify_user and context.notify_callback:
                        await context.notify_callback(
                            context.user_id, f"Research complete: {topic}\n\n{doc_result['url']}"
                        )

                    return ActionResult.success_result(
                        f"Research complete. Report available at: {doc_result['url']}",
                        result_data,
                    )

            # Return report in response if not writing to doc
            result_data["report"] = report
            return ActionResult.success_result(
                f"Research complete on: {topic}",
                result_data,
            )

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return ActionResult.error_result(f"Research failed: {e}")

    async def _generate_search_queries(
        self,
        claude: ClaudeClient,
        topic: str,
        depth: str,
    ) -> list[str]:
        """Generate diverse search queries to cover the topic thoroughly."""
        num_queries = {"quick": 3, "moderate": 5, "thorough": 8}.get(depth, 5)

        prompt = f"""Generate {num_queries} diverse search queries to thoroughly research this topic:

Topic: {topic}

Requirements:
- Each query should explore a different angle or aspect
- Include queries for recent developments, comparisons, expert opinions
- Make queries specific enough to get relevant results
- Include at least one query for recent news/developments

Return ONLY the queries, one per line, no numbering or bullets."""

        try:
            response = await claude.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="You are a research assistant. Generate effective search queries.",
                max_tokens=500,
            )

            queries = [q.strip() for q in response.content.strip().split("\n") if q.strip() and len(q.strip()) > 5]

            # Always include the original topic as a query
            if topic not in queries:
                queries.insert(0, topic)

            return queries[: self.max_searches]

        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            # Fallback to basic queries
            return [
                topic,
                f"{topic} latest developments",
                f"{topic} comparison review",
                f"best {topic}",
                f"{topic} expert analysis",
            ][:num_queries]

    async def _gather_sources(
        self,
        context: ActionContext,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """Execute searches and gather source information."""
        from gru.actions.registry import get_registry

        registry = get_registry()
        sources = []
        seen_urls = set()

        for query in queries:
            try:
                # Use web search action
                result = await registry.execute(
                    "web_search",
                    context,
                    query=query,
                )

                if result.success and result.data.get("results"):
                    for item in result.data["results"][:5]:
                        url = item.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(
                                {
                                    "query": query,
                                    "title": item.get("title", ""),
                                    "url": url,
                                    "snippet": item.get("snippet", ""),
                                }
                            )

                # Small delay between searches
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
                continue

            if len(sources) >= self.max_sources:
                break

        return sources

    async def _synthesize_report(
        self,
        claude: ClaudeClient,
        topic: str,
        sources: list[dict[str, Any]],
        output_format: str,
        depth: str,
    ) -> str:
        """Synthesize gathered information into a coherent report."""
        # Format sources for Claude
        sources_text = "\n\n".join([f"Source: {s['title']}\nURL: {s['url']}\nContent: {s['snippet']}" for s in sources])

        format_instructions = {
            "report": """Write a comprehensive research report covering the executive summary, key findings organized by theme, detailed analysis, and recommendations.""",
            "bullets": """Write a structured summary covering key points, important details, and recommendations.""",
            "summary": """Write a concise summary covering main findings, key insights, and recommendations.""",
        }.get(output_format, "Write a comprehensive report")

        length_guide = {
            "quick": "Keep it concise, around 300-500 words.",
            "moderate": "Aim for 500-800 words with good detail.",
            "thorough": "Be comprehensive, 800-1200 words with detailed analysis.",
        }.get(depth, "")

        prompt = f"""Research Topic: {topic}

Based on the following sources, {format_instructions}

{length_guide}

SOURCES:
{sources_text}

Write the report now. Be analytical and insightful. Mention sources naturally in the text.
At the end, list the source URLs on separate lines."""

        try:
            response = await claude.send_message(
                messages=[{"role": "user", "content": prompt}],
                system="""You are an expert research analyst. Write in plain text only. No markdown formatting, no asterisks, no hashtags, no bullet points. Use natural paragraphs and conversational language. Separate sections with blank lines, not headers.""",
                max_tokens=4096,
            )

            report = response.content.strip()

            # Strip markdown formatting
            report = self._strip_markdown(report)

            # Add metadata header (plain text)
            header = f"""Research Report: {topic}
Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
Sources Analyzed: {len(sources)}

"""
            return header + report

        except Exception as e:
            logger.error(f"Report synthesis failed: {e}")
            raise

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        import re

        # Remove headers (# ## ### etc)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Remove bullet points but keep content
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # Remove numbered lists markers but keep content
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Clean up extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def _write_to_doc(
        self,
        title: str,
        content: str,
    ) -> dict[str, Any] | None:
        """Write report to Google Doc."""
        from gru.actions.services.google import get_google_connector

        connector = get_google_connector()
        if not connector or not connector.is_authenticated():
            logger.warning("Google not configured, skipping doc creation")
            return None

        try:
            result = await connector.create_document(title, content)
            return result
        except Exception as e:
            logger.error(f"Failed to create Google Doc: {e}")
            return None


class QuickAnswerAction(Action):
    """Get a quick answer to a factual question using web search."""

    name = "quick_answer"
    description = "Get a quick, factual answer to a question"
    category = "research"
    requires_auth = False

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("question") and not params.get("query"):
            return False, "question is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        question = params.get("question") or params.get("query")

        claude = get_research_claude()
        if not claude:
            return ActionResult.error_result("Claude client not configured")

        from gru.actions.registry import get_registry

        registry = get_registry()

        try:
            # Single focused search
            search_result = await registry.execute(
                "web_search",
                context,
                query=question,
            )

            sources = []
            if search_result.success and search_result.data.get("results"):
                sources = search_result.data["results"][:5]

            if not sources:
                # Try to answer from Claude's knowledge
                response = await claude.send_message(
                    messages=[{"role": "user", "content": question}],
                    system="Answer concisely and factually. If you're not sure, say so.",
                    max_tokens=500,
                )
                return ActionResult.success_result(
                    response.content,
                    {"source": "knowledge", "question": question},
                )

            # Synthesize answer from sources
            sources_text = "\n".join([f"- {s.get('title', '')}: {s.get('snippet', '')}" for s in sources])

            response = await claude.send_message(
                messages=[
                    {
                        "role": "user",
                        "content": f"""Question: {question}

Sources:
{sources_text}

Provide a concise, accurate answer based on these sources.""",
                    }
                ],
                system="Answer questions concisely and accurately. Cite sources when relevant.",
                max_tokens=500,
            )

            return ActionResult.success_result(
                response.content,
                {
                    "source": "web_search",
                    "question": question,
                    "sources_count": len(sources),
                },
            )

        except Exception as e:
            logger.error(f"Quick answer failed: {e}")
            return ActionResult.error_result(f"Failed to answer: {e}")
