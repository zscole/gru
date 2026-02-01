"""Browse tools for fetching and extracting web content."""

from __future__ import annotations

import logging
import re

import httpx

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)


async def fetch_url(url: str, extract_text: bool = True) -> dict:
    """Fetch a URL and optionally extract text content."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Gru/1.0)",
                },
                timeout=30,
            )
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "application/json" in content_type:
            return {"type": "json", "data": resp.json()}

        if "text/html" in content_type:
            html = resp.text
            if extract_text:
                # Simple HTML to text conversion
                text = _html_to_text(html)
                # Truncate if too long
                if len(text) > 10000:
                    text = text[:10000] + "\n\n[Content truncated]"
                return {"type": "html", "text": text, "url": str(resp.url)}
            return {"type": "html", "html": html[:50000], "url": str(resp.url)}

        if "text/" in content_type:
            return {"type": "text", "text": resp.text[:50000]}

        return {"type": content_type, "size": len(resp.content)}

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"error": str(e)}


def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion."""
    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.I)

    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Convert common elements to text
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"<p[^>]*>", "\n\n", html, flags=re.I)
    html = re.sub(r"</p>", "", html, flags=re.I)
    html = re.sub(r"<div[^>]*>", "\n", html, flags=re.I)
    html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.I)
    html = re.sub(r"<h[1-6][^>]*>", "\n\n", html, flags=re.I)
    html = re.sub(r"</h[1-6]>", "\n", html, flags=re.I)

    # Remove remaining tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    html = html.replace("&nbsp;", " ")
    html = html.replace("&amp;", "&")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")

    # Clean up whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r" {2,}", " ", html)

    return html.strip()


async def extract_from_url(url: str, what: str) -> dict:
    """Fetch a URL and extract specific information."""
    result = await fetch_url(url, extract_text=True)

    if "error" in result:
        return result

    text = result.get("text", result.get("html", ""))

    return {
        "url": url,
        "content": text,
        "instruction": f"Extract from this content: {what}",
    }


def register_browse_tools() -> None:
    """Register all browse tools."""
    register_tool(
        name="fetch_url",
        description="Fetch content from a URL. Use this to read web pages, APIs, or download content.",
        parameters={
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "extract_text": {
                "type": "boolean",
                "description": "Extract text from HTML (default true)",
                "optional": True,
            },
        },
        handler=fetch_url,
    )

    register_tool(
        name="extract_from_url",
        description="Fetch a URL and extract specific information from it.",
        parameters={
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "what": {
                "type": "string",
                "description": "What information to extract",
            },
        },
        handler=extract_from_url,
    )
