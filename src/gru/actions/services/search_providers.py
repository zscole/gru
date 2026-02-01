"""Search providers for web search with fallback chain."""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from abc import ABC, abstractmethod

import aiohttp

logger = logging.getLogger(__name__)


class SearchResult:
    """A single search result."""

    def __init__(self, title: str, url: str, snippet: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchProvider(ABC):
    """Base class for search providers."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute a search and return results."""
        pass

    def is_available(self) -> bool:
        """Check if this provider is available (API keys configured, etc.)."""
        return True


class StartpageProvider(SearchProvider):
    """Startpage search - privacy-focused, proxies Google results."""

    name = "startpage"
    requires_api_key = False

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.startpage.com/sp/search?query={encoded}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with aiohttp.ClientSession() as session, session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning(f"Startpage returned status {resp.status}")
                return []

            html = await resp.text()

        results = []
        # Startpage results have class "result-link" for the main link
        # and class "w-gl__result-title" for titles

        # Pattern to find result links
        re.compile(r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>.*?</a>', re.DOTALL | re.IGNORECASE)

        # Find all result containers
        re.compile(
            r'<div[^>]*class="[^"]*w-gl__result[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', re.DOTALL | re.IGNORECASE
        )

        # Simpler approach: find result-link anchors
        for match in re.finditer(
            r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE
        ):
            if len(results) >= num_results:
                break

            url = match.group(1)
            title_html = match.group(2)

            # Clean title
            title = re.sub(r"<[^>]+>", "", title_html).strip()

            # Skip if not a real URL
            if not url.startswith("http"):
                continue

            # Try to find snippet near this result
            snippet = ""

            if title and url:
                results.append(SearchResult(title, url, snippet))

        logger.info(f"Startpage returned {len(results)} results for: {query}")
        return results


class BingProvider(SearchProvider):
    """Bing HTML search - no API key needed."""

    name = "bing"
    requires_api_key = False

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        encoded = urllib.parse.quote_plus(query)
        # Add setlang and mkt for English US results
        url = f"https://www.bing.com/search?q={encoded}&setlang=en-US&mkt=en-US"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with aiohttp.ClientSession() as session, session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning(f"Bing returned status {resp.status}")
                return []

            html = await resp.text()

        results = []
        # Bing results are in <li class="b_algo"> elements
        # Title in <h2><a href="...">title</a></h2>
        # Snippet in <p> or <div class="b_caption">

        # Pattern for result blocks
        algo_pattern = re.compile(r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)

        for match in algo_pattern.finditer(html):
            if len(results) >= num_results:
                break

            block = match.group(1)

            # Extract URL and title from <h2><a>
            link_match = re.search(
                r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL | re.IGNORECASE
            )
            if not link_match:
                continue

            url = link_match.group(1)
            title = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()

            # Extract snippet
            snippet = ""
            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL | re.IGNORECASE)
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()[:200]

            if title and url.startswith("http"):
                results.append(SearchResult(title, url, snippet))

        logger.info(f"Bing returned {len(results)} results for: {query}")
        return results


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo HTML search - no API key needed."""

    name = "duckduckgo"
    requires_api_key = False

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session, session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning(f"DuckDuckGo returned status {resp.status}")
                return []

            html = await resp.text()

        results = []
        # Parse results from HTML
        # DuckDuckGo HTML results are in <a class="result__a"> tags
        re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        # Simpler pattern for just links and titles
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
        )
        snippet_pattern = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (href, title) in enumerate(links[:num_results]):
            # Clean up the URL (DuckDuckGo uses redirect URLs)
            actual_url = href
            if "uddg=" in href:
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    actual_url = urllib.parse.unquote(match.group(1))

            # Clean HTML tags from title
            clean_title = re.sub(r"<[^>]+>", "", title).strip()

            # Get corresponding snippet
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()[:200]

            if clean_title and actual_url.startswith("http"):
                results.append(SearchResult(clean_title, actual_url, snippet))

        logger.info(f"DuckDuckGo returned {len(results)} results for: {query}")
        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""

    name = "brave"
    requires_api_key = True

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            return []

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": num_results}

        async with aiohttp.ClientSession() as session, session.get(url, headers=headers, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"Brave Search returned status {resp.status}")
                return []

            data = await resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", "")[:200],
                )
            )

        logger.info(f"Brave Search returned {len(results)} results for: {query}")
        return results


class SerpAPIProvider(SearchProvider):
    """SerpAPI provider (Google results via API)."""

    name = "serpapi"
    requires_api_key = True

    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            return []

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results,
        }

        async with aiohttp.ClientSession() as session, session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"SerpAPI returned status {resp.status}")
                return []

            data = await resp.json()

        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "")[:200],
                )
            )

        logger.info(f"SerpAPI returned {len(results)} results for: {query}")
        return results


class GoogleCSEProvider(SearchProvider):
    """Google Custom Search Engine provider."""

    name = "google_cse"
    requires_api_key = True

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CSE_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")

    def is_available(self) -> bool:
        return bool(self.api_key and self.cse_id)

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        if not self.api_key or not self.cse_id:
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, 10),  # CSE max is 10
        }

        async with aiohttp.ClientSession() as session, session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"Google CSE returned status {resp.status}")
                return []

            data = await resp.json()

        results = []
        for item in data.get("items", [])[:num_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "")[:200],
                )
            )

        logger.info(f"Google CSE returned {len(results)} results for: {query}")
        return results


class SearchProviderChain:
    """Chain of search providers with fallback."""

    def __init__(self, providers: list[SearchProvider] | None = None):
        if providers is None:
            # Default provider chain - Startpage first (proxies Google, most reliable)
            providers = [
                StartpageProvider(),
                BingProvider(),
                DuckDuckGoProvider(),
                BraveSearchProvider(),
                SerpAPIProvider(),
                GoogleCSEProvider(),
            ]
        self.providers = providers

    def get_available_providers(self) -> list[SearchProvider]:
        """Get list of available providers."""
        return [p for p in self.providers if p.is_available()]

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search using available providers, with fallback."""
        for provider in self.providers:
            if not provider.is_available():
                continue

            try:
                results = await provider.search(query, num_results)
                if results:
                    return results
                logger.info(f"{provider.name} returned no results, trying next provider")
            except Exception as e:
                logger.warning(f"{provider.name} failed: {e}, trying next provider")
                continue

        logger.warning(f"All search providers failed for: {query}")
        return []


# Singleton instance
_search_chain: SearchProviderChain | None = None


def get_search_chain() -> SearchProviderChain:
    """Get the search provider chain singleton."""
    global _search_chain
    if _search_chain is None:
        _search_chain = SearchProviderChain()
    return _search_chain
