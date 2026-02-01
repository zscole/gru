"""Research tools for AI trend monitoring and analysis.

This module provides tools for collecting data from various sources,
analyzing trends, and generating reports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_orchestrator: Orchestrator | None = None
_config: Any = None

# Research data storage
RESEARCH_DIR = Path.home() / ".gru" / "research"


def set_research_dependencies(config: Any, orchestrator: Orchestrator) -> None:
    """Set dependencies for research tools."""
    global _config, _orchestrator
    _config = config
    _orchestrator = orchestrator
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


async def _fetch_json(url: str, headers: dict | None = None) -> dict | list | None:
    """Fetch JSON from URL."""
    try:
        async with aiohttp.ClientSession() as session, session.get(url, headers=headers, timeout=30) as resp:
            if resp.status == 200:
                return await resp.json()
            logger.warning(f"HTTP {resp.status} fetching {url}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


async def _fetch_html(url: str, headers: dict | None = None) -> str | None:
    """Fetch HTML from URL."""
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if headers:
        default_headers.update(headers)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=default_headers, timeout=30) as resp:
                if resp.status == 200:
                    return await resp.text()
                logger.warning(f"HTTP {resp.status} fetching {url}")
                return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


async def fetch_x_ai(
    accounts: str = "kaboroevich,_akhaliq,AndrewYNg,ylaboratory,EMostaque", limit: int = 20
) -> dict[str, Any]:
    """Fetch AI-related posts from X/Twitter via Nitter instances.

    Args:
        accounts: Comma-separated list of X/Twitter handles to monitor
        limit: Maximum posts to fetch
    """
    # Nitter instances (public Twitter frontends) - try multiple in case some are down
    nitter_instances = [
        "nitter.privacydev.net",
        "nitter.poast.org",
        "nitter.cz",
        "nitter.1d4.us",
    ]

    results = []

    for account in accounts.split(","):
        account = account.strip().lstrip("@")

        for instance in nitter_instances:
            try:
                url = f"https://{instance}/{account}"
                html = await _fetch_html(url)

                if not html:
                    continue

                # Parse tweets from HTML (simple extraction)
                # Nitter uses timeline-item class for tweets
                import re

                # Extract tweet content - look for tweet-content class
                tweet_pattern = r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>'
                tweets = re.findall(tweet_pattern, html, re.DOTALL)

                # Extract tweet links
                link_pattern = r'<a class="tweet-link"[^>]*href="([^"]*)"'
                links = re.findall(link_pattern, html)

                # Extract timestamps
                time_pattern = r'<span class="tweet-date"[^>]*><a[^>]*title="([^"]*)"'
                times = re.findall(time_pattern, html)

                for i, tweet in enumerate(tweets[: limit // len(accounts.split(",")) + 1]):
                    # Clean HTML tags
                    clean_text = re.sub(r"<[^>]+>", "", tweet).strip()

                    if clean_text and len(clean_text) > 20:  # Skip very short tweets
                        results.append(
                            {
                                "source": f"x/@{account}",
                                "content": clean_text[:500],
                                "url": f"https://x.com{links[i]}" if i < len(links) else f"https://x.com/{account}",
                                "timestamp": times[i] if i < len(times) else "",
                                "author": account,
                            }
                        )

                break  # Success with this instance, move to next account

            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed for {account}: {e}")
                continue

        await asyncio.sleep(0.5)  # Rate limit between accounts

    return {
        "posts": results[:limit],
        "count": len(results[:limit]),
        "accounts": accounts,
        "note": "Fetched via Nitter (may be incomplete if instances are down)",
    }


async def fetch_huggingface_trending(limit: int = 20) -> dict[str, Any]:
    """Fetch trending models and spaces from Hugging Face.

    Args:
        limit: Maximum items to fetch
    """
    results = []

    # Fetch trending models
    models_url = "https://huggingface.co/api/models?sort=trending&limit=" + str(limit)
    models = await _fetch_json(models_url)

    if models:
        for model in models[: limit // 2]:
            results.append(
                {
                    "source": "huggingface/models",
                    "name": model.get("id", ""),
                    "url": f"https://huggingface.co/{model.get('id', '')}",
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "tags": model.get("tags", [])[:5],
                    "pipeline_tag": model.get("pipeline_tag", ""),
                }
            )

    # Fetch trending spaces
    spaces_url = "https://huggingface.co/api/spaces?sort=trending&limit=" + str(limit)
    spaces = await _fetch_json(spaces_url)

    if spaces:
        for space in spaces[: limit // 2]:
            results.append(
                {
                    "source": "huggingface/spaces",
                    "name": space.get("id", ""),
                    "url": f"https://huggingface.co/spaces/{space.get('id', '')}",
                    "likes": space.get("likes", 0),
                    "sdk": space.get("sdk", ""),
                }
            )

    return {
        "items": results,
        "count": len(results),
        "categories": ["models", "spaces"],
    }


async def fetch_producthunt_ai(limit: int = 15) -> dict[str, Any]:
    """Fetch AI product launches from Product Hunt.

    Args:
        limit: Maximum products to fetch
    """
    # Product Hunt doesn't have a simple public API, so we scrape the AI topic page
    url = "https://www.producthunt.com/topics/artificial-intelligence"
    html = await _fetch_html(url)

    results = []

    if html:
        import re

        # Extract product cards - look for data-test="post-item" or similar
        # This is a simplified extraction
        name_pattern = r'<a[^>]*data-test="post-name"[^>]*>([^<]+)</a>'
        names = re.findall(name_pattern, html)

        tagline_pattern = r'<a[^>]*data-test="post-tagline"[^>]*>([^<]+)</a>'
        taglines = re.findall(tagline_pattern, html)

        link_pattern = r'href="(/posts/[^"?]+)'
        links = re.findall(link_pattern, html)

        # Deduplicate links
        seen_links = set()
        unique_links = []
        for link in links:
            if link not in seen_links:
                seen_links.add(link)
                unique_links.append(link)

        for i in range(min(limit, len(names))):
            results.append(
                {
                    "source": "producthunt",
                    "name": names[i] if i < len(names) else "",
                    "tagline": taglines[i] if i < len(taglines) else "",
                    "url": f"https://www.producthunt.com{unique_links[i]}" if i < len(unique_links) else "",
                }
            )

    return {
        "products": results,
        "count": len(results),
        "category": "artificial-intelligence",
    }


async def fetch_tech_news(limit: int = 20) -> dict[str, Any]:
    """Fetch AI news from tech news RSS feeds.

    Args:
        limit: Maximum articles to fetch
    """
    # RSS feeds for AI/tech news
    feeds = [
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
        ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
        ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
        ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
        ("Ars Technica AI", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    ]

    results = []

    for feed_name, feed_url in feeds:
        try:
            xml = await _fetch_html(feed_url)
            if not xml:
                continue

            import re

            # Parse RSS items
            items = xml.split("<item>")[1:]  # Skip content before first item

            for item in items[: limit // len(feeds) + 1]:
                # Extract title
                title_match = re.search(r"<title>(?:<!\[CDATA\[)?([^<\]]+)(?:\]\]>)?</title>", item)
                title = title_match.group(1).strip() if title_match else ""

                # Extract link
                link_match = re.search(r"<link>([^<]+)</link>", item)
                link = link_match.group(1).strip() if link_match else ""

                # Extract pubDate
                date_match = re.search(r"<pubDate>([^<]+)</pubDate>", item)
                pub_date = date_match.group(1).strip() if date_match else ""

                # Extract description
                desc_match = re.search(r"<description>(?:<!\[CDATA\[)?(.+?)(?:\]\]>)?</description>", item, re.DOTALL)
                description = ""
                if desc_match:
                    description = re.sub(r"<[^>]+>", "", desc_match.group(1))[:300].strip()

                if title and link:
                    # Filter for AI relevance
                    ai_keywords = [
                        "ai",
                        "gpt",
                        "llm",
                        "openai",
                        "anthropic",
                        "claude",
                        "gemini",
                        "machine learning",
                        "neural",
                        "model",
                        "chatbot",
                        "artificial intelligence",
                    ]
                    title_lower = title.lower()
                    is_ai = any(kw in title_lower for kw in ai_keywords)

                    if is_ai or "AI" in title:
                        results.append(
                            {
                                "source": feed_name,
                                "title": title,
                                "url": link,
                                "published": pub_date,
                                "summary": description,
                            }
                        )

        except Exception as e:
            logger.debug(f"Failed to fetch {feed_name}: {e}")
            continue

        await asyncio.sleep(0.3)

    # Sort by date if possible, otherwise just return
    return {
        "articles": results[:limit],
        "count": len(results[:limit]),
        "feeds": [f[0] for f in feeds],
    }


async def fetch_ai_newsletters() -> dict[str, Any]:
    """Fetch content from AI newsletters and blogs."""
    sources = [
        ("Import AI", "https://importai.substack.com/feed"),
        ("The Batch", "https://www.deeplearning.ai/the-batch/feed/"),
        ("AI Weekly", "https://aiweekly.co/feed"),
        ("Last Week in AI", "https://lastweekin.ai/feed"),
    ]

    results = []

    for name, url in sources:
        try:
            xml = await _fetch_html(url)
            if not xml:
                continue

            import re

            items = xml.split("<item>")[1:]

            for item in items[:3]:  # Just get latest few from each
                title_match = re.search(r"<title>(?:<!\[CDATA\[)?([^<\]]+)(?:\]\]>)?</title>", item)
                title = title_match.group(1).strip() if title_match else ""

                link_match = re.search(r"<link>([^<]+)</link>", item)
                link = link_match.group(1).strip() if link_match else ""

                date_match = re.search(r"<pubDate>([^<]+)</pubDate>", item)
                pub_date = date_match.group(1).strip() if date_match else ""

                if title and link:
                    results.append(
                        {
                            "source": name,
                            "title": title,
                            "url": link,
                            "published": pub_date,
                        }
                    )

        except Exception as e:
            logger.debug(f"Failed to fetch newsletter {name}: {e}")

        await asyncio.sleep(0.3)

    return {
        "newsletters": results,
        "count": len(results),
        "sources": [s[0] for s in sources],
    }


async def fetch_reddit_ai(
    subreddits: str = "MachineLearning,LocalLLaMA,artificial,singularity", limit: int = 25, timeframe: str = "day"
) -> dict[str, Any]:
    """Fetch top posts from AI-related subreddits.

    Args:
        subreddits: Comma-separated list of subreddit names
        limit: Number of posts per subreddit
        timeframe: Time filter (hour, day, week, month, year, all)
    """
    results = []

    for subreddit in subreddits.split(","):
        subreddit = subreddit.strip()
        url = f"https://www.reddit.com/r/{subreddit}/top.json?t={timeframe}&limit={limit}"

        headers = {"User-Agent": "Gru-Research-Bot/1.0"}
        data = await _fetch_json(url, headers)

        if data and "data" in data and "children" in data["data"]:
            for post in data["data"]["children"]:
                p = post["data"]
                results.append(
                    {
                        "source": f"reddit/r/{subreddit}",
                        "title": p.get("title", ""),
                        "url": f"https://reddit.com{p.get('permalink', '')}",
                        "score": p.get("score", 0),
                        "comments": p.get("num_comments", 0),
                        "created": datetime.fromtimestamp(p.get("created_utc", 0)).isoformat(),
                        "author": p.get("author", ""),
                        "selftext": p.get("selftext", "")[:500] if p.get("selftext") else None,
                        "external_url": p.get("url") if not p.get("is_self") else None,
                    }
                )

        # Rate limit
        await asyncio.sleep(1)

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "posts": results[: limit * 2],  # Return top posts across all subreddits
        "count": len(results),
        "subreddits": subreddits,
        "timeframe": timeframe,
    }


async def fetch_github_trending(language: str = "", since: str = "daily") -> dict[str, Any]:
    """Fetch trending repositories from GitHub.

    Args:
        language: Filter by programming language (empty for all)
        since: Time range (daily, weekly, monthly)
    """
    # GitHub doesn't have an official trending API, so we use the unofficial one
    url = f"https://api.github.com/search/repositories?q=created:>{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}+stars:>10&sort=stars&order=desc&per_page=30"

    if language:
        url += f"+language:{language}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Gru-Research-Bot/1.0",
    }

    # Add GitHub token if available
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GRU_GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    data = await _fetch_json(url, headers)

    results = []
    if data and "items" in data:
        for repo in data["items"]:
            results.append(
                {
                    "source": "github",
                    "name": repo.get("full_name", ""),
                    "url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": repo.get("language", ""),
                    "created": repo.get("created_at", ""),
                    "updated": repo.get("updated_at", ""),
                    "topics": repo.get("topics", []),
                }
            )

    return {
        "repositories": results,
        "count": len(results),
        "language": language or "all",
        "since": since,
    }


async def fetch_hackernews_top(limit: int = 30, min_score: int = 50) -> dict[str, Any]:
    """Fetch top stories from Hacker News.

    Args:
        limit: Maximum number of stories to fetch
        min_score: Minimum score filter
    """
    # Get top story IDs
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    story_ids = await _fetch_json(url)

    if not story_ids:
        return {"stories": [], "count": 0, "error": "Failed to fetch story IDs"}

    results = []
    ai_keywords = [
        "ai",
        "gpt",
        "llm",
        "claude",
        "openai",
        "anthropic",
        "machine learning",
        "neural",
        "transformer",
        "diffusion",
        "model",
        "agent",
        "langchain",
        "embedding",
        "vector",
        "rag",
        "fine-tune",
        "lora",
        "inference",
    ]

    for story_id in story_ids[:100]:  # Check more than limit to filter
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        story = await _fetch_json(story_url)

        if not story or story.get("score", 0) < min_score:
            continue

        title = story.get("title", "").lower()
        # Filter for AI-related content
        is_ai_related = any(kw in title for kw in ai_keywords)

        if is_ai_related or story.get("score", 0) > 200:  # High score or AI-related
            results.append(
                {
                    "source": "hackernews",
                    "title": story.get("title", ""),
                    "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                    "hn_url": f"https://news.ycombinator.com/item?id={story_id}",
                    "score": story.get("score", 0),
                    "comments": story.get("descendants", 0),
                    "author": story.get("by", ""),
                    "created": datetime.fromtimestamp(story.get("time", 0)).isoformat(),
                    "is_ai_related": is_ai_related,
                }
            )

        if len(results) >= limit:
            break

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.1)

    return {
        "stories": results,
        "count": len(results),
        "min_score": min_score,
    }


async def fetch_arxiv_ai(query: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL", max_results: int = 20) -> dict[str, Any]:
    """Fetch recent AI papers from ArXiv.

    Args:
        query: ArXiv search query
        max_results: Maximum number of papers
    """
    import urllib.parse

    # Search for papers from last 2 days
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        async with aiohttp.ClientSession() as session, session.get(url, timeout=30) as resp:
            if resp.status != 200:
                return {"papers": [], "error": f"HTTP {resp.status}"}

            text = await resp.text()
    except Exception as e:
        return {"papers": [], "error": str(e)}

    # Parse XML (simple parsing without external deps)
    results = []

    # Extract entries
    entries = text.split("<entry>")[1:]  # Skip the first split (before first entry)

    for entry in entries:
        try:
            # Extract fields with simple string parsing
            def extract(tag: str, text: str = entry) -> str:
                start = text.find(f"<{tag}>")
                end = text.find(f"</{tag}>")
                if start != -1 and end != -1:
                    return text[start + len(tag) + 2 : end].strip()
                return ""

            title = extract("title").replace("\n", " ")
            summary = extract("summary").replace("\n", " ")[:500]

            # Extract link
            link_start = entry.find('href="http://arxiv.org/abs/')
            if link_start != -1:
                link_end = entry.find('"', link_start + 6)
                arxiv_url = entry[link_start + 6 : link_end]
            else:
                arxiv_url = ""

            # Extract authors
            authors = []
            for author_block in entry.split("<author>")[1:]:
                name_match = author_block.split("<name>")
                if len(name_match) > 1:
                    name = name_match[1].split("</name>")[0]
                    authors.append(name)

            if title:
                results.append(
                    {
                        "source": "arxiv",
                        "title": title,
                        "url": arxiv_url,
                        "summary": summary,
                        "authors": authors[:5],  # Limit authors
                    }
                )
        except Exception as e:
            logger.debug(f"Error parsing arxiv entry: {e}")
            continue

    return {
        "papers": results,
        "count": len(results),
        "query": query,
    }


async def collect_all_sources() -> dict[str, Any]:
    """Collect data from all research sources in parallel."""

    tasks = [
        fetch_reddit_ai(),
        fetch_github_trending(),
        fetch_hackernews_top(),
        fetch_arxiv_ai(),
        fetch_x_ai(),
        fetch_huggingface_trending(),
        fetch_producthunt_ai(),
        fetch_tech_news(),
        fetch_ai_newsletters(),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    source_names = [
        "reddit",
        "github",
        "hackernews",
        "arxiv",
        "x_twitter",
        "huggingface",
        "producthunt",
        "tech_news",
        "newsletters",
    ]

    collected = {"collected_at": datetime.now().isoformat()}
    for i, name in enumerate(source_names):
        collected[name] = results[i] if not isinstance(results[i], Exception) else {"error": str(results[i])}

    # Save to file for persistence
    today = datetime.now().strftime("%Y-%m-%d")
    data_file = RESEARCH_DIR / f"collected_{today}.json"

    try:
        with open(data_file, "w") as f:
            json.dump(collected, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save research data: {e}")

    # Summary - count items from each source
    total_items = 0
    sources_summary = {}

    for name in source_names:
        src = collected.get(name, {})
        if "error" in src:
            sources_summary[name] = "error"
        else:
            # Different sources use different keys for their lists
            count = src.get("count", 0)
            if count == 0:
                # Try to count items directly
                for key in [
                    "posts",
                    "repositories",
                    "stories",
                    "papers",
                    "items",
                    "products",
                    "articles",
                    "newsletters",
                ]:
                    if key in src:
                        count = len(src[key])
                        break
            sources_summary[name] = count
            total_items += count

    return {
        "status": "collected",
        "total_items": total_items,
        "sources": sources_summary,
        "data_file": str(data_file),
    }


async def get_collected_data(date: str | None = None) -> dict[str, Any]:
    """Get previously collected research data.

    Args:
        date: Date in YYYY-MM-DD format (default: today)
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    data_file = RESEARCH_DIR / f"collected_{date}.json"

    if not data_file.exists():
        return {"error": f"No data found for {date}"}

    try:
        with open(data_file) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


async def save_research_report(report: str, date: str | None = None) -> dict[str, Any]:
    """Save a research report.

    Args:
        report: The report content (markdown)
        date: Date in YYYY-MM-DD format (default: today)
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    report_file = RESEARCH_DIR / f"report_{date}.md"

    try:
        with open(report_file, "w") as f:
            f.write(report)
        return {"status": "saved", "file": str(report_file)}
    except Exception as e:
        return {"error": str(e)}


async def get_research_report(date: str | None = None) -> dict[str, Any]:
    """Get a saved research report.

    Args:
        date: Date in YYYY-MM-DD format (default: today)
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    report_file = RESEARCH_DIR / f"report_{date}.md"

    if not report_file.exists():
        return {"error": f"No report found for {date}"}

    try:
        with open(report_file) as f:
            return {"report": f.read(), "date": date, "file": str(report_file)}
    except Exception as e:
        return {"error": str(e)}


DAILY_RESEARCH_PROMPT = """You are conducting daily AI research for Zak Cole.

FOCUS: Technical developments you can BUILD with. Not general AI news, corporate drama, policy, or hype.

STYLE REQUIREMENTS - READ CAREFULLY:
- NO emojis anywhere (not in README, not in reports, not in commit messages)
- NO buzzwords (revolutionary, game-changing, cutting-edge, next-gen, etc.)
- NO fluff or filler text
- NO mock data, NO fake data, NO simulated results - REAL functionality ONLY
- Write like a senior engineer, not a marketing department
- Be direct and factual
- Professional grade output only
- If you can't build something functional, skip it and pick something else

WHAT TO LOOK FOR:
- New open source models with weights available
- Tools, libraries, frameworks you can pip/npm install
- Papers with code implementations on GitHub
- New APIs or SDKs
- Techniques with working demos
- Repos gaining traction fast

IGNORE:
- Corporate news (layoffs, acquisitions, leaks)
- Policy/regulation news
- Opinion pieces and drama
- Vague announcements without code
- Rumors and speculation

Your task:

1. **BUILDABLE TECH** - What new tools/models/techniques were released?
   - Only include things with actual code or weights
   - Note: GitHub link, HuggingFace model, pip package, etc.
   - Skip anything that's just an announcement without substance

2. **TOP PICK** - The ONE most interesting thing you can build with TODAY
   - Must have working code/weights available
   - Explain what it enables (be specific, not vague)
   - Why it matters technically

3. **PROOF OF CONCEPT** - Write working code demonstrating the top pick
   - MUST be functional with REAL data/APIs - NO mock data, NO fake data, NO simulated results
   - If it requires an API, use a real API (free tier or public)
   - If it's a model, actually load and run the model
   - If you can't make it functional, SKIP IT and pick something else
   - Test it and verify it produces real output
   - Clear setup instructions (pip install, etc.)
   - Clean code, no excessive comments

4. **GITHUB** - Create a NEW standalone repo (not nested)
   - IMPORTANT: Create repo in /tmp, NOT in your current working directory
   - Steps:
     ```bash
     cd /tmp
     REPO_NAME="ai-poc-$(date +%Y-%m-%d)-short-description"
     mkdir $REPO_NAME && cd $REPO_NAME
     git init
     # ... add your code files here ...
     git add .
     git commit -m "Initial commit"
     gh repo create $REPO_NAME --public --source=. --push
     ```
   - Each PoC must be its own standalone GitHub repo
   - README format (no emojis, no badges):
     ```
     # [Project Name]

     [One sentence: what this does]

     ## What it demonstrates
     [2-3 sentences on the technical capability]

     ## Setup
     pip install -r requirements.txt

     ## Usage
     python main.py

     ## Output
     [Example output or what to expect]

     ## Source
     [Link to original paper/repo/model]
     ```
   - TEST the code before committing - run it and verify it works

5. **DELIVER REPORT** - Use deliver_report with this EXACT format:

TL;DR: [One sentence summary of what you built and why it matters]

GitHub: [full URL to the repo]

Tweet (copy/paste ready):
[2-3 sentences about what this is and why it's interesting. No emojis. No hashtags. Include the GitHub link at the end.]

Other notable releases today:
- [One liner about release 1]
- [One liner about release 2]

That's it. Nothing else. No greetings, no sign-offs, no extra commentary.

IMPORTANT: You MUST use deliver_report at the end. User only sees what you send via that tool.

WORKFLOW:
1. Scan the data for BUILDABLE items (has code, weights, package, or API)
2. Pick the most impressive one
3. Build a working demo
4. Test it, push to GitHub
5. Use deliver_report tool to send the brief

COLLECTED DATA:
{collected_data}

Begin your research analysis now.
"""


async def start_daily_research(notify_chat_id: str | None = None) -> dict[str, Any]:
    """Start the daily AI research workflow.

    This spawns a coordinator agent that:
    1. Collects data from all sources
    2. Analyzes trends
    3. Writes and tests PoC code
    4. Pushes to GitHub
    5. Generates report
    6. Delivers to user

    Args:
        notify_chat_id: Telegram chat ID to send the final report to
    """
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    # First, collect data from all sources
    collected = await collect_all_sources()

    if collected.get("total_items", 0) == 0:
        return {"error": "Failed to collect any data from sources"}

    # Load the full collected data
    data = await get_collected_data()
    if "error" in data:
        return data

    # Format collected data for the agent
    collected_summary = json.dumps(data, indent=2)

    # Build the task with collected data
    task = DAILY_RESEARCH_PROMPT.format(collected_data=collected_summary)

    # Add history of previous PoCs to avoid repeats
    poc_history = _get_poc_history_context()
    if poc_history:
        task = poc_history + "\n\n" + task

    # Add notification instruction if chat_id provided
    if notify_chat_id:
        task += f"\n\nWhen complete, the report will be sent to Telegram chat {notify_chat_id}."

    try:
        # Spawn the research coordinator agent
        agent = await _orchestrator.spawn_agent(
            task=task,
            name="daily-research",
            supervised=False,
            live_output=False,  # Don't stream raw tool output to user
        )

        return {
            "status": "started",
            "agent_id": agent["id"],
            "data_collected": collected["total_items"],
            "sources": collected["sources"],
            "message": "Daily research started. I'll analyze trends, write a PoC, and deliver your morning report.",
        }
    except Exception as e:
        logger.error(f"Failed to start daily research: {e}")
        return {"error": str(e)}


async def schedule_daily_research(time: str = "06:00", notify_chat_id: str | None = None) -> dict[str, Any]:
    """Schedule daily research to run automatically.

    Args:
        time: Time to run in HH:MM format (24-hour)
        notify_chat_id: Telegram chat ID to send reports to
    """
    if not _orchestrator or not _orchestrator.proactive:
        return {"error": "Proactive engine not available"}

    from gru.proactive import TriggerType

    # Create the trigger
    trigger_id = await _orchestrator.proactive.add_trigger(
        name="daily_ai_research",
        trigger_type=TriggerType.SCHEDULED,
        action=f"research:daily:{notify_chat_id or 'default'}",
        schedule=time,
        config={"notify_chat_id": notify_chat_id},
    )

    return {
        "status": "scheduled",
        "trigger_id": trigger_id,
        "time": time,
        "message": f"Daily AI research scheduled for {time}. You'll receive a report each morning.",
    }


async def cancel_daily_research() -> dict[str, Any]:
    """Cancel the scheduled daily research."""
    if not _orchestrator or not _orchestrator.proactive:
        return {"error": "Proactive engine not available"}

    # Find and remove the trigger
    triggers = await _orchestrator.proactive.list_triggers()

    for trigger in triggers:
        if trigger["name"] == "daily_ai_research":
            await _orchestrator.proactive.remove_trigger(trigger["id"])
            return {"status": "cancelled", "message": "Daily research schedule cancelled."}

    return {"error": "No daily research schedule found"}


# Thresholds for "breaking news" detection
BREAKING_THRESHOLDS = {
    "reddit_score": 500,  # Reddit post with 500+ upvotes
    "hn_score": 200,  # HN story with 200+ points
    "github_stars": 100,  # New repo with 100+ stars in a day
    "hf_likes": 50,  # HF model/space with 50+ likes
}

# Higher thresholds for "groundbreaking" - triggers immediate PoC generation
GROUNDBREAKING_THRESHOLDS = {
    "reddit_score": 2000,  # Viral Reddit post
    "hn_score": 500,  # Top HN story
    "github_stars": 500,  # Exploding repo
    "hf_likes": 200,  # Hot model/space
}

# Keywords that indicate groundbreaking content (in titles)
GROUNDBREAKING_KEYWORDS = [
    "breakthrough",
    "revolutionary",
    "first ever",
    "world first",
    "beats gpt-4",
    "beats claude",
    "beats gemini",
    "surpasses",
    "open source",
    "opensource",
    "released",
    "announcing",
    "state-of-the-art",
    "sota",
    "new model",
    "outperforms",
]

# Keywords that indicate BUILDABLE/TECHNICAL content (what we want)
TECHNICAL_KEYWORDS = [
    # Models and weights
    "open source",
    "opensource",
    "open-source",
    "weights",
    "model release",
    "fine-tune",
    "finetune",
    "lora",
    "qlora",
    "quantized",
    "gguf",
    "ggml",
    # Frameworks and tools
    "framework",
    "library",
    "sdk",
    "api",
    "toolkit",
    "cli",
    "package",
    "pip install",
    "npm install",
    "github.com",
    "huggingface",
    # Techniques
    "implementation",
    "tutorial",
    "how to",
    "guide",
    "code",
    "demo",
    "inference",
    "training",
    "benchmark",
    "eval",
    "dataset",
    # Specific tech
    "llm",
    "diffusion",
    "transformer",
    "embedding",
    "vector",
    "rag",
    "agent",
    "langchain",
    "llamaindex",
    "vllm",
    "ollama",
    "mlx",
    "whisper",
    "stable diffusion",
    "flux",
    "comfyui",
    # Actions
    "released",
    "launching",
    "introducing",
    "announcing",
    "built",
    "created",
    "developed",
    "open-sourced",
]

# Keywords that indicate NON-TECHNICAL content (filter out)
NOISE_KEYWORDS = [
    # Corporate drama
    "lawsuit",
    "sued",
    "suing",
    "legal",
    "court",
    "trial",
    "fired",
    "layoff",
    "layoffs",
    "hiring",
    "hired",
    "ceo",
    "leak",
    "leaked",
    "leaking",
    "whistleblow",
    "secrets",
    "acquisition",
    "acquires",
    "acquired",
    "merger",
    "ipo",
    # Policy and regulation
    "regulation",
    "regulatory",
    "congress",
    "senate",
    "legislation",
    "bill",
    "law",
    "policy",
    "government",
    "eu ai act",
    "executive order",
    # Opinion and drama
    "opinion",
    "editorial",
    "controversy",
    "drama",
    "beef",
    "twitter",
    "tweet",
    "drama",
    "fight",
    "argues",
    "slams",
    "warns",
    "warning",
    "danger",
    "risk",
    "threat",
    "scary",
    # Vague hype
    "could",
    "might",
    "may",
    "possibly",
    "rumor",
    "reportedly",
    "sources say",
    "allegedly",
    "speculation",
]

# Track what we've already notified about (to avoid duplicates)
_notified_items: set[str] = set()
_poc_generated_items: set[str] = set()  # Track items we've already generated PoCs for
_poc_topics: set[str] = set()  # Track topic keywords to avoid similar PoCs

# File to persist PoC history across restarts
POC_HISTORY_FILE = RESEARCH_DIR / "poc_history.txt"


def _normalize_topic(title: str) -> set[str]:
    """Extract normalized keywords from a title for deduplication."""
    import re

    # Remove common words and extract key terms
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "for",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "with",
        "ai",
        "poc",
        "new",
        "first",
    }
    words = re.findall(r"\b[a-z]{3,}\b", title.lower())
    return {w for w in words if w not in stopwords}


def _is_similar_topic(title: str, threshold: float = 0.5) -> bool:
    """Check if a topic is similar to one we've already built a PoC for."""
    if not _poc_topics:
        return False

    title_keywords = _normalize_topic(title)
    if not title_keywords:
        return False

    for existing_keywords in _poc_topics:
        if not existing_keywords:
            continue
        # Check overlap ratio
        existing_set = set(existing_keywords.split(","))
        overlap = len(title_keywords & existing_set)
        max_size = max(len(title_keywords), len(existing_set))
        if max_size > 0 and overlap / max_size >= threshold:
            return True
    return False


def _load_poc_history() -> list[str]:
    """Load list of previously created PoCs."""
    global _poc_topics
    if POC_HISTORY_FILE.exists():
        lines = POC_HISTORY_FILE.read_text().strip().split("\n")
        # Also populate topics set from history
        for line in lines:
            if "|" in line:
                title = line.split("|")[0].strip()
                keywords = _normalize_topic(title)
                if keywords:
                    _poc_topics.add(",".join(sorted(keywords)))
        return lines
    return []


def _save_poc_to_history(title: str, url: str) -> None:
    """Save a PoC to history."""
    global _poc_topics
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(POC_HISTORY_FILE, "a") as f:
        f.write(f"{title} | {url}\n")

    # Also add to in-memory topic tracking
    keywords = _normalize_topic(title)
    if keywords:
        _poc_topics.add(",".join(sorted(keywords)))


def save_poc_from_report(report: str) -> bool:
    """Extract GitHub URL from a PoC report and save to history.

    Called by orchestrator when a PoC agent delivers a report.
    Returns True if a PoC was saved.
    """
    import re

    # Extract GitHub URL from report
    github_match = re.search(r"https://github\.com/[^\s\)>\]]+", report)
    if not github_match:
        return False

    github_url = github_match.group(0).rstrip(".,;:")

    # Extract title from TL;DR line or first line
    title = "Unknown PoC"
    tldr_match = re.search(r"TL;DR:\s*(.+?)(?:\n|$)", report)
    if tldr_match:
        title = tldr_match.group(1).strip()[:100]
    else:
        # Fallback: use repo name from URL
        repo_match = re.search(r"github\.com/[^/]+/([^/\s]+)", github_url)
        if repo_match:
            title = repo_match.group(1)

    _save_poc_to_history(title, github_url)
    logger.info(f"Saved PoC to history: {title} | {github_url}")
    return True


def _get_poc_history_context() -> str:
    """Get context string of previous PoCs for prompts."""
    history = _load_poc_history()
    if not history:
        return ""
    recent = history[-20:]  # Last 20
    return "PREVIOUS POCs (do NOT repeat these topics):\n" + "\n".join(f"- {h}" for h in recent)


def _is_technical_content(title: str, description: str = "") -> bool:
    """Check if content is technical/buildable (not noise)."""
    text = (title + " " + description).lower()

    # First check if it's noise - reject if so
    if any(kw in text for kw in NOISE_KEYWORDS):
        return False

    # Then check if it has technical indicators
    return any(kw in text for kw in TECHNICAL_KEYWORDS)


def _is_buildable(item: dict) -> bool:
    """Check if an item represents something you can actually build with."""
    title = item.get("title", "")
    description = item.get("description", "") or item.get("summary", "") or ""
    url = item.get("url", "").lower()

    # GitHub repos are almost always buildable
    if "github.com" in url:
        return True

    # HuggingFace models/spaces are buildable
    if "huggingface.co" in url:
        return True

    # ArXiv papers with code links
    if "arxiv" in url and ("github" in description.lower() or "code" in description.lower()):
        return True

    # Check content for technical keywords
    return _is_technical_content(title, description)


async def _check_for_breaking_news(collected: dict) -> list[dict]:
    """Analyze collected data for breaking/significant TECHNICAL items only."""
    breaking = []

    # Check Reddit - only technical/buildable posts
    reddit = collected.get("reddit", {})
    for post in reddit.get("posts", []):
        score = post.get("score", 0)
        item_id = f"reddit:{post.get('url', '')}"
        if score >= BREAKING_THRESHOLDS["reddit_score"] and item_id not in _notified_items:
            item = {
                "source": "Reddit",
                "type": "viral_post",
                "title": post.get("title", ""),
                "description": post.get("selftext", ""),
                "url": post.get("url", ""),
                "score": score,
                "why": f"Viral post with {score} upvotes",
            }
            # Only include if it's buildable/technical
            if _is_buildable(item):
                breaking.append(item)
                _notified_items.add(item_id)

    # Check HackerNews - only technical/buildable stories
    hn = collected.get("hackernews", {})
    for story in hn.get("stories", []):
        score = story.get("score", 0)
        item_id = f"hn:{story.get('hn_url', '')}"
        if score >= BREAKING_THRESHOLDS["hn_score"] and item_id not in _notified_items:
            item = {
                "source": "HackerNews",
                "type": "top_story",
                "title": story.get("title", ""),
                "url": story.get("url", ""),
                "score": score,
                "why": f"Top HN story with {score} points",
            }
            if _is_buildable(item):
                breaking.append(item)
                _notified_items.add(item_id)

    # Check GitHub - repos are inherently buildable, always include
    github = collected.get("github", {})
    for repo in github.get("repositories", []):
        stars = repo.get("stars", 0)
        item_id = f"github:{repo.get('url', '')}"
        if stars >= BREAKING_THRESHOLDS["github_stars"] and item_id not in _notified_items:
            breaking.append(
                {
                    "source": "GitHub",
                    "type": "trending_repo",
                    "title": repo.get("name", ""),
                    "description": repo.get("description", "")[:200],
                    "url": repo.get("url", ""),
                    "stars": stars,
                    "language": repo.get("language", ""),
                    "topics": repo.get("topics", []),
                    "why": f"Trending repo with {stars} stars",
                }
            )
            _notified_items.add(item_id)

    # Check Hugging Face - models/spaces are inherently buildable
    hf = collected.get("huggingface", {})
    for item in hf.get("items", []):
        likes = item.get("likes", 0)
        item_id = f"hf:{item.get('url', '')}"
        if likes >= BREAKING_THRESHOLDS["hf_likes"] and item_id not in _notified_items:
            breaking.append(
                {
                    "source": "HuggingFace",
                    "type": "trending_model",
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "likes": likes,
                    "pipeline_tag": item.get("pipeline_tag", ""),
                    "why": f"Trending on HuggingFace with {likes} likes",
                }
            )
            _notified_items.add(item_id)

    # Check for ArXiv papers - only those with code/implementations
    arxiv = collected.get("arxiv", {})
    for paper in arxiv.get("papers", []):
        title = paper.get("title", "")
        summary = paper.get("summary", "")
        item_id = f"arxiv:{paper.get('url', '')}"

        if item_id not in _notified_items:
            # Only include papers that mention code, github, implementation, etc.
            text = (title + " " + summary).lower()
            has_code = any(
                kw in text for kw in ["github", "code", "implementation", "released", "open source", "weights"]
            )
            has_breakthrough = any(
                kw in text for kw in ["state-of-the-art", "sota", "outperforms", "beats", "surpasses"]
            )

            if has_code or has_breakthrough:
                # Double-check it's not noise
                if not any(kw in text for kw in NOISE_KEYWORDS):
                    breaking.append(
                        {
                            "source": "ArXiv",
                            "type": "paper_with_code",
                            "title": title,
                            "description": summary[:300],
                            "url": paper.get("url", ""),
                            "authors": paper.get("authors", [])[:3],
                            "why": "Paper with code/implementation" if has_code else "SOTA paper",
                        }
                    )
                    _notified_items.add(item_id)

    # Skip general tech news - too much noise. Only process if it's about a specific release
    news = collected.get("tech_news", {})
    release_keywords = ["open source", "releases code", "releases model", "launches api", "now available on github"]
    for article in news.get("articles", []):
        title = article.get("title", "")
        summary = article.get("summary", "")
        item_id = f"news:{article.get('url', '')}"

        if item_id not in _notified_items:
            text = (title + " " + summary).lower()
            # Only include actual releases, not announcements or news
            if any(kw in text for kw in release_keywords) and not any(kw in text for kw in NOISE_KEYWORDS):
                breaking.append(
                    {
                        "source": article.get("source", "Tech News"),
                        "type": "release",
                        "title": title,
                        "description": summary[:200],
                        "url": article.get("url", ""),
                        "why": "New release/tool available",
                    }
                )
                _notified_items.add(item_id)

    return breaking


def _is_groundbreaking(item: dict) -> bool:
    """Check if an item is groundbreaking (warrants immediate PoC)."""
    title = item.get("title", "").lower()
    description = item.get("description", "").lower()

    # Check for groundbreaking keywords
    has_keyword = any(kw in title or kw in description for kw in GROUNDBREAKING_KEYWORDS)

    # Check for exceptionally high metrics
    score = item.get("score", 0)
    stars = item.get("stars", 0)
    likes = item.get("likes", 0)

    high_engagement = (
        score >= GROUNDBREAKING_THRESHOLDS.get("reddit_score", 2000)
        or score >= GROUNDBREAKING_THRESHOLDS.get("hn_score", 500)
        or stars >= GROUNDBREAKING_THRESHOLDS.get("github_stars", 500)
        or likes >= GROUNDBREAKING_THRESHOLDS.get("hf_likes", 200)
    )

    # Groundbreaking if high engagement OR has groundbreaking keywords with decent engagement
    return high_engagement or (has_keyword and (score >= 100 or stars >= 50 or likes >= 25))


async def _generate_groundbreaking_poc(item: dict) -> dict[str, Any]:
    """Spawn an agent to immediately analyze and create a PoC for a groundbreaking item."""
    if not _orchestrator:
        return {"error": "Orchestrator not available"}

    title = item.get("title", "")
    item_id = f"{item.get('source', 'unknown')}:{item.get('url', '')}"

    # Check if we already processed this exact item
    if item_id in _poc_generated_items:
        return {"status": "already_generated", "item": title}

    # Load history if not loaded (populates _poc_topics)
    if not _poc_topics:
        _load_poc_history()

    # Check if this is similar to a topic we've already built
    if _is_similar_topic(title):
        logger.info(f"Skipping similar topic: {title}")
        _poc_generated_items.add(item_id)  # Mark as processed to avoid retry
        return {"status": "similar_topic_exists", "item": title}

    _poc_generated_items.add(item_id)

    # Add this topic to tracking
    keywords = _normalize_topic(title)
    if keywords:
        _poc_topics.add(",".join(sorted(keywords)))

    # Get history of previous PoCs to avoid repeats
    poc_history = _get_poc_history_context()
    history_section = f"{poc_history}\n\n" if poc_history else ""

    task = f"""{history_section}Significant AI development detected. Analyze and create a PoC quickly.

STYLE REQUIREMENTS:
- NO emojis anywhere
- NO buzzwords (revolutionary, game-changing, cutting-edge, etc.)
- Write like a senior engineer
- Be direct and factual
- Professional output only

ITEM DETAILS:
- Source: {item.get("source", "Unknown")}
- Title: {item.get("title", "Unknown")}
- URL: {item.get("url", "")}
- Context: {item.get("why", "High engagement")}
- Details: {item.get("description", "")[:500] if item.get("description") else "N/A"}

TASKS:
1. ANALYZE - What is this? What does it actually do?
2. VERIFY - Is this legit? What's the real technical contribution?
3. POC CODE - Write working demo code
   - MUST use REAL data/APIs - NO mock data, NO fake data, NO simulations
   - If it's a model, actually download and run it
   - If it needs an API, use a real one (free tier or public)
   - If you can't make it functional with real data, SKIP and report that
   - Test it, run it, verify real output
   - Clear setup instructions
   - Clean code, minimal comments
4. GITHUB - Create NEW standalone repo in /tmp (not nested):
   ```bash
   cd /tmp
   REPO_NAME="ai-poc-$(date +%Y-%m-%d)-short-name"
   mkdir $REPO_NAME && cd $REPO_NAME
   git init
   # add code files
   git add .
   git commit -m "Initial commit"
   gh repo create $REPO_NAME --public --source=. --push
   ```
   - TEST the code before committing
   - README (no emojis, no badges):
     # [Name]
     [One sentence]
     ## What it does
     ## Setup
     ## Usage
     ## Source
5. REPORT - Use deliver_report with this EXACT format:

TL;DR: [One sentence - what this is and why it matters]

GitHub: [full URL]

Tweet (copy/paste ready):
[2-3 sentences about what this is, why it's interesting, what it enables. No emojis. No hashtags. Include the GitHub link.]

That's it. Nothing else.

You MUST use deliver_report. User only sees what you send via that tool.
"""

    try:
        agent = await _orchestrator.spawn_agent(
            task=task,
            name="groundbreaking-poc",
            supervised=False,
            live_output=False,  # Don't stream raw tool output
        )

        # Notify that we're working on it
        if _orchestrator._notify_callback:
            _orchestrator._notify_callback(
                "proactive",
                f"Found something interesting: {item['title']}\n\nLooking into it now, will send you a summary shortly.",
            )

        return {
            "status": "poc_generation_started",
            "agent_id": agent["id"],
            "item": item["title"],
        }
    except Exception as e:
        logger.error(f"Failed to start groundbreaking PoC: {e}")
        return {"error": str(e)}


async def _notify_breaking_news(breaking_items: list[dict], chat_id: str | None = None) -> None:
    """Send notifications for breaking news items."""
    if not _orchestrator or not breaking_items:
        return

    for item in breaking_items[:5]:  # Limit to top 5 to avoid spam
        message = f"[{item['source']}] {item['title']}\n\n"
        message += f"{item['why']}\n"
        message += f"Link: {item['url']}"

        if _orchestrator._notify_callback:
            _orchestrator._notify_callback("proactive", message)

        await asyncio.sleep(1)  # Small delay between notifications


async def check_breaking_news(notify: bool = True) -> dict[str, Any]:
    """Check all sources for breaking news right now.

    Args:
        notify: Whether to send notifications for findings
    """
    # Collect fresh data
    collected = await collect_all_sources()

    if "error" in collected:
        return collected

    # Check for breaking items
    breaking = await _check_for_breaking_news(
        await get_collected_data()  # Get the full data, not just summary
    )

    groundbreaking_pocs = []

    if breaking:
        # Check for groundbreaking items that warrant immediate PoC generation
        for item in breaking:
            if _is_groundbreaking(item):
                logger.info(f"GROUNDBREAKING item detected: {item.get('title', 'Unknown')}")
                poc_result = await _generate_groundbreaking_poc(item)
                groundbreaking_pocs.append(poc_result)

        # Notify about regular breaking news (non-groundbreaking)
        if notify:
            regular_breaking = [i for i in breaking if not _is_groundbreaking(i)]
            if regular_breaking:
                await _notify_breaking_news(regular_breaking)

    return {
        "status": "checked",
        "breaking_items": len(breaking),
        "groundbreaking_items": len(groundbreaking_pocs),
        "poc_generations": groundbreaking_pocs,
        "items": breaking,
        "total_sources_checked": collected.get("total_items", 0),
    }


async def start_realtime_monitoring(interval_hours: int = 2, notify_chat_id: str | None = None) -> dict[str, Any]:
    """Start continuous monitoring for AI news. Checks every few hours and alerts on breaking news.

    Args:
        interval_hours: How often to check (default: every 2 hours)
        notify_chat_id: Telegram chat ID for notifications
    """
    if not _orchestrator or not _orchestrator.proactive:
        return {"error": "Proactive engine not available"}

    from gru.proactive import TriggerType

    # Create interval trigger for continuous monitoring
    trigger_id = await _orchestrator.proactive.add_trigger(
        name="realtime_ai_monitor",
        trigger_type=TriggerType.INTERVAL,
        action=f"research:monitor:{notify_chat_id or 'default'}",
        interval_minutes=interval_hours * 60,
        config={"notify_chat_id": notify_chat_id, "interval_hours": interval_hours},
    )

    # Also do an immediate check
    initial_check = await check_breaking_news(notify=True)

    return {
        "status": "monitoring_started",
        "trigger_id": trigger_id,
        "interval_hours": interval_hours,
        "message": f"Real-time monitoring active. Checking every {interval_hours} hours. You'll be notified immediately when something significant happens.",
        "initial_check": initial_check,
    }


async def stop_realtime_monitoring() -> dict[str, Any]:
    """Stop the real-time monitoring."""
    if not _orchestrator or not _orchestrator.proactive:
        return {"error": "Proactive engine not available"}

    triggers = await _orchestrator.proactive.list_triggers()

    for trigger in triggers:
        if trigger["name"] == "realtime_ai_monitor":
            await _orchestrator.proactive.remove_trigger(trigger["id"])
            return {"status": "stopped", "message": "Real-time monitoring stopped."}

    return {"error": "No real-time monitoring active"}


async def get_monitoring_status() -> dict[str, Any]:
    """Get current status of research monitoring."""
    if not _orchestrator or not _orchestrator.proactive:
        return {"error": "Proactive engine not available"}

    triggers = await _orchestrator.proactive.list_triggers()

    status = {
        "daily_research": None,
        "realtime_monitoring": None,
        "notified_items_count": len(_notified_items),
    }

    for trigger in triggers:
        if trigger["name"] == "daily_ai_research":
            status["daily_research"] = {
                "active": True,
                "schedule": trigger.get("schedule"),
                "last_fired": trigger.get("last_fired"),
            }
        elif trigger["name"] == "realtime_ai_monitor":
            status["realtime_monitoring"] = {
                "active": True,
                "last_fired": trigger.get("last_fired"),
                "fire_count": trigger.get("fire_count", 0),
            }

    return status


def register_research_tools() -> None:
    """Register research tools."""

    register_tool(
        name="fetch_reddit_ai",
        description="Fetch top posts from AI-related subreddits (MachineLearning, LocalLLaMA, etc.)",
        parameters={
            "subreddits": {
                "type": "string",
                "description": "Comma-separated subreddit names (default: MachineLearning,LocalLLaMA,artificial,singularity)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Number of posts per subreddit (default: 25)",
                "optional": True,
            },
            "timeframe": {
                "type": "string",
                "description": "Time filter: hour, day, week, month, year, all (default: day)",
                "optional": True,
            },
        },
        handler=fetch_reddit_ai,
    )

    register_tool(
        name="fetch_github_trending",
        description="Fetch trending repositories from GitHub, especially AI/ML related.",
        parameters={
            "language": {
                "type": "string",
                "description": "Filter by programming language (empty for all)",
                "optional": True,
            },
            "since": {
                "type": "string",
                "description": "Time range: daily, weekly, monthly (default: daily)",
                "optional": True,
            },
        },
        handler=fetch_github_trending,
    )

    register_tool(
        name="fetch_hackernews_top",
        description="Fetch top AI-related stories from Hacker News.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum stories to fetch (default: 30)",
                "optional": True,
            },
            "min_score": {
                "type": "integer",
                "description": "Minimum score filter (default: 50)",
                "optional": True,
            },
        },
        handler=fetch_hackernews_top,
    )

    register_tool(
        name="fetch_arxiv_ai",
        description="Fetch recent AI/ML papers from ArXiv.",
        parameters={
            "query": {
                "type": "string",
                "description": "ArXiv search query (default: AI/ML/NLP categories)",
                "optional": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum papers to fetch (default: 20)",
                "optional": True,
            },
        },
        handler=fetch_arxiv_ai,
    )

    register_tool(
        name="fetch_x_ai",
        description="Fetch AI-related posts from X/Twitter via Nitter (public frontend). Monitors key AI accounts.",
        parameters={
            "accounts": {
                "type": "string",
                "description": "Comma-separated X/Twitter handles to monitor (default: top AI accounts)",
                "optional": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum posts to fetch (default: 20)",
                "optional": True,
            },
        },
        handler=fetch_x_ai,
    )

    register_tool(
        name="fetch_huggingface_trending",
        description="Fetch trending models and spaces from Hugging Face.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum items to fetch (default: 20)",
                "optional": True,
            },
        },
        handler=fetch_huggingface_trending,
    )

    register_tool(
        name="fetch_producthunt_ai",
        description="Fetch AI product launches from Product Hunt.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum products to fetch (default: 15)",
                "optional": True,
            },
        },
        handler=fetch_producthunt_ai,
    )

    register_tool(
        name="fetch_tech_news",
        description="Fetch AI news from tech news RSS feeds (TechCrunch, VentureBeat, MIT Tech Review, The Verge, Ars Technica).",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum articles to fetch (default: 20)",
                "optional": True,
            },
        },
        handler=fetch_tech_news,
    )

    register_tool(
        name="fetch_ai_newsletters",
        description="Fetch content from AI newsletters (Import AI, The Batch, AI Weekly, Last Week in AI).",
        parameters={},
        handler=fetch_ai_newsletters,
    )

    register_tool(
        name="collect_all_sources",
        description="Collect data from ALL research sources in parallel: Reddit, GitHub, HackerNews, ArXiv, X/Twitter, Hugging Face, Product Hunt, tech news, and AI newsletters.",
        parameters={},
        handler=collect_all_sources,
    )

    register_tool(
        name="start_daily_research",
        description="Start the daily AI research workflow. Collects data, analyzes trends, writes PoC code, pushes to GitHub, and generates a morning report.",
        parameters={
            "notify_chat_id": {
                "type": "string",
                "description": "Telegram chat ID to send the final report to",
                "optional": True,
            },
        },
        handler=start_daily_research,
    )

    register_tool(
        name="schedule_daily_research",
        description="Schedule daily AI research to run automatically at a specific time.",
        parameters={
            "time": {
                "type": "string",
                "description": "Time to run in HH:MM format, 24-hour (default: 06:00)",
                "optional": True,
            },
            "notify_chat_id": {
                "type": "string",
                "description": "Telegram chat ID to send reports to",
                "optional": True,
            },
        },
        handler=schedule_daily_research,
    )

    register_tool(
        name="cancel_daily_research",
        description="Cancel the scheduled daily research.",
        parameters={},
        handler=cancel_daily_research,
    )

    register_tool(
        name="get_collected_data",
        description="Get previously collected research data.",
        parameters={
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format (default: today)",
                "optional": True,
            },
        },
        handler=get_collected_data,
    )

    register_tool(
        name="save_research_report",
        description="Save a research report.",
        parameters={
            "report": {
                "type": "string",
                "description": "The report content (markdown)",
            },
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format (default: today)",
                "optional": True,
            },
        },
        handler=save_research_report,
    )

    register_tool(
        name="get_research_report",
        description="Get a saved research report.",
        parameters={
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format (default: today)",
                "optional": True,
            },
        },
        handler=get_research_report,
    )

    register_tool(
        name="check_breaking_news",
        description="Check all sources right now for breaking AI news. Sends notifications for significant findings.",
        parameters={
            "notify": {
                "type": "boolean",
                "description": "Whether to send notifications for findings (default: true)",
                "optional": True,
            },
        },
        handler=check_breaking_news,
    )

    register_tool(
        name="start_realtime_monitoring",
        description="Start continuous monitoring for AI news. Checks every few hours and alerts on breaking news.",
        parameters={
            "interval_hours": {
                "type": "integer",
                "description": "How often to check in hours (default: 2)",
                "optional": True,
            },
            "notify_chat_id": {
                "type": "string",
                "description": "Telegram chat ID for notifications",
                "optional": True,
            },
        },
        handler=start_realtime_monitoring,
    )

    register_tool(
        name="stop_realtime_monitoring",
        description="Stop the real-time AI news monitoring.",
        parameters={},
        handler=stop_realtime_monitoring,
    )

    register_tool(
        name="get_monitoring_status",
        description="Get current status of research monitoring (daily research schedule and real-time monitoring).",
        parameters={},
        handler=get_monitoring_status,
    )
