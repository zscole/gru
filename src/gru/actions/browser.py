"""Browser abstraction using Playwright for headless/headed automation."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Lazy import playwright to allow module loading without it installed
_playwright_available = False
try:
    from playwright.async_api import (
        Browser as PlaywrightBrowser,
        BrowserContext,
        Page,
        async_playwright,
    )
    _playwright_available = True
except ImportError:
    PlaywrightBrowser = Any
    BrowserContext = Any
    Page = Any
    async_playwright = None

if TYPE_CHECKING:
    from playwright.async_api import Playwright

logger = logging.getLogger(__name__)


def _check_playwright() -> None:
    """Check if playwright is available, raise helpful error if not."""
    if not _playwright_available:
        raise ImportError(
            "playwright is required for browser automation. "
            "Install it with: pip install playwright && playwright install"
        )


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""

    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    timeout: int = 30000  # ms
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str | None = None
    proxy: str | None = None
    storage_dir: Path | None = None  # For persistent sessions
    slow_mo: int = 0  # Slow down actions for debugging (ms)
    screenshots_dir: Path | None = None  # Save screenshots on errors

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> BrowserConfig:
        """Create config from environment variables."""
        import os

        headless_str = os.getenv("GRU_BROWSER_MODE", "headless")
        headless = headless_str.lower() != "headed"

        config = cls(
            headless=headless,
            browser_type=os.getenv("GRU_BROWSER_TYPE", "chromium"),
            timeout=int(os.getenv("GRU_BROWSER_TIMEOUT", "30000")),
            slow_mo=int(os.getenv("GRU_BROWSER_SLOW_MO", "0")),
        )

        if data_dir:
            config.storage_dir = data_dir / "browser_sessions"
            config.screenshots_dir = data_dir / "screenshots"

        return config


class Browser:
    """Browser automation wrapper with session management."""

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: PlaywrightBrowser | None = None
        self._contexts: dict[str, BrowserContext] = {}
        self._default_context: BrowserContext | None = None

    async def start(self) -> None:
        """Start the browser."""
        _check_playwright()

        if self._browser:
            return

        self._playwright = await async_playwright().start()

        # Select browser type
        browser_types = {
            "chromium": self._playwright.chromium,
            "firefox": self._playwright.firefox,
            "webkit": self._playwright.webkit,
        }
        browser_type = browser_types.get(self.config.browser_type, self._playwright.chromium)

        # Launch options
        launch_options: dict[str, Any] = {
            "headless": self.config.headless,
            "slow_mo": self.config.slow_mo,
        }

        if self.config.proxy:
            launch_options["proxy"] = {"server": self.config.proxy}

        self._browser = await browser_type.launch(**launch_options)
        logger.info(f"Browser started: {self.config.browser_type} (headless={self.config.headless})")

        # Create directories if needed
        if self.config.storage_dir:
            self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        if self.config.screenshots_dir:
            self.config.screenshots_dir.mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        """Stop the browser and save sessions."""
        for name, context in self._contexts.items():
            await self._save_context_state(name, context)
            await context.close()
        self._contexts.clear()

        if self._default_context:
            await self._default_context.close()
            self._default_context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser stopped")

    async def get_context(self, name: str | None = None) -> BrowserContext:
        """Get or create a browser context.

        Named contexts persist their state (cookies, storage) between sessions.
        Use named contexts for services that require login (e.g., "ubereats", "doordash").
        """
        if not self._browser:
            await self.start()

        if name is None:
            if not self._default_context:
                self._default_context = await self._create_context()
            return self._default_context

        if name not in self._contexts:
            self._contexts[name] = await self._create_context(name)

        return self._contexts[name]

    async def _create_context(self, name: str | None = None) -> BrowserContext:
        """Create a new browser context, optionally loading saved state."""
        if not self._browser:
            raise RuntimeError("Browser not started")

        context_options: dict[str, Any] = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        }

        if self.config.user_agent:
            context_options["user_agent"] = self.config.user_agent

        # Load saved state if available
        if name and self.config.storage_dir:
            state_file = self.config.storage_dir / f"{name}_state.json"
            if state_file.exists():
                try:
                    context_options["storage_state"] = str(state_file)
                    logger.info(f"Loaded saved state for context: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load state for {name}: {e}")

        return await self._browser.new_context(**context_options)

    async def _save_context_state(self, name: str, context: BrowserContext) -> None:
        """Save context state for persistence."""
        if not self.config.storage_dir:
            return

        state_file = self.config.storage_dir / f"{name}_state.json"
        try:
            state = await context.storage_state()
            state_file.write_text(json.dumps(state, indent=2))
            logger.info(f"Saved state for context: {name}")
        except Exception as e:
            logger.warning(f"Failed to save state for {name}: {e}")

    async def new_page(self, context_name: str | None = None) -> Page:
        """Create a new page in the specified context."""
        context = await self.get_context(context_name)
        page = await context.new_page()
        page.set_default_timeout(self.config.timeout)
        return page

    async def screenshot(self, page: Page, name: str) -> Path | None:
        """Take a screenshot and save it."""
        if not self.config.screenshots_dir:
            return None

        path = self.config.screenshots_dir / f"{name}.png"
        await page.screenshot(path=str(path))
        logger.info(f"Screenshot saved: {path}")
        return path

    async def run_with_page(
        self,
        callback,
        context_name: str | None = None,
        screenshot_on_error: bool = True,
    ):
        """Run a callback with a page, handling cleanup and errors.

        Usage:
            async def do_something(page):
                await page.goto("https://example.com")
                return await page.title()

            result = await browser.run_with_page(do_something, "mycontext")
        """
        page = await self.new_page(context_name)
        try:
            return await callback(page)
        except Exception as e:
            if screenshot_on_error:
                import time
                await self.screenshot(page, f"error_{int(time.time())}")
            raise
        finally:
            await page.close()

    def is_running(self) -> bool:
        """Check if browser is running."""
        return self._browser is not None

    async def clear_context(self, name: str) -> bool:
        """Clear a context's saved state and cookies."""
        if name in self._contexts:
            await self._contexts[name].close()
            del self._contexts[name]

        if self.config.storage_dir:
            state_file = self.config.storage_dir / f"{name}_state.json"
            if state_file.exists():
                state_file.unlink()
                logger.info(f"Cleared state for context: {name}")
                return True

        return False

    async def list_contexts(self) -> list[str]:
        """List all saved context names."""
        contexts = list(self._contexts.keys())

        if self.config.storage_dir and self.config.storage_dir.exists():
            for state_file in self.config.storage_dir.glob("*_state.json"):
                name = state_file.stem.replace("_state", "")
                if name not in contexts:
                    contexts.append(name)

        return contexts


# Singleton browser instance for shared use
_browser_instance: Browser | None = None


async def get_browser(config: BrowserConfig | None = None) -> Browser:
    """Get the shared browser instance."""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = Browser(config)
    return _browser_instance


async def shutdown_browser() -> None:
    """Shutdown the shared browser instance."""
    global _browser_instance
    if _browser_instance:
        await _browser_instance.stop()
        _browser_instance = None
