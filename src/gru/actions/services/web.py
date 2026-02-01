"""Generic web interaction actions."""

from __future__ import annotations

import asyncio
import logging

from gru.actions.base import Action, ActionContext, ActionResult

logger = logging.getLogger(__name__)


class NavigateAction(Action):
    """Navigate to a URL."""

    name = "navigate"
    description = "Navigate to a URL in the browser"
    category = "web"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        url = params.get("url")
        if not url:
            return False, "url is required"
        if not url.startswith(("http://", "https://")):
            return False, "url must start with http:// or https://"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        url = params["url"]
        wait_until = params.get("wait_until", "domcontentloaded")
        context_name = params.get("context_name")

        async def navigate(page):
            await page.goto(url, wait_until=wait_until)
            title = await page.title()
            return {
                "url": page.url,
                "title": title,
            }

        try:
            result = await context.browser.run_with_page(navigate, context_name)
            return ActionResult.success_result(
                f"Navigated to {result['title']}",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Navigation failed: {e}")


class ClickAction(Action):
    """Click an element on the page."""

    name = "click"
    description = "Click an element using a selector"
    category = "web"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("selector"):
            return False, "selector is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        selector = params["selector"]
        url = params.get("url")
        context_name = params.get("context_name")

        async def click(page):
            if url:
                await page.goto(url, wait_until="domcontentloaded")

            await page.click(selector)
            await page.wait_for_load_state("domcontentloaded")

            return {
                "clicked": selector,
                "url": page.url,
            }

        try:
            result = await context.browser.run_with_page(click, context_name)
            return ActionResult.success_result(
                f"Clicked {selector}",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Click failed: {e}")


class TypeAction(Action):
    """Type text into an input field."""

    name = "type_text"
    description = "Type text into an input field"
    category = "web"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("selector"):
            return False, "selector is required"
        if params.get("text") is None:
            return False, "text is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        selector = params["selector"]
        text = params["text"]
        url = params.get("url")
        clear_first = params.get("clear_first", True)
        press_enter = params.get("press_enter", False)
        context_name = params.get("context_name")

        async def type_text(page):
            if url:
                await page.goto(url, wait_until="domcontentloaded")

            if clear_first:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)

            if press_enter:
                await page.press(selector, "Enter")
                await page.wait_for_load_state("domcontentloaded")

            return {
                "typed": text[:50] + "..." if len(text) > 50 else text,
                "selector": selector,
                "url": page.url,
            }

        try:
            result = await context.browser.run_with_page(type_text, context_name)
            return ActionResult.success_result(
                f"Typed text into {selector}",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Type failed: {e}")


class ExtractAction(Action):
    """Extract content from a page."""

    name = "extract"
    description = "Extract text or data from a page"
    category = "web"

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        url = params.get("url")
        selector = params.get("selector")
        extract_type = params.get("type", "text")  # text, html, attribute
        attribute = params.get("attribute")
        multiple = params.get("multiple", False)
        context_name = params.get("context_name")

        async def extract(page):
            if url:
                await page.goto(url, wait_until="domcontentloaded")

            if selector:
                if multiple:
                    elements = await page.query_selector_all(selector)
                    results = []
                    for el in elements:
                        if extract_type == "text":
                            results.append(await el.inner_text())
                        elif extract_type == "html":
                            results.append(await el.inner_html())
                        elif extract_type == "attribute" and attribute:
                            results.append(await el.get_attribute(attribute))
                    return {"extracted": results, "count": len(results)}
                else:
                    element = await page.query_selector(selector)
                    if not element:
                        return {"extracted": None, "error": "Element not found"}

                    if extract_type == "text":
                        text = await element.inner_text()
                    elif extract_type == "html":
                        text = await element.inner_html()
                    elif extract_type == "attribute" and attribute:
                        text = await element.get_attribute(attribute)
                    else:
                        text = await element.inner_text()

                    return {"extracted": text}
            else:
                # Extract full page
                if extract_type == "text":
                    body = await page.query_selector("body")
                    text = await body.inner_text() if body else ""
                    return {"extracted": text[:5000]}  # Limit size
                elif extract_type == "html":
                    html = await page.content()
                    return {"extracted": html[:10000]}

            return {"extracted": None}

        try:
            result = await context.browser.run_with_page(extract, context_name)
            return ActionResult.success_result(
                "Extracted content",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Extract failed: {e}")


class ScreenshotAction(Action):
    """Take a screenshot of a page."""

    name = "screenshot"
    description = "Take a screenshot of a page or element"
    category = "web"

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        url = params.get("url")
        selector = params.get("selector")
        name = params.get("name", "screenshot")
        full_page = params.get("full_page", False)
        context_name = params.get("context_name")

        async def screenshot(page):
            if url:
                await page.goto(url, wait_until="networkidle")

            if selector:
                element = await page.query_selector(selector)
                if element:
                    path = await context.browser.screenshot(page, name)
                    await element.screenshot(path=str(path) if path else None)
                    return {"path": str(path) if path else None, "element": selector}

            path = await context.browser.screenshot(page, name)
            return {"path": str(path) if path else None, "full_page": full_page}

        try:
            result = await context.browser.run_with_page(screenshot, context_name)
            return ActionResult.success_result(
                "Screenshot saved",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Screenshot failed: {e}")


class WaitAction(Action):
    """Wait for a condition or duration."""

    name = "wait"
    description = "Wait for an element, navigation, or duration"
    category = "web"

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        wait_type = params.get("type", "duration")  # duration, selector, navigation
        duration = params.get("duration", 1000)  # ms
        selector = params.get("selector")
        url = params.get("url")
        context_name = params.get("context_name")

        if wait_type == "duration":
            await asyncio.sleep(duration / 1000)
            return ActionResult.success_result(f"Waited {duration}ms")

        async def wait(page):
            if url:
                await page.goto(url, wait_until="domcontentloaded")

            if wait_type == "selector" and selector:
                await page.wait_for_selector(selector)
                return {"waited_for": selector}
            elif wait_type == "navigation":
                await page.wait_for_load_state("networkidle")
                return {"url": page.url}

            return {}

        try:
            result = await context.browser.run_with_page(wait, context_name)
            return ActionResult.success_result("Wait completed", result)
        except Exception as e:
            return ActionResult.error_result(f"Wait failed: {e}")
