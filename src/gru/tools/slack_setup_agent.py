"""Automated Slack setup agent.

Uses Playwright to automate Slack app creation and OAuth token retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_slack_setup_agent(app_name: str = "Gru Assistant") -> dict:
    """Fully automated Slack app setup.

    Creates a Slack app, configures OAuth scopes, installs to workspace,
    and extracts the user token.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}

    data_dir = Path(os.path.expanduser("~/gru/data"))
    if not data_dir.exists():
        data_dir = Path(os.path.expanduser("~/.gru"))
    data_dir.mkdir(parents=True, exist_ok=True)

    token_path = data_dir / "slack_user_token.json"

    results = {
        "steps_completed": [],
        "errors": [],
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            channel="chrome",
        )

        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Step 1: Go to Slack API
            logger.info("Navigating to Slack API...")
            await page.goto("https://api.slack.com/apps", wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            # Check if we need to log in
            if "slack.com/signin" in page.url or "slack.com/workspace-signin" in page.url:
                results["status"] = "login_required"
                results["message"] = "Please log in to Slack in the browser window. I'll wait."

                # Wait for user to log in (up to 5 minutes)
                try:
                    await page.wait_for_url("**/api.slack.com/**", timeout=300000)
                    results["steps_completed"].append("slack_login")
                except Exception:
                    return {
                        "status": "timeout",
                        "error": "Login timeout. Please try again.",
                    }

            results["steps_completed"].append("api_accessed")
            await page.wait_for_timeout(2000)

            # Step 2: Create new app
            logger.info("Creating Slack app...")

            # Click "Create New App"
            create_btn = await page.query_selector('button:has-text("Create New App")')
            if not create_btn:
                create_btn = await page.query_selector('a:has-text("Create New App")')

            if create_btn:
                await create_btn.click()
                await page.wait_for_timeout(2000)

                # Select "From scratch"
                from_scratch = await page.query_selector('button:has-text("From scratch")')
                if from_scratch:
                    await from_scratch.click()
                    await page.wait_for_timeout(1500)

                # Fill app name
                name_input = await page.query_selector('input[placeholder*="App Name"]')
                if not name_input:
                    name_input = await page.query_selector('input[name="name"]')
                if name_input:
                    await name_input.fill(app_name)
                    await page.wait_for_timeout(500)

                # Select workspace from dropdown
                workspace_select = await page.query_selector('select')
                if workspace_select:
                    # Get first option value
                    options = await workspace_select.query_selector_all('option')
                    if len(options) > 1:
                        value = await options[1].get_attribute('value')
                        if value:
                            await workspace_select.select_option(value)

                # Click Create App
                create_app_btn = await page.query_selector('button:has-text("Create App")')
                if create_app_btn:
                    await create_app_btn.click()
                    await page.wait_for_timeout(3000)
                    results["steps_completed"].append("app_created")

            # Step 3: Configure OAuth scopes
            logger.info("Configuring OAuth scopes...")

            # Navigate to OAuth & Permissions
            oauth_link = await page.query_selector('a:has-text("OAuth & Permissions")')
            if oauth_link:
                await oauth_link.click()
                await page.wait_for_timeout(2000)
            else:
                # Try direct navigation
                current_url = page.url
                if "/apps/" in current_url:
                    app_id = current_url.split("/apps/")[1].split("/")[0].split("?")[0]
                    await page.goto(f"https://api.slack.com/apps/{app_id}/oauth")
                    await page.wait_for_timeout(2000)

            # Add User Token Scopes
            user_scopes = [
                "channels:history",
                "channels:read",
                "groups:history",
                "groups:read",
                "im:history",
                "im:read",
                "mpim:history",
                "mpim:read",
                "users:read",
            ]

            # Find and click "Add an OAuth Scope" under User Token Scopes
            user_scopes_section = await page.query_selector('text=User Token Scopes')
            if user_scopes_section:
                # Scroll to it
                await user_scopes_section.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)

            add_scope_btn = await page.query_selector('button:has-text("Add an OAuth Scope")')

            # Try to add each scope
            for scope in user_scopes:
                try:
                    # Click add scope button
                    add_buttons = await page.query_selector_all('button:has-text("Add an OAuth Scope")')
                    if len(add_buttons) > 1:
                        # Second one is usually for user scopes
                        await add_buttons[1].click()
                    elif add_buttons:
                        await add_buttons[0].click()

                    await page.wait_for_timeout(500)

                    # Type the scope
                    scope_input = await page.query_selector('input[placeholder*="Search"]')
                    if scope_input:
                        await scope_input.fill(scope)
                        await page.wait_for_timeout(500)

                        # Click the matching option
                        scope_option = await page.query_selector(f'div[data-qa*="{scope}"]')
                        if not scope_option:
                            scope_option = await page.query_selector(f'text={scope}')
                        if scope_option:
                            await scope_option.click()
                            await page.wait_for_timeout(300)

                except Exception as e:
                    results["errors"].append(f"Failed to add scope {scope}: {e}")

            results["steps_completed"].append("scopes_configured")

            # Step 4: Install to workspace
            logger.info("Installing to workspace...")

            install_btn = await page.query_selector('button:has-text("Install to Workspace")')
            if not install_btn:
                install_btn = await page.query_selector('a:has-text("Install to Workspace")')

            if install_btn:
                await install_btn.click()
                await page.wait_for_timeout(3000)

                # Click Allow on the OAuth consent
                allow_btn = await page.query_selector('button:has-text("Allow")')
                if allow_btn:
                    await allow_btn.click()
                    await page.wait_for_timeout(3000)

                results["steps_completed"].append("installed_to_workspace")

            # Step 5: Extract user token
            logger.info("Extracting user token...")

            # Navigate back to OAuth page to get the token
            await page.wait_for_timeout(2000)

            # Look for the user token on the page
            page_content = await page.content()

            import re
            token_match = re.search(r'(xoxp-[a-zA-Z0-9-]+)', page_content)

            if token_match:
                user_token = token_match.group(1)

                # Save token
                token_data = {
                    "token": user_token,
                    "app_name": app_name,
                }
                token_path.write_text(json.dumps(token_data, indent=2))

                results["steps_completed"].append("token_extracted")
                results["status"] = "success"
                results["message"] = "Slack setup complete! I can now read your DMs, mentions, and channels."
            else:
                # Token might be hidden, try clicking to reveal
                token_section = await page.query_selector('text=User OAuth Token')
                if token_section:
                    await token_section.scroll_into_view_if_needed()

                    # Look for copy button or reveal button
                    copy_btn = await page.query_selector('button[aria-label*="Copy"]')
                    if copy_btn:
                        await copy_btn.click()
                        # Token should be in clipboard but we can't easily access it

                results["status"] = "partial"
                results["message"] = (
                    "App created and installed but couldn't auto-extract token. "
                    "Copy the 'User OAuth Token' from the page and send it to me."
                )

        except Exception as e:
            logger.error(f"Slack setup error: {e}")
            results["status"] = "error"
            results["error"] = str(e)

            # Take screenshot for debugging
            screenshot_path = "/tmp/slack_setup_error.png"
            await page.screenshot(path=screenshot_path)
            results["screenshot"] = screenshot_path

        finally:
            await browser.close()

    return results


async def check_slack_token_valid(token: str) -> dict:
    """Verify a Slack token is valid."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {token}"},
            )
            data = resp.json()

            if data.get("ok"):
                return {
                    "valid": True,
                    "user": data.get("user"),
                    "team": data.get("team"),
                    "user_id": data.get("user_id"),
                }
            else:
                return {
                    "valid": False,
                    "error": data.get("error"),
                }
    except Exception as e:
        return {"valid": False, "error": str(e)}
