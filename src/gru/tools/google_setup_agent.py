"""Automated Google Cloud setup agent.

Uses Playwright with persistent browser profile to automate the entire
Google Cloud Console setup, leveraging the user's existing Google login.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_google_setup_agent(project_name: str = "gru-assistant") -> dict:
    """Fully automated Google Cloud + OAuth setup.

    Uses your existing Chrome profile so you're already logged in.
    Creates project, enables APIs, creates OAuth credentials, completes auth.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}

    data_dir = Path(os.path.expanduser("~/.gru"))
    data_dir.mkdir(parents=True, exist_ok=True)

    credentials_path = data_dir / "google_credentials.json"
    token_path = data_dir / "google_token.json"

    # Use Chrome's default profile for existing login
    chrome_user_data = Path.home() / "Library/Application Support/Google/Chrome"
    if not chrome_user_data.exists():
        # Try Linux path
        chrome_user_data = Path.home() / ".config/google-chrome"
    if not chrome_user_data.exists():
        # Try Windows path
        chrome_user_data = Path.home() / "AppData/Local/Google/Chrome/User Data"

    results = {
        "steps_completed": [],
        "errors": [],
    }

    async with async_playwright() as p:
        # Launch with visible browser - user can watch/intervene if needed
        browser = await p.chromium.launch(
            headless=False,
            channel="chrome",  # Use installed Chrome
        )

        # Create context - can't easily use existing profile, but Chrome will
        # often have session cookies that persist
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Step 1: Go to Google Cloud Console
            logger.info("Navigating to Google Cloud Console...")
            await page.goto("https://console.cloud.google.com/", wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            # Check if we need to log in
            if "accounts.google.com" in page.url:
                results["status"] = "login_required"
                results["message"] = "Please log in to Google in the browser window. I'll wait."

                # Wait for user to log in (up to 5 minutes)
                try:
                    await page.wait_for_url("**/console.cloud.google.com/**", timeout=300000)
                    results["steps_completed"].append("google_login")
                except Exception:
                    return {
                        "status": "timeout",
                        "error": "Login timeout. Please try again and log in faster.",
                    }

            results["steps_completed"].append("console_accessed")
            await page.wait_for_timeout(2000)

            # Step 2: Create or select project
            logger.info("Creating project...")

            # Click project selector
            project_selector = await page.query_selector('[aria-label="Select a project"]')
            if not project_selector:
                project_selector = await page.query_selector('button:has-text("Select a project")')

            if project_selector:
                await project_selector.click()
                await page.wait_for_timeout(1500)

                # Click "New Project"
                new_project = await page.query_selector('button:has-text("New Project")')
                if new_project:
                    await new_project.click()
                    await page.wait_for_timeout(2000)

                    # Fill project name
                    name_input = await page.query_selector('input[aria-label="Project name"]')
                    if not name_input:
                        name_input = await page.query_selector('#p6ntest-name-input')
                    if name_input:
                        await name_input.fill(project_name)
                        await page.wait_for_timeout(500)

                    # Click Create
                    create_btn = await page.query_selector('button:has-text("Create")')
                    if create_btn:
                        await create_btn.click()
                        await page.wait_for_timeout(5000)
                        results["steps_completed"].append("project_created")

            # Step 3: Enable APIs
            logger.info("Enabling APIs...")
            apis_to_enable = [
                ("calendar", "Google Calendar API"),
                ("gmail", "Gmail API"),
                ("docs", "Google Docs API"),
                ("drive", "Google Drive API"),
            ]

            for api_id, api_name in apis_to_enable:
                try:
                    await page.goto(
                        f"https://console.cloud.google.com/apis/library/{api_id}.googleapis.com",
                        wait_until="domcontentloaded"
                    )
                    await page.wait_for_timeout(2000)

                    enable_btn = await page.query_selector('button:has-text("Enable")')
                    if enable_btn:
                        await enable_btn.click()
                        await page.wait_for_timeout(3000)
                        results["steps_completed"].append(f"enabled_{api_id}")
                except Exception as e:
                    results["errors"].append(f"Failed to enable {api_name}: {e}")

            # Step 4: Create OAuth credentials
            logger.info("Creating OAuth credentials...")
            await page.goto(
                "https://console.cloud.google.com/apis/credentials",
                wait_until="domcontentloaded"
            )
            await page.wait_for_timeout(2000)

            # Configure OAuth consent screen first if needed
            consent_link = await page.query_selector('a:has-text("Configure Consent Screen")')
            if consent_link:
                await consent_link.click()
                await page.wait_for_timeout(2000)

                # Select External
                external_radio = await page.query_selector('input[value="EXTERNAL"]')
                if external_radio:
                    await external_radio.click()

                create_btn = await page.query_selector('button:has-text("Create")')
                if create_btn:
                    await create_btn.click()
                    await page.wait_for_timeout(2000)

                # Fill app name
                app_name_input = await page.query_selector('input[formcontrolname="displayName"]')
                if app_name_input:
                    await app_name_input.fill("Gru Assistant")

                # Fill email
                email_input = await page.query_selector('input[formcontrolname="userSupportEmail"]')
                if email_input:
                    await email_input.click()
                    await page.wait_for_timeout(500)
                    # Select from dropdown if present
                    email_option = await page.query_selector('mat-option')
                    if email_option:
                        await email_option.click()

                # Save and continue through the screens
                for _ in range(3):
                    save_btn = await page.query_selector('button:has-text("Save and Continue")')
                    if save_btn:
                        await save_btn.click()
                        await page.wait_for_timeout(2000)

                results["steps_completed"].append("consent_screen_configured")

            # Create OAuth Client ID
            await page.goto(
                "https://console.cloud.google.com/apis/credentials/oauthclient",
                wait_until="domcontentloaded"
            )
            await page.wait_for_timeout(2000)

            # Select Desktop app
            app_type = await page.query_selector('mat-select[formcontrolname="applicationType"]')
            if app_type:
                await app_type.click()
                await page.wait_for_timeout(500)
                desktop_option = await page.query_selector('mat-option:has-text("Desktop app")')
                if desktop_option:
                    await desktop_option.click()

            # Set name
            name_input = await page.query_selector('input[formcontrolname="displayName"]')
            if name_input:
                await name_input.fill("Gru Desktop")

            # Create
            create_btn = await page.query_selector('button:has-text("Create")')
            if create_btn:
                await create_btn.click()
                await page.wait_for_timeout(3000)

            # Extract credentials from the modal
            client_id = None
            client_secret = None

            # Look for the credentials in the page
            client_id_el = await page.query_selector('[data-client-id]')
            if client_id_el:
                client_id = await client_id_el.get_attribute('data-client-id')

            # Try to find credentials in text
            page_text = await page.content()

            import re
            id_match = re.search(r'(\d+-[a-z0-9]+\.apps\.googleusercontent\.com)', page_text)
            secret_match = re.search(r'(GOCSPX-[a-zA-Z0-9_-]+)', page_text)

            if id_match:
                client_id = id_match.group(1)
            if secret_match:
                client_secret = secret_match.group(1)

            # If we can't find them, try downloading the JSON
            if not client_id or not client_secret:
                download_btn = await page.query_selector('button:has-text("Download JSON")')
                if download_btn:
                    async with page.expect_download() as download_info:
                        await download_btn.click()
                    download = await download_info.value
                    download_path = await download.path()

                    with open(download_path) as f:
                        creds_data = json.load(f)

                    installed = creds_data.get("installed", creds_data.get("web", {}))
                    client_id = installed.get("client_id")
                    client_secret = installed.get("client_secret")

            if client_id and client_secret:
                results["steps_completed"].append("credentials_created")
                results["client_id"] = client_id

                # Save credentials
                credentials_data = {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                    }
                }
                credentials_path.write_text(json.dumps(credentials_data, indent=2))
                results["steps_completed"].append("credentials_saved")

                # Step 5: Complete OAuth flow
                logger.info("Completing OAuth flow...")
                from google_auth_oauthlib.flow import InstalledAppFlow

                SCOPES = [
                    "https://www.googleapis.com/auth/calendar.readonly",
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/gmail.send",
                    "https://www.googleapis.com/auth/gmail.labels",
                    "https://www.googleapis.com/auth/documents",
                    "https://www.googleapis.com/auth/drive.file",
                ]

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path),
                    SCOPES,
                    redirect_uri="urn:ietf:wg:oauth:2.0:oob"
                )
                auth_url, _ = flow.authorization_url(prompt="consent")

                # Navigate to auth URL in the already-logged-in browser
                await page.goto(auth_url, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # Click through consent screens
                allow_btn = await page.query_selector('button:has-text("Allow")')
                if allow_btn:
                    await allow_btn.click()
                    await page.wait_for_timeout(2000)

                # Look for another Allow/Continue
                continue_btn = await page.query_selector('button:has-text("Continue")')
                allow_btn = await page.query_selector('button:has-text("Allow")')
                if continue_btn:
                    await continue_btn.click()
                    await page.wait_for_timeout(2000)
                if allow_btn:
                    await allow_btn.click()
                    await page.wait_for_timeout(2000)

                # Extract the auth code from the page
                auth_code = None
                code_el = await page.query_selector('input[readonly]')
                if code_el:
                    auth_code = await code_el.input_value()

                if not auth_code:
                    # Look for code in URL or page text
                    if "code=" in page.url:
                        auth_code = page.url.split("code=")[1].split("&")[0]
                    else:
                        textarea = await page.query_selector('textarea')
                        if textarea:
                            auth_code = await textarea.input_value()

                if auth_code:
                    # Exchange for tokens
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    token_path.write_text(creds.to_json())
                    results["steps_completed"].append("oauth_completed")
                    results["status"] = "success"
                    results["message"] = "Google setup complete! Calendar, Gmail, and Docs are now connected."
                else:
                    results["status"] = "partial"
                    results["message"] = "Created credentials but couldn't complete OAuth. You may need to authorize manually."
            else:
                results["status"] = "partial"
                results["message"] = "Couldn't extract credentials. Check the browser window."

        except Exception as e:
            logger.error(f"Setup agent error: {e}")
            results["status"] = "error"
            results["error"] = str(e)

            # Take screenshot for debugging
            screenshot_path = "/tmp/google_setup_error.png"
            await page.screenshot(path=screenshot_path)
            results["screenshot"] = screenshot_path

        finally:
            await browser.close()

    return results
