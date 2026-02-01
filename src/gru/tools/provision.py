"""API key provisioning tools - automate signup and key generation."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""

    name: str
    signup_url: str
    console_url: str
    key_pattern: str  # Regex to identify the key format
    free_tier: bool = True
    requires_billing: bool = False
    instructions: str = ""


# Supported providers
PROVIDERS = {
    "google_maps": ProviderConfig(
        name="Google Maps",
        signup_url="https://console.cloud.google.com/",
        console_url="https://console.cloud.google.com/apis/credentials",
        key_pattern=r"AIza[0-9A-Za-z\-_]{35}",
        free_tier=True,
        requires_billing=True,  # $200 free credit but needs billing
        instructions="Create project, enable Maps APIs, create API key",
    ),
    "brave_search": ProviderConfig(
        name="Brave Search",
        signup_url="https://brave.com/search/api/",
        console_url="https://api.search.brave.com/app/keys",
        key_pattern=r"BSA[a-zA-Z0-9]{32}",
        free_tier=True,
        requires_billing=False,
    ),
    "openweathermap": ProviderConfig(
        name="OpenWeatherMap",
        signup_url="https://home.openweathermap.org/users/sign_up",
        console_url="https://home.openweathermap.org/api_keys",
        key_pattern=r"[a-f0-9]{32}",
        free_tier=True,
        requires_billing=False,
    ),
    "serper": ProviderConfig(
        name="Serper (Google Search API)",
        signup_url="https://serper.dev/signup",
        console_url="https://serper.dev/api-key",
        key_pattern=r"[a-f0-9]{40,}",
        free_tier=True,  # 2500 free queries
        requires_billing=False,
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        signup_url="https://console.anthropic.com/",
        console_url="https://console.anthropic.com/settings/keys",
        key_pattern=r"sk-ant-api[a-zA-Z0-9\-_]{90,}",
        free_tier=False,
        requires_billing=True,
    ),
}


async def list_providers() -> dict:
    """List available API providers and their status."""
    result = []
    for key, provider in PROVIDERS.items():
        # Check if key is already configured
        env_var = f"{key.upper()}_API_KEY"
        configured = bool(os.getenv(env_var))

        result.append(
            {
                "id": key,
                "name": provider.name,
                "configured": configured,
                "free_tier": provider.free_tier,
                "requires_billing": provider.requires_billing,
                "signup_url": provider.signup_url,
            }
        )

    return {"providers": result}


async def provision_api_key(
    provider: str,
    email: str,
    headless: bool = False,
) -> dict:
    """Provision an API key by automating the signup process.

    This opens a browser, navigates through the signup flow, and
    extracts the API key. User interaction may be required for
    email verification and ToS acceptance.
    """
    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        return {"error": f"Unknown provider: {provider}. Available: {available}"}

    config = PROVIDERS[provider]

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}

    logger.info(f"Starting provisioning for {config.name}")

    async with async_playwright() as p:
        # Launch visible browser so user can interact
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Navigate to signup
            await page.goto(config.signup_url, wait_until="domcontentloaded")

            # Provider-specific flows
            if provider == "google_maps":
                result = await _provision_google_maps(page, email, config)
            elif provider == "brave_search":
                result = await _provision_brave_search(page, email, config)
            elif provider == "openweathermap":
                result = await _provision_openweathermap(page, email, config)
            elif provider == "serper":
                result = await _provision_serper(page, email, config)
            else:
                result = await _provision_generic(page, email, config)

            return result

        except Exception as e:
            logger.error(f"Provisioning failed: {e}")
            # Take screenshot for debugging
            screenshot_path = f"/tmp/provision_{provider}_error.png"
            await page.screenshot(path=screenshot_path)
            return {
                "error": str(e),
                "screenshot": screenshot_path,
                "hint": "Check screenshot for current page state",
            }
        finally:
            await browser.close()


async def _provision_google_maps(page, email: str, config: ProviderConfig) -> dict:
    """Handle Google Cloud/Maps API provisioning."""
    # Google requires OAuth login - can't fully automate
    # Guide the user through the process

    await page.wait_for_timeout(2000)

    return {
        "status": "manual_required",
        "provider": "google_maps",
        "steps": [
            "1. Sign in with your Google account",
            "2. Create a new project (or select existing)",
            "3. Go to APIs & Services > Credentials",
            "4. Click 'Create Credentials' > 'API Key'",
            "5. Copy the key and tell me what it is",
        ],
        "console_url": config.console_url,
        "note": "Google requires interactive login. Complete the steps and paste the key.",
    }


async def _provision_brave_search(page, email: str, config: ProviderConfig) -> dict:
    """Handle Brave Search API provisioning."""
    await page.wait_for_timeout(2000)

    # Look for signup/login form
    # Brave uses GitHub OAuth primarily

    return {
        "status": "manual_required",
        "provider": "brave_search",
        "steps": [
            "1. Click 'Get Started' or 'Sign Up'",
            "2. Sign in with GitHub (recommended) or email",
            "3. Go to API Keys section",
            "4. Create a new API key",
            "5. Copy the key and tell me what it is",
        ],
        "console_url": config.console_url,
    }


async def _provision_openweathermap(page, email: str, config: ProviderConfig) -> dict:
    """Handle OpenWeatherMap API provisioning."""
    await page.wait_for_timeout(2000)

    # Try to fill signup form
    try:
        # Check if we're on signup page
        await page.query_selector('input[name="user[username]"]')
        email_field = await page.query_selector('input[name="user[email]"]')

        if email_field:
            await email_field.fill(email)

            return {
                "status": "form_ready",
                "provider": "openweathermap",
                "steps": [
                    "1. Fill in username and password",
                    "2. Complete the captcha",
                    "3. Click 'Create Account'",
                    "4. Verify your email",
                    "5. Go to API Keys tab",
                    "6. Copy your API key and tell me",
                ],
                "console_url": config.console_url,
                "note": "Email pre-filled. Complete registration and paste the key.",
            }
    except Exception:
        pass

    return {
        "status": "manual_required",
        "provider": "openweathermap",
        "signup_url": config.signup_url,
        "console_url": config.console_url,
    }


async def _provision_serper(page, email: str, config: ProviderConfig) -> dict:
    """Handle Serper API provisioning."""
    await page.wait_for_timeout(2000)

    try:
        # Look for email field
        email_field = await page.query_selector('input[type="email"]')
        if email_field:
            await email_field.fill(email)

            return {
                "status": "form_ready",
                "provider": "serper",
                "steps": [
                    "1. Complete the signup form",
                    "2. Verify your email",
                    "3. Copy your API key from the dashboard",
                    "4. Tell me the key",
                ],
                "console_url": config.console_url,
            }
    except Exception:
        pass

    return {
        "status": "manual_required",
        "provider": "serper",
        "signup_url": config.signup_url,
        "console_url": config.console_url,
    }


async def _provision_generic(page, email: str, config: ProviderConfig) -> dict:
    """Generic provisioning - guide user through manual process."""
    return {
        "status": "manual_required",
        "provider": config.name,
        "signup_url": config.signup_url,
        "console_url": config.console_url,
        "instructions": config.instructions,
    }


async def save_api_key(provider: str, key: str) -> dict:
    """Save an API key to configuration."""
    if provider not in PROVIDERS:
        return {"error": f"Unknown provider: {provider}"}

    config = PROVIDERS[provider]

    # Validate key format
    if not re.match(config.key_pattern, key):
        return {
            "error": f"Key doesn't match expected format for {config.name}",
            "expected_pattern": config.key_pattern,
        }

    # Determine env var name
    env_var_map = {
        "google_maps": "GOOGLE_MAPS_API_KEY",
        "brave_search": "BRAVE_API_KEY",
        "openweathermap": "OPENWEATHERMAP_API_KEY",
        "serper": "SERPER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")

    # Save to .env file
    env_path = os.path.expanduser("~/.gru/.env")
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    # Read existing
    existing = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing[k] = v

    # Update
    existing[env_var] = key

    # Write back
    with open(env_path, "w") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")

    # Also set in current environment
    os.environ[env_var] = key

    return {
        "status": "saved",
        "provider": config.name,
        "env_var": env_var,
        "env_file": env_path,
        "note": "Key saved. Restart Gru to use it, or it's already active in this session.",
    }


async def setup_google_oauth(
    client_id: str | None = None, client_secret: str | None = None, auth_code: str | None = None
) -> dict:
    """Set up Google OAuth for Calendar, Gmail, Docs access.

    Three-step process:
    1. Call without params -> get instructions for creating OAuth credentials
    2. Call with client_id + client_secret -> save creds, get auth URL
    3. Call with auth_code -> exchange for tokens, complete setup
    """
    import json
    from pathlib import Path

    data_dir = Path(os.path.expanduser("~/.gru"))
    data_dir.mkdir(parents=True, exist_ok=True)

    credentials_path = data_dir / "google_credentials.json"
    token_path = data_dir / "google_token.json"

    scopes = [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.labels",
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive.file",
    ]

    # Check if already configured
    if token_path.exists() and not auth_code:
        return {
            "status": "already_configured",
            "note": "Google OAuth is already set up. To reconfigure, delete ~/.gru/google_token.json",
        }

    # Step 3: Exchange auth code for tokens
    if auth_code:
        if not credentials_path.exists():
            return {"error": "No credentials saved. Provide client_id and client_secret first."}

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            flow.fetch_token(code=auth_code)
            creds = flow.credentials

            token_path.write_text(creds.to_json())

            return {
                "status": "success",
                "note": "Google OAuth configured! Calendar, Gmail, and Docs access is now available.",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Step 1: No credentials provided - give instructions
    if not client_id or not client_secret:
        return {
            "status": "credentials_needed",
            "steps": [
                "1. Go to console.cloud.google.com",
                "2. Create or select a project",
                "3. Enable: Calendar API, Gmail API, Docs API, Drive API",
                "4. Go to Credentials, create OAuth client ID (Desktop app)",
                "5. Give me the Client ID and Client Secret",
            ],
            "console_url": "https://console.cloud.google.com/apis/credentials",
        }

    # Step 2: Save credentials and generate auth URL
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

    # Generate auth URL for manual flow
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path), scopes, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
        )
        auth_url, _ = flow.authorization_url(prompt="consent")

        return {
            "status": "auth_needed",
            "auth_url": auth_url,
            "instructions": "Open this URL, sign in, authorize, then give me the code shown on the page.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_google_status() -> dict:
    """Check Google OAuth configuration status."""
    from pathlib import Path

    data_dir = Path(os.path.expanduser("~/.gru"))
    credentials_path = data_dir / "google_credentials.json"
    token_path = data_dir / "google_token.json"

    status = {
        "credentials_configured": credentials_path.exists(),
        "token_exists": token_path.exists(),
        "authenticated": False,
    }

    if token_path.exists():
        try:
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(token_path))
            status["authenticated"] = creds.valid
            status["expired"] = creds.expired
        except Exception as e:
            status["error"] = str(e)

    return status


def register_provision_tools() -> None:
    """Register provisioning tools."""
    register_tool(
        name="list_api_providers",
        description="List available API providers and whether they're configured.",
        parameters={},
        handler=list_providers,
    )

    register_tool(
        name="provision_api_key",
        description="Start the process to provision an API key for a service. Opens a browser to guide through signup.",
        parameters={
            "provider": {
                "type": "string",
                "description": "Provider ID (google_maps, brave_search, openweathermap, serper)",
            },
            "email": {
                "type": "string",
                "description": "Email address for account creation",
            },
            "headless": {
                "type": "boolean",
                "description": "Run browser in headless mode (default false - shows browser)",
                "optional": True,
            },
        },
        handler=provision_api_key,
    )

    register_tool(
        name="save_api_key",
        description="Save an API key after the user provides it. Validates format and stores securely.",
        parameters={
            "provider": {
                "type": "string",
                "description": "Provider ID (google_maps, brave_search, etc.)",
            },
            "key": {
                "type": "string",
                "description": "The API key to save",
            },
        },
        handler=save_api_key,
    )

    register_tool(
        name="setup_google_oauth",
        description="Set up Google OAuth for Calendar, Gmail, and Docs access. Three steps: (1) call with no params for instructions, (2) call with client_id and client_secret to get auth URL, (3) call with auth_code to complete setup.",
        parameters={
            "client_id": {
                "type": "string",
                "description": "OAuth Client ID from Google Cloud Console",
                "optional": True,
            },
            "client_secret": {
                "type": "string",
                "description": "OAuth Client Secret from Google Cloud Console",
                "optional": True,
            },
            "auth_code": {
                "type": "string",
                "description": "Authorization code from Google OAuth page",
                "optional": True,
            },
        },
        handler=setup_google_oauth,
    )

    register_tool(
        name="check_google_status",
        description="Check if Google OAuth is configured and authenticated.",
        parameters={},
        handler=check_google_status,
    )

    # Import and register the automated setup agent
    from gru.tools.google_setup_agent import run_google_setup_agent

    register_tool(
        name="auto_setup_google",
        description="Fully automated Google Cloud setup. Opens a browser, creates project, enables APIs, creates OAuth credentials, and completes authentication. User may need to log in if not already. Use this when user wants Google Calendar/Gmail/Docs set up without manual steps.",
        parameters={
            "project_name": {
                "type": "string",
                "description": "Name for the Google Cloud project (default: gru-assistant)",
                "optional": True,
            },
        },
        handler=run_google_setup_agent,
    )
