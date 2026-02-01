"""Authentication management for action services."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.actions.browser import Browser

logger = logging.getLogger(__name__)


@dataclass
class ServiceAuth:
    """Authentication state for a service."""

    service: str
    authenticated: bool = False
    last_check: datetime | None = None
    login_url: str | None = None
    user_info: dict[str, Any] | None = None


class AuthManager:
    """Manages authentication state for browser-based services."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir
        self._auth_state: dict[str, ServiceAuth] = {}

        # Known service login URLs
        self._login_urls = {
            "ubereats": "https://www.ubereats.com/login",
            "doordash": "https://www.doordash.com/consumer/login",
            "google": "https://accounts.google.com/signin",
            "amazon": "https://www.amazon.com/ap/signin",
        }

        # Selectors to check if logged in
        self._auth_checks = {
            "ubereats": {
                "url_pattern": "ubereats.com",
                "logged_in_selector": "[data-testid='user-menu']",
                "logged_out_selector": "[data-testid='sign-in-button']",
            },
            "doordash": {
                "url_pattern": "doordash.com",
                "logged_in_selector": "[data-anchor-id='AccountButton']",
                "logged_out_selector": "[data-anchor-id='SignUpButton']",
            },
        }

    def get_login_url(self, service: str) -> str | None:
        """Get login URL for a service."""
        return self._login_urls.get(service)

    def get_auth_state(self, service: str) -> ServiceAuth:
        """Get authentication state for a service."""
        if service not in self._auth_state:
            self._auth_state[service] = ServiceAuth(
                service=service,
                login_url=self.get_login_url(service),
            )
        return self._auth_state[service]

    async def check_auth(self, browser: Browser, service: str) -> bool:
        """Check if currently authenticated to a service.

        Args:
            browser: Browser instance
            service: Service name (e.g., "ubereats")

        Returns:
            True if authenticated
        """
        check_config = self._auth_checks.get(service)
        if not check_config:
            # Unknown service, assume authenticated if we have saved state
            return self.get_auth_state(service).authenticated

        try:
            page = await browser.new_page(service)

            # Navigate to a page that requires auth
            login_url = self.get_login_url(service)
            if login_url:
                await page.goto(login_url, wait_until="domcontentloaded")

            # Check for logged-in indicator
            logged_in_selector = check_config.get("logged_in_selector")
            if logged_in_selector:
                try:
                    await page.wait_for_selector(logged_in_selector, timeout=5000)
                    self._auth_state[service] = ServiceAuth(
                        service=service,
                        authenticated=True,
                        last_check=datetime.now(),
                        login_url=login_url,
                    )
                    await page.close()
                    return True
                except Exception:
                    pass

            # Check for logged-out indicator
            logged_out_selector = check_config.get("logged_out_selector")
            if logged_out_selector:
                try:
                    await page.wait_for_selector(logged_out_selector, timeout=5000)
                    self._auth_state[service] = ServiceAuth(
                        service=service,
                        authenticated=False,
                        last_check=datetime.now(),
                        login_url=login_url,
                    )
                    await page.close()
                    return False
                except Exception:
                    pass

            await page.close()
            return False

        except Exception as e:
            logger.error(f"Auth check failed for {service}: {e}")
            return False

    async def login_interactive(
        self,
        browser: Browser,
        service: str,
        timeout: int = 120000,
    ) -> bool:
        """Start interactive login flow.

        Opens the login page and waits for user to complete login.
        Only works in headed mode.

        Args:
            browser: Browser instance (must be headed)
            service: Service name
            timeout: Max time to wait for login (ms)

        Returns:
            True if login succeeded
        """
        login_url = self.get_login_url(service)
        if not login_url:
            logger.error(f"Unknown service: {service}")
            return False

        if browser.config.headless:
            logger.warning(f"Interactive login requires headed browser mode for {service}")
            return False

        try:
            page = await browser.new_page(service)
            await page.goto(login_url)

            logger.info(f"Waiting for user to complete login for {service}...")

            # Wait for login to complete by checking for auth indicator
            check_config = self._auth_checks.get(service, {})
            logged_in_selector = check_config.get("logged_in_selector")

            if logged_in_selector:
                await page.wait_for_selector(logged_in_selector, timeout=timeout)

            self._auth_state[service] = ServiceAuth(
                service=service,
                authenticated=True,
                last_check=datetime.now(),
                login_url=login_url,
            )

            await page.close()
            logger.info(f"Login successful for {service}")
            return True

        except Exception as e:
            logger.error(f"Login failed for {service}: {e}")
            return False

    async def logout(self, browser: Browser, service: str) -> bool:
        """Clear authentication for a service.

        Args:
            browser: Browser instance
            service: Service name

        Returns:
            True if logout succeeded
        """
        try:
            await browser.clear_context(service)
            self._auth_state[service] = ServiceAuth(
                service=service,
                authenticated=False,
                last_check=datetime.now(),
                login_url=self.get_login_url(service),
            )
            logger.info(f"Logged out of {service}")
            return True
        except Exception as e:
            logger.error(f"Logout failed for {service}: {e}")
            return False

    def list_services(self) -> list[dict[str, Any]]:
        """List all known services and their auth state."""
        services = []
        for service in self._login_urls:
            state = self.get_auth_state(service)
            services.append({
                "service": service,
                "authenticated": state.authenticated,
                "last_check": state.last_check.isoformat() if state.last_check else None,
                "login_url": state.login_url,
            })
        return services


# Singleton auth manager
_auth_manager: AuthManager | None = None


def get_auth_manager(storage_dir: Path | None = None) -> AuthManager:
    """Get the shared auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(storage_dir)
    return _auth_manager
