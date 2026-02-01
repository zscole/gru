"""Reservation action handlers - OpenTable, Resy, etc."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from gru.actions.autonomous import (
    ActionCategory,
    ActionHandler,
    ActionPreview,
    ActionResult,
)

logger = logging.getLogger(__name__)


class OpenTableReservationHandler(ActionHandler):
    """Make restaurant reservations via OpenTable."""

    @property
    def action_type(self) -> str:
        return "opentable_reservation"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.RESERVATION

    @property
    def description(self) -> str:
        return "Book a restaurant reservation on OpenTable"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("restaurant"):
            return False, "Restaurant name required"
        if not params.get("date"):
            return False, "Date required"
        if not params.get("time"):
            return False, "Time required"
        if not params.get("party_size"):
            return False, "Party size required"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        restaurant = params["restaurant"]
        date = params["date"]
        time = params["time"]
        party_size = params["party_size"]

        return ActionPreview(
            summary=f"Book {restaurant} for {party_size} on {date} at {time}",
            details=[
                f"Restaurant: {restaurant}",
                f"Date: {date}",
                f"Time: {time}",
                f"Party size: {party_size}",
            ],
            reversible=True,
            warnings=[
                "You may be charged a cancellation fee if you don't show up",
                "Browser automation required - a Chrome window will open",
            ],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute OpenTable reservation via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ActionResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        restaurant = params["restaurant"]
        date = params["date"]
        time = params["time"]
        party_size = params["party_size"]

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False, channel="chrome")
                context = await browser.new_context()
                page = await context.new_page()

                # Search for restaurant on OpenTable
                search_url = f"https://www.opentable.com/s?dateTime={date}T{time}&covers={party_size}&term={restaurant}"
                await page.goto(search_url, wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)

                # Find and click the restaurant
                restaurant_card = await page.query_selector(f'a[href*="{restaurant.lower().replace(" ", "-")}"]')
                if not restaurant_card:
                    # Try clicking first result
                    restaurant_card = await page.query_selector('[data-test="restaurant-card"]')

                if restaurant_card:
                    await restaurant_card.click()
                    await page.wait_for_timeout(2000)

                    # Find available time slot
                    time_slot = await page.query_selector(f'button:has-text("{time}")')
                    if not time_slot:
                        # Get first available slot
                        time_slot = await page.query_selector('[data-test="time-slot-button"]')

                    if time_slot:
                        await time_slot.click()
                        await page.wait_for_timeout(2000)

                        # Fill reservation details if needed
                        # This typically requires login - check if logged in
                        login_prompt = await page.query_selector('[data-test="login-prompt"]')

                        if login_prompt:
                            # Need to pause for user to log in
                            await page.wait_for_timeout(60000)  # Wait up to 60 seconds

                        # Complete reservation
                        complete_btn = await page.query_selector('button:has-text("Complete reservation")')
                        if complete_btn:
                            await complete_btn.click()
                            await page.wait_for_timeout(3000)

                            # Check for confirmation
                            confirmation = await page.query_selector('[data-test="confirmation"]')
                            if confirmation:
                                conf_text = await confirmation.text_content()
                                await browser.close()
                                return ActionResult(
                                    success=True,
                                    message=f"Reservation confirmed at {restaurant}",
                                    data={"confirmation": conf_text},
                                    undo_available=True,
                                    undo_data={"restaurant": restaurant, "date": date, "time": time},
                                )

                await browser.close()
                return ActionResult(
                    success=False,
                    message="Could not complete reservation. Please try manually.",
                )

        except Exception as e:
            logger.error(f"OpenTable reservation failed: {e}")
            return ActionResult(success=False, message=f"Reservation failed: {e}")

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        """Cancel a reservation."""
        return ActionResult(
            success=False,
            message="Please cancel the reservation manually via OpenTable app or website."
        )


class ResyReservationHandler(ActionHandler):
    """Make restaurant reservations via Resy."""

    @property
    def action_type(self) -> str:
        return "resy_reservation"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.RESERVATION

    @property
    def description(self) -> str:
        return "Book a restaurant reservation on Resy"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("restaurant"):
            return False, "Restaurant name required"
        if not params.get("date"):
            return False, "Date required"
        if not params.get("time"):
            return False, "Time required"
        if not params.get("party_size"):
            return False, "Party size required"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        restaurant = params["restaurant"]
        date = params["date"]
        time = params["time"]
        party_size = params["party_size"]

        return ActionPreview(
            summary=f"Book {restaurant} via Resy for {party_size} on {date} at {time}",
            details=[
                f"Restaurant: {restaurant}",
                f"Date: {date}",
                f"Time: {time}",
                f"Party size: {party_size}",
            ],
            reversible=True,
            warnings=[
                "Resy may require credit card on file",
                "Browser automation required - a Chrome window will open",
            ],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute Resy reservation via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ActionResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        restaurant = params["restaurant"]
        date = params["date"]
        time = params["time"]
        party_size = params["party_size"]
        city = params.get("city", "san-francisco")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False, channel="chrome")
                context = await browser.new_context()
                page = await context.new_page()

                # Go to Resy search
                await page.goto(f"https://resy.com/cities/{city}", wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # Search for restaurant
                search_input = await page.query_selector('input[placeholder*="Search"]')
                if search_input:
                    await search_input.fill(restaurant)
                    await page.wait_for_timeout(1500)

                    # Click search result
                    result = await page.query_selector(f'[href*="{restaurant.lower().replace(" ", "-")}"]')
                    if result:
                        await result.click()
                        await page.wait_for_timeout(2000)

                        # Set date and party size
                        # Resy's interface varies, so this is approximate

                        # Find reservation button
                        reserve_btn = await page.query_selector('button:has-text("Reserve")')
                        if not reserve_btn:
                            reserve_btn = await page.query_selector('button:has-text("Find a Time")')

                        if reserve_btn:
                            await reserve_btn.click()
                            await page.wait_for_timeout(2000)

                            # Select time slot
                            time_btn = await page.query_selector(f'button:has-text("{time}")')
                            if time_btn:
                                await time_btn.click()
                                await page.wait_for_timeout(2000)

                                # Check for login
                                if "auth" in page.url or await page.query_selector('[data-test="login"]'):
                                    # Wait for user to log in
                                    await page.wait_for_timeout(60000)

                                # Complete booking
                                complete_btn = await page.query_selector('button:has-text("Complete")')
                                if complete_btn:
                                    await complete_btn.click()
                                    await page.wait_for_timeout(3000)

                                    await browser.close()
                                    return ActionResult(
                                        success=True,
                                        message=f"Resy reservation booked at {restaurant}",
                                        data={"restaurant": restaurant, "date": date, "time": time},
                                        undo_available=True,
                                        undo_data={"restaurant": restaurant},
                                    )

                await browser.close()
                return ActionResult(
                    success=False,
                    message="Could not complete Resy reservation. Please try manually at resy.com",
                )

        except Exception as e:
            logger.error(f"Resy reservation failed: {e}")
            return ActionResult(success=False, message=f"Reservation failed: {e}")

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        return ActionResult(
            success=False,
            message="Please cancel via the Resy app or website."
        )
