"""Payment action handlers - Venmo, etc."""

from __future__ import annotations

import logging
from typing import Any

from gru.actions.autonomous import (
    ActionCategory,
    ActionHandler,
    ActionPreview,
    ActionResult,
)

logger = logging.getLogger(__name__)


class VenmoPaymentHandler(ActionHandler):
    """Send money via Venmo."""

    @property
    def action_type(self) -> str:
        return "venmo_payment"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.PAYMENT

    @property
    def description(self) -> str:
        return "Send a Venmo payment"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("recipient"):
            return False, "Recipient username or phone required"
        if not params.get("amount"):
            return False, "Amount required"
        if params.get("amount", 0) <= 0:
            return False, "Amount must be positive"
        if params.get("amount", 0) > 500:
            return False, "Amount exceeds safety limit ($500). Increase manually if needed."
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        recipient = params["recipient"]
        amount = params["amount"]
        note = params.get("note", "")

        return ActionPreview(
            summary=f"Send ${amount:.2f} to {recipient} via Venmo",
            details=[
                f"To: {recipient}",
                f"Amount: ${amount:.2f}",
                f"Note: {note}" if note else "Note: (none)",
            ],
            reversible=False,
            cost=amount,
            warnings=[
                "Money transfers cannot be automatically reversed",
                "Verify recipient before confirming",
            ],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute Venmo payment via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ActionResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium",
            )

        recipient = params["recipient"]
        amount = params["amount"]
        note = params.get("note", "From Gru")

        try:
            async with async_playwright() as p:
                # Use persistent context to potentially reuse Venmo login
                import os

                user_data_dir = os.path.expanduser("~/.gru/browser-profiles/venmo")
                os.makedirs(user_data_dir, exist_ok=True)
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False,
                    channel="chrome",
                )
                page = browser.pages[0] if browser.pages else await browser.new_page()

                # Go to Venmo
                await page.goto("https://venmo.com/pay", wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)

                # Check if logged in
                if "account/sign-in" in page.url:
                    # Need to log in - wait for user
                    logger.info("Venmo login required - waiting for user...")
                    try:
                        await page.wait_for_url("**/venmo.com/**", timeout=120000)
                    except Exception:
                        await browser.close()
                        return ActionResult(success=False, message="Venmo login timeout. Please try again.")

                # Navigate to pay page
                await page.goto("https://venmo.com/pay", wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # Enter recipient
                recipient_input = await page.query_selector('input[placeholder*="Name, @username"]')
                if not recipient_input:
                    recipient_input = await page.query_selector('input[aria-label*="recipient"]')

                if recipient_input:
                    await recipient_input.fill(recipient)
                    await page.wait_for_timeout(1500)

                    # Select from dropdown
                    suggestion = await page.query_selector('[data-testid="recipient-suggestion"]')
                    if suggestion:
                        await suggestion.click()
                        await page.wait_for_timeout(1000)

                    # Enter amount
                    amount_input = await page.query_selector('input[placeholder*="0"]')
                    if not amount_input:
                        amount_input = await page.query_selector('input[aria-label*="amount"]')

                    if amount_input:
                        await amount_input.fill(str(amount))
                        await page.wait_for_timeout(500)

                        # Enter note
                        note_input = await page.query_selector('textarea[placeholder*="What"]')
                        if not note_input:
                            note_input = await page.query_selector("textarea")

                        if note_input:
                            await note_input.fill(note)
                            await page.wait_for_timeout(500)

                        # Click Pay button
                        pay_btn = await page.query_selector('button:has-text("Pay")')
                        if pay_btn:
                            await pay_btn.click()
                            await page.wait_for_timeout(3000)

                            # Confirm if needed
                            confirm_btn = await page.query_selector('button:has-text("Confirm")')
                            if confirm_btn:
                                await confirm_btn.click()
                                await page.wait_for_timeout(3000)

                            # Check for success
                            success_indicator = await page.query_selector('[data-testid="payment-success"]')
                            if success_indicator or "success" in page.url:
                                await browser.close()
                                return ActionResult(
                                    success=True,
                                    message=f"Sent ${amount:.2f} to {recipient} via Venmo",
                                    data={"recipient": recipient, "amount": amount},
                                )

                await browser.close()
                return ActionResult(
                    success=False,
                    message="Could not complete Venmo payment. Please try manually at venmo.com",
                )

        except Exception as e:
            logger.error(f"Venmo payment failed: {e}")
            return ActionResult(success=False, message=f"Payment failed: {e}")
