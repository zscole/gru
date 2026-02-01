"""Purchase action handlers - DoorDash, Amazon, etc."""

from __future__ import annotations

import logging
import os
from typing import Any

from gru.actions.autonomous import (
    ActionCategory,
    ActionHandler,
    ActionPreview,
    ActionResult,
)

logger = logging.getLogger(__name__)


class DoorDashOrderHandler(ActionHandler):
    """Order food via DoorDash."""

    @property
    def action_type(self) -> str:
        return "doordash_order"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.PURCHASE

    @property
    def description(self) -> str:
        return "Order food from DoorDash"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("restaurant"):
            return False, "Restaurant name required"
        if not params.get("items"):
            return False, "Items to order required"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        restaurant = params["restaurant"]
        items = params["items"]
        address = params.get("address", "default address")
        reorder = params.get("reorder", False)

        if reorder:
            return ActionPreview(
                summary=f"Reorder from {restaurant} via DoorDash",
                details=[
                    f"Restaurant: {restaurant}",
                    "Reordering previous order",
                    f"Deliver to: {address}",
                ],
                reversible=True,
                warnings=["Order will be placed with your saved payment method"],
            )

        items_list = items if isinstance(items, list) else [items]
        return ActionPreview(
            summary=f"Order from {restaurant} via DoorDash",
            details=[
                f"Restaurant: {restaurant}",
                "Items:",
                *[f"  - {item}" for item in items_list],
                f"Deliver to: {address}",
            ],
            reversible=True,
            warnings=[
                "Order will be placed with your saved payment method",
                "Customizations may not be available via automation",
            ],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute DoorDash order via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ActionResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium",
            )

        restaurant = params["restaurant"]
        items = params.get("items", [])
        reorder = params.get("reorder", False)

        try:
            async with async_playwright() as p:
                import os

                user_data_dir = os.path.expanduser("~/.gru/browser-profiles/doordash")
                os.makedirs(user_data_dir, exist_ok=True)
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False,
                    channel="chrome",
                )
                page = browser.pages[0] if browser.pages else await browser.new_page()

                if reorder:
                    # Go to orders page
                    await page.goto("https://www.doordash.com/orders", wait_until="domcontentloaded")
                    await page.wait_for_timeout(3000)

                    # Check for login
                    if "identity" in page.url:
                        logger.info("DoorDash login required - waiting...")
                        try:
                            await page.wait_for_url("**/doordash.com/orders**", timeout=120000)
                        except Exception:
                            await browser.close()
                            return ActionResult(success=False, message="Login timeout")

                    # Find recent order from restaurant
                    order_card = await page.query_selector(f'[data-testid="order-card"]:has-text("{restaurant}")')
                    if order_card:
                        reorder_btn = await order_card.query_selector('button:has-text("Reorder")')
                        if reorder_btn:
                            await reorder_btn.click()
                            await page.wait_for_timeout(3000)

                            # Proceed to checkout
                            checkout_btn = await page.query_selector('button:has-text("Checkout")')
                            if checkout_btn:
                                await checkout_btn.click()
                                await page.wait_for_timeout(2000)

                                # Place order
                                place_order_btn = await page.query_selector('button:has-text("Place Order")')
                                if place_order_btn:
                                    await place_order_btn.click()
                                    await page.wait_for_timeout(5000)

                                    # Check for confirmation
                                    if "confirmation" in page.url or await page.query_selector(
                                        '[data-testid="order-confirmation"]'
                                    ):
                                        await browser.close()
                                        return ActionResult(
                                            success=True,
                                            message=f"DoorDash reorder placed from {restaurant}",
                                            data={"restaurant": restaurant, "type": "reorder"},
                                            undo_available=True,
                                            undo_data={"action": "cancel"},
                                        )

                else:
                    # Search for restaurant
                    await page.goto(
                        f"https://www.doordash.com/search/store/{restaurant}/", wait_until="domcontentloaded"
                    )
                    await page.wait_for_timeout(3000)

                    # Check for login
                    if "identity" in page.url:
                        logger.info("DoorDash login required - waiting...")
                        try:
                            await page.wait_for_url("**/doordash.com/**", timeout=120000)
                            await page.goto(f"https://www.doordash.com/search/store/{restaurant}/")
                            await page.wait_for_timeout(3000)
                        except Exception:
                            await browser.close()
                            return ActionResult(success=False, message="Login timeout")

                    # Click first restaurant result
                    restaurant_link = await page.query_selector('[data-testid="store-card"]')
                    if restaurant_link:
                        await restaurant_link.click()
                        await page.wait_for_timeout(2000)

                        # Add items to cart
                        items_list = items if isinstance(items, list) else [items]
                        for item in items_list:
                            # Search for item
                            item_elem = await page.query_selector(f'button:has-text("{item}")')
                            if item_elem:
                                await item_elem.click()
                                await page.wait_for_timeout(1000)

                                # Add to cart
                                add_btn = await page.query_selector('button:has-text("Add to Cart")')
                                if add_btn:
                                    await add_btn.click()
                                    await page.wait_for_timeout(1000)

                        # Go to cart and checkout
                        cart_btn = await page.query_selector('[data-testid="cart-button"]')
                        if cart_btn:
                            await cart_btn.click()
                            await page.wait_for_timeout(2000)

                            checkout_btn = await page.query_selector('button:has-text("Checkout")')
                            if checkout_btn:
                                await checkout_btn.click()
                                await page.wait_for_timeout(2000)

                                place_order_btn = await page.query_selector('button:has-text("Place Order")')
                                if place_order_btn:
                                    await place_order_btn.click()
                                    await page.wait_for_timeout(5000)

                                    if "confirmation" in page.url:
                                        await browser.close()
                                        return ActionResult(
                                            success=True,
                                            message=f"DoorDash order placed from {restaurant}",
                                            data={"restaurant": restaurant, "items": items_list},
                                            undo_available=True,
                                            undo_data={"action": "cancel"},
                                        )

                await browser.close()
                return ActionResult(
                    success=False,
                    message="Could not complete DoorDash order. Please order manually at doordash.com",
                )

        except Exception as e:
            logger.error(f"DoorDash order failed: {e}")
            return ActionResult(success=False, message=f"Order failed: {e}")

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        return ActionResult(success=False, message="Please cancel your order via the DoorDash app.")


class AmazonOrderHandler(ActionHandler):
    """Order items from Amazon."""

    @property
    def action_type(self) -> str:
        return "amazon_order"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.PURCHASE

    @property
    def description(self) -> str:
        return "Order items from Amazon"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("item") and not params.get("asin"):
            return False, "Item name or ASIN required"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        item = params.get("item") or params.get("asin")
        quantity = params.get("quantity", 1)
        buy_now = params.get("buy_now", False)

        warnings = [
            "Order will use your default payment method",
            "Order will ship to your default address",
        ]
        if buy_now:
            warnings.append("Buy Now will complete purchase immediately")

        return ActionPreview(
            summary=f"Order {quantity}x '{item}' from Amazon",
            details=[
                f"Item: {item}",
                f"Quantity: {quantity}",
                f"Method: {'Buy Now' if buy_now else 'Add to Cart'}",
            ],
            reversible=True,
            warnings=warnings,
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        """Execute Amazon order via browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ActionResult(
                success=False,
                message="Playwright not installed. Run: pip install playwright && playwright install chromium",
            )

        item = params.get("item")
        asin = params.get("asin")
        params.get("quantity", 1)
        buy_now = params.get("buy_now", False)

        try:
            async with async_playwright() as p:
                user_data_dir = os.path.expanduser("~/.gru/browser-profiles/amazon")
                os.makedirs(user_data_dir, exist_ok=True)
                browser = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False,
                    channel="chrome",
                )
                page = browser.pages[0] if browser.pages else await browser.new_page()

                if asin:
                    # Go directly to product page
                    await page.goto(f"https://www.amazon.com/dp/{asin}", wait_until="domcontentloaded")
                else:
                    # Search for item
                    await page.goto("https://www.amazon.com", wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)

                    # Check for login
                    sign_in = await page.query_selector("#nav-link-accountList")
                    if sign_in:
                        sign_in_text = await sign_in.text_content()
                        if "Sign in" in sign_in_text:
                            logger.info("Amazon login may be required")

                    # Search
                    search_box = await page.query_selector("#twotabsearchtextbox")
                    if search_box:
                        await search_box.fill(item)
                        await page.keyboard.press("Enter")
                        await page.wait_for_timeout(3000)

                        # Click first result
                        first_result = await page.query_selector('[data-component-type="s-search-result"] h2 a')
                        if first_result:
                            await first_result.click()
                            await page.wait_for_timeout(2000)

                # On product page
                if buy_now:
                    # Click Buy Now
                    buy_now_btn = await page.query_selector("#buy-now-button")
                    if buy_now_btn:
                        await buy_now_btn.click()
                        await page.wait_for_timeout(3000)

                        # May need to handle login here
                        if "signin" in page.url:
                            logger.info("Amazon login required - waiting...")
                            try:
                                await page.wait_for_url("**/amazon.com/**", timeout=120000)
                            except Exception:
                                await browser.close()
                                return ActionResult(success=False, message="Login timeout")

                        # Place order
                        place_order_btn = await page.query_selector("#submitOrderButtonId")
                        if not place_order_btn:
                            place_order_btn = await page.query_selector('[name="placeYourOrder1"]')

                        if place_order_btn:
                            await place_order_btn.click()
                            await page.wait_for_timeout(5000)

                            # Check for confirmation
                            if "thankyou" in page.url or await page.query_selector("#thank-you-page"):
                                await browser.close()
                                return ActionResult(
                                    success=True,
                                    message=f"Amazon order placed for {item or asin}",
                                    data={"item": item, "asin": asin},
                                    undo_available=True,
                                    undo_data={"action": "cancel"},
                                )
                else:
                    # Add to cart
                    add_to_cart_btn = await page.query_selector("#add-to-cart-button")
                    if add_to_cart_btn:
                        await add_to_cart_btn.click()
                        await page.wait_for_timeout(2000)

                        await browser.close()
                        return ActionResult(
                            success=True,
                            message=f"Added {item or asin} to Amazon cart",
                            data={"item": item, "asin": asin, "added_to_cart": True},
                        )

                await browser.close()
                return ActionResult(
                    success=False,
                    message="Could not complete Amazon order. Please order manually at amazon.com",
                )

        except Exception as e:
            logger.error(f"Amazon order failed: {e}")
            return ActionResult(success=False, message=f"Order failed: {e}")

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        return ActionResult(success=False, message="Please cancel your order via Amazon.com > Your Orders.")
