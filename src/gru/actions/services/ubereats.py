"""Uber Eats actions for searching and ordering food."""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse

from gru.actions.auth import get_auth_manager
from gru.actions.base import Action, ActionContext, ActionResult

logger = logging.getLogger(__name__)

UBEREATS_BASE = "https://www.ubereats.com"
CONTEXT_NAME = "ubereats"


class UberEatsSearchAction(Action):
    """Search for restaurants and food on Uber Eats."""

    name = "ubereats_search"
    description = "Search for restaurants or food items on Uber Eats"
    category = "food_delivery"
    requires_auth = False  # Can browse without login

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("query"):
            return False, "query is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        query = params["query"]
        location = params.get("location") or context.location
        num_results = params.get("num_results", 10)
        params.get("sort_by", "relevance")  # relevance, rating, delivery_time

        # Build URL with location
        location_str = ""
        if location:
            if isinstance(location, dict) and "address" in location:
                location_str = location["address"]
            elif isinstance(location, str):
                location_str = location

        encoded_query = urllib.parse.quote_plus(query)

        async def search(page):
            # Navigate to Uber Eats
            await page.goto(UBEREATS_BASE, wait_until="domcontentloaded")

            # Set location if provided
            if location_str:
                try:
                    # Click on location/address input
                    location_input = await page.wait_for_selector(
                        "[data-testid='location-typeahead-input'], input[placeholder*='address']", timeout=5000
                    )
                    if location_input:
                        await location_input.fill(location_str)
                        await asyncio.sleep(1)
                        # Select first suggestion
                        await page.keyboard.press("ArrowDown")
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"Could not set location: {e}")

            # Navigate to search
            search_url = f"{UBEREATS_BASE}/search?q={encoded_query}"
            await page.goto(search_url, wait_until="networkidle")

            # Wait for results
            await asyncio.sleep(2)

            results = []

            # Try to find restaurant cards
            cards = await page.query_selector_all("[data-testid='store-card']")
            if not cards:
                # Fallback selectors
                cards = await page.query_selector_all("a[href*='/store/']")

            for card in cards[:num_results]:
                try:
                    # Get name
                    name_el = await card.query_selector("h3, [data-testid='store-title']")
                    name = await name_el.inner_text() if name_el else None

                    if not name:
                        # Try aria-label
                        name = await card.get_attribute("aria-label")
                        if name:
                            name = name.split(",")[0]

                    if not name:
                        continue

                    # Get link
                    href = await card.get_attribute("href")
                    if not href and card.tag_name != "a":
                        link_el = await card.query_selector("a")
                        href = await link_el.get_attribute("href") if link_el else None

                    # Get rating
                    rating = None
                    rating_el = await card.query_selector("[aria-label*='rating']")
                    if rating_el:
                        rating_text = await rating_el.get_attribute("aria-label")
                        rating_match = re.search(r"([\d.]+)", rating_text or "")
                        if rating_match:
                            rating = float(rating_match.group(1))

                    # Get delivery time
                    delivery_time = None
                    time_el = await card.query_selector("[data-testid='delivery-time']")
                    if time_el:
                        delivery_time = await time_el.inner_text()

                    # Get delivery fee
                    fee = None
                    fee_el = await card.query_selector("[data-testid='delivery-fee']")
                    if fee_el:
                        fee = await fee_el.inner_text()

                    results.append(
                        {
                            "name": name.strip(),
                            "url": f"{UBEREATS_BASE}{href}" if href and not href.startswith("http") else href,
                            "rating": rating,
                            "delivery_time": delivery_time,
                            "delivery_fee": fee,
                        }
                    )

                except Exception as e:
                    logger.debug(f"Error parsing restaurant card: {e}")
                    continue

            return {
                "query": query,
                "location": location_str,
                "results": results,
                "count": len(results),
            }

        try:
            result = await context.browser.run_with_page(search, CONTEXT_NAME)

            if result["count"] == 0:
                return ActionResult.error_result(f"No restaurants found for '{query}'")

            top = result["results"][0]
            message = f"Found {result['count']} restaurants. Top: {top['name']}"
            if top.get("rating"):
                message += f" ({top['rating']}*)"
            if top.get("delivery_time"):
                message += f" - {top['delivery_time']}"

            return ActionResult.success_result(message, result)

        except Exception as e:
            return ActionResult.error_result(f"Search failed: {e}")


class UberEatsCartAction(Action):
    """View or modify Uber Eats cart."""

    name = "ubereats_cart"
    description = "View current Uber Eats cart"
    category = "food_delivery"
    requires_auth = True

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        action = params.get("action", "view")  # view, clear

        # Check auth
        auth_manager = get_auth_manager()
        if not await auth_manager.check_auth(context.browser, "ubereats"):
            return ActionResult.auth_required("ubereats", f"{UBEREATS_BASE}/login")

        async def manage_cart(page):
            await page.goto(UBEREATS_BASE, wait_until="domcontentloaded")

            # Find and click cart button
            cart_btn = await page.wait_for_selector("[data-testid='cart-button'], [aria-label*='cart']", timeout=5000)

            if not cart_btn:
                return {"items": [], "total": None, "error": "Cart not found"}

            await cart_btn.click()
            await asyncio.sleep(1)

            if action == "clear":
                # Find clear/remove buttons
                remove_btns = await page.query_selector_all("[data-testid='remove-item']")
                for btn in remove_btns:
                    await btn.click()
                    await asyncio.sleep(0.5)
                return {"cleared": True, "items": []}

            # Get cart items
            items = []
            item_els = await page.query_selector_all("[data-testid='cart-item']")

            for item_el in item_els:
                try:
                    name_el = await item_el.query_selector("[data-testid='cart-item-name']")
                    price_el = await item_el.query_selector("[data-testid='cart-item-price']")
                    qty_el = await item_el.query_selector("[data-testid='cart-item-quantity']")

                    items.append(
                        {
                            "name": await name_el.inner_text() if name_el else "Unknown",
                            "price": await price_el.inner_text() if price_el else None,
                            "quantity": await qty_el.inner_text() if qty_el else "1",
                        }
                    )
                except Exception:
                    continue

            # Get total
            total = None
            total_el = await page.query_selector("[data-testid='cart-total']")
            if total_el:
                total = await total_el.inner_text()

            return {
                "items": items,
                "total": total,
                "item_count": len(items),
            }

        try:
            result = await context.browser.run_with_page(manage_cart, CONTEXT_NAME)

            if action == "clear":
                return ActionResult.success_result("Cart cleared", result)

            if not result.get("items"):
                return ActionResult.success_result("Cart is empty", result)

            message = f"Cart has {result['item_count']} items"
            if result.get("total"):
                message += f" - Total: {result['total']}"

            return ActionResult.success_result(message, result)

        except Exception as e:
            return ActionResult.error_result(f"Cart action failed: {e}")


class UberEatsOrderAction(Action):
    """Order food from Uber Eats."""

    name = "ubereats_order"
    description = "Add items to cart and place an order on Uber Eats"
    category = "food_delivery"
    requires_auth = True
    requires_confirmation = True  # Always confirm before payment

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("restaurant") and not params.get("restaurant_url"):
            return False, "restaurant or restaurant_url is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        restaurant = params.get("restaurant")
        restaurant_url = params.get("restaurant_url")
        items = params.get("items", [])  # List of item names to order
        item_query = params.get("item_query")  # Or search for items
        place_order = params.get("place_order", False)  # Actually checkout
        scheduled_time = params.get("scheduled_time")  # Optional scheduling

        # Check auth
        auth_manager = get_auth_manager()
        if not await auth_manager.check_auth(context.browser, "ubereats"):
            return ActionResult.auth_required("ubereats", f"{UBEREATS_BASE}/login")

        async def order(page):
            # Navigate to restaurant
            if restaurant_url:
                await page.goto(restaurant_url, wait_until="networkidle")
            elif restaurant:
                # Search for restaurant first
                search_url = f"{UBEREATS_BASE}/search?q={urllib.parse.quote_plus(restaurant)}"
                await page.goto(search_url, wait_until="networkidle")
                await asyncio.sleep(2)

                # Click first result
                first_result = await page.query_selector("[data-testid='store-card'] a, a[href*='/store/']")
                if first_result:
                    await first_result.click()
                    await page.wait_for_load_state("networkidle")
                else:
                    return {"error": f"Restaurant '{restaurant}' not found"}

            await asyncio.sleep(2)

            # Get restaurant info
            restaurant_name = None
            name_el = await page.query_selector("h1, [data-testid='store-title']")
            if name_el:
                restaurant_name = await name_el.inner_text()

            # Find and add items
            added_items = []

            if items:
                # Add specific items by name
                for item_name in items:
                    item_cards = await page.query_selector_all("[data-testid='menu-item']")
                    for card in item_cards:
                        try:
                            card_name = await card.inner_text()
                            if item_name.lower() in card_name.lower():
                                await card.click()
                                await asyncio.sleep(1)

                                # Click add to cart
                                add_btn = await page.wait_for_selector(
                                    "[data-testid='add-to-cart'], button:has-text('Add to Cart')", timeout=5000
                                )
                                if add_btn:
                                    await add_btn.click()
                                    added_items.append(item_name)
                                    await asyncio.sleep(1)

                                # Close modal if open
                                close_btn = await page.query_selector("[data-testid='modal-close']")
                                if close_btn:
                                    await close_btn.click()
                                break
                        except Exception:
                            continue

            elif item_query:
                # Search for items on menu
                menu_items = await page.query_selector_all("[data-testid='menu-item']")
                for card in menu_items[:5]:  # Check first 5 items
                    try:
                        card_text = await card.inner_text()
                        if item_query.lower() in card_text.lower():
                            await card.click()
                            await asyncio.sleep(1)

                            add_btn = await page.wait_for_selector(
                                "[data-testid='add-to-cart'], button:has-text('Add')", timeout=5000
                            )
                            if add_btn:
                                await add_btn.click()
                                added_items.append(card_text.split("\n")[0])
                                await asyncio.sleep(1)
                            break
                    except Exception:
                        continue

            # Get cart summary
            cart_info = {"items": added_items}

            if place_order:
                # Click checkout
                checkout_btn = await page.query_selector("[data-testid='checkout-button'], button:has-text('Checkout')")
                if checkout_btn:
                    await checkout_btn.click()
                    await asyncio.sleep(2)

                    # Check for order total
                    total_el = await page.query_selector("[data-testid='order-total']")
                    if total_el:
                        cart_info["total"] = await total_el.inner_text()

                    # Get delivery time estimate
                    time_el = await page.query_selector("[data-testid='delivery-time']")
                    if time_el:
                        cart_info["estimated_delivery"] = await time_el.inner_text()

                    # If scheduled time, set it
                    if scheduled_time:
                        schedule_btn = await page.query_selector("[data-testid='schedule-button']")
                        if schedule_btn:
                            await schedule_btn.click()
                            # TODO: Set the specific time

                    # Return confirmation needed (don't auto-complete payment)
                    cart_info["checkout_ready"] = True
                    cart_info["message"] = "Ready to place order. Confirm to complete payment."

            return {
                "restaurant": restaurant_name,
                "added_items": added_items,
                "cart": cart_info,
                "place_order": place_order,
            }

        try:
            result = await context.browser.run_with_page(order, CONTEXT_NAME)

            if result.get("error"):
                return ActionResult.error_result(result["error"])

            if not result.get("added_items"):
                return ActionResult.error_result("Could not add any items to cart")

            message = f"Added {len(result['added_items'])} items from {result['restaurant']}"

            if result.get("cart", {}).get("checkout_ready"):
                # Return confirmation required
                return ActionResult.confirm_required(
                    message + " - Ready for checkout",
                    {
                        "restaurant": result["restaurant"],
                        "items": result["added_items"],
                        "total": result.get("cart", {}).get("total"),
                        "estimated_delivery": result.get("cart", {}).get("estimated_delivery"),
                    },
                )

            return ActionResult.success_result(message, result)

        except Exception as e:
            return ActionResult.error_result(f"Order failed: {e}")
