"""Search actions for finding information and local businesses."""

from __future__ import annotations

import logging
import re
import urllib.parse

from gru.actions.base import Action, ActionContext, ActionResult

logger = logging.getLogger(__name__)


class WebSearchAction(Action):
    """Perform a web search using multiple providers with fallback."""

    name = "web_search"
    description = "Search the web using multiple search providers"
    category = "search"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("query"):
            return False, "query is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        from gru.actions.services.search_providers import get_search_chain

        query = params["query"]
        num_results = params.get("num_results", 5)

        try:
            chain = get_search_chain()
            results = await chain.search(query, num_results)

            result_dicts = [r.to_dict() for r in results]

            return ActionResult.success_result(
                f"Found {len(results)} results for '{query}'",
                {
                    "query": query,
                    "results": result_dicts,
                    "count": len(results),
                },
            )
        except Exception as e:
            return ActionResult.error_result(f"Search failed: {e}")


class LocalSearchAction(Action):
    """Search for local businesses and places."""

    name = "local_search"
    description = "Search for local businesses using Google Maps"
    category = "search"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("query"):
            return False, "query is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        query = params["query"]
        location = params.get("location") or context.location
        num_results = params.get("num_results", 5)

        # Build search URL
        search_query = query
        if location:
            if isinstance(location, dict) and "address" in location:
                search_query = f"{query} near {location['address']}"
            elif isinstance(location, str):
                search_query = f"{query} near {location}"

        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://www.google.com/maps/search/{encoded_query}"

        async def search(page):
            await page.goto(url, wait_until="domcontentloaded")

            # Wait for results to load
            await page.wait_for_selector("[role='feed']", timeout=15000)

            results = []
            # Get result cards
            result_elements = await page.query_selector_all("[role='feed'] > div > div > a")

            for _i, element in enumerate(result_elements[:num_results]):
                try:
                    # Get the aria-label which contains name and rating
                    label = await element.get_attribute("aria-label")
                    href = await element.get_attribute("href")

                    if label:
                        # Parse the label (format: "Name. Rating stars. Price. Type")
                        parts = label.split(".")
                        name = parts[0].strip() if parts else label

                        # Try to extract rating
                        rating = None
                        rating_match = re.search(r"([\d.]+)\s*stars?", label, re.I)
                        if rating_match:
                            rating = float(rating_match.group(1))

                        results.append(
                            {
                                "name": name,
                                "rating": rating,
                                "url": href,
                                "full_label": label,
                            }
                        )
                except Exception:
                    continue

            return {
                "query": query,
                "location": location,
                "results": results,
                "count": len(results),
            }

        try:
            result = await context.browser.run_with_page(search)
            return ActionResult.success_result(
                f"Found {result['count']} places for '{query}'",
                result,
            )
        except Exception as e:
            return ActionResult.error_result(f"Local search failed: {e}")


class DistanceAction(Action):
    """Get distance and directions between two locations."""

    name = "get_distance"
    description = "Get distance and travel time between two locations"
    category = "search"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("destination"):
            return False, "destination is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        destination = params["destination"]
        origin = params.get("origin") or params.get("location") or context.location

        if not origin:
            return ActionResult.error_result("I need to know your location to calculate distance. What's your address?")

        # Format origin if it's a dict
        if isinstance(origin, dict) and "address" in origin:
            origin = origin["address"]

        # Build Google Maps directions URL
        origin_encoded = urllib.parse.quote_plus(str(origin))
        dest_encoded = urllib.parse.quote_plus(str(destination))
        url = f"https://www.google.com/maps/dir/{origin_encoded}/{dest_encoded}"

        async def get_distance(page):
            await page.goto(url, wait_until="domcontentloaded")

            # Wait for directions to load
            try:
                await page.wait_for_selector("[data-trip-index]", timeout=15000)
            except Exception:
                # Try alternate selector
                await page.wait_for_selector(".section-directions-trip", timeout=10000)

            results = []

            # Get route options
            route_elements = await page.query_selector_all("[data-trip-index]")
            if not route_elements:
                route_elements = await page.query_selector_all(".section-directions-trip")

            for element in route_elements[:3]:
                try:
                    text = await element.inner_text()
                    lines = [x.strip() for x in text.split("\n") if x.strip()]

                    # Parse duration and distance from the text
                    duration = None
                    distance = None

                    for line in lines:
                        # Look for duration (e.g., "15 min", "1 hr 30 min")
                        if re.search(r"\d+\s*(min|hr|hour)", line, re.I) and not duration:
                            duration = line
                        # Look for distance (e.g., "5.2 mi", "8.4 km")
                        if re.search(r"[\d.]+\s*(mi|km|miles|kilometers)", line, re.I) and not distance:
                            distance = line

                    if duration or distance:
                        results.append(
                            {
                                "duration": duration,
                                "distance": distance,
                                "raw": " | ".join(lines[:3]),
                            }
                        )
                except Exception:
                    continue

            return results

        try:
            routes = await context.browser.run_with_page(get_distance)

            if not routes:
                return ActionResult.error_result(f"Could not find directions from {origin} to {destination}")

            # Format the best route
            best = routes[0]
            duration = best.get("duration", "unknown")
            distance = best.get("distance", "unknown")

            message = f"{destination} is {distance} from your location, about {duration} by car."

            return ActionResult.success_result(
                message,
                {
                    "origin": origin,
                    "destination": destination,
                    "duration": duration,
                    "distance": distance,
                    "routes": routes,
                    "maps_url": url,
                },
            )

        except Exception as e:
            return ActionResult.error_result(f"Distance lookup failed: {e}")


class RestaurantSearchAction(Action):
    """Search for restaurants with filtering options."""

    name = "restaurant_search"
    description = "Search for restaurants with cuisine, rating, and price filters"
    category = "search"

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        # Either query or cuisine must be provided
        if not params.get("query") and not params.get("cuisine"):
            return False, "query or cuisine is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        query = params.get("query", "")
        cuisine = params.get("cuisine")
        location = params.get("location") or context.location
        min_rating = params.get("min_rating", 0)
        max_price = params.get("max_price")  # 1-4 ($ to $$$$)
        open_now = params.get("open_now", False)
        num_results = params.get("num_results", 10)

        # Build search query
        search_parts = []
        if query:
            search_parts.append(query)
        if cuisine:
            search_parts.append(f"{cuisine} restaurant")
        elif not query:
            search_parts.append("restaurant")

        if open_now:
            search_parts.append("open now")

        search_query = " ".join(search_parts)

        if location:
            if isinstance(location, dict) and "address" in location:
                search_query = f"{search_query} near {location['address']}"
            elif isinstance(location, str):
                search_query = f"{search_query} near {location}"

        # Use Google Maps for restaurant search
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://www.google.com/maps/search/{encoded_query}"

        async def search(page):
            await page.goto(url, wait_until="domcontentloaded")

            try:
                await page.wait_for_selector("[role='feed']", timeout=15000)
            except Exception:
                return {
                    "query": search_query,
                    "results": [],
                    "count": 0,
                    "error": "No results found",
                }

            results = []
            result_elements = await page.query_selector_all("[role='feed'] > div > div")

            for element in result_elements:
                if len(results) >= num_results:
                    break

                try:
                    # Find the link/card
                    link = await element.query_selector("a[aria-label]")
                    if not link:
                        continue

                    label = await link.get_attribute("aria-label")
                    href = await link.get_attribute("href")

                    if not label:
                        continue

                    # Parse restaurant info
                    name = label.split(".")[0].strip()

                    # Extract rating
                    rating = None
                    rating_match = re.search(r"([\d.]+)\s*stars?", label, re.I)
                    if rating_match:
                        rating = float(rating_match.group(1))

                    # Skip if below min rating
                    if rating and rating < min_rating:
                        continue

                    # Extract price level (count $ signs)
                    price_level = None
                    price_match = re.search(r"(\$+)", label)
                    if price_match:
                        price_level = len(price_match.group(1))

                    # Skip if above max price
                    if max_price and price_level and price_level > max_price:
                        continue

                    # Get additional details from the card
                    details = {}

                    # Try to get address
                    addr_el = await element.query_selector("[data-tooltip]")
                    if addr_el:
                        details["address"] = await addr_el.inner_text()

                    results.append(
                        {
                            "name": name,
                            "rating": rating,
                            "price_level": price_level,
                            "url": href,
                            **details,
                        }
                    )

                except Exception:
                    continue

            # Sort by rating if available
            results.sort(key=lambda x: x.get("rating") or 0, reverse=True)

            return {
                "query": search_query,
                "filters": {
                    "cuisine": cuisine,
                    "min_rating": min_rating,
                    "max_price": max_price,
                    "open_now": open_now,
                },
                "results": results,
                "count": len(results),
            }

        try:
            result = await context.browser.run_with_page(search)

            if result["count"] == 0:
                return ActionResult.error_result(
                    f"No restaurants found for '{search_query}'",
                    result.get("error"),
                )

            # Format response with top result highlighted
            top = result["results"][0] if result["results"] else None
            message = f"Found {result['count']} restaurants"
            if top:
                rating_str = f" ({top['rating']}*)" if top.get("rating") else ""
                message = f"Top: {top['name']}{rating_str}. {result['count']} total."

            return ActionResult.success_result(message, result)

        except Exception as e:
            return ActionResult.error_result(f"Restaurant search failed: {e}")
