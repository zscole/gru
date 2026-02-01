"""Google Maps tools - directions, places, distance."""

from __future__ import annotations

import logging
import os

import httpx

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)

MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


async def get_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
) -> dict:
    """Get directions between two locations via Google Maps API."""
    if not MAPS_API_KEY:
        return {"error": "GOOGLE_MAPS_API_KEY not configured"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "key": MAPS_API_KEY,
            },
            timeout=30,
        )
        data = resp.json()

    if data["status"] != "OK":
        return {"error": data.get("error_message", data["status"])}

    route = data["routes"][0]["legs"][0]
    return {
        "distance": route["distance"]["text"],
        "duration": route["duration"]["text"],
        "start_address": route["start_address"],
        "end_address": route["end_address"],
        "summary": f"{route['distance']['text']}, about {route['duration']['text']} by {mode}",
    }


async def find_places(
    query: str,
    location: str | None = None,
    radius: int = 5000,
) -> dict:
    """Find places via Google Maps API."""
    if not MAPS_API_KEY:
        return {"error": "GOOGLE_MAPS_API_KEY not configured"}

    params: dict = {
        "query": query,
        "key": MAPS_API_KEY,
    }

    # Geocode location if provided
    if location:
        async with httpx.AsyncClient() as client:
            geo_resp = await client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": location, "key": MAPS_API_KEY},
                timeout=30,
            )
            geo_data = geo_resp.json()

        if geo_data["status"] == "OK":
            lat = geo_data["results"][0]["geometry"]["location"]["lat"]
            lng = geo_data["results"][0]["geometry"]["location"]["lng"]
            params["location"] = f"{lat},{lng}"
            params["radius"] = radius

    # Search for places
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params=params,
            timeout=30,
        )
        data = resp.json()

    if data["status"] != "OK":
        return {"error": data.get("error_message", data["status"])}

    results = []
    for p in data["results"][:10]:
        place = {
            "name": p["name"],
            "address": p.get("formatted_address"),
        }
        if p.get("rating"):
            place["rating"] = p["rating"]
        if p.get("price_level"):
            place["price_level"] = "$" * p["price_level"]
        if p.get("opening_hours", {}).get("open_now") is not None:
            place["open_now"] = p["opening_hours"]["open_now"]
        results.append(place)

    return {"results": results, "count": len(results)}


async def get_place_details(place_name: str, location: str | None = None) -> dict:
    """Get detailed information about a specific place."""
    # First find the place
    places = await find_places(place_name, location, radius=10000)
    if "error" in places:
        return places

    if not places.get("results"):
        return {"error": f"Could not find place: {place_name}"}

    # Return the first match with available details
    place = places["results"][0]
    return {
        "name": place["name"],
        "address": place.get("address"),
        "rating": place.get("rating"),
        "price_level": place.get("price_level"),
        "open_now": place.get("open_now"),
    }


def register_maps_tools() -> None:
    """Register all maps tools."""
    register_tool(
        name="get_directions",
        description="Get directions, distance, and travel time between two locations. Use this when someone asks 'how far', 'how long to get to', 'directions to', etc.",
        parameters={
            "origin": {
                "type": "string",
                "description": "Starting address or location",
            },
            "destination": {
                "type": "string",
                "description": "Destination address or location",
            },
            "mode": {
                "type": "string",
                "description": "Travel mode",
                "enum": ["driving", "walking", "bicycling", "transit"],
                "optional": True,
            },
        },
        handler=get_directions,
    )

    register_tool(
        name="find_places",
        description="Search for places like restaurants, stores, services, etc. Use this when someone asks to find places nearby or in an area.",
        parameters={
            "query": {
                "type": "string",
                "description": "What to search for (e.g., 'sushi restaurant', 'coffee shop', 'pharmacy')",
            },
            "location": {
                "type": "string",
                "description": "Search near this address or location",
                "optional": True,
            },
            "radius": {
                "type": "integer",
                "description": "Search radius in meters (default 5000)",
                "optional": True,
            },
        },
        handler=find_places,
    )

    register_tool(
        name="get_place_details",
        description="Get detailed information about a specific place including address, rating, hours.",
        parameters={
            "place_name": {
                "type": "string",
                "description": "Name of the place to look up",
            },
            "location": {
                "type": "string",
                "description": "Location to search near (helps find the right place)",
                "optional": True,
            },
        },
        handler=get_place_details,
    )
