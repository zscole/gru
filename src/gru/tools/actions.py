"""Autonomous action tools for Claude."""

from __future__ import annotations

import logging

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)

_action_engine = None


def set_action_engine(engine) -> None:
    """Set the autonomous action engine."""
    global _action_engine
    _action_engine = engine
    logger.debug("Autonomous action engine set for tools")


async def send_email(to: str, subject: str, body: str, cc: str | None = None) -> dict:
    """Send an email to someone.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body text
        cc: Optional CC recipients
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "send_email",
        {"to": to, "subject": subject, "body": body, "cc": cc},
    )


async def send_slack(channel: str | None = None, user: str | None = None, message: str = "") -> dict:
    """Send a Slack message.

    Args:
        channel: Channel name (without #)
        user: User to DM (if not sending to channel)
        message: Message to send
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "send_slack_message",
        {"channel": channel, "user": user, "message": message},
    )


async def create_event(
    title: str,
    start: str,
    end: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    description: str | None = None,
) -> dict:
    """Create a calendar event.

    Args:
        title: Event title
        start: Start time (e.g., "tomorrow at 3pm", "2024-01-15 14:00")
        end: End time (optional, defaults to 1 hour after start)
        location: Event location
        attendees: List of email addresses to invite
        description: Event description
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "create_calendar_event",
        {
            "title": title,
            "start": start,
            "end": end,
            "location": location,
            "attendees": attendees or [],
            "description": description,
        },
    )


async def book_restaurant(
    restaurant: str,
    date: str,
    time: str,
    party_size: int,
    platform: str = "opentable",
    city: str | None = None,
) -> dict:
    """Book a restaurant reservation.

    Args:
        restaurant: Restaurant name
        date: Date (e.g., "2024-01-15", "tomorrow", "this Saturday")
        time: Time (e.g., "7:00 PM", "19:00")
        party_size: Number of people
        platform: Booking platform - "opentable" or "resy"
        city: City for Resy searches
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    action_type = "resy_reservation" if platform.lower() == "resy" else "opentable_reservation"

    params = {
        "restaurant": restaurant,
        "date": date,
        "time": time,
        "party_size": party_size,
    }
    if city:
        params["city"] = city

    return await _action_engine.request_action(action_type, params)


async def send_venmo(recipient: str, amount: float, note: str = "") -> dict:
    """Send money via Venmo.

    Args:
        recipient: Venmo username, phone number, or email
        amount: Amount to send (max $500 for safety)
        note: Payment note/description
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "venmo_payment",
        {"recipient": recipient, "amount": amount, "note": note},
    )


async def order_doordash(
    restaurant: str,
    items: list[str] | None = None,
    reorder: bool = False,
) -> dict:
    """Order food from DoorDash.

    Args:
        restaurant: Restaurant name
        items: List of items to order (optional if reordering)
        reorder: If True, reorder from a recent order at this restaurant
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "doordash_order",
        {"restaurant": restaurant, "items": items or [], "reorder": reorder},
    )


async def order_amazon(
    item: str | None = None,
    asin: str | None = None,
    quantity: int = 1,
    buy_now: bool = False,
) -> dict:
    """Order an item from Amazon.

    Args:
        item: Item name/search query
        asin: Amazon product ID (if known)
        quantity: Number to order
        buy_now: If True, complete purchase immediately; if False, just add to cart
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.request_action(
        "amazon_order",
        {"item": item, "asin": asin, "quantity": quantity, "buy_now": buy_now},
    )


async def confirm_action(action_id: str) -> dict:
    """Confirm a pending action.

    Args:
        action_id: ID of the action to confirm
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.confirm_action(action_id)


async def cancel_action(action_id: str) -> dict:
    """Cancel a pending action.

    Args:
        action_id: ID of the action to cancel
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return await _action_engine.cancel_action(action_id)


async def list_pending_actions() -> dict:
    """List all pending actions awaiting confirmation."""
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    pending = await _action_engine.get_pending_actions()
    return {"pending_actions": pending, "count": len(pending)}


async def list_available_actions() -> dict:
    """List all available autonomous actions."""
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    return {"actions": _action_engine.registry.list_actions()}


async def action_history(limit: int = 20) -> dict:
    """Get recent action history.

    Args:
        limit: Maximum number of actions to return
    """
    if not _action_engine:
        return {"error": "Action engine not initialized"}

    history = await _action_engine.get_action_history(limit=limit)
    return {"history": history, "count": len(history)}


def register_action_tools() -> None:
    """Register autonomous action tools."""

    register_tool(
        name="send_email",
        description="Send an email. Requires confirmation before sending.",
        parameters={
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body text"},
            "cc": {"type": "string", "description": "CC recipients (comma-separated)", "optional": True},
        },
        handler=send_email,
    )

    register_tool(
        name="send_slack",
        description="Send a Slack message to a channel or user. Requires confirmation.",
        parameters={
            "channel": {"type": "string", "description": "Channel name (without #)", "optional": True},
            "user": {"type": "string", "description": "User to DM", "optional": True},
            "message": {"type": "string", "description": "Message to send"},
        },
        handler=send_slack,
    )

    register_tool(
        name="create_event",
        description="Create a calendar event. Requires confirmation.",
        parameters={
            "title": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "Start time (e.g., 'tomorrow at 3pm')"},
            "end": {"type": "string", "description": "End time", "optional": True},
            "location": {"type": "string", "description": "Event location", "optional": True},
            "attendees": {"type": "array", "description": "Email addresses to invite", "optional": True},
            "description": {"type": "string", "description": "Event description", "optional": True},
        },
        handler=create_event,
    )

    register_tool(
        name="book_restaurant",
        description="Book a restaurant reservation via OpenTable or Resy. Requires confirmation.",
        parameters={
            "restaurant": {"type": "string", "description": "Restaurant name"},
            "date": {"type": "string", "description": "Date (e.g., 'tomorrow', 'this Saturday')"},
            "time": {"type": "string", "description": "Time (e.g., '7:00 PM')"},
            "party_size": {"type": "integer", "description": "Number of people"},
            "platform": {"type": "string", "description": "Platform: 'opentable' or 'resy'", "optional": True},
            "city": {"type": "string", "description": "City for Resy searches", "optional": True},
        },
        handler=book_restaurant,
    )

    register_tool(
        name="send_venmo",
        description="Send money via Venmo. Requires confirmation. Max $500.",
        parameters={
            "recipient": {"type": "string", "description": "Venmo username, phone, or email"},
            "amount": {"type": "number", "description": "Amount to send"},
            "note": {"type": "string", "description": "Payment note", "optional": True},
        },
        handler=send_venmo,
    )

    register_tool(
        name="order_doordash",
        description="Order food from DoorDash. Requires confirmation.",
        parameters={
            "restaurant": {"type": "string", "description": "Restaurant name"},
            "items": {"type": "array", "description": "Items to order", "optional": True},
            "reorder": {"type": "boolean", "description": "Reorder previous order", "optional": True},
        },
        handler=order_doordash,
    )

    register_tool(
        name="order_amazon",
        description="Order from Amazon. Requires confirmation for Buy Now.",
        parameters={
            "item": {"type": "string", "description": "Item name/search", "optional": True},
            "asin": {"type": "string", "description": "Amazon product ID", "optional": True},
            "quantity": {"type": "integer", "description": "Quantity", "optional": True},
            "buy_now": {"type": "boolean", "description": "Complete purchase immediately", "optional": True},
        },
        handler=order_amazon,
    )

    register_tool(
        name="confirm_action",
        description="Confirm a pending action to execute it.",
        parameters={
            "action_id": {"type": "string", "description": "ID of the action to confirm"},
        },
        handler=confirm_action,
    )

    register_tool(
        name="cancel_action",
        description="Cancel a pending action.",
        parameters={
            "action_id": {"type": "string", "description": "ID of the action to cancel"},
        },
        handler=cancel_action,
    )

    register_tool(
        name="list_pending_actions",
        description="List all pending actions awaiting confirmation.",
        parameters={},
        handler=list_pending_actions,
    )

    register_tool(
        name="list_available_actions",
        description="List all autonomous actions Gru can perform.",
        parameters={},
        handler=list_available_actions,
    )

    register_tool(
        name="action_history",
        description="Get history of recent autonomous actions.",
        parameters={
            "limit": {"type": "integer", "description": "Max actions to return", "optional": True},
        },
        handler=action_history,
    )
