"""Google tools - Calendar, Gmail, Docs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.connectors.google import GoogleConnector

logger = logging.getLogger(__name__)

# Connector set by orchestrator
_connector: GoogleConnector | None = None


def set_google_connector(connector: GoogleConnector) -> None:
    """Set the Google connector (called by orchestrator)."""
    global _connector
    _connector = connector
    logger.debug("Google connector set for tools")


def _get_connector():
    """Get the Google connector."""
    global _connector
    # Try to get from orchestrator first
    if _connector is not None:
        return _connector

    # Fallback: try to load from disk (for CLI usage)
    from gru.connectors.google import GoogleConnector
    data_dir = Path.home() / ".gru"
    fallback = GoogleConnector(data_dir)
    if fallback.load_token():
        _connector = fallback
        return _connector

    return None


async def get_calendar_events(days_ahead: int = 1) -> dict:
    """Get upcoming calendar events."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Google not configured. Run 'gru google setup' and 'gru google login' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Google. Run 'gru google login' first."}

    try:
        from datetime import timezone

        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

        events_result = connector._calendar_service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=20,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = events_result.get("items", [])

        results = []
        for event in events:
            start = event.get("start", {})
            start_time = start.get("dateTime", start.get("date", ""))

            # Parse time for display
            time_str = ""
            if "T" in start_time:
                try:
                    dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    time_str = dt.strftime("%I:%M %p")
                except:
                    time_str = start_time
            else:
                time_str = "All day"

            results.append({
                "summary": event.get("summary", "Untitled"),
                "time": time_str,
                "location": event.get("location"),
                "description": event.get("description", "")[:100] if event.get("description") else None,
            })

        return {
            "events": results,
            "count": len(results),
            "period": f"next {days_ahead} day(s)",
        }

    except Exception as e:
        logger.error(f"Calendar fetch failed: {e}")
        return {"error": str(e)}


async def get_emails(max_results: int = 10, unread_only: bool = True) -> dict:
    """Get recent emails."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Google not configured. Run 'gru google setup' and 'gru google login' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Google. Run 'gru google login' first."}

    try:
        query = "is:unread" if unread_only else ""

        results = connector._gmail_service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results,
        ).execute()

        messages = results.get("messages", [])
        emails = []

        for msg_meta in messages[:max_results]:
            msg_id = msg_meta.get("id", "")

            msg = connector._gmail_service.users().messages().get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

            from_header = headers.get("From", "Unknown")
            if "<" in from_header:
                sender = from_header.split("<")[0].strip().strip('"')
            else:
                sender = from_header.split("@")[0]

            emails.append({
                "from": sender,
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
            })

        return {
            "emails": emails,
            "count": len(emails),
            "unread_only": unread_only,
        }

    except Exception as e:
        logger.error(f"Email fetch failed: {e}")
        return {"error": str(e)}


async def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Google not configured. Run 'gru google setup' and 'gru google login' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Google. Run 'gru google login' first."}

    try:
        result = await connector.send_email(to, subject, body)
        return {
            "status": "sent",
            "to": to,
            "subject": subject,
        }
    except Exception as e:
        logger.error(f"Send email failed: {e}")
        return {"error": str(e)}


async def create_doc(title: str, content: str = "") -> dict:
    """Create a Google Doc."""
    connector = _get_connector()

    if connector is None:
        return {"error": "Google not configured. Run 'gru google setup' and 'gru google login' first."}

    if not connector.is_authenticated():
        return {"error": "Not authenticated with Google. Run 'gru google login' first."}

    try:
        result = await connector.create_document(title, content)
        return {
            "status": "created",
            "title": title,
            "url": result.get("url"),
            "document_id": result.get("document_id"),
        }
    except Exception as e:
        logger.error(f"Create doc failed: {e}")
        return {"error": str(e)}


def register_google_tools() -> None:
    """Register Google tools."""
    register_tool(
        name="get_calendar",
        description="Get upcoming calendar events. Use this when user asks about their schedule, meetings, or what's on their calendar.",
        parameters={
            "days_ahead": {
                "type": "integer",
                "description": "Number of days to look ahead (default 1)",
                "optional": True,
            },
        },
        handler=get_calendar_events,
    )

    register_tool(
        name="get_emails",
        description="Get recent emails. Use this when user asks about their inbox or emails.",
        parameters={
            "max_results": {
                "type": "integer",
                "description": "Maximum emails to return (default 10)",
                "optional": True,
            },
            "unread_only": {
                "type": "boolean",
                "description": "Only show unread emails (default true)",
                "optional": True,
            },
        },
        handler=get_emails,
    )

    register_tool(
        name="send_email",
        description="Send an email via Gmail.",
        parameters={
            "to": {
                "type": "string",
                "description": "Recipient email address",
            },
            "subject": {
                "type": "string",
                "description": "Email subject",
            },
            "body": {
                "type": "string",
                "description": "Email body text",
            },
        },
        handler=send_email,
    )

    register_tool(
        name="create_doc",
        description="Create a new Google Doc.",
        parameters={
            "title": {
                "type": "string",
                "description": "Document title",
            },
            "content": {
                "type": "string",
                "description": "Initial content for the document",
                "optional": True,
            },
        },
        handler=create_doc,
    )
