"""Calendar action handlers - Create, update, delete events."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from gru.actions.autonomous import (
    ActionCategory,
    ActionHandler,
    ActionPreview,
    ActionResult,
)

logger = logging.getLogger(__name__)

_google_connector = None


def set_google_connector(connector) -> None:
    global _google_connector
    _google_connector = connector


class CreateEventHandler(ActionHandler):
    """Create a Google Calendar event."""

    @property
    def action_type(self) -> str:
        return "create_calendar_event"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.CALENDAR

    @property
    def description(self) -> str:
        return "Create a calendar event"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("title"):
            return False, "Event title required"
        if not params.get("start"):
            return False, "Start time required"
        if not _google_connector:
            return False, "Google Calendar not connected"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        title = params["title"]
        start = params["start"]
        end = params.get("end", "1 hour after start")
        attendees = params.get("attendees", [])
        location = params.get("location", "")

        details = [
            f"Event: {title}",
            f"When: {start}",
        ]
        if end != "1 hour after start":
            details.append(f"Until: {end}")
        if location:
            details.append(f"Location: {location}")
        if attendees:
            details.append(f"Attendees: {', '.join(attendees)}")

        return ActionPreview(
            summary=f"Create event: {title}",
            details=details,
            reversible=True,
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Google Calendar not connected")

        try:
            title = params["title"]
            start = params["start"]
            end = params.get("end")
            attendees = params.get("attendees", [])
            location = params.get("location", "")
            description = params.get("description", "")

            # Parse start time
            start_dt = self._parse_datetime(start)
            if not start_dt:
                return ActionResult(success=False, message=f"Could not parse start time: {start}")

            # Calculate end time
            end_dt = self._parse_datetime(end) if end else start_dt + timedelta(hours=1)

            # Build event
            event = {
                "summary": title,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": "America/Los_Angeles",
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": "America/Los_Angeles",
                },
            }

            if location:
                event["location"] = location
            if description:
                event["description"] = description
            if attendees:
                event["attendees"] = [{"email": a} for a in attendees]

            # Create event
            result = (
                _google_connector._calendar_service.events()
                .insert(
                    calendarId="primary",
                    body=event,
                    sendUpdates="all" if attendees else "none",
                )
                .execute()
            )

            return ActionResult(
                success=True,
                message=f"Created event: {title}",
                data={"event_id": result.get("id"), "link": result.get("htmlLink")},
                undo_available=True,
                undo_data={"event_id": result.get("id")},
            )

        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return ActionResult(success=False, message=f"Failed: {e}")

    def _parse_datetime(self, dt_str: str) -> datetime | None:
        """Parse various datetime formats."""
        import re

        from dateutil import parser as dateparser

        try:
            # Try dateutil parser first
            return dateparser.parse(dt_str)
        except Exception:
            pass

        # Try relative times
        now = datetime.now()

        if "tomorrow" in dt_str.lower():
            base = now + timedelta(days=1)
            time_match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", dt_str.lower())
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                if time_match.group(3) == "pm" and hour < 12:
                    hour += 12
                return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return base.replace(hour=9, minute=0, second=0, microsecond=0)

        return None

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Google Calendar not connected")

        try:
            event_id = undo_data["event_id"]
            _google_connector._calendar_service.events().delete(
                calendarId="primary",
                eventId=event_id,
            ).execute()
            return ActionResult(success=True, message="Event deleted")
        except Exception as e:
            return ActionResult(success=False, message=f"Could not delete: {e}")


class UpdateEventHandler(ActionHandler):
    """Update an existing calendar event."""

    @property
    def action_type(self) -> str:
        return "update_calendar_event"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.CALENDAR

    @property
    def description(self) -> str:
        return "Update a calendar event"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("event_id") and not params.get("event_title"):
            return False, "Event ID or title required"
        if not _google_connector:
            return False, "Google Calendar not connected"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        event_id = params.get("event_id", "")
        event_title = params.get("event_title", "")
        changes = []

        if params.get("new_title"):
            changes.append(f"Title -> {params['new_title']}")
        if params.get("new_start"):
            changes.append(f"Start -> {params['new_start']}")
        if params.get("new_location"):
            changes.append(f"Location -> {params['new_location']}")
        if params.get("add_attendees"):
            changes.append(f"Add attendees: {', '.join(params['add_attendees'])}")

        return ActionPreview(
            summary=f"Update event: {event_title or event_id}",
            details=changes or ["No changes specified"],
            reversible=True,
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Google Calendar not connected")

        try:
            event_id = params.get("event_id")

            # Find event by title if no ID
            if not event_id and params.get("event_title"):
                event_id = await self._find_event_by_title(params["event_title"])
                if not event_id:
                    return ActionResult(success=False, message=f"Could not find event: {params['event_title']}")

            # Get current event
            event = (
                _google_connector._calendar_service.events()
                .get(
                    calendarId="primary",
                    eventId=event_id,
                )
                .execute()
            )

            # Store original for undo
            original = dict(event)

            # Apply updates
            if params.get("new_title"):
                event["summary"] = params["new_title"]
            if params.get("new_location"):
                event["location"] = params["new_location"]
            if params.get("new_description"):
                event["description"] = params["new_description"]
            if params.get("add_attendees"):
                existing = event.get("attendees", [])
                for email in params["add_attendees"]:
                    existing.append({"email": email})
                event["attendees"] = existing

            # Update
            result = (
                _google_connector._calendar_service.events()
                .update(
                    calendarId="primary",
                    eventId=event_id,
                    body=event,
                    sendUpdates="all",
                )
                .execute()
            )

            return ActionResult(
                success=True,
                message=f"Updated event: {result.get('summary')}",
                data={"event_id": event_id},
                undo_available=True,
                undo_data={"event_id": event_id, "original": original},
            )

        except Exception as e:
            logger.error(f"Failed to update event: {e}")
            return ActionResult(success=False, message=f"Failed: {e}")

    async def _find_event_by_title(self, title: str) -> str | None:
        """Find event ID by title."""
        try:
            now = datetime.utcnow().isoformat() + "Z"
            results = (
                _google_connector._calendar_service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=50,
                    singleEvents=True,
                    q=title,
                )
                .execute()
            )

            events = results.get("items", [])
            for event in events:
                if title.lower() in event.get("summary", "").lower():
                    return event["id"]
            return None
        except Exception:
            return None

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Google Calendar not connected")

        try:
            event_id = undo_data["event_id"]
            original = undo_data["original"]

            _google_connector._calendar_service.events().update(
                calendarId="primary",
                eventId=event_id,
                body=original,
            ).execute()
            return ActionResult(success=True, message="Event restored")
        except Exception as e:
            return ActionResult(success=False, message=f"Could not restore: {e}")


class DeleteEventHandler(ActionHandler):
    """Delete a calendar event."""

    @property
    def action_type(self) -> str:
        return "delete_calendar_event"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.CALENDAR

    @property
    def description(self) -> str:
        return "Delete a calendar event"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("event_id") and not params.get("event_title"):
            return False, "Event ID or title required"
        if not _google_connector:
            return False, "Google Calendar not connected"
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        event_title = params.get("event_title", params.get("event_id", ""))

        return ActionPreview(
            summary=f"Delete event: {event_title}",
            details=[
                f"Event: {event_title}",
                "This will cancel the event and notify attendees",
            ],
            reversible=False,
            warnings=["Deleted events cannot be recovered"],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Google Calendar not connected")

        try:
            event_id = params.get("event_id")

            # Find event by title if no ID
            if not event_id and params.get("event_title"):
                event_id = await UpdateEventHandler()._find_event_by_title(params["event_title"])
                if not event_id:
                    return ActionResult(success=False, message=f"Could not find event: {params['event_title']}")

            _google_connector._calendar_service.events().delete(
                calendarId="primary",
                eventId=event_id,
                sendUpdates="all",
            ).execute()

            return ActionResult(
                success=True,
                message="Event deleted",
                data={"event_id": event_id},
            )

        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return ActionResult(success=False, message=f"Failed: {e}")
