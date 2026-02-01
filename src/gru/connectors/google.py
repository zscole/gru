"""Google Calendar and Gmail connector for Gru."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

if TYPE_CHECKING:
    from gru.proactive import ProactiveEngine

logger = logging.getLogger(__name__)

# Scopes for Calendar, Gmail, and Docs
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",  # For creating docs
]


class GoogleConnector:
    """Connector for Google Calendar and Gmail."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.credentials_path = data_dir / "google_credentials.json"
        self.token_path = data_dir / "google_token.json"
        self._credentials: Credentials | None = None
        self._calendar_service: Any = None
        self._gmail_service: Any = None
        self._docs_service: Any = None
        self._drive_service: Any = None
        self._last_calendar_sync: datetime | None = None
        self._last_email_sync: datetime | None = None
        self._seen_email_ids: set[str] = set()
        self._seen_event_ids: set[str] = set()

    def is_configured(self) -> bool:
        """Check if Google credentials are configured."""
        return self.credentials_path.exists()

    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._credentials is not None and self._credentials.valid

    async def setup_credentials(self, client_id: str, client_secret: str) -> None:
        """Set up OAuth credentials file."""
        credentials_data = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        self.credentials_path.write_text(json.dumps(credentials_data, indent=2))
        logger.info("Google credentials saved")

    def authenticate(self, headless: bool = False) -> bool:
        """Perform OAuth authentication flow.

        Args:
            headless: If True, print URL for manual auth instead of opening browser

        Returns:
            True if authentication succeeded
        """
        if not self.is_configured():
            logger.error("Google credentials not configured. Run 'gru google setup' first.")
            return False

        creds = None

        # Load existing token if available
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception as e:
                logger.warning(f"Failed to load existing token: {e}")

        # Refresh or get new credentials
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}")
                creds = None

        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                if headless:
                    # For headless/remote environments
                    creds = flow.run_console()
                else:
                    # Opens browser for auth
                    creds = flow.run_local_server(port=0)
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return False

        # Save token for future use
        self.token_path.write_text(creds.to_json())
        self._credentials = creds

        # Initialize services
        self._calendar_service = build("calendar", "v3", credentials=creds)
        self._gmail_service = build("gmail", "v1", credentials=creds)
        self._docs_service = build("docs", "v1", credentials=creds)
        self._drive_service = build("drive", "v3", credentials=creds)

        logger.info("Google authentication successful")
        return True

    def load_token(self) -> bool:
        """Load saved token without full auth flow."""
        if not self.token_path.exists():
            return False

        try:
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json())

            if creds.valid:
                self._credentials = creds
                self._calendar_service = build("calendar", "v3", credentials=creds)
                self._gmail_service = build("gmail", "v1", credentials=creds)
                self._docs_service = build("docs", "v1", credentials=creds)
                self._drive_service = build("drive", "v3", credentials=creds)
                return True
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")

        return False

    async def sync_calendar(
        self,
        proactive: ProactiveEngine,
        hours_ahead: int = 24,
    ) -> list[dict[str, Any]]:
        """Sync upcoming calendar events and create observations.

        Args:
            proactive: ProactiveEngine to add observations to
            hours_ahead: How many hours ahead to look

        Returns:
            List of events found
        """
        if not self._calendar_service and not self.load_token():
            logger.warning("Not authenticated with Google")
            return []

        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(hours=hours_ahead)).isoformat() + "Z"

        try:
            events_result = (
                self._calendar_service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=20,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            new_events = []

            for event in events:
                event_id = event.get("id", "")
                if event_id in self._seen_event_ids:
                    continue

                self._seen_event_ids.add(event_id)
                new_events.append(event)

                # Parse event details
                summary = event.get("summary", "Untitled event")
                start = event.get("start", {})
                start_time = start.get("dateTime", start.get("date", ""))

                # Calculate time until event
                try:
                    if "T" in start_time:
                        event_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        hours_until = (event_dt.replace(tzinfo=None) - now).total_seconds() / 3600
                    else:
                        # All-day event
                        hours_until = 24
                except Exception:
                    hours_until = 24

                # Determine importance based on proximity
                if hours_until <= 1:
                    importance = 0.95
                    category = "deadline"
                elif hours_until <= 4:
                    importance = 0.8
                    category = "reminder"
                else:
                    importance = 0.6
                    category = "reminder"

                # Get attendees
                attendees = event.get("attendees", [])
                attendee_names = [a.get("email", "").split("@")[0] for a in attendees[:3]]

                # Create observation
                content = f"Calendar: {summary}"
                if attendee_names:
                    content += f" (with {', '.join(attendee_names)})"
                if hours_until <= 24:
                    if hours_until < 1:
                        content += f" in {int(hours_until * 60)} minutes"
                    else:
                        content += f" in {hours_until:.1f} hours"

                await proactive.add_observation(
                    content=content,
                    category=category,
                    importance=importance,
                    source="google_calendar",
                    expires_in_hours=int(hours_until) + 1,
                    metadata={
                        "event_id": event_id,
                        "start_time": start_time,
                        "attendees": attendee_names,
                    },
                )

            self._last_calendar_sync = datetime.now()
            logger.info(f"Calendar sync: {len(new_events)} new events")
            return new_events

        except Exception as e:
            logger.error(f"Calendar sync failed: {e}")
            return []

    async def sync_email(
        self,
        proactive: ProactiveEngine,
        max_results: int = 10,
        important_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Sync recent emails and create observations for important ones.

        Args:
            proactive: ProactiveEngine to add observations to
            max_results: Maximum emails to fetch
            important_only: Only fetch emails marked important

        Returns:
            List of emails found
        """
        if not self._gmail_service and not self.load_token():
            logger.warning("Not authenticated with Google")
            return []

        try:
            # Build query
            query = "is:unread"
            if important_only:
                query += " is:important"

            results = (
                self._gmail_service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=max_results,
                )
                .execute()
            )

            messages = results.get("messages", [])
            new_emails = []

            for msg_meta in messages:
                msg_id = msg_meta.get("id", "")
                if msg_id in self._seen_email_ids:
                    continue

                self._seen_email_ids.add(msg_id)

                # Fetch full message
                try:
                    msg = (
                        self._gmail_service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=msg_id,
                            format="metadata",
                            metadataHeaders=["From", "Subject", "Date"],
                        )
                        .execute()
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch email {msg_id}: {e}")
                    continue

                new_emails.append(msg)

                # Parse headers
                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                from_header = headers.get("From", "Unknown")
                subject = headers.get("Subject", "(no subject)")

                # Extract sender name
                if "<" in from_header:
                    sender = from_header.split("<")[0].strip().strip('"')
                else:
                    sender = from_header.split("@")[0]

                # Check labels for importance signals
                labels = msg.get("labelIds", [])
                is_important = "IMPORTANT" in labels
                is_starred = "STARRED" in labels

                # Determine importance
                if is_starred:
                    importance = 0.9
                elif is_important:
                    importance = 0.8
                else:
                    importance = 0.5

                # Skip low-importance emails for observations
                if importance < 0.6:
                    continue

                # Create observation
                content = f"Email from {sender}: {subject[:60]}"
                if len(subject) > 60:
                    content += "..."

                await proactive.add_observation(
                    content=content,
                    category="follow_up",
                    importance=importance,
                    source="gmail",
                    expires_in_hours=48,
                    metadata={
                        "message_id": msg_id,
                        "from": from_header,
                        "subject": subject,
                    },
                )

            self._last_email_sync = datetime.now()
            logger.info(f"Email sync: {len(new_emails)} new emails")
            return new_emails

        except Exception as e:
            logger.error(f"Email sync failed: {e}")
            return []

    async def sync_all(self, proactive: ProactiveEngine) -> dict[str, int]:
        """Sync both calendar and email.

        Returns:
            Dict with counts of new items
        """
        events = await self.sync_calendar(proactive)
        emails = await self.sync_email(proactive)

        return {
            "events": len(events),
            "emails": len(emails),
        }

    async def create_document(
        self,
        title: str,
        content: str | None = None,
    ) -> dict[str, Any]:
        """Create a new Google Doc.

        Args:
            title: Document title
            content: Optional initial content (plain text or simple formatting)

        Returns:
            Dict with document ID, title, and URL
        """
        if not self._docs_service and not self.load_token():
            raise RuntimeError("Not authenticated with Google")

        try:
            # Create the document
            doc = self._docs_service.documents().create(body={"title": title}).execute()

            doc_id = doc.get("documentId")
            doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

            logger.info(f"Created Google Doc: {title} ({doc_id})")

            # Add content if provided
            if content and doc_id:
                await self.write_to_document(doc_id, content)

            return {
                "document_id": doc_id,
                "title": title,
                "url": doc_url,
            }

        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            raise

    async def write_to_document(
        self,
        document_id: str,
        content: str,
        insert_at_end: bool = True,
    ) -> bool:
        """Write content to a Google Doc.

        Args:
            document_id: The document ID
            content: Text content to write
            insert_at_end: If True, append to end; if False, insert at beginning

        Returns:
            True if successful
        """
        if not self._docs_service and not self.load_token():
            raise RuntimeError("Not authenticated with Google")

        try:
            # Get current document to find end index
            doc = self._docs_service.documents().get(documentId=document_id).execute()

            # Find insertion index
            if insert_at_end:
                # Get the end of the document body
                end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)
                index = max(1, end_index - 1)
            else:
                index = 1

            # Build the request
            requests = [
                {
                    "insertText": {
                        "location": {"index": index},
                        "text": content,
                    }
                }
            ]

            self._docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()

            logger.info(f"Wrote {len(content)} chars to doc {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write to document: {e}")
            raise

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
    ) -> dict[str, Any]:
        """Send an email via Gmail.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text or HTML)
            html: If True, body is HTML

        Returns:
            Dict with message ID and thread ID
        """
        if not self._gmail_service and not self.load_token():
            raise RuntimeError("Not authenticated with Google")

        import base64
        from email.mime.text import MIMEText

        try:
            # Create message
            mime_type = "html" if html else "plain"
            message = MIMEText(body, mime_type)
            message["to"] = to
            message["subject"] = subject

            # Encode the message
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

            # Send
            result = self._gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()

            logger.info(f"Sent email to {to}: {subject}")

            return {
                "message_id": result.get("id"),
                "thread_id": result.get("threadId"),
                "to": to,
                "subject": subject,
            }

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise

    async def get_user_email(self) -> str | None:
        """Get the authenticated user's email address."""
        if not self._gmail_service and not self.load_token():
            return None

        try:
            profile = self._gmail_service.users().getProfile(userId="me").execute()
            return profile.get("emailAddress")
        except Exception as e:
            logger.warning(f"Failed to get user email: {e}")
            return None

    def get_status(self) -> dict[str, Any]:
        """Get connector status."""
        return {
            "configured": self.is_configured(),
            "authenticated": self.is_authenticated(),
            "last_calendar_sync": self._last_calendar_sync.isoformat() if self._last_calendar_sync else None,
            "last_email_sync": self._last_email_sync.isoformat() if self._last_email_sync else None,
            "seen_events": len(self._seen_event_ids),
            "seen_emails": len(self._seen_email_ids),
        }

    def clear_seen_cache(self) -> None:
        """Clear the seen items cache (useful for re-syncing)."""
        self._seen_email_ids.clear()
        self._seen_event_ids.clear()
        logger.info("Cleared seen items cache")


async def setup_google_triggers(
    proactive: ProactiveEngine,
    connector: GoogleConnector,
) -> None:
    """Set up automatic sync triggers for Google services."""
    from gru.proactive import TriggerType

    # Check if triggers already exist
    existing = await proactive.list_triggers()
    existing_names = {t["name"] for t in existing}

    if "google_calendar_sync" not in existing_names:
        await proactive.add_trigger(
            name="google_calendar_sync",
            trigger_type=TriggerType.INTERVAL,
            action="check:google_calendar",
            interval_minutes=15,
            config={"connector": "google"},
        )

    if "google_email_sync" not in existing_names:
        await proactive.add_trigger(
            name="google_email_sync",
            trigger_type=TriggerType.INTERVAL,
            action="check:google_email",
            interval_minutes=5,
            config={"connector": "google"},
        )

    logger.info("Google sync triggers configured")
