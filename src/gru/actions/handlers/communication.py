"""Communication action handlers - Email, Slack, SMS."""

from __future__ import annotations

import base64
import logging
from email.mime.text import MIMEText
from typing import Any

from gru.actions.autonomous import (
    ActionCategory,
    ActionHandler,
    ActionPreview,
    ActionResult,
)

logger = logging.getLogger(__name__)

# Connectors set by orchestrator
_google_connector = None
_slack_connector = None


def set_google_connector(connector) -> None:
    global _google_connector
    _google_connector = connector


def set_slack_connector(connector) -> None:
    global _slack_connector
    _slack_connector = connector


class SendEmailHandler(ActionHandler):
    """Send an email via Gmail."""

    @property
    def action_type(self) -> str:
        return "send_email"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.COMMUNICATION

    @property
    def description(self) -> str:
        return "Send an email"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("to"):
            return False, "Recipient email required"
        if not params.get("subject"):
            return False, "Subject required"
        if not params.get("body"):
            return False, "Email body required"
        if not _google_connector:
            return False, "Gmail not connected. Run 'gru google login' first."
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        to = params["to"]
        subject = params["subject"]
        body = params["body"]

        return ActionPreview(
            summary=f"Send email to {to}: {subject[:50]}",
            details=[
                f"To: {to}",
                f"Subject: {subject}",
                f"Body: {body[:100]}..." if len(body) > 100 else f"Body: {body}",
            ],
            reversible=False,
            warnings=["Email cannot be unsent once delivered"],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        if not _google_connector:
            return ActionResult(success=False, message="Gmail not connected")

        try:
            to = params["to"]
            subject = params["subject"]
            body = params["body"]
            cc = params.get("cc")
            bcc = params.get("bcc")

            # Create message
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc

            # Encode and send
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            result = _google_connector._gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()

            return ActionResult(
                success=True,
                message=f"Email sent to {to}",
                data={"message_id": result.get("id")},
            )

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return ActionResult(success=False, message=f"Failed to send email: {e}")


class SendSlackMessageHandler(ActionHandler):
    """Send a Slack message."""

    @property
    def action_type(self) -> str:
        return "send_slack_message"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.COMMUNICATION

    @property
    def description(self) -> str:
        return "Send a Slack message"

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("channel") and not params.get("user"):
            return False, "Channel or user required"
        if not params.get("message"):
            return False, "Message required"
        if not _slack_connector:
            return False, "Slack not connected. Run 'gru slack setup' first."
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        target = params.get("channel") or params.get("user")
        message = params["message"]

        return ActionPreview(
            summary=f"Send Slack message to {target}",
            details=[
                f"To: {target}",
                f"Message: {message[:100]}..." if len(message) > 100 else f"Message: {message}",
            ],
            reversible=True,  # Can delete message
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        if not _slack_connector:
            return ActionResult(success=False, message="Slack not connected")

        try:
            channel = params.get("channel")
            user = params.get("user")
            message = params["message"]

            # If user specified, find DM channel
            if user and not channel:
                channel = await _slack_connector.get_dm_channel(user)
                if not channel:
                    return ActionResult(success=False, message=f"Could not find user: {user}")

            # Send message
            result = await _slack_connector.send_message(channel, message)

            return ActionResult(
                success=True,
                message=f"Slack message sent to {channel}",
                data={"channel": channel, "ts": result.get("ts")},
                undo_available=True,
                undo_data={"channel": channel, "ts": result.get("ts")},
            )

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return ActionResult(success=False, message=f"Failed: {e}")

    async def undo(self, params: dict[str, Any], undo_data: dict[str, Any]) -> ActionResult:
        if not _slack_connector:
            return ActionResult(success=False, message="Slack not connected")

        try:
            channel = undo_data["channel"]
            ts = undo_data["ts"]
            await _slack_connector.delete_message(channel, ts)
            return ActionResult(success=True, message="Message deleted")
        except Exception as e:
            return ActionResult(success=False, message=f"Could not delete: {e}")


class SendSMSHandler(ActionHandler):
    """Send an SMS via Twilio."""

    @property
    def action_type(self) -> str:
        return "send_sms"

    @property
    def category(self) -> ActionCategory:
        return ActionCategory.COMMUNICATION

    @property
    def description(self) -> str:
        return "Send an SMS text message"

    @property
    def requires_confirmation(self) -> bool:
        return True

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        if not params.get("to"):
            return False, "Phone number required"
        if not params.get("message"):
            return False, "Message required"
        # Check for Twilio credentials in environment
        import os

        if not os.environ.get("TWILIO_ACCOUNT_SID"):
            return False, "Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
        return True, ""

    async def preview(self, params: dict[str, Any]) -> ActionPreview:
        to = params["to"]
        message = params["message"]

        return ActionPreview(
            summary=f"Send SMS to {to}",
            details=[
                f"To: {to}",
                f"Message: {message[:100]}..." if len(message) > 100 else f"Message: {message}",
            ],
            reversible=False,
            cost=0.01,  # Approximate Twilio cost
            warnings=["SMS messages cannot be unsent"],
        )

    async def execute(self, params: dict[str, Any]) -> ActionResult:
        import os

        try:
            from twilio.rest import Client

            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            from_number = os.environ.get("TWILIO_PHONE_NUMBER")

            if not all([account_sid, auth_token, from_number]):
                return ActionResult(success=False, message="Twilio not fully configured")

            client = Client(account_sid, auth_token)

            message = client.messages.create(
                body=params["message"],
                from_=from_number,
                to=params["to"],
            )

            return ActionResult(
                success=True,
                message=f"SMS sent to {params['to']}",
                data={"sid": message.sid},
            )

        except ImportError:
            return ActionResult(success=False, message="Twilio library not installed")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return ActionResult(success=False, message=f"Failed: {e}")
