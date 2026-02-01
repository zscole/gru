"""Google Docs and Gmail actions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gru.actions.base import Action, ActionContext, ActionResult

if TYPE_CHECKING:
    from gru.connectors.google import GoogleConnector

logger = logging.getLogger(__name__)

# Global connector reference (set by orchestrator)
_google_connector: GoogleConnector | None = None


def set_google_connector(connector: GoogleConnector) -> None:
    """Set the Google connector for actions to use."""
    global _google_connector
    _google_connector = connector


def get_google_connector() -> GoogleConnector | None:
    """Get the Google connector."""
    return _google_connector


class CreateDocumentAction(Action):
    """Create a new Google Doc."""

    name = "create_document"
    description = "Create a new Google Doc with optional content"
    category = "google"
    requires_auth = True

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("title"):
            return False, "title is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        title = params["title"]
        content = params.get("content", "")

        connector = get_google_connector()
        if not connector:
            return ActionResult.error_result(
                "Google connector not configured. Run 'gru google login' first."
            )

        if not connector.is_authenticated():
            return ActionResult.auth_required("google", None)

        try:
            result = await connector.create_document(title, content)

            return ActionResult.success_result(
                f"Created document: {result['title']}",
                {
                    "document_id": result["document_id"],
                    "url": result["url"],
                    "title": result["title"],
                }
            )

        except Exception as e:
            return ActionResult.error_result(f"Failed to create document: {e}")


class WriteDocumentAction(Action):
    """Write content to an existing Google Doc."""

    name = "write_document"
    description = "Write or append content to a Google Doc"
    category = "google"
    requires_auth = True

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("document_id"):
            return False, "document_id is required"
        if not params.get("content"):
            return False, "content is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        document_id = params["document_id"]
        content = params["content"]
        append = params.get("append", True)

        connector = get_google_connector()
        if not connector:
            return ActionResult.error_result(
                "Google connector not configured. Run 'gru google login' first."
            )

        if not connector.is_authenticated():
            return ActionResult.auth_required("google", None)

        try:
            await connector.write_to_document(document_id, content, insert_at_end=append)

            doc_url = f"https://docs.google.com/document/d/{document_id}/edit"

            return ActionResult.success_result(
                f"Wrote {len(content)} characters to document",
                {
                    "document_id": document_id,
                    "url": doc_url,
                    "chars_written": len(content),
                }
            )

        except Exception as e:
            return ActionResult.error_result(f"Failed to write to document: {e}")


class SendEmailAction(Action):
    """Send an email via Gmail."""

    name = "send_email"
    description = "Send an email to a recipient"
    category = "google"
    requires_auth = True
    requires_confirmation = True  # Always confirm before sending

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("to"):
            return False, "to (recipient email) is required"
        if not params.get("subject"):
            return False, "subject is required"
        if not params.get("body"):
            return False, "body is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        to = params["to"]
        subject = params["subject"]
        body = params["body"]
        html = params.get("html", False)

        connector = get_google_connector()
        if not connector:
            return ActionResult.error_result(
                "Google connector not configured. Run 'gru google login' first."
            )

        if not connector.is_authenticated():
            return ActionResult.auth_required("google", None)

        try:
            result = await connector.send_email(to, subject, body, html)

            return ActionResult.success_result(
                f"Email sent to {to}",
                {
                    "message_id": result["message_id"],
                    "to": to,
                    "subject": subject,
                }
            )

        except Exception as e:
            return ActionResult.error_result(f"Failed to send email: {e}")


class CompileDocumentAction(Action):
    """Compile a conversation into a structured document."""

    name = "compile_document"
    description = "Compile conversation history into a formatted document (e.g., PRD, meeting notes)"
    category = "google"
    requires_auth = True

    async def validate_params(self, **params) -> tuple[bool, str | None]:
        if not params.get("title"):
            return False, "title is required"
        if not params.get("conversation") and not params.get("content"):
            return False, "conversation or content is required"
        return True, None

    async def execute(self, context: ActionContext, **params) -> ActionResult:
        title = params["title"]
        conversation = params.get("conversation", [])
        content = params.get("content", "")
        doc_type = params.get("doc_type", "document")  # prd, meeting_notes, summary, document
        email_to = params.get("email_to")  # Optional: email the link

        connector = get_google_connector()
        if not connector:
            return ActionResult.error_result(
                "Google connector not configured. Run 'gru google login' first."
            )

        if not connector.is_authenticated():
            return ActionResult.auth_required("google", None)

        # If we have a conversation, we need to compile it
        # This would typically use Claude to format the conversation
        if conversation and not content:
            content = self._format_conversation(conversation, doc_type)

        if not content:
            return ActionResult.error_result("No content to write")

        try:
            # Create the document
            doc_result = await connector.create_document(title, content)
            doc_url = doc_result["url"]

            result_data = {
                "document_id": doc_result["document_id"],
                "url": doc_url,
                "title": title,
            }

            # Email the link if requested
            if email_to:
                email_subject = f"Document: {title}"
                email_body = f"Hi,\n\nI've created a document for you:\n\n{title}\n{doc_url}\n\nBest regards"

                email_result = await connector.send_email(
                    to=email_to,
                    subject=email_subject,
                    body=email_body,
                )
                result_data["email_sent_to"] = email_to
                result_data["email_message_id"] = email_result["message_id"]

            message = f"Created '{title}'"
            if email_to:
                message += f" and sent link to {email_to}"

            return ActionResult.success_result(message, result_data)

        except Exception as e:
            return ActionResult.error_result(f"Failed to compile document: {e}")

    def _format_conversation(self, conversation: list[dict], doc_type: str) -> str:
        """Format a conversation into document content.

        This is a basic formatter - for production, you'd use Claude
        to intelligently structure the content.
        """
        lines = []

        if doc_type == "prd":
            lines.append("PRODUCT REQUIREMENTS DOCUMENT")
            lines.append("=" * 40)
            lines.append("")
            lines.append("Generated from conversation")
            lines.append("")
            lines.append("---")
            lines.append("")

        elif doc_type == "meeting_notes":
            lines.append("MEETING NOTES")
            lines.append("=" * 40)
            lines.append("")

        # Add conversation content
        for msg in conversation:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            if isinstance(content, str):
                lines.append(f"[{role}]")
                lines.append(content)
                lines.append("")

        return "\n".join(lines)
