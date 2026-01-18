"""Tests for webhook server."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import AioHTTPTestCase

from gru.config import Config
from gru.webhook import WebhookServer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        telegram_token="test_token",
        telegram_admin_ids=[123],
        anthropic_api_key="test_api_key",
        webhook_enabled=True,
        webhook_host="127.0.0.1",
        webhook_port=8080,
        webhook_secret="test_secret",
    )


@pytest.fixture
def config_no_secret():
    """Create test config without webhook secret."""
    return Config(
        telegram_token="test_token",
        telegram_admin_ids=[123],
        anthropic_api_key="test_api_key",
        webhook_enabled=True,
        webhook_host="127.0.0.1",
        webhook_port=8080,
        webhook_secret="",
    )


@pytest.fixture
def config_disabled():
    """Create test config with webhook disabled."""
    return Config(
        telegram_token="test_token",
        telegram_admin_ids=[123],
        anthropic_api_key="test_api_key",
        webhook_enabled=False,
    )


@pytest.fixture
def orchestrator():
    """Create mock orchestrator."""
    mock = MagicMock()
    mock.notify = AsyncMock()
    return mock


@pytest.fixture
def webhook_server(config, orchestrator):
    """Create webhook server instance."""
    return WebhookServer(config, orchestrator)


@pytest.fixture
def webhook_server_no_secret(config_no_secret, orchestrator):
    """Create webhook server instance without secret."""
    return WebhookServer(config_no_secret, orchestrator)


def create_vercel_payload(
    event_type: str = "deployment.succeeded",
    branch: str = "main",
    url: str = "my-app-abc123.vercel.app",
    state: str = "READY",
    error_message: str | None = None,
) -> dict:
    """Create a Vercel webhook payload."""
    deployment = {
        "meta": {"githubCommitRef": branch},
        "url": url,
        "state": state,
    }
    if error_message:
        deployment["errorMessage"] = error_message

    return {
        "type": event_type,
        "payload": {"deployment": deployment},
    }


def sign_payload(payload: bytes, secret: str) -> str:
    """Sign a payload with HMAC-SHA1."""
    return hmac.new(secret.encode(), payload, hashlib.sha1).hexdigest()


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, webhook_server):
        """Test health endpoint returns ok status."""
        # Create a mock request
        request = MagicMock()
        response = await webhook_server._handle_health(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "ok"


# =============================================================================
# Vercel Webhook Tests
# =============================================================================


class TestVercelWebhook:
    """Tests for Vercel webhook handling."""

    @pytest.mark.asyncio
    async def test_valid_signature(self, webhook_server, config):
        """Test webhook with valid signature passes."""
        payload = create_vercel_payload()
        body = json.dumps(payload).encode()
        signature = sign_payload(body, config.webhook_secret)

        request = MagicMock()
        request.headers = {"x-vercel-signature": signature}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server._handle_vercel(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_invalid_signature(self, webhook_server, config):
        """Test webhook with invalid signature is rejected."""
        payload = create_vercel_payload()
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {"x-vercel-signature": "invalid_signature"}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server._handle_vercel(request)
        assert response.status == 401
        body_json = json.loads(response.body)
        # Exact match to kill mutant #19
        assert body_json["error"] == "Invalid signature"

    @pytest.mark.asyncio
    async def test_missing_signature(self, webhook_server, config):
        """Test webhook with missing signature is rejected."""
        payload = create_vercel_payload()
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server._handle_vercel(request)
        assert response.status == 401

    @pytest.mark.asyncio
    async def test_no_secret_configured(self, webhook_server_no_secret):
        """Test webhook passes when no secret is configured."""
        payload = create_vercel_payload()
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_invalid_json(self, webhook_server_no_secret):
        """Test webhook with invalid JSON returns 400."""
        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=b"not valid json")

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 400
        body_json = json.loads(response.body)
        # Exact match to kill mutant #24
        assert body_json["error"] == "Invalid JSON"

    @pytest.mark.asyncio
    async def test_deployment_succeeded_notifies_agent(self, webhook_server_no_secret, orchestrator):
        """Test deployment.succeeded event notifies agent."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-abc123",
            url="my-app.vercel.app",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

        # Should notify the agent
        orchestrator.notify.assert_called_once()
        call_args = orchestrator.notify.call_args
        assert call_args[0][0] == "abc123"  # agent_id
        assert "https://my-app.vercel.app" in call_args[0][1]  # message

    @pytest.mark.asyncio
    async def test_deployment_succeeded_adds_https_prefix(self, webhook_server_no_secret, orchestrator):
        """Test https:// prefix is added to URL if missing."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-test",
            url="my-app.vercel.app",  # No protocol
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        await webhook_server_no_secret._handle_vercel(request)

        call_args = orchestrator.notify.call_args
        assert "https://my-app.vercel.app" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_deployment_succeeded_keeps_https_prefix(self, webhook_server_no_secret, orchestrator):
        """Test https:// URL is kept as-is."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-test",
            url="https://my-app.vercel.app",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        await webhook_server_no_secret._handle_vercel(request)

        call_args = orchestrator.notify.call_args
        # Should not have double https://
        assert "https://https://" not in call_args[0][1]
        assert "https://my-app.vercel.app" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_deployment_error_notifies_agent(self, webhook_server_no_secret, orchestrator):
        """Test deployment.error event notifies agent."""
        payload = create_vercel_payload(
            event_type="deployment.error",
            branch="gru-agent-def456",
            error_message="Build failed",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

        # Should notify the agent of error
        orchestrator.notify.assert_called_once()
        call_args = orchestrator.notify.call_args
        assert call_args[0][0] == "def456"  # agent_id
        assert "failed" in call_args[0][1]
        assert "Build failed" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_non_gru_branch_ignored(self, webhook_server_no_secret, orchestrator):
        """Test deployments from non-gru branches are ignored."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="main",
            url="my-app.vercel.app",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

        # Should NOT notify since branch doesn't start with gru-agent-
        orchestrator.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_event_type_ignored(self, webhook_server_no_secret, orchestrator):
        """Test unknown event types are ignored."""
        payload = create_vercel_payload(
            event_type="deployment.created",
            branch="gru-agent-test",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

        # Should NOT notify for unknown event types
        orchestrator.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_payload(self, webhook_server_no_secret):
        """Test handling of minimal/empty payload."""
        payload = {}
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        # Should not crash, just return ok
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_missing_type_key(self, webhook_server_no_secret, orchestrator):
        """Test payload without type key doesn't notify.

        Kills mutant: event_type = payload.get("type", "") -> "XXXX"
        """
        # Payload without "type" key
        payload = {
            "payload": {
                "deployment": {
                    "meta": {"githubCommitRef": "gru-agent-test"},
                    "url": "test.vercel.app",
                    "state": "READY",
                }
            }
        }
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200
        # Should NOT notify because event_type is empty (not a known type)
        orchestrator.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_branch_metadata(self, webhook_server_no_secret, orchestrator):
        """Test payload without branch metadata doesn't notify.

        Kills mutant: branch = ... .get("githubCommitRef", "") -> "XXXX"
        """
        payload = {
            "type": "deployment.succeeded",
            "payload": {
                "deployment": {
                    "meta": {},  # No githubCommitRef
                    "url": "test.vercel.app",
                    "state": "READY",
                }
            },
        }
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200
        # Should NOT notify because branch is empty (doesn't start with gru-agent-)
        orchestrator.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_url_https_prefix_exact_format(self, webhook_server_no_secret, orchestrator):
        """Test that https:// prefix is added correctly without extra characters.

        Kills mutant: preview_url = f"https://{preview_url}" -> "XXhttps://..."
        """
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-format",
            url="my-app.vercel.app",  # No protocol
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        await webhook_server_no_secret._handle_vercel(request)

        call_args = orchestrator.notify.call_args
        message = call_args[0][1]
        # Verify exact format - no extra XX characters
        assert "https://my-app.vercel.app" in message
        assert "XXhttps" not in message
        assert "vercel.appXX" not in message

    @pytest.mark.asyncio
    async def test_missing_preview_url_no_notification(self, webhook_server_no_secret, orchestrator):
        """Test that missing URL prevents notification.

        Kills mutant #38: preview_url default "" -> "XXXX"
        """
        payload = {
            "type": "deployment.succeeded",
            "payload": {
                "deployment": {
                    "meta": {"githubCommitRef": "gru-agent-nourl"},
                    # No "url" key
                    "state": "READY",
                }
            },
        }
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200
        # Should NOT notify because preview_url is empty/falsy
        orchestrator.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_deployment_error_default_message(self, webhook_server_no_secret, orchestrator):
        """Test deployment error with missing errorMessage uses default.

        Kills mutant #63: default "Unknown error" -> "XXUnknown errorXX"
        """
        payload = {
            "type": "deployment.error",
            "payload": {
                "deployment": {
                    "meta": {"githubCommitRef": "gru-agent-err"},
                    "url": "test.vercel.app",
                    "state": "ERROR",
                    # No "errorMessage" key - should use default
                }
            },
        }
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200

        orchestrator.notify.assert_called_once()
        call_args = orchestrator.notify.call_args
        message = call_args[0][1]
        # Verify exact default message format
        assert "Unknown error" in message
        assert "XXUnknown" not in message

    @pytest.mark.asyncio
    async def test_deployment_error_message_format(self, webhook_server_no_secret, orchestrator):
        """Test error notification message format.

        Kills mutant #65: "Preview deployment failed" -> "XXPreview..."
        """
        payload = create_vercel_payload(
            event_type="deployment.error",
            branch="gru-agent-errfmt",
            error_message="Build timeout",
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        await webhook_server_no_secret._handle_vercel(request)

        call_args = orchestrator.notify.call_args
        message = call_args[0][1]
        # Verify exact message prefix
        assert message.startswith("Preview deployment failed:")
        assert "XXPreview" not in message

    @pytest.mark.asyncio
    async def test_vercel_response_body_format(self, webhook_server_no_secret):
        """Test vercel handler returns correct response body.

        Kills mutants #66, #67: {"status": "ok"} mutations
        """
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="main",  # Non-gru branch to avoid notification
        )
        body = json.dumps(payload).encode()

        request = MagicMock()
        request.headers = {}
        request.read = AsyncMock(return_value=body)

        response = await webhook_server_no_secret._handle_vercel(request)
        assert response.status == 200
        body_json = json.loads(response.body)
        # Exact key and value match
        assert body_json == {"status": "ok"}


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Tests for server start/stop."""

    @pytest.mark.asyncio
    async def test_start_when_enabled(self, config, orchestrator):
        """Test server starts when enabled."""
        server = WebhookServer(config, orchestrator)

        mock_runner = MagicMock()
        mock_runner.setup = AsyncMock()
        mock_site = MagicMock()
        mock_site.start = AsyncMock()

        with (
            patch("gru.webhook.web.AppRunner", return_value=mock_runner),
            patch("gru.webhook.web.TCPSite", return_value=mock_site),
        ):
            await server.start()

            mock_runner.setup.assert_called_once()
            mock_site.start.assert_called_once()
            assert server._runner is not None

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, config_disabled, orchestrator):
        """Test server does not start when disabled."""
        server = WebhookServer(config_disabled, orchestrator)

        await server.start()

        # Runner should not be created
        assert server._runner is None

    @pytest.mark.asyncio
    async def test_stop_with_runner(self, config, orchestrator):
        """Test server stop cleans up runner."""
        server = WebhookServer(config, orchestrator)
        mock_runner = MagicMock()
        mock_runner.cleanup = AsyncMock()
        server._runner = mock_runner

        await server.stop()

        mock_runner.cleanup.assert_called_once()
        # Runner is set to None after stop
        assert server._runner is None

    @pytest.mark.asyncio
    async def test_stop_without_runner(self, config, orchestrator):
        """Test server stop handles no runner gracefully."""
        server = WebhookServer(config, orchestrator)
        server._runner = None

        # Should not raise
        await server.stop()


# =============================================================================
# Route Setup Tests
# =============================================================================


class TestRouteSetup:
    """Tests for route configuration."""

    def test_routes_registered(self, webhook_server):
        """Test that routes are properly registered."""
        routes = webhook_server._app.router.routes()
        route_paths = [r.resource.canonical for r in routes]

        assert "/webhook/vercel" in route_paths
        assert "/health" in route_paths


# =============================================================================
# Integration Tests with aiohttp test client
# =============================================================================


class TestWebhookIntegration(AioHTTPTestCase):
    """Integration tests using aiohttp test client."""

    async def get_application(self):
        """Create the application for testing."""
        self.config = Config(
            telegram_token="test_token",
            telegram_admin_ids=[123],
            anthropic_api_key="test_api_key",
            webhook_enabled=True,
            webhook_secret="",
        )
        self.orchestrator = MagicMock()
        self.orchestrator.notify = AsyncMock()

        server = WebhookServer(self.config, self.orchestrator)
        return server._app

    async def test_health_endpoint_integration(self):
        """Test health endpoint via HTTP request."""
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"

    async def test_vercel_webhook_integration(self):
        """Test Vercel webhook via HTTP request."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-test123",
            url="test.vercel.app",
        )

        resp = await self.client.request(
            "POST",
            "/webhook/vercel",
            json=payload,
        )
        assert resp.status == 200

        # Verify notification was sent
        self.orchestrator.notify.assert_called_once()

    async def test_invalid_json_integration(self):
        """Test invalid JSON via HTTP request."""
        resp = await self.client.request(
            "POST",
            "/webhook/vercel",
            data=b"not json",
        )
        assert resp.status == 400


class TestWebhookIntegrationWithSecret(AioHTTPTestCase):
    """Integration tests with signature verification."""

    async def get_application(self):
        """Create the application for testing with secret."""
        self.secret = "test_webhook_secret"
        self.config = Config(
            telegram_token="test_token",
            telegram_admin_ids=[123],
            anthropic_api_key="test_api_key",
            webhook_enabled=True,
            webhook_secret=self.secret,
        )
        self.orchestrator = MagicMock()
        self.orchestrator.notify = AsyncMock()

        server = WebhookServer(self.config, self.orchestrator)
        return server._app

    async def test_valid_signature_integration(self):
        """Test valid signature via HTTP request."""
        payload = create_vercel_payload(
            event_type="deployment.succeeded",
            branch="gru-agent-signed",
            url="signed.vercel.app",
        )
        body = json.dumps(payload).encode()
        signature = sign_payload(body, self.secret)

        resp = await self.client.request(
            "POST",
            "/webhook/vercel",
            data=body,
            headers={"x-vercel-signature": signature, "Content-Type": "application/json"},
        )
        assert resp.status == 200

    async def test_invalid_signature_integration(self):
        """Test invalid signature via HTTP request."""
        payload = create_vercel_payload()
        body = json.dumps(payload).encode()

        resp = await self.client.request(
            "POST",
            "/webhook/vercel",
            data=body,
            headers={"x-vercel-signature": "wrong_signature", "Content-Type": "application/json"},
        )
        assert resp.status == 401
