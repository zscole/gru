"""Tests for Claude API client."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from gru.claude import (
    DEFAULT_TOOLS,
    ClaudeClient,
    Response,
    ToolDefinition,
    ToolResult,
    ToolUse,
    retry_with_backoff,
)
from gru.config import Config


@dataclass
class MockTextBlock:
    """Mock Anthropic text block."""

    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    """Mock Anthropic tool use block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockUsage:
    """Mock Anthropic usage."""

    input_tokens: int = 10
    output_tokens: int = 20


@dataclass
class MockResponse:
    """Mock Anthropic response."""

    content: list = None
    stop_reason: str = "end_turn"
    usage: MockUsage = None

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = MockUsage()


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        telegram_token="test_token",
        telegram_admin_ids=[123],
        anthropic_api_key="test_api_key",
        default_model="test-model",
        max_tokens=1000,
    )


@pytest.fixture
def client(config):
    """Create Claude client with mocked Anthropic client."""
    with patch("gru.claude.anthropic.AsyncAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        client = ClaudeClient(config)
        client._client = mock_instance
        yield client


def test_tool_definition():
    """Test ToolDefinition dataclass."""
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {}},
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.input_schema == {"type": "object", "properties": {}}


def test_tool_use():
    """Test ToolUse dataclass."""
    tool_use = ToolUse(id="tu_123", name="bash", input={"command": "ls"})
    assert tool_use.id == "tu_123"
    assert tool_use.name == "bash"
    assert tool_use.input == {"command": "ls"}


def test_tool_result():
    """Test ToolResult dataclass."""
    result = ToolResult(tool_use_id="tu_123", content="output")
    assert result.tool_use_id == "tu_123"
    assert result.content == "output"
    assert result.is_error is False

    error_result = ToolResult(tool_use_id="tu_456", content="error", is_error=True)
    assert error_result.is_error is True


def test_response():
    """Test Response dataclass."""
    response = Response(
        content="Hello",
        tool_uses=[],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert response.content == "Hello"
    assert response.tool_uses == []
    assert response.stop_reason == "end_turn"


def test_default_tools_exist():
    """Test that default tools are defined."""
    assert len(DEFAULT_TOOLS) > 0

    tool_names = [t.name for t in DEFAULT_TOOLS]
    assert "bash" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "search_files" in tool_names
    assert "request_human_input" in tool_names
    assert "send_message_to_agent" in tool_names
    assert "get_shared_context" in tool_names
    assert "set_shared_context" in tool_names


def test_default_tools_have_schemas():
    """Test that all default tools have valid schemas."""
    for tool in DEFAULT_TOOLS:
        assert tool.name, "Tool must have a name"
        assert tool.description, "Tool must have a description"
        assert isinstance(tool.input_schema, dict), "Tool must have input schema"
        assert tool.input_schema.get("type") == "object", "Schema must be object type"
        assert "properties" in tool.input_schema, "Schema must have properties"


@pytest.mark.asyncio
async def test_send_message_text_response(client):
    """Test sending a message and receiving text response."""
    mock_response = MockResponse(
        content=[MockTextBlock(text="Hello, world!")],
        stop_reason="end_turn",
    )
    client._client.messages.create = AsyncMock(return_value=mock_response)

    response = await client.send_message(
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.content == "Hello, world!"
    assert response.tool_uses == []
    assert response.stop_reason == "end_turn"
    assert response.usage["input_tokens"] == 10
    assert response.usage["output_tokens"] == 20


@pytest.mark.asyncio
async def test_send_message_with_system_prompt(client):
    """Test sending message with system prompt."""
    mock_response = MockResponse(content=[MockTextBlock(text="Response")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.send_message(
        messages=[{"role": "user", "content": "Hi"}],
        system="You are a helpful assistant.",
    )

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_send_message_with_tools(client):
    """Test sending message with tools."""
    mock_response = MockResponse(content=[MockTextBlock(text="Response")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    tools = [
        ToolDefinition(
            name="test_tool",
            description="Test",
            input_schema={"type": "object", "properties": {}},
        )
    ]

    await client.send_message(
        messages=[{"role": "user", "content": "Hi"}],
        tools=tools,
    )

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_send_message_tool_use_response(client):
    """Test receiving a tool use response."""
    mock_response = MockResponse(
        content=[
            MockTextBlock(text="I'll help with that."),
            MockToolUseBlock(id="tu_123", name="bash", input={"command": "ls -la"}),
        ],
        stop_reason="tool_use",
    )
    client._client.messages.create = AsyncMock(return_value=mock_response)

    response = await client.send_message(
        messages=[{"role": "user", "content": "List files"}],
        tools=DEFAULT_TOOLS,
    )

    assert response.content == "I'll help with that."
    assert len(response.tool_uses) == 1
    assert response.tool_uses[0].id == "tu_123"
    assert response.tool_uses[0].name == "bash"
    assert response.tool_uses[0].input == {"command": "ls -la"}
    assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_send_message_multiple_tool_uses(client):
    """Test receiving multiple tool uses in one response."""
    mock_response = MockResponse(
        content=[
            MockTextBlock(text="Running commands."),
            MockToolUseBlock(id="tu_1", name="bash", input={"command": "ls"}),
            MockToolUseBlock(id="tu_2", name="read_file", input={"path": "test.txt"}),
        ],
        stop_reason="tool_use",
    )
    client._client.messages.create = AsyncMock(return_value=mock_response)

    response = await client.send_message(
        messages=[{"role": "user", "content": "Do tasks"}],
        tools=DEFAULT_TOOLS,
    )

    assert len(response.tool_uses) == 2
    assert response.tool_uses[0].name == "bash"
    assert response.tool_uses[1].name == "read_file"


@pytest.mark.asyncio
async def test_send_message_custom_model(client):
    """Test sending message with custom model."""
    mock_response = MockResponse(content=[MockTextBlock(text="Response")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.send_message(
        messages=[{"role": "user", "content": "Hi"}],
        model="claude-opus-4-20250514",
    )

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-20250514"


@pytest.mark.asyncio
async def test_send_message_custom_max_tokens(client):
    """Test sending message with custom max tokens."""
    mock_response = MockResponse(content=[MockTextBlock(text="Response")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.send_message(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=500,
    )

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 500


@pytest.mark.asyncio
async def test_send_with_tool_results(client):
    """Test sending tool results back to Claude."""
    mock_response = MockResponse(content=[MockTextBlock(text="Done!")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    messages = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running command."},
                {"type": "tool_use", "id": "tu_123", "name": "bash", "input": {"command": "ls"}},
            ],
        },
    ]

    tool_results = [
        ToolResult(tool_use_id="tu_123", content="file1.txt\nfile2.txt"),
    ]

    response = await client.send_with_tool_results(
        messages=messages,
        tool_results=tool_results,
        tools=DEFAULT_TOOLS,
    )

    assert response.content == "Done!"

    # Verify the tool result was added to messages
    call_kwargs = client._client.messages.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    assert len(sent_messages) == 3  # Original 2 + tool result
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["content"][0]["type"] == "tool_result"
    assert sent_messages[-1]["content"][0]["tool_use_id"] == "tu_123"


@pytest.mark.asyncio
async def test_send_with_tool_results_error(client):
    """Test sending error tool results."""
    mock_response = MockResponse(content=[MockTextBlock(text="I see the error.")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    tool_results = [
        ToolResult(tool_use_id="tu_123", content="Command failed", is_error=True),
    ]

    await client.send_with_tool_results(
        messages=[{"role": "user", "content": "Run command"}],
        tool_results=tool_results,
    )

    call_kwargs = client._client.messages.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    tool_result = sent_messages[-1]["content"][0]
    assert tool_result["is_error"] is True


@pytest.mark.asyncio
async def test_client_uses_config_defaults(client, config):
    """Test that client uses config defaults."""
    mock_response = MockResponse(content=[MockTextBlock(text="Response")])
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.send_message(messages=[{"role": "user", "content": "Hi"}])

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == config.default_model
    assert call_kwargs["max_tokens"] == config.max_tokens


@pytest.mark.asyncio
async def test_empty_text_content(client):
    """Test handling response with no text content."""
    mock_response = MockResponse(
        content=[MockToolUseBlock(id="tu_1", name="bash", input={"command": "ls"})],
        stop_reason="tool_use",
    )
    client._client.messages.create = AsyncMock(return_value=mock_response)

    response = await client.send_message(
        messages=[{"role": "user", "content": "List"}],
        tools=DEFAULT_TOOLS,
    )

    assert response.content == ""
    assert len(response.tool_uses) == 1


# Retry logic tests


@pytest.mark.asyncio
async def test_retry_success_first_try():
    """Test retry succeeds on first attempt."""

    async def success_func():
        return "success"

    result = await retry_with_backoff(success_func)
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_success_after_failures():
    """Test retry succeeds after transient failures."""
    call_count = 0

    async def eventual_success():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise anthropic.RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limited"}},
            )
        return "success"

    result = await retry_with_backoff(
        eventual_success,
        max_retries=3,
        base_delay=0.01,  # Short delay for tests
    )
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    """Test retry raises after max retries."""
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise anthropic.APIConnectionError(request=MagicMock())

    with pytest.raises(anthropic.APIConnectionError):
        await retry_with_backoff(
            always_fail,
            max_retries=2,
            base_delay=0.01,
        )

    assert call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_retry_non_retryable_exception():
    """Test non-retryable exceptions are raised immediately."""
    call_count = 0

    async def bad_request():
        nonlocal call_count
        call_count += 1
        raise anthropic.BadRequestError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body={"error": {"message": "Bad request"}},
        )

    with pytest.raises(anthropic.BadRequestError):
        await retry_with_backoff(
            bad_request,
            max_retries=3,
            base_delay=0.01,
        )

    assert call_count == 1  # Should not retry


@pytest.mark.asyncio
async def test_retry_with_arguments():
    """Test retry passes arguments correctly."""

    async def func_with_args(a, b, c=None):
        return f"{a}-{b}-{c}"

    result = await retry_with_backoff(func_with_args, "x", "y", c="z")
    assert result == "x-y-z"


@pytest.mark.asyncio
async def test_send_message_with_retry(client):
    """Test send_message retries on rate limit."""
    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise anthropic.RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limited"}},
            )
        return MockResponse(content=[MockTextBlock(text="Success")])

    client._client.messages.create = mock_create

    with patch("gru.claude.BASE_DELAY", 0.01):
        response = await client.send_message(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert response.content == "Success"
    assert call_count == 2
