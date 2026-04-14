"""Integration tests for web tools — uses mocked HTTP to avoid network calls."""

from unittest.mock import patch

import pytest

from relay.tools.impl.web import fetch_web_content
from tests.fixtures.tool_helpers import make_tool_call, run_tool


@pytest.mark.asyncio
@patch("relay.tools.impl.web.trafilatura.extract")
@patch("relay.tools.impl.web.trafilatura.fetch_url")
async def test_fetch_web_content(mock_fetch, mock_extract, create_test_graph):
    mock_fetch.return_value = "<html><body><h1>Test Page</h1><p>Content</p></body></html>"
    mock_extract.return_value = "# Test Page\n\nContent"

    app = create_test_graph([fetch_web_content])
    state = make_tool_call("fetch_web_content", url="https://example.com")
    result = await run_tool(app, state)

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs
    assert "Test Page" in tool_msgs[0].content


@pytest.mark.asyncio
@patch("relay.tools.impl.web.trafilatura.extract")
@patch("relay.tools.impl.web.trafilatura.fetch_url")
async def test_fetch_web_content_no_content(mock_fetch, mock_extract, create_test_graph):
    mock_fetch.return_value = "<html><body></body></html>"
    mock_extract.return_value = None

    app = create_test_graph([fetch_web_content])
    state = make_tool_call("fetch_web_content", url="https://example.com")
    result = await run_tool(app, state)

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs
    assert "No main content could be extracted" in tool_msgs[0].content


@pytest.mark.asyncio
@patch("relay.tools.impl.web.trafilatura.fetch_url")
async def test_fetch_web_content_download_failure(mock_fetch, create_test_graph):
    mock_fetch.return_value = None

    app = create_test_graph([fetch_web_content])
    state = make_tool_call("fetch_web_content", url="https://invalid.example")
    result = await run_tool(app, state)

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs
    # ToolNode wraps ToolException into the message content.
    assert "Could not download" in tool_msgs[0].content
