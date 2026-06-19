import pytest
from fastmcp import FastMCP

from api_agent.tools import register_all_tools


@pytest.mark.asyncio
async def test_execute_tool_not_registered():
    mcp = FastMCP("test")

    register_all_tools(mcp)

    names = {t.name for t in await mcp.list_tools(run_middleware=False)}
    assert "_query" in names
    assert "_execute" not in names


@pytest.mark.asyncio
async def test_query_tool_has_mcp_metadata():
    mcp = FastMCP("test")
    register_all_tools(mcp)

    tool = next(t for t in await mcp.list_tools(run_middleware=False) if t.name == "_query")

    assert tool.title == "Ask API"
    assert "natural-language question" in (tool.description or "")
    assert "return_directly" in tool.parameters.get("properties", {})
    assert "return_directly" not in tool.parameters.get("required", [])
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is None
    assert tool.annotations.openWorldHint is True
