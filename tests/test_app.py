import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from api_agent.__main__ import create_app, create_public_mcp
from api_agent.agent.prompts import DECISION_GUIDANCE


def _route_paths(app) -> set[str]:
    return {route.path for route in app.routes if isinstance(route, Route)}


@pytest.mark.asyncio
async def test_public_mcp_has_only_public_tools():
    mcp = create_public_mcp()

    tools = await mcp.list_tools(run_middleware=False)
    names = {tool.name for tool in tools}

    assert names == {"_query"}
    query_tool = tools[0]
    assert "r_*" not in (query_tool.description or "")
    assert "specific tool" not in (query_tool.description or "")


def test_decision_guidance_prefers_recipe_tools_first():
    assert "Check available recipe tools before direct API/SQL calls" in DECISION_GUIDANCE


def test_create_app_mounts_public_mcp_and_health_only():
    app = create_app()

    assert _route_paths(app) == {"/mcp", "/health"}


def test_mcp_path_does_not_redirect_to_slash_path():
    app = create_app()

    with TestClient(app, follow_redirects=False) as client:
        response = client.get("/mcp")
        slash_response = client.get("/mcp/")

    assert response.status_code == 405
    assert "location" not in response.headers
    assert slash_response.status_code == 307
    assert slash_response.headers["location"] == "http://testserver/mcp"
