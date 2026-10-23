"""Tests for GraphQL client read-only enforcement."""

from unittest.mock import patch

import httpx
import pytest

from api_agent.graphql.client import execute_query

ENDPOINT = "https://api.example.com/graphql"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "mutation { deleteUser(id: 1) }",
        ",mutation { deleteUser(id: 1) }",
        "\ufeffmutation { deleteUser(id: 1) }",
        "subscription { events }",
        "query A { a } mutation B { deleteUser(id: 1) }",
    ],
)
async def test_non_query_operations_are_blocked(query):
    with patch("api_agent.graphql.client.bounded_request") as send:
        result = await execute_query(query, endpoint=ENDPOINT)

    assert result == {"success": False, "error": "Mutations are not allowed (read-only mode)"}
    send.assert_not_called()


@pytest.mark.asyncio
async def test_unparseable_query_is_not_sent():
    with patch("api_agent.graphql.client.bounded_request") as send:
        result = await execute_query("{ users(", endpoint=ENDPOINT)

    assert result["success"] is False
    assert result["error"].startswith("Invalid GraphQL query")
    send.assert_not_called()


@pytest.mark.asyncio
async def test_queries_with_mutation_named_fields_are_sent():
    response = httpx.Response(
        200, json={"data": {"mutationLog": []}}, request=httpx.Request("POST", ENDPOINT)
    )
    with patch("api_agent.graphql.client.bounded_request", return_value=response):
        result = await execute_query("{ mutationLog { id } }", endpoint=ENDPOINT)

    assert result == {"success": True, "data": {"mutationLog": []}}
