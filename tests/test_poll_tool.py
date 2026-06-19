"""Tests for poll_until_done tool behavior."""

import pytest
from agents.tool_context import ToolContext

TOOL_CONTEXT = ToolContext(
    context=None,
    tool_name="poll_until_done",
    tool_call_id="test",
    tool_arguments="{}",
    run_config=None,
)


class TestPollBlocking:
    """Test that poll_until_done respects allow_unsafe_paths."""

    @pytest.mark.asyncio
    async def test_post_blocked_without_whitelist(self):
        import json

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=(),  # No paths allowed
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        result = await poll_tool.on_invoke_tool(
            TOOL_CONTEXT,
            json.dumps(
                {
                    "method": "POST",
                    "path": "/search",
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "body": json.dumps({"polling": {"count": 1}}),
                }
            ),
        )
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "not allowed" in result_dict.get("error", "")

    @pytest.mark.asyncio
    async def test_post_allowed_with_whitelist(self):
        import json

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        result = await poll_tool.on_invoke_tool(
            TOOL_CONTEXT,
            json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "body": json.dumps({"polling": {"count": 1}}),
                }
            ),
        )
        result_dict = json.loads(result)
        # Will fail with connection error but NOT blocked
        assert "not allowed" not in result_dict.get("error", "")


class TestPollGuardrails:
    """Test guardrails prevent LLM mistakes."""

    @pytest.mark.asyncio
    async def test_done_field_not_found_returns_error(self):
        """If done_field doesn't exist in response, return error with available keys."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        # Mock response without the expected done_field
        mock_response = {"status": "pending", "results": []}

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            return_value={"success": True, "data": mock_response},
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "done_field": "polling.completed",  # doesn't exist
                        "done_value": "true",
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "not found" in result_dict["error"]
            assert "polling.completed" in result_dict["error"]
            # Should show available keys for debugging
            assert "status" in result_dict["error"] or "keys" in result_dict["error"]

    @pytest.mark.asyncio
    async def test_agent_specified_delay_ms(self):
        """Agent-specified delay_ms should override response delay."""
        import json
        import time
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        call_times = []

        async def mock_request(*args, **kwargs):
            call_times.append(time.time())
            # Server asks for 60s delay but agent specifies 100ms
            return {
                "success": True,
                "data": {"polling": {"completed": len(call_times) >= 2, "delayMs": 60000}},
            }

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "done_field": "polling.completed",
                        "done_value": "true",
                        "delay_ms": 100,  # Agent overrides to 100ms
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert len(call_times) == 2
            actual_delay = call_times[1] - call_times[0]
            assert actual_delay < 1.0  # 100ms + tolerance

    @pytest.mark.asyncio
    async def test_delay_ms_is_capped(self, monkeypatch):
        """Agent-specified delay_ms is capped by server settings."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        monkeypatch.setattr("api_agent.rest.polling.settings.MAX_POLL_DELAY_MS", 25)

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")
        calls = 0

        async def mock_request(*args, **kwargs):
            nonlocal calls
            calls += 1
            return {
                "success": True,
                "data": {"polling": {"completed": calls >= 2}},
            }

        with (
            patch(
                "api_agent.rest.polling.execute_request",
                new_callable=AsyncMock,
                side_effect=mock_request,
            ),
            patch("api_agent.rest.polling.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "done_field": "polling.completed",
                        "done_value": "true",
                        "delay_ms": 60000,
                    }
                ),
            )

        assert json.loads(result)["success"] is True
        sleep.assert_awaited_once_with(0.025)

    @pytest.mark.asyncio
    async def test_max_polls_error_shows_last_value(self):
        """max_polls exceeded should show last done_field value."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "data": {"polling": {"completed": False}},
            },
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "done_field": "polling.completed",
                        "done_value": "true",
                        "delay_ms": 1,
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "max_polls" in result_dict["error"].lower() or "exceeded" in result_dict["error"]
            # Should show what the actual value was
            assert "false" in result_dict["error"].lower() or "False" in result_dict["error"]

    @pytest.mark.asyncio
    async def test_max_polls_does_not_sleep_after_final_attempt(self, monkeypatch):
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        monkeypatch.setattr("api_agent.rest.polling.settings.MAX_POLLS", 1)

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        with (
            patch(
                "api_agent.rest.polling.execute_request",
                new_callable=AsyncMock,
                return_value={
                    "success": True,
                    "data": {"polling": {"completed": False}},
                },
            ),
            patch("api_agent.rest.polling.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "done_field": "polling.completed",
                        "done_value": "true",
                        "delay_ms": 60000,
                    }
                ),
            )

        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert result_dict["attempts"] == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_increment_polling_count(self):
        """polling.count in body should auto-increment between polls."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        received_bodies = []

        async def mock_request(*args, body=None, **kwargs):
            import copy

            received_bodies.append(copy.deepcopy(body))
            return {
                "success": True,
                "data": {"polling": {"completed": len(received_bodies) >= 3}},
            }

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/search/flights",
                        "body": json.dumps({"polling": {"count": 1}}),
                        "done_field": "polling.completed",
                        "done_value": "true",
                        "delay_ms": 1,
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            # Check counts incremented: 1, 2, 3
            assert received_bodies[0]["polling"]["count"] == 1
            assert received_bodies[1]["polling"]["count"] == 2
            assert received_bodies[2]["polling"]["count"] == 3

    @pytest.mark.asyncio
    async def test_numeric_done_field_zero_means_done(self):
        """retry.next == 0 should be detected as done (Flights API pattern)."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/flights/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # retry.next = 2, 1, 0 (0 means done)
            return {
                "success": True,
                "data": {"retry": {"next": max(0, 3 - call_count)}, "trips": []},
            }

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/flights/search",
                        "done_field": "retry.next",
                        "done_value": "0",  # 0 means done
                        "delay_ms": 1,
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert call_count == 3  # Should poll 3 times until retry.next == 0

    @pytest.mark.asyncio
    async def test_invalid_body_json_returns_error(self):
        """Invalid body JSON should return a friendly error."""
        import json

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        result = await poll_tool.on_invoke_tool(
            TOOL_CONTEXT,
            json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "body": "not-json",
                }
            ),
        )
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "invalid body json" in result_dict["error"].lower()

    @pytest.mark.asyncio
    async def test_works_without_polling_count_in_body(self):
        """Should work fine without polling.count in body."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.context import RequestContext
        from api_agent.rest.polling import create_poll_tool

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/status/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = create_poll_tool(ctx, "https://api.example.com")

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "success": True,
                "data": {"status": {"done": call_count >= 2}},
            }

        with patch(
            "api_agent.rest.polling.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            result = await poll_tool.on_invoke_tool(
                TOOL_CONTEXT,
                json.dumps(
                    {
                        "method": "POST",
                        "path": "/status/check",
                        "body": json.dumps({"query": "test"}),  # No polling.count
                        "done_field": "status.done",
                        "done_value": "true",
                        "delay_ms": 1,
                    }
                ),
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
