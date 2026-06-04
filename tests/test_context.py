"""Tests for request context extraction from headers."""

from unittest.mock import patch

import pytest

from api_agent.context import (
    MissingHeaderError,
    RequestContext,
    extract_api_name,
    get_full_hostname,
    get_request_context,
    get_tool_name_prefix,
    parse_request_context,
)


class TestGetRequestContext:
    """Test header extraction and validation."""

    def test_parse_request_context_is_pure(self):
        ctx = parse_request_context(
            {
                "x-target-url": "https://api.example.com/openapi.json",
                "x-api-type": "rest",
                "x-target-headers": '{"Authorization": "Bearer xxx"}',
                "x-base-url": "https://api.example.com/v1",
                "x-allow-unsafe-paths": '["/search"]',
                "x-poll-paths": '["/jobs"]',
                "x-recipe-learn-rate": "1",
                "x-debug": "true",
            }
        )

        assert ctx.target_url == "https://api.example.com/openapi.json"
        assert ctx.api_type == "rest"
        assert ctx.target_headers == {"Authorization": "Bearer xxx"}
        assert ctx.base_url == "https://api.example.com/v1"
        assert ctx.allow_unsafe_paths == ("/search",)
        assert ctx.include_result is False
        assert ctx.poll_paths == ("/jobs",)
        assert ctx.learning_rate == 1.0
        assert ctx.debug is True

    def test_learning_rate_rejects_invalid_values(self):
        headers = {
            "x-target-url": "https://api.example.com/openapi.json",
            "x-api-type": "rest",
        }

        with pytest.raises(MissingHeaderError, match="X-Recipe-Learn-Rate"):
            parse_request_context({**headers, "x-recipe-learn-rate": "always"})

        with pytest.raises(MissingHeaderError, match="between 0 and 1"):
            parse_request_context({**headers, "x-recipe-learn-rate": "1.2"})

    @patch("api_agent.context.get_http_headers")
    def test_extracts_all_headers(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/graphql",
            "x-api-type": "graphql",
            "x-target-headers": '{"Authorization": "Bearer xxx"}',
            "x-include-result": "true",
        }
        ctx = get_request_context()
        assert ctx.target_url == "https://api.example.com/graphql"
        assert ctx.api_type == "graphql"
        assert ctx.target_headers == {"Authorization": "Bearer xxx"}
        assert ctx.allow_unsafe_paths == ()
        assert ctx.include_result is True

    @patch("api_agent.context.get_http_headers")
    def test_allow_unsafe_paths_parsed(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-allow-unsafe-paths": '["/search", "/_search", "/api/*/query"]',
        }
        ctx = get_request_context()
        assert ctx.allow_unsafe_paths == ("/search", "/_search", "/api/*/query")

    @patch("api_agent.context.get_http_headers")
    def test_allow_unsafe_paths_invalid_json_defaults_empty(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-allow-unsafe-paths": "not-json",
        }
        ctx = get_request_context()
        assert ctx.allow_unsafe_paths == ()

    @patch("api_agent.context.get_http_headers")
    def test_missing_target_url_raises(self, mock_headers):
        mock_headers.return_value = {"x-api-type": "graphql"}
        with pytest.raises(MissingHeaderError, match="X-Target-URL"):
            get_request_context()

    @patch("api_agent.context.get_http_headers")
    def test_missing_api_type_raises(self, mock_headers):
        mock_headers.return_value = {"x-target-url": "https://api.example.com"}
        with pytest.raises(MissingHeaderError, match="X-API-Type"):
            get_request_context()

    @patch("api_agent.context.get_http_headers")
    def test_invalid_api_type_raises(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "invalid",
        }
        with pytest.raises(MissingHeaderError, match="must be 'graphql' or 'rest'"):
            get_request_context()

    @patch("api_agent.context.get_http_headers")
    def test_rest_api_type(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/openapi.json",
            "x-api-type": "rest",
        }
        ctx = get_request_context()
        assert ctx.api_type == "rest"

    @patch("api_agent.context.get_http_headers")
    def test_invalid_json_headers_defaults_empty(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
            "x-target-headers": "not-json",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {}

    @patch("api_agent.context.get_http_headers")
    def test_missing_headers_defaults_empty(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {}

    @patch("api_agent.context.get_http_headers")
    def test_case_insensitive_headers(self, mock_headers):
        # HTTP headers are case-insensitive, fastmcp lowercases them
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
        }
        ctx = get_request_context()
        assert ctx.target_url == "https://api.example.com"

    @patch("api_agent.context.get_http_headers")
    def test_passthrough_headers_merged(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/graphql",
            "x-api-type": "graphql",
            "x-target-headers": '{"X-Api-Key": "from-json"}',
            "x-passthrough-headers": '["x-request-id", "x-correlation-id"]',
            "x-request-id": "req-abc",
            "x-correlation-id": "corr-xyz",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {
            "X-Api-Key": "from-json",
            "X-Request-Id": "req-abc",
            "X-Correlation-Id": "corr-xyz",
        }

    @patch("api_agent.context.get_http_headers")
    def test_passthrough_headers_invalid_json_ignored(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/graphql",
            "x-api-type": "graphql",
            "x-passthrough-headers": "not-json",
            "x-request-id": "req-abc",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {}

    @patch("api_agent.context.get_http_headers")
    def test_passthrough_headers_forwards_authorization_when_listed(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/graphql",
            "x-api-type": "graphql",
            "x-passthrough-headers": '["authorization"]',
            "authorization": "Bearer from-client",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {"Authorization": "Bearer from-client"}
        mock_headers.assert_called_once_with(include={"authorization"})

    @patch("api_agent.context.get_http_headers")
    def test_passthrough_skips_missing_headers(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/graphql",
            "x-api-type": "graphql",
            "x-passthrough-headers": '["x-missing", "x-request-id"]',
            "x-request-id": "only-this",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {"X-Request-Id": "only-this"}


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_frozen_immutable(self):
        ctx = RequestContext(
            target_url="https://api.example.com",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.target_url = "new"  # ty: ignore[invalid-assignment]  # intentional write to frozen field


class TestBaseUrl:
    """Test X-Base-URL header extraction."""

    @patch("api_agent.context.get_http_headers")
    def test_base_url_extracted(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/openapi.json",
            "x-api-type": "rest",
            "x-base-url": "https://api.example.com",
        }
        ctx = get_request_context()
        assert ctx.base_url == "https://api.example.com"

    @patch("api_agent.context.get_http_headers")
    def test_base_url_defaults_none(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/openapi.json",
            "x-api-type": "rest",
        }
        ctx = get_request_context()
        assert ctx.base_url is None

    @patch("api_agent.context.get_http_headers")
    def test_base_url_empty_string_normalizes_to_none(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com/openapi.json",
            "x-api-type": "rest",
            "x-base-url": "",
        }
        ctx = get_request_context()
        assert ctx.base_url is None

    @patch("api_agent.context.get_http_headers")
    def test_target_headers_empty_string_defaults_empty(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
            "x-target-headers": "",
        }
        ctx = get_request_context()
        assert ctx.target_headers == {}

    @patch("api_agent.context.get_http_headers")
    def test_allow_unsafe_paths_empty_string_defaults_empty(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-allow-unsafe-paths": "",
        }
        ctx = get_request_context()
        assert ctx.allow_unsafe_paths == ()


class TestIncludeResult:
    """Test X-Include-Result header extraction."""

    @pytest.mark.parametrize("value", ["true", "yes", "1", "TRUE", "Yes"])
    @patch("api_agent.context.get_http_headers")
    def test_truthy_values(self, mock_headers, value):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
            "x-include-result": value,
        }
        ctx = get_request_context()
        assert ctx.include_result is True

    @pytest.mark.parametrize("value", ["false", "no", "0", "", None])
    @patch("api_agent.context.get_http_headers")
    def test_falsy_values(self, mock_headers, value):
        headers = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "graphql",
        }
        if value is not None:
            headers["x-include-result"] = value
        mock_headers.return_value = headers
        ctx = get_request_context()
        assert ctx.include_result is False


class TestPollPaths:
    """Test X-Poll-Paths header extraction."""

    @patch("api_agent.context.get_http_headers")
    def test_poll_paths_parsed(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-poll-paths": '["/search", "/flights/*"]',
        }
        ctx = get_request_context()
        assert ctx.poll_paths == ("/search", "/flights/*")

    @pytest.mark.parametrize("value", ["not-json", "", None])
    @patch("api_agent.context.get_http_headers")
    def test_invalid_or_missing_defaults_empty(self, mock_headers, value):
        headers = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
        }
        if value is not None:
            headers["x-poll-paths"] = value
        mock_headers.return_value = headers
        ctx = get_request_context()
        assert ctx.poll_paths == ()


class TestGetToolNamePrefix:
    """Test tool name prefix generation (semantic, no hash)."""

    def test_simple_url_skips_api(self):
        url = "https://api.example.com/openapi.json"
        result = get_tool_name_prefix(url)
        assert result == "example"  # skips 'api' and 'com'

    def test_complex_subdomain(self):
        url = "https://flights-service.internal.example.com/openapi.json"
        result = get_tool_name_prefix(url)
        assert result == "flights_service_example"  # skips internal, com

    def test_consistent_result(self):
        url = "https://api.example.com/openapi.json"
        result1 = get_tool_name_prefix(url)
        result2 = get_tool_name_prefix(url)
        assert result1 == result2

    def test_different_urls_different_result(self):
        url1 = "https://api.example.com/openapi.json"
        url2 = "https://payments.example.net/openapi.json"
        assert get_tool_name_prefix(url1) != get_tool_name_prefix(url2)

    def test_empty_url(self):
        assert get_tool_name_prefix("") == "api"

    def test_none_url(self):
        assert get_tool_name_prefix(None) == "api"

    def test_truncated_to_32_chars(self):
        url = "https://very-long-subdomain-name.another-long-part.yet-more.example.com"
        result = get_tool_name_prefix(url)
        assert len(result) <= 32


class TestGetFullHostname:
    """Test full hostname extraction for descriptions."""

    def test_extracts_hostname(self):
        url = "https://flights-service.example.com/openapi.json"
        assert get_full_hostname(url) == "flights-service.example.com"

    def test_empty_url(self):
        assert get_full_hostname("") == "api"

    def test_none_url(self):
        assert get_full_hostname(None) == "api"


class TestExtractApiName:
    """Test API name extraction from headers."""

    def test_explicit_header_takes_priority(self):
        headers = {
            "x-api-name": "My Custom API",
            "x-target-url": "https://other-api.example.com",
        }
        assert extract_api_name(headers) == "my_custom_api"

    def test_explicit_header_preserves_hyphen(self):
        headers = {
            "x-api-name": "weather-alerts",
            "x-target-url": "https://other-api.example.com",
        }
        assert extract_api_name(headers) == "weather-alerts"

    def test_falls_back_to_url_prefix(self):
        headers = {"x-target-url": "https://flights-api.example.com/api"}
        result = extract_api_name(headers)
        assert result == "flights_api_example"

    def test_explicit_name_truncated_to_32(self):
        headers = {"x-api-name": "very_long_api_name_that_exceeds_the_32_char_limit"}
        result = extract_api_name(headers)
        assert len(result) <= 32

    def test_no_headers_returns_default(self):
        headers = {}
        assert extract_api_name(headers) == "api"
