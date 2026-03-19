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
)


class TestGetRequestContext:
    """Test header extraction and validation."""

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
            ctx.target_url = "new"  # type: ignore[misc]  # intentional write to frozen field


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


class TestAuthHeaders:
    """Test X-Auth-* header extraction."""

    @patch("api_agent.context.get_http_headers")
    def test_auth_headers_parsed(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-auth-url": "https://auth0.example.com/token",
            "x-auth-body": '{"grant_type": "client_credentials", "client_id": "abc"}',
            "x-auth-token-path": "data.token",
        }
        ctx = get_request_context()
        assert ctx.auth_url == "https://auth0.example.com/token"
        assert ctx.auth_body == {"grant_type": "client_credentials", "client_id": "abc"}
        assert ctx.auth_token_path == "data.token"

    @patch("api_agent.context.get_http_headers")
    def test_auth_headers_default_when_absent(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
        }
        ctx = get_request_context()
        assert ctx.auth_url is None
        assert ctx.auth_body is None
        assert ctx.auth_token_path == "access_token"

    @patch("api_agent.context.get_http_headers")
    def test_auth_body_invalid_json_defaults_none(self, mock_headers):
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-auth-url": "https://auth0.example.com/token",
            "x-auth-body": "not-json",
        }
        ctx = get_request_context()
        assert ctx.auth_url == "https://auth0.example.com/token"
        assert ctx.auth_body is None

    @patch("api_agent.context.get_http_headers")
    def test_auth_body_ignored_without_auth_url(self, mock_headers):
        """auth_body not parsed if auth_url is missing."""
        mock_headers.return_value = {
            "x-target-url": "https://api.example.com",
            "x-api-type": "rest",
            "x-auth-body": '{"grant_type": "client_credentials"}',
        }
        ctx = get_request_context()
        assert ctx.auth_url is None
        assert ctx.auth_body is None


class TestGetToolNamePrefix:
    """Test tool name prefix generation (semantic, no hash)."""

    def test_simple_url_skips_api(self):
        url = "https://api.example.com/openapi.json"
        result = get_tool_name_prefix(url)
        assert result == "example"  # skips 'api' and 'com'

    def test_complex_subdomain(self):
        url = "https://flights-api-qa.internal.example.com/openapi.json"
        result = get_tool_name_prefix(url)
        assert result == "flights_api_qa_example"  # skips internal, com

    def test_consistent_result(self):
        url = "https://api.example.com/openapi.json"
        result1 = get_tool_name_prefix(url)
        result2 = get_tool_name_prefix(url)
        assert result1 == result2

    def test_different_urls_different_result(self):
        url1 = "https://api.example.com/openapi.json"
        url2 = "https://api.stripe.com/openapi.json"
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
        url = "https://flights-api-qa.example.com/openapi.json"
        assert get_full_hostname(url) == "flights-api-qa.example.com"

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
