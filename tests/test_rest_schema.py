"""Tests for REST/OpenAPI schema context generation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api_agent.rest.schema_loader import (
    _rewrite_swagger_ref,
    _swagger_param_to_oas3,
    _swagger_request_body_to_oas3,
    _swagger_responses_to_oas3,
    _swagger_security_to_oas3,
    _swagger_servers_from_spec,
    _format_params,
    _format_schema,
    _infer_string_format,
    _schema_to_type,
    build_schema_context,
    load_openapi_spec,
    normalize_swagger2_to_oas3,
)


class TestSchemaToType:
    """Test OpenAPI type to compact notation conversion."""

    def test_string(self):
        assert _schema_to_type({"type": "string"}) == "str"

    def test_integer(self):
        assert _schema_to_type({"type": "integer"}) == "int"

    def test_number(self):
        assert _schema_to_type({"type": "number"}) == "float"

    def test_boolean(self):
        assert _schema_to_type({"type": "boolean"}) == "bool"

    def test_array(self):
        assert _schema_to_type({"type": "array", "items": {"type": "string"}}) == "str[]"

    def test_array_of_objects(self):
        assert (
            _schema_to_type({"type": "array", "items": {"$ref": "#/components/schemas/User"}})
            == "User[]"
        )

    def test_ref(self):
        assert _schema_to_type({"$ref": "#/components/schemas/User"}) == "User"

    def test_object(self):
        assert _schema_to_type({"type": "object"}) == "object"

    def test_dict_type(self):
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        assert _schema_to_type(schema) == "dict[str, str]"

    def test_empty(self):
        assert _schema_to_type({}) == "any"

    def test_none(self):
        assert _schema_to_type(None) == "any"

    def test_nullable_type_array(self):
        """OpenAPI 3.1 nullable types as array."""
        assert _schema_to_type({"type": ["string", "null"]}) == "str"
        assert _schema_to_type({"type": ["integer", "null"]}) == "int"
        assert _schema_to_type({"type": ["null"]}) == "any"

    def test_string_with_format(self):
        """String format preserved in notation."""
        assert _schema_to_type({"type": "string", "format": "date-time"}) == "str(date-time)"
        assert _schema_to_type({"type": "string", "format": "date"}) == "str(date)"
        assert _schema_to_type({"type": "string", "format": "uri"}) == "str(uri)"
        assert _schema_to_type({"type": "string", "format": "email"}) == "str(email)"
        assert _schema_to_type({"type": "string"}) == "str"  # no format

    def test_string_format_inferred_from_field_name(self):
        """Format inferred from field name when not in schema."""
        # datetime inferred
        assert _schema_to_type({"type": "string"}, field_name="departDateTime") == "str(date-time)"
        assert _schema_to_type({"type": "string"}, field_name="arrivalDateTime") == "str(date-time)"
        # date inferred
        assert _schema_to_type({"type": "string"}, field_name="birthDate") == "str(date)"
        # explicit format takes precedence
        assert (
            _schema_to_type({"type": "string", "format": "uri"}, field_name="dateTime")
            == "str(uri)"
        )
        # no inference for unrelated names
        assert _schema_to_type({"type": "string"}, field_name="name") == "str"
        # "update" excluded to avoid false positives like "updatedAt"
        assert _schema_to_type({"type": "string"}, field_name="updateDate") == "str"


class TestInferStringFormat:
    """Test format inference from field names."""

    def test_datetime_field(self):
        assert _infer_string_format("departDateTime") == "date-time"
        assert _infer_string_format("arrivalDateTime") == "date-time"
        assert _infer_string_format("createdDatetime") == "date-time"

    def test_date_field(self):
        assert _infer_string_format("birthDate") == "date"
        assert _infer_string_format("startDate") == "date"

    def test_time_field(self):
        assert _infer_string_format("openTime") == "time"
        assert _infer_string_format("checkInTime") == "time"
        assert _infer_string_format("departureTime") == "time"

    def test_excludes_update(self):
        """Avoid false positives for 'updatedAt' style fields."""
        assert _infer_string_format("updateDate") == ""
        assert _infer_string_format("lastUpdated") == ""

    def test_no_match(self):
        assert _infer_string_format("name") == ""
        assert _infer_string_format("email") == ""
        assert _infer_string_format("") == ""


class TestFormatParams:
    """Test parameter formatting."""

    def test_required_param(self):
        params = [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}]
        assert _format_params(params) == "id: str"

    def test_optional_param_stripped(self):
        """Optional params are now stripped entirely."""
        params = [
            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}}
        ]
        assert _format_params(params) == ""  # Optional stripped

    def test_path_param_always_required(self):
        params = [{"name": "id", "in": "path", "schema": {"type": "string"}}]
        assert _format_params(params) == "id: str"

    def test_multiple_params_only_required(self):
        """Only required params shown, optional stripped."""
        params = [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
        ]
        assert _format_params(params) == "id: str"  # limit stripped


class TestFormatSchema:
    """Test schema formatting."""

    def test_object_schema_only_required(self):
        """Only required fields shown, optional stripped."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
            "required": ["id"],
        }
        result = _format_schema("User", schema)
        assert "User {" in result
        assert "id: str!" in result
        assert "name" not in result  # optional field stripped

    def test_enum_schema(self):
        schema = {"type": "string", "enum": ["active", "inactive"]}
        result = _format_schema("Status", schema)
        assert "Status: enum(active | inactive)" in result

    def test_malformed_required_with_list(self):
        """Handle malformed OpenAPI where required contains nested lists."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
            "required": ["id", ["nested", "list"]],
        }
        result = _format_schema("User", schema)
        assert "id: str!" in result
        assert "name" not in result  # optional stripped

    def test_malformed_required_with_dict(self):
        """Handle malformed OpenAPI where required contains dicts."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": [{"field": "id"}],
        }
        result = _format_schema("User", schema)
        assert "id" not in result  # dict in required is filtered, so id is optional → stripped

    def test_malformed_required_mixed_types(self):
        """Handle required with mixed valid and invalid types."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", None, 123, ["x"], "b"],
        }
        result = _format_schema("Test", schema)
        assert "a: str!" in result
        assert "b: str!" in result


class TestBuildSchemaContext:
    """Test OpenAPI schema context generation."""

    @pytest.fixture
    def openapi_spec(self):
        """Realistic OpenAPI 3.x spec fixture."""
        return {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                            },
                            {
                                "name": "offset",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                            },
                        ],
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/User"},
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
                "/users/{id}": {
                    "get": {
                        "summary": "Get user",
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/User"}
                                    }
                                }
                            }
                        },
                    }
                },
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["id", "name"],
                    }
                },
                "securitySchemes": {
                    "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
                },
            },
        }

    def test_endpoints_section(self, openapi_spec):
        ctx = build_schema_context(openapi_spec)
        assert "<endpoints>" in ctx
        # Optional params stripped, so GET /users has no params
        assert "GET /users() -> User[]" in ctx
        assert "GET /users/{id}(id: str) -> User" in ctx

    def test_endpoints_with_summary(self, openapi_spec):
        ctx = build_schema_context(openapi_spec)
        assert "# List users" in ctx
        assert "# Get user" in ctx

    def test_schemas_section(self, openapi_spec):
        ctx = build_schema_context(openapi_spec)
        assert "<schemas>" in ctx
        assert "User {" in ctx
        assert "id: str!" in ctx
        assert "name: str!" in ctx
        assert "email" not in ctx  # optional field stripped

    def test_auth_section(self, openapi_spec):
        ctx = build_schema_context(openapi_spec)
        assert "<auth>" in ctx
        assert "bearerAuth: HTTP bearer JWT" in ctx

    def test_empty_spec(self):
        ctx = build_schema_context({})
        assert ctx == ""

    def test_api_key_auth(self):
        spec = {
            "openapi": "3.0.0",
            "paths": {},
            "components": {
                "securitySchemes": {
                    "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
                }
            },
        }
        ctx = build_schema_context(spec)
        assert "apiKey: API key in header 'X-API-Key'" in ctx

    def test_post_endpoint_with_body(self):
        """POST endpoints show request body type."""
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/search": {
                    "post": {
                        "summary": "Search flights",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SearchRequest"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/SearchResponse"}
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {"schemas": {}},
        }
        ctx = build_schema_context(spec)
        assert "POST /search(body: SearchRequest!) -> SearchResponse" in ctx

    def test_post_endpoint_optional_body(self):
        """POST with optional body."""
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/update": {
                    "put": {
                        "requestBody": {
                            "required": False,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Data"}
                                }
                            },
                        },
                        "responses": {"200": {}},
                    }
                }
            },
        }
        ctx = build_schema_context(spec)
        assert "PUT /update(body: Data)" in ctx
        assert "body: Data!" not in ctx  # not required


class TestSwagger2Normalization:
    def test_normalize_swagger2_basic_shapes(self):
        swagger_spec = {
            "swagger": "2.0",
            "info": {"title": "OKR API", "version": "2.0"},
            "host": "api.example.com",
            "basePath": "/v1",
            "schemes": ["https"],
            "paths": {
                "/users/{id}": {
                    "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
                    "get": {
                        "summary": "Get user",
                        "responses": {"200": {"schema": {"$ref": "#/definitions/User"}}},
                    },
                    "post": {
                        "summary": "Update user",
                        "parameters": [
                            {
                                "name": "body",
                                "in": "body",
                                "required": True,
                                "schema": {"$ref": "#/definitions/UpdateUser"},
                            }
                        ],
                        "responses": {"200": {"schema": {"$ref": "#/definitions/User"}}},
                    },
                }
            },
            "definitions": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                "UpdateUser": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "securityDefinitions": {"basicAuth": {"type": "basic"}},
        }

        normalized = normalize_swagger2_to_oas3(swagger_spec)

        assert normalized["openapi"].startswith("3.")
        assert normalized["servers"] == [{"url": "https://api.example.com/v1"}]
        assert "components" in normalized
        assert "schemas" in normalized["components"]
        assert "User" in normalized["components"]["schemas"]
        assert normalized["components"]["securitySchemes"]["basicAuth"] == {
            "type": "http",
            "scheme": "basic",
        }

        get_op = normalized["paths"]["/users/{id}"]["get"]
        assert get_op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/User"
        )

        post_op = normalized["paths"]["/users/{id}"]["post"]
        assert post_op["requestBody"]["required"] is True
        assert (
            post_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            == "#/components/schemas/UpdateUser"
        )

    def test_build_context_from_normalized_swagger2(self):
        swagger_spec = {
            "swagger": "2.0",
            "paths": {
                "/users/{id}": {
                    "get": {
                        "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
                        "responses": {"200": {"schema": {"$ref": "#/definitions/User"}}},
                    }
                }
            },
            "definitions": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                }
            },
        }
        normalized = normalize_swagger2_to_oas3(swagger_spec)
        ctx = build_schema_context(normalized)
        assert "GET /users/{id}(id: str) -> User" in ctx
        assert "User { id: str! }" in ctx


class TestSwagger2Helpers:
    def test_rewrite_swagger_ref(self):
        assert _rewrite_swagger_ref("#/definitions/User") == "#/components/schemas/User"

    def test_swagger_param_conversion(self):
        result = _swagger_param_to_oas3({"name": "limit", "in": "query", "type": "integer"})
        assert result is not None
        assert result["schema"]["type"] == "integer"

    def test_swagger_request_body_extraction(self):
        body, remaining = _swagger_request_body_to_oas3(
            [
                {"in": "body", "name": "data", "required": True, "schema": {"type": "object"}},
                {"in": "query", "name": "limit", "type": "integer"},
            ]
        )
        assert body is not None
        assert body["required"] is True
        assert len(remaining) == 1

    def test_swagger_response_conversion(self):
        responses = {"200": {"description": "ok", "schema": {"$ref": "#/definitions/User"}}}
        result = _swagger_responses_to_oas3(responses)
        assert result["200"]["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/User"
        )

    def test_swagger_security_conversion(self):
        result = _swagger_security_to_oas3({"basicAuth": {"type": "basic"}})
        assert result["basicAuth"] == {"type": "http", "scheme": "basic"}

    def test_swagger_servers_from_spec(self):
        result = _swagger_servers_from_spec(
            {"host": "api.example.com", "basePath": "/v1", "schemes": ["https"]}
        )
        assert result == [{"url": "https://api.example.com/v1"}]


def _mock_http_response(status: int, text: str):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status
    mock_resp.text = text
    if status >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def _patch_http(text: str, status: int = 200):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=_mock_http_response(status, text))
    return patch("httpx.AsyncClient", return_value=mock_client)


class TestLoadOpenApiSpec:
    @pytest.mark.asyncio
    async def test_swagger_2_spec_normalized(self):
        spec = json.dumps({"swagger": "2.0", "info": {}, "paths": {}, "host": "api.example.com"})
        with _patch_http(spec):
            result = await load_openapi_spec("https://api.example.com/openapi.json")
        assert result.get("openapi", "").startswith("3.")
