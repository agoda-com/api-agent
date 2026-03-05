"""OpenAPI 3.x spec loader and compact schema context builder."""

import json
import logging
from typing import Any

import httpx
import yaml

from ..config import settings

logger = logging.getLogger(__name__)


def _rewrite_swagger_ref(ref: Any) -> Any:
    """Rewrite Swagger 2 refs to OpenAPI 3 refs."""
    if not isinstance(ref, str):
        return ref
    return ref.replace("#/definitions/", "#/components/schemas/")


def _rewrite_refs(value: Any) -> Any:
    """Recursively rewrite refs from Swagger 2 to OpenAPI 3 paths."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "$ref":
                out[k] = _rewrite_swagger_ref(v)
            else:
                out[k] = _rewrite_refs(v)
        return out
    if isinstance(value, list):
        return [_rewrite_refs(v) for v in value]
    return value


def _swagger_param_to_oas3(param: Any) -> dict[str, Any] | None:
    """Convert non-body Swagger 2 parameter to OpenAPI 3 parameter."""
    if not isinstance(param, dict):
        return None
    if param.get("in") == "body":
        return None

    converted = _rewrite_refs(param)
    if "schema" in converted and isinstance(converted["schema"], dict):
        return converted

    # Swagger 2 allows schema fields directly on parameter.
    schema: dict[str, Any] = {}
    for key in ["type", "format", "items", "enum", "default", "minimum", "maximum"]:
        if key in converted:
            schema[key] = converted[key]  # already rewritten
    if schema:
        converted["schema"] = schema
    return converted


def _swagger_request_body_to_oas3(parameters: list[Any]) -> tuple[dict[str, Any] | None, list[Any]]:
    """Extract Swagger 2 body parameter and return OAS3 requestBody + remaining params."""
    request_body: dict[str, Any] | None = None
    remaining: list[Any] = []
    for p in parameters:
        if isinstance(p, dict) and p.get("in") == "body" and request_body is None:
            schema = p.get("schema", {})
            if isinstance(schema, dict):
                request_body = {
                    "required": bool(p.get("required", False)),
                    "content": {"application/json": {"schema": _rewrite_refs(schema)}},
                }
            continue
        remaining.append(p)
    return request_body, remaining


def _swagger_responses_to_oas3(responses: Any) -> dict[str, Any]:
    """Convert Swagger 2 responses shape to OpenAPI 3 responses shape."""
    if not isinstance(responses, dict):
        return {}

    out: dict[str, Any] = {}
    for code, resp in responses.items():
        if not isinstance(resp, dict):
            continue
        converted = _rewrite_refs(resp)
        schema = converted.pop("schema", None)
        if isinstance(schema, dict):
            converted["content"] = {"application/json": {"schema": schema}}
        out[str(code)] = converted
    return out


def _swagger_security_to_oas3(security_definitions: Any) -> dict[str, Any]:
    """Convert Swagger 2 securityDefinitions to OAS3 securitySchemes."""
    if not isinstance(security_definitions, dict):
        return {}

    out: dict[str, Any] = {}
    for name, scheme in security_definitions.items():
        if not isinstance(scheme, dict):
            continue
        scheme_type = scheme.get("type", "")
        converted = _rewrite_refs(scheme)
        if scheme_type == "basic":
            converted = {"type": "http", "scheme": "basic"}
        elif scheme_type == "oauth2":
            flow = scheme.get("flow", "")
            flows: dict[str, Any] = {}
            scopes = scheme.get("scopes", {})
            if flow == "accessCode":
                flows["authorizationCode"] = {
                    "authorizationUrl": scheme.get("authorizationUrl", ""),
                    "tokenUrl": scheme.get("tokenUrl", ""),
                    "scopes": scopes if isinstance(scopes, dict) else {},
                }
            elif flow == "application":
                flows["clientCredentials"] = {
                    "tokenUrl": scheme.get("tokenUrl", ""),
                    "scopes": scopes if isinstance(scopes, dict) else {},
                }
            elif flow == "password":
                flows["password"] = {
                    "tokenUrl": scheme.get("tokenUrl", ""),
                    "scopes": scopes if isinstance(scopes, dict) else {},
                }
            else:
                flows["implicit"] = {
                    "authorizationUrl": scheme.get("authorizationUrl", ""),
                    "scopes": scopes if isinstance(scopes, dict) else {},
                }
            converted = {"type": "oauth2", "flows": flows}
        out[name] = converted
    return out


def _swagger_servers_from_spec(swagger_spec: dict[str, Any]) -> list[dict[str, str]]:
    """Build OAS3 servers from Swagger 2 host/basePath/schemes fields."""
    host = swagger_spec.get("host", "")
    base_path = swagger_spec.get("basePath", "")
    if not isinstance(host, str) or not host:
        return []
    if not isinstance(base_path, str):
        base_path = ""
    if base_path and not base_path.startswith("/"):
        base_path = f"/{base_path}"

    schemes = swagger_spec.get("schemes", [])
    if not isinstance(schemes, list):
        schemes = []
    scheme_list = [s for s in schemes if isinstance(s, str) and s]
    if not scheme_list:
        scheme_list = ["https"]
    return [{"url": f"{s}://{host}{base_path}"} for s in scheme_list]


def normalize_swagger2_to_oas3(swagger_spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize Swagger 2.0 spec into minimal OpenAPI 3.x structure."""
    out: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": _rewrite_refs(swagger_spec.get("info", {}))
        if isinstance(swagger_spec.get("info"), dict)
        else {},
        "paths": {},
        "components": {
            "schemas": _rewrite_refs(swagger_spec.get("definitions", {}))
            if isinstance(swagger_spec.get("definitions"), dict)
            else {},
            "securitySchemes": _swagger_security_to_oas3(
                swagger_spec.get("securityDefinitions", {})
            ),
        },
    }

    servers = _swagger_servers_from_spec(swagger_spec)
    if servers:
        out["servers"] = servers

    paths = swagger_spec.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}

    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        out_path_item: dict[str, Any] = {}

        path_level_params_raw = path_item.get("parameters", [])
        if not isinstance(path_level_params_raw, list):
            path_level_params_raw = []
        path_level_params = []
        for p in path_level_params_raw:
            converted_param = _swagger_param_to_oas3(p)
            if converted_param:
                path_level_params.append(converted_param)
        if path_level_params:
            out_path_item["parameters"] = path_level_params

        for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue

            new_op = _rewrite_refs(op)
            raw_params = op.get("parameters", [])
            if not isinstance(raw_params, list):
                raw_params = []
            request_body, non_body_params = _swagger_request_body_to_oas3(raw_params)

            converted_params = []
            for p in non_body_params:
                converted_param = _swagger_param_to_oas3(p)
                if converted_param:
                    converted_params.append(converted_param)
            if converted_params:
                new_op["parameters"] = converted_params
            else:
                new_op.pop("parameters", None)

            if request_body:
                new_op["requestBody"] = request_body

            new_op["responses"] = _swagger_responses_to_oas3(op.get("responses", {}))
            out_path_item[method] = new_op

        out["paths"][path] = out_path_item

    return out


async def load_openapi_spec(
    spec_url: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load OpenAPI 3.x spec from URL.

    Args:
        spec_url: URL to OpenAPI 3.x spec
        headers: Optional auth headers

    Returns:
        Parsed OpenAPI spec dict, or empty dict on error.
    """
    if not spec_url:
        return {}

    request_headers = dict(headers) if headers else {}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(spec_url, headers=request_headers)
            resp.raise_for_status()
            raw = resp.text

        # Parse JSON or YAML
        if raw.strip().startswith("{"):
            spec = json.loads(raw)
        else:
            spec = yaml.safe_load(raw)

        if not isinstance(spec, dict):
            logger.warning("OpenAPI spec root is not an object")
            return {}

        # Validate/normalize API schema shape
        openapi_version = spec.get("openapi", "")
        if isinstance(openapi_version, str) and openapi_version.startswith("3."):
            return spec

        swagger_version = spec.get("swagger", "")
        if isinstance(swagger_version, str) and swagger_version.startswith("2."):
            logger.info("Detected Swagger 2.0 spec, normalizing to OpenAPI 3.0 shape")
            return normalize_swagger2_to_oas3(spec)

        logger.warning(
            f"Unsupported API schema version. openapi={openapi_version!r}, swagger={swagger_version!r}"
        )
        return {}

    except Exception as e:
        logger.exception(f"Failed to load OpenAPI spec: {e}")
        return {}


def get_base_url_from_spec(spec: dict[str, Any], spec_url: str = "") -> str:
    """Extract base URL from OpenAPI spec's servers[0], or derive from spec URL."""
    servers = spec.get("servers", [])
    if isinstance(servers, list) and servers:
        first = servers[0]
        if isinstance(first, dict):
            url = first.get("url", "")
            if isinstance(url, str):
                return url

    # Fallback: derive from spec URL (e.g., https://api.example.com/openapi.json -> https://api.example.com)
    if spec_url:
        from urllib.parse import urlparse

        parsed = urlparse(spec_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    return ""


def _infer_string_format(field_name: str) -> str:
    """Infer string format from field name when not in schema."""
    if not field_name:
        return ""
    name_lower = field_name.lower()
    if "datetime" in name_lower:
        return "date-time"
    if "date" in name_lower and "update" not in name_lower:
        return "date"
    if "time" in name_lower and "update" not in name_lower:
        return "time"
    return ""


def _schema_to_type(
    schema: Any,
    schemas: dict[str, Any] | None = None,
    field_name: str = "",
) -> str:
    """Convert JSON Schema to compact type notation."""
    # OpenAPI 3.1 / JSON Schema allows boolean schemas (true/false)
    if schema is True or schema is False:
        return "any"

    if not schema:
        return "any"

    if not isinstance(schema, dict):
        return "any"

    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        # Extract type name from #/components/schemas/TypeName
        if isinstance(ref, str):
            return ref.split("/")[-1]
        return "any"

    schema_type = schema.get("type", "any")
    if not isinstance(schema_type, (str, list)):
        return "any"
    # OpenAPI 3.1 allows type arrays like ["string", "null"] - use first non-null type
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if non_null else "any"

    if schema_type == "array":
        items = schema.get("items", {})
        return f"{_schema_to_type(items, schemas)}[]"

    if schema_type == "object":
        # Check for additionalProperties (dict type)
        additional = schema.get("additionalProperties")
        if additional is True:
            return "dict[str, any]"
        if isinstance(additional, dict):
            val_type = _schema_to_type(additional, schemas)
            return f"dict[str, {val_type}]"
        if additional is not None:
            return "dict[str, any]"
        return "object"

    # Preserve string format (date-time, date, uri, etc.)
    if schema_type == "string":
        fmt = schema.get("format", "") or _infer_string_format(field_name)
        return f"str({fmt})" if fmt else "str"

    type_map = {
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    return type_map.get(schema_type, schema_type)


def _format_params(params: list[Any]) -> str:
    """Format ONLY required parameters. Optional params stripped."""
    parts = []
    for p in params:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "")
        required = p.get("required", p.get("in") == "path")  # path params always required
        if not required:
            continue  # Skip optional params
        schema = p.get("schema", {})
        type_str = _schema_to_type(schema, field_name=name)
        parts.append(f"{name}: {type_str}")
    return ", ".join(parts)


def _extract_response_type(responses: Any) -> str:
    """Extract return type from responses."""
    if not isinstance(responses, dict):
        return "any"

    # Check 200/201 responses
    for code in ["200", "201", "default"]:
        if code in responses:
            resp = responses[code]
            if not isinstance(resp, dict):
                continue
            content = resp.get("content", {})
            if not isinstance(content, dict):
                continue
            json_content = content.get("application/json", {})
            if not isinstance(json_content, dict):
                continue
            schema = json_content.get("schema", {})
            if schema:
                return _schema_to_type(schema)
    return "any"


def _format_schema(name: str, schema: Any) -> str:
    """Format schema definition with ONLY required fields.

    Optional fields are stripped to prevent agent from inventing values.
    """
    if schema is True or schema is False or not isinstance(schema, dict):
        return f"{name}: {_schema_to_type(schema)}"

    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        raw_required = schema.get("required", [])
        if not isinstance(raw_required, list):
            raw_required = []
        required = set(r for r in raw_required if isinstance(r, str))
        fields = []
        # Only include required fields
        for field_name, field_schema in props.items():
            if field_name not in required:
                continue  # Skip optional fields
            field_type = _schema_to_type(field_schema, field_name=field_name)
            fields.append(f"{field_name}: {field_type}!")
        return f"{name} {{ {', '.join(fields)} }}"

    elif schema.get("enum"):
        enum_vals = schema.get("enum")
        if not isinstance(enum_vals, list):
            enum_vals = [enum_vals]
        vals = " | ".join(str(v) for v in enum_vals)
        return f"{name}: enum({vals})"

    return f"{name}: {_schema_to_type(schema)}"


def build_schema_context(spec: dict[str, Any]) -> str:
    """Build compact schema context from OpenAPI spec.

    Format:
        <endpoints>
        GET /users(limit?: int, offset?: int) -> User[]  # List users
        GET /users/{id}() -> User  # Get user by ID

        <schemas>
        User { id: str!, name: str!, email?: str }

        <auth>
        Bearer token in Authorization header
    """
    if not spec:
        return ""

    lines = ["<endpoints>"]

    # Process paths
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    for path, path_item in paths.items():
        # Skip OpenAPI extension keys (e.g., x-foo) and malformed entries
        if not isinstance(path, str) or not path.startswith("/"):
            continue
        if not isinstance(path_item, dict):
            continue
        for method in ["get", "post", "put", "delete", "patch"]:
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue

            params = op.get("parameters", [])
            if not isinstance(params, list):
                params = []
            # Also include path-level parameters
            path_params = path_item.get("parameters", [])
            if not isinstance(path_params, list):
                path_params = []
            params = path_params + params

            # Extract request body type for POST/PUT/PATCH
            body_type = ""
            if method in ["post", "put", "patch"]:
                req_body = op.get("requestBody", {})
                if isinstance(req_body, dict):
                    content = req_body.get("content", {})
                    if isinstance(content, dict):
                        json_content = content.get("application/json", {})
                        if isinstance(json_content, dict):
                            body_schema = json_content.get("schema", {})
                            if body_schema:
                                required = bool(req_body.get("required", False))
                                body_type = _schema_to_type(body_schema)
                                suffix = "!" if required else ""
                                body_type = f"body: {body_type}{suffix}"

            param_str = _format_params(params)
            # Prepend body type if present
            if body_type:
                param_str = f"{body_type}, {param_str}" if param_str else body_type

            response_type = _extract_response_type(op.get("responses", {}))
            summary = op.get("description") or op.get("summary") or op.get("operationId", "")
            if not isinstance(summary, str):
                summary = ""

            desc = f"  # {summary}" if summary else ""
            lines.append(f"{method.upper()} {path}({param_str}) -> {response_type}{desc}")

    # Process schemas
    components = spec.get("components", {})
    if not isinstance(components, dict):
        components = {}
    schemas = components.get("schemas", {})
    if isinstance(schemas, dict) and schemas:
        lines.append("\n<schemas>")
        for name, schema in schemas.items():
            lines.append(_format_schema(name, schema))

    # Auth info
    security_schemes = components.get("securitySchemes", {})
    if isinstance(security_schemes, dict) and security_schemes:
        lines.append("\n<auth>")
        for name, scheme in security_schemes.items():
            if not isinstance(scheme, dict):
                continue
            scheme_type = scheme.get("type", "")
            if scheme_type == "http":
                bearer_format = scheme.get("bearerFormat", "")
                lines.append(f"{name}: HTTP {scheme.get('scheme', '')} {bearer_format}".strip())
            elif scheme_type == "apiKey":
                in_loc = scheme.get("in", "")
                key_name = scheme.get("name", "")
                lines.append(f"{name}: API key in {in_loc} '{key_name}'")
            elif scheme_type == "oauth2":
                lines.append(f"{name}: OAuth2")
            else:
                lines.append(f"{name}: {scheme_type}")

    return "\n".join(lines)


async def fetch_schema_context(
    spec_url: str,
    headers: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """Fetch and build schema context.

    Args:
        spec_url: URL to OpenAPI spec
        headers: Optional auth headers

    Returns:
        Tuple of (truncated_context, base_url, raw_spec_json)
    """
    spec = await load_openapi_spec(spec_url, headers)
    if not spec:
        return "", "", ""

    # Raw spec JSON for grep-like search (preserves all info)
    try:
        raw_spec_json = json.dumps(spec, indent=2)
    except TypeError:
        raw_spec_json = json.dumps(spec, indent=2, default=str)

    # Build DSL for LLM context
    try:
        dsl_context = build_schema_context(spec)
    except Exception:
        logger.exception("Failed to build OpenAPI schema context")
        dsl_context = ""
    base_url = get_base_url_from_spec(spec, spec_url)

    # Truncate DSL if too large
    context = dsl_context
    if len(context) > settings.MAX_SCHEMA_CHARS:
        context = (
            context[: settings.MAX_SCHEMA_CHARS]
            + "\n[SCHEMA TRUNCATED - use search_schema() to explore]"
        )

    return context, base_url, raw_spec_json
