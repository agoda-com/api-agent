"""FastMCP middleware for dynamic tool naming per session."""

import json
import re
from collections.abc import Sequence

from fastmcp.exceptions import NotFoundError, ToolError, ValidationError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools import Tool as FastMCPTool
from fastmcp.tools import ToolResult
from mcp import types as mt
from mcp.types import TextContent

from .config import settings
from .context import MissingHeaderError, extract_api_name, get_full_hostname, get_request_context
from .description import get_downstream_description
from .recipe.contracts import (
    get_recipe_description,
    get_recipe_tool_args,
    get_recipe_tool_name,
    has_recipe_contract,
)
from .recipe.naming import sanitize_tool_name
from .recipe.runner import execute_recipe_tool, load_schema_and_base_url
from .recipe.search import build_api_id
from .recipe.tooling import build_recipe_docstring, create_params_model
from .store import ASYNC_API_AGENT_STORE, sha256_hex

# Internal tool name suffix pattern
INTERNAL_TOOL_PATTERN = re.compile(r"^_(.+)$")
# Keep exposed recipe tool names below 50 chars including "r_".
MAX_TOOL_NAME_LEN = 49
RECIPE_NAME_PREFIX = "r"
RECIPE_PREFIX_STR = f"{RECIPE_NAME_PREFIX}_"
SPECIFIC_TOOL_HINT = (
    "If another listed tool directly matches the request, use that specific tool before this "
    "general question tool."
)


def _get_tool_suffix(internal_name: str) -> str:
    """Extract suffix from internal tool name (_query -> query)."""
    match = INTERNAL_TOOL_PATTERN.match(internal_name)
    return match.group(1) if match else internal_name


def _prefer_specific_tools(description: str) -> str:
    """Add a model-facing hint when specific tools are available."""
    if SPECIFIC_TOOL_HINT in description:
        return description
    return f"{description.rstrip()}\n\n{SPECIFIC_TOOL_HINT}"


def _tool_title(name: str) -> str:
    """Build a compact human title from a tool name."""
    return " ".join(part for part in name.replace("_", " ").split()).title()


def _max_slug_length() -> int:
    """Calculate max slug length that fits within tool name limit."""
    return max(1, MAX_TOOL_NAME_LEN - len(RECIPE_PREFIX_STR))


def _build_recipe_tool_name(slug: str) -> str:
    """Build MCP tool name from a pre-sanitized slug."""
    base = f"{RECIPE_NAME_PREFIX}_{slug}"
    if len(base) <= MAX_TOOL_NAME_LEN:
        return base
    max_slug = _max_slug_length()
    return f"{RECIPE_NAME_PREFIX}_{slug[:max_slug]}"


def _build_recipe_input_schema(params_spec: dict, tool_name: str) -> dict:
    """Build flat JSON Schema for recipe tool input.

    All public tool args are top-level required fields.
    Uses Pydantic ``create_params_model`` for schema generation.
    """
    Model = create_params_model(params_spec, tool_name)
    schema = Model.model_json_schema()

    schema.pop("title", None)
    schema["additionalProperties"] = False

    return schema


async def _list_recipe_tools(
    req_ctx,
    raw_schema: str,
    base_url: str,
) -> list[FastMCPTool]:
    if not settings.ENABLE_RECIPES:
        return []

    if not raw_schema:
        return []

    schema_hash = sha256_hex(raw_schema)
    api_id = build_api_id(req_ctx, req_ctx.api_type, base_url)
    recipes = await ASYNC_API_AGENT_STORE.list_recipes(
        api_id=api_id,
        schema_hash=schema_hash,
    )
    tools: list[FastMCPTool] = []

    # Group by tool slug (truncated to fit name) and pick most recent
    max_slug_len = _max_slug_length()
    by_slug: dict[str, list[dict]] = {}
    for r in recipes:
        if not has_recipe_contract(r):
            continue
        name = get_recipe_tool_name(r) or "recipe"
        slug = sanitize_tool_name(name)[:max_slug_len]
        by_slug.setdefault(slug, []).append(r)

    for slug, group in by_slug.items():
        group.sort(key=lambda r: (r.get("last_used_at", 0), r.get("created_at", 0)), reverse=True)
        r = group[0]
        tool_name = _build_recipe_tool_name(slug)
        params_spec = get_recipe_tool_args(r)
        description = get_recipe_description(r)
        if not description.strip():
            continue
        desc = build_recipe_docstring(
            r.get("question", ""),
            [],
            req_ctx.api_type,
            params_spec,
            description=description,
        )
        title = _tool_title(get_recipe_tool_name(r) or slug)
        tools.append(
            FastMCPTool(
                name=tool_name,
                title=title,
                description=desc,
                parameters=_build_recipe_input_schema(params_spec, slug),
                annotations=mt.ToolAnnotations(
                    title=title,
                    openWorldHint=True,
                ),
                tags={"recipe"},
            )
        )

    return tools


class DynamicToolNamingMiddleware(Middleware):
    """Middleware that dynamically names tools based on session context."""

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next,
    ) -> Sequence[FastMCPTool]:
        """Transform tool names and descriptions based on session headers."""
        tools: Sequence[FastMCPTool] = await call_next(context)

        try:
            headers = get_http_headers()
        except LookupError:
            # No HTTP context (e.g., stdio transport) - return unchanged
            return tools

        try:
            req_ctx = get_request_context()
        except MissingHeaderError as e:
            raise RuntimeError(str(e)) from e

        raw_schema, base_url = await load_schema_and_base_url(req_ctx)
        if not raw_schema:
            schema_type = "GraphQL" if req_ctx.api_type == "graphql" else "OpenAPI"
            raise RuntimeError(
                f"Failed to load {schema_type} schema. Check X-Target-URL and auth headers."
            )

        target_url = headers.get("x-target-url", "")
        # Short prefix for tool name, full hostname for description
        name_prefix = extract_api_name(headers)
        full_hostname = get_full_hostname(target_url)
        schema_hash = sha256_hex(raw_schema)
        api_id = build_api_id(req_ctx, req_ctx.api_type, base_url)
        downstream_description = await get_downstream_description(
            api_type=req_ctx.api_type,
            hostname=full_hostname,
            raw_schema=raw_schema,
            api_id=api_id,
            schema_hash=schema_hash,
        )
        recipe_tools = await _list_recipe_tools(req_ctx, raw_schema, base_url)
        has_specific_tools = bool(recipe_tools)

        transformed = []
        for tool in tools:
            suffix = _get_tool_suffix(tool.name)
            primary_prefix = f"{name_prefix}_"
            alt_prefix = f"{name_prefix.replace('-', '_')}_"
            if suffix.startswith(primary_prefix):
                suffix = suffix.removeprefix(primary_prefix)
            elif suffix.startswith(alt_prefix):
                suffix = suffix.removeprefix(alt_prefix)
            new_name = f"{name_prefix}_{suffix}"
            if suffix == "query":
                new_desc = downstream_description
                if has_specific_tools:
                    new_desc = _prefer_specific_tools(new_desc)
            else:
                new_desc = tool.description or ""
            title = _tool_title(new_name)

            modified_tool = tool.model_copy(
                update={
                    "name": new_name,
                    "title": title,
                    "description": new_desc,
                    "annotations": mt.ToolAnnotations(
                        title=title,
                        openWorldHint=True,
                    ),
                }
            )
            transformed.append(modified_tool)

        return [*transformed, *sorted(recipe_tools, key=lambda tool: tool.name)]

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next,
    ) -> ToolResult:
        """Validate and transform tool name back to internal name."""
        try:
            headers = get_http_headers()
        except LookupError:
            # No HTTP context - pass through unchanged
            return await call_next(context)

        api_name = extract_api_name(headers)
        tool_name = context.message.name

        # Handle recipe tools directly
        if tool_name.startswith(RECIPE_PREFIX_STR):
            recipe_slug = tool_name.removeprefix(RECIPE_PREFIX_STR)
            try:
                req_ctx = get_request_context()
            except MissingHeaderError as e:
                raise ToolError(str(e)) from e

            arguments = context.message.arguments or {}
            if not isinstance(arguments, dict):
                raise ValidationError("Invalid arguments: expected object.")

            params = {k: v for k, v in arguments.items() if k != "return_directly"} or None

            raw_schema, base_url = await load_schema_and_base_url(req_ctx)
            if not raw_schema:
                raise ToolError("schema not loaded")

            schema_hash = sha256_hex(raw_schema)
            api_id = build_api_id(req_ctx, req_ctx.api_type, base_url)
            recipe_meta = await ASYNC_API_AGENT_STORE.find_recipe_by_tool_slug(
                api_id=api_id,
                schema_hash=schema_hash,
                tool_slug=recipe_slug,
                max_slug_len=_max_slug_length(),
            )
            if not recipe_meta:
                raise NotFoundError(f"recipe not found: {recipe_slug}")

            recipe_id = recipe_meta["recipe_id"]
            result = await execute_recipe_tool(
                req_ctx,
                recipe_id,
                params,
                True,
                raw_schema=raw_schema,
                base_url=base_url,
            )

            # Parse and handle errors from recipe execution
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) and parsed.get("success") is False:
                err_msg = parsed.get("error", "recipe execution failed")
                if isinstance(err_msg, str) and err_msg.startswith(
                    ("missing required param:", "unexpected params:", "invalid param type:")
                ):
                    raise ValidationError(err_msg)
                raise ToolError(err_msg)

            return ToolResult(content=[TextContent(type="text", text=result)])

        # Validate tool name matches session's API for non-recipe tools
        expected_prefix = f"{api_name}_"
        if not tool_name.startswith(expected_prefix):
            raise NotFoundError(
                f"Tool '{tool_name}' not valid for API '{api_name}'. "
                f"Expected tool name starting with '{expected_prefix}' "
                "or recipe tool prefix 'r_'."
            )

        # Transform back to internal name (_suffix)
        suffix = tool_name.removeprefix(expected_prefix)
        if not suffix:
            raise NotFoundError(
                f"Tool '{tool_name}' not valid for API '{api_name}'. Missing tool suffix."
            )
        internal_name = f"_{suffix}"

        # Create modified context with internal tool name
        modified_params = mt.CallToolRequestParams(
            name=internal_name,
            arguments=context.message.arguments,
        )
        modified_context = context.copy(message=modified_params)

        return await call_next(modified_context)
