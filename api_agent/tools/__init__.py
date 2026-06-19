"""MCP tools."""

import logging

from fastmcp import FastMCP

from .query import register_query_tool

logger = logging.getLogger(__name__)


def register_public_tools(mcp: FastMCP) -> None:
    """Register public tools with generic internal names.

    Internal names are transformed by middleware to session-specific names.
    """
    register_query_tool(mcp)

    logger.info("Registered public tools: _query")


register_all_tools = register_public_tools
