"""API Agent MCP Server - Universal GraphQL/REST to MCP gateway."""

import logging
from typing import Literal, cast

import uvicorn
from fastmcp import FastMCP
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import settings
from .middleware import DynamicToolNamingMiddleware
from .tools import register_public_tools

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OriginGuard:
    """Reject browser requests from origins not allowed by CORS (incl. DNS rebinding)."""

    def __init__(self, app: ASGIApp, allowed: list[str]) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and "*" not in self.allowed:
            origin = Headers(scope=scope).get("origin")
            if origin and origin not in self.allowed:
                await PlainTextResponse("Origin not allowed", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_public_mcp() -> FastMCP:
    """Create public MCP server."""
    mcp = FastMCP(settings.MCP_NAME, strict_input_validation=True)
    register_public_tools(mcp)
    mcp.add_middleware(DynamicToolNamingMiddleware())
    return mcp


def create_app():
    """Create ASGI application."""
    public_mcp = create_public_mcp()

    @public_mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok"})

    cors_origins = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
    middleware = [
        Middleware(OriginGuard, allowed=cors_origins),  # type: ignore[arg-type]
        Middleware(
            CORSMiddleware,  # type: ignore[arg-type]  # Starlette middleware typing
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=[
                "Content-Type",
                "Authorization",
                "MCP-Session-Id",
                "mcp-protocol-version",
            ],
            allow_credentials=False,
            max_age=600,
        ),
    ]

    transport = cast(Literal["http", "streamable-http", "sse"], settings.TRANSPORT)
    return public_mcp.http_app(
        path="/mcp",
        middleware=middleware,
        stateless_http=settings.STATELESS_HTTP,
        transport=transport,
    )


def main():
    """Run server."""
    logger.info(f"Starting API Agent on {settings.HOST}:{settings.PORT}")
    logger.info("Endpoint config via headers: X-Target-URL, X-API-Type, X-Target-Headers")
    uvicorn.run(create_app(), host=settings.HOST, port=settings.PORT, log_level="info")


if __name__ == "__main__":
    main()
