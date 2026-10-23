"""Outbound HTTP with response size and total time limits."""

import asyncio
from typing import Any

import httpx

MAX_RESPONSE_BYTES = 10_000_000
TOTAL_TIMEOUT_SECONDS = 60


async def bounded_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Send one request and return a fully read response, or raise if over limits."""
    body = bytearray()
    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(method, url, **kwargs) as resp:
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError(f"Response exceeds {MAX_RESPONSE_BYTES} bytes")
    headers = [
        (k, v)
        for k, v in resp.headers.multi_items()
        if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
    ]
    return httpx.Response(
        resp.status_code, headers=headers, content=bytes(body), request=resp.request
    )
