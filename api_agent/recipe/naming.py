"""Recipe tool naming helpers."""

import re


def sanitize_tool_name(name: str | None) -> str:
    """Normalize tool name to a safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    for prefix in ("r_", "api_", "rest_", "graphql_"):
        if slug.startswith(prefix):
            slug = slug.removeprefix(prefix)
            break
    if slug and not slug[0].isalpha():
        slug = f"recipe_{slug}"
    return slug or "recipe"
