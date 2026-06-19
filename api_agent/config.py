"""Configuration settings for API Agent MCP server."""

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import computed_field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_TOML_MAP: dict[str, dict[str, str]] = {
    "mcp": {
        "name": "MCP_NAME",
    },
    "server": {
        "host": "HOST",
        "port": "PORT",
        "transport": "TRANSPORT",
        "stateless_http": "STATELESS_HTTP",
        "cors_allowed_origins": "CORS_ALLOWED_ORIGINS",
        "debug": "DEBUG",
    },
    "model": {
        "api": "MODEL_API",
        "name": "MODEL_NAME",
        "openai_base_url": "OPENAI_BASE_URL",
        "reasoning_effort": "REASONING_EFFORT",
    },
    "agent": {
        "max_turns": "MAX_AGENT_TURNS",
        "max_response_chars": "MAX_RESPONSE_CHARS",
        "max_schema_chars": "MAX_SCHEMA_CHARS",
        "max_preview_rows": "MAX_PREVIEW_ROWS",
        "max_tool_response_chars": "MAX_TOOL_RESPONSE_CHARS",
    },
    "polling": {
        "max_polls": "MAX_POLLS",
        "default_delay_ms": "DEFAULT_POLL_DELAY_MS",
        "max_delay_ms": "MAX_POLL_DELAY_MS",
    },
    "recipes": {
        "enabled": "ENABLE_RECIPES",
        "store": "STORAGE_BACKEND",
        "max_size": "RECIPE_CACHE_SIZE",
        "learn_rate": "RECIPE_LEARN_RATE",
        "namespace": "STORAGE_NAMESPACE",
    },
    "storage": {
        "backend": "STORAGE_BACKEND",
        "namespace": "STORAGE_NAMESPACE",
    },
    "redis": {
        "url": "REDIS_URL",
    },
    "description": {
        "model_name": "DESCRIPTION_MODEL_NAME",
        "timeout_seconds": "DESCRIPTION_TIMEOUT_SECONDS",
    },
}

_ENV_OVERRIDES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "PORT")


def _load_toml_settings() -> dict[str, Any]:
    path = Path(os.environ.get("API_AGENT_CONFIG", "api-agent.toml"))
    values: dict[str, Any] = {}

    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

        for section, mapping in _TOML_MAP.items():
            section_values = raw.get(section, {})
            if not isinstance(section_values, dict):
                continue
            for toml_name, field_name in mapping.items():
                if toml_name in section_values:
                    values[field_name] = section_values[toml_name]

    for field_name in _ENV_OVERRIDES:
        if field_name in os.environ:
            values[field_name] = os.environ[field_name]
    return values


class ApiAgentTomlSettingsSource(PydanticBaseSettingsSource):
    """Read api-agent.toml into existing flat settings fields."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return _load_toml_settings()


class Settings(BaseSettings):
    """Settings loaded from api-agent.toml.

    API_AGENT_CONFIG may select the TOML path. OpenAI secrets may come from env.
    """

    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            ApiAgentTomlSettingsSource(settings_cls),
        )

    # MCP Server
    MCP_NAME: str = "API Agent"

    @computed_field
    @property
    def MCP_SLUG(self) -> str:
        """Slugified MCP_NAME for identifiers."""
        return re.sub(r"[^a-z0-9]+", "_", self.MCP_NAME.lower()).strip("_")

    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    MODEL_NAME: str = "gpt-5.5"
    MODEL_API: Literal["responses", "chat_completions"] = "responses"
    REASONING_EFFORT: str = "low"
    DESCRIPTION_MODEL_NAME: str = "gpt-5.4-mini"
    DESCRIPTION_TIMEOUT_SECONDS: float = 15.0

    # Agent limits
    MAX_AGENT_TURNS: int = 30
    MAX_RESPONSE_CHARS: int = 50000
    MAX_SCHEMA_CHARS: int = 32000
    MAX_PREVIEW_ROWS: int = 10  # Rows to show before suggesting pagination
    MAX_TOOL_RESPONSE_CHARS: int = 32000  # ~8K tokens, cap tool responses for LLM context

    # Polling limits
    MAX_POLLS: int = 20  # Max poll attempts
    DEFAULT_POLL_DELAY_MS: int = 3000  # Default delay if agent doesn't specify
    MAX_POLL_DELAY_MS: int = 3000  # Cap delay between poll attempts

    # Server
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 3000
    TRANSPORT: str = "streamable-http"
    STATELESS_HTTP: bool = True
    CORS_ALLOWED_ORIGINS: str = "*"

    # Recipes
    ENABLE_RECIPES: bool = True
    RECIPE_CACHE_SIZE: int = 1000
    RECIPE_LEARN_RATE: float = 0.2

    # Storage
    STORAGE_BACKEND: str = "memory"
    STORAGE_NAMESPACE: str = "api-agent"

    # Redis-backed storage
    REDIS_URL: str = ""


settings = Settings()
