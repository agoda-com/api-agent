from api_agent import config


def _write_config(tmp_path, text: str):
    path = tmp_path / "api-agent.toml"
    path.write_text(text.strip())
    return path


def test_settings_load_app_config_from_toml(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        """
[mcp]
name = "Custom API Agent"

[server]
host = "127.0.0.1"
port = 3001
transport = "http"
stateless_http = false
cors_allowed_origins = "https://example.com"
debug = true

[model]
api = "chat_completions"
name = "gpt-4.1"
openai_base_url = "https://toml.example/v1"
reasoning_effort = ""

[description]
model_name = "gpt-5.4-mini"
timeout_seconds = 1.5

[agent]
max_turns = 7
max_response_chars = 12345
max_schema_chars = 23456
max_preview_rows = 3
max_tool_response_chars = 34567

[polling]
max_polls = 4
default_delay_ms = 500
max_delay_ms = 1000

[recipes]
enabled = false
max_size = 12
learn_rate = 0.5

[storage]
backend = "redis"
namespace = "test-agent"

[redis]
url = "redis://localhost:6379/2"
""",
    )
    monkeypatch.setenv("API_AGENT_CONFIG", str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    settings = config.Settings()

    assert settings.MCP_NAME == "Custom API Agent"
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 3001
    assert settings.TRANSPORT == "http"
    assert settings.STATELESS_HTTP is False
    assert settings.CORS_ALLOWED_ORIGINS == "https://example.com"
    assert settings.DEBUG is True
    assert settings.MODEL_API == "chat_completions"
    assert settings.MODEL_NAME == "gpt-4.1"
    assert settings.OPENAI_BASE_URL == "https://toml.example/v1"
    assert settings.REASONING_EFFORT == ""
    assert settings.DESCRIPTION_MODEL_NAME == "gpt-5.4-mini"
    assert settings.DESCRIPTION_TIMEOUT_SECONDS == 1.5
    assert settings.MAX_AGENT_TURNS == 7
    assert settings.MAX_RESPONSE_CHARS == 12345
    assert settings.MAX_SCHEMA_CHARS == 23456
    assert settings.MAX_PREVIEW_ROWS == 3
    assert settings.MAX_TOOL_RESPONSE_CHARS == 34567
    assert settings.MAX_POLLS == 4
    assert settings.DEFAULT_POLL_DELAY_MS == 500
    assert settings.MAX_POLL_DELAY_MS == 1000
    assert settings.ENABLE_RECIPES is False
    assert settings.STORAGE_BACKEND == "redis"
    assert settings.RECIPE_CACHE_SIZE == 12
    assert settings.RECIPE_LEARN_RATE == 0.5
    assert settings.STORAGE_NAMESPACE == "test-agent"
    assert settings.REDIS_URL == "redis://localhost:6379/2"


def test_env_does_not_override_app_settings(monkeypatch):
    monkeypatch.setenv("API_AGENT_CONFIG", "/tmp/no-such-api-agent.toml")
    monkeypatch.setenv("API_AGENT_MODEL_NAME", "env-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = config.Settings()

    assert settings.MODEL_NAME == "gpt-5.5"
    assert settings.DESCRIPTION_TIMEOUT_SECONDS == 15.0


def test_openai_api_key_loads_from_env(monkeypatch):
    monkeypatch.setenv("API_AGENT_CONFIG", "/tmp/no-such-api-agent.toml")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = config.Settings()

    assert settings.OPENAI_API_KEY == "env-key"


def test_openai_base_url_env_overrides_toml(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        """
[model]
openai_base_url = "https://toml.example/v1"
""",
    )
    monkeypatch.setenv("API_AGENT_CONFIG", str(path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")

    settings = config.Settings()

    assert settings.OPENAI_BASE_URL == "https://env.example/v1"


def test_port_env_overrides_toml(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        """
[server]
port = 3001
""",
    )
    monkeypatch.setenv("API_AGENT_CONFIG", str(path))
    monkeypatch.setenv("PORT", "4321")

    settings = config.Settings()

    assert settings.PORT == 4321
