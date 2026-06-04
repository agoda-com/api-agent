from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from api_agent.agent import model as agent_model


def test_shared_model_uses_responses_api():
    assert isinstance(agent_model.model, OpenAIResponsesModel)


def test_create_openai_model_supports_both_apis():
    client = AsyncOpenAI(api_key="test")

    assert isinstance(
        agent_model.create_openai_model("responses", "gpt-5.5", client),
        OpenAIResponsesModel,
    )
    assert isinstance(
        agent_model.create_openai_model("chat_completions", "gpt-4.1", client),
        OpenAIChatCompletionsModel,
    )


def test_run_config_serializes_tool_execution():
    run_config = agent_model.get_run_config()

    assert run_config.tool_execution is not None
    assert run_config.tool_execution.max_function_tool_concurrency == 1
