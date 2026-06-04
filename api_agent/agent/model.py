"""Shared OpenAI model configuration for API agents."""

from typing import Literal, cast

from agents import ModelSettings, RunConfig, set_default_openai_api
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.run import CallModelData, ModelInputData, ToolExecutionConfig
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from ..config import settings
from .progress import get_turn_context, increment_turn

OPENAI_API_MODE = settings.MODEL_API
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

# Set default API mode
set_default_openai_api(OPENAI_API_MODE)

# Shared OpenAI client
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)


def create_openai_model(api: str, model_name: str, openai_client: AsyncOpenAI):
    """Create the Agents SDK model adapter for the configured OpenAI API."""
    if api == "responses":
        return OpenAIResponsesModel(model=model_name, openai_client=openai_client)
    if api == "chat_completions":
        return OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)
    raise ValueError(f"Unsupported model API: {api}")


# Shared model instance
model = create_openai_model(OPENAI_API_MODE, settings.MODEL_NAME, client)


async def _inject_turn(call_data: CallModelData) -> ModelInputData:
    """Inject turn count into instructions before each LLM call."""
    increment_turn()
    turn_info = get_turn_context(settings.MAX_AGENT_TURNS)
    existing_instructions = call_data.model_data.instructions or ""
    call_data.model_data.instructions = f"{existing_instructions}\n\n{turn_info}"
    return call_data.model_data


def get_run_config() -> RunConfig:
    """Get RunConfig with optional reasoning settings and turn injection."""
    model_settings = None
    if settings.REASONING_EFFORT:
        effort = cast(ReasoningEffort, settings.REASONING_EFFORT)
        model_settings = ModelSettings(reasoning=Reasoning(effort=effort))

    return RunConfig(
        model_settings=model_settings,
        call_model_input_filter=_inject_turn,
        tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
    )
