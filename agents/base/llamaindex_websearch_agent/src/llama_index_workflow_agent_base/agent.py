from typing import Callable

from llama_index.llms.openai_like import OpenAILike
from agents.base.llamaindex_websearch_agent.src.llama_index_workflow_agent_base import TOOLS
from agents.base.llamaindex_websearch_agent.src.llama_index_workflow_agent_base.workflow import FunctionCallingAgent

from apps.utils import get_env_var


def get_workflow_closure(
        model_id: str = None,
        base_url: str = None
) -> Callable:
    """Workflow generator closure."""

    api_key = get_env_var("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is required. Please set it in environment variables or .env file")

    if not base_url:
        base_url = get_env_var("BASE_URL")
        if not base_url:
            raise ValueError("BASE_URL is required. Please set it in environment variables or .env file")

    if not model_id:
        model_id = get_env_var("MODEL_ID")
        if not model_id:
            raise ValueError("MODEL_ID is required. Please set it in environment variables or .env file")

    default_system_prompt = "You are a helpful AI assistant, please respond to the user's query to the best of your ability!"
    context_window = 4096

    client = OpenAILike(
        model=model_id,
        api_key=api_key,
        api_base=base_url,
        context_window=context_window,  # Bypass model name validation for custom models
        is_chat_model=True,  # Use chat completions endpoint instead of completions
        is_function_calling_model=True,  # Enable function calling/tools support
    )

    def get_agent(system_prompt: str = default_system_prompt) -> FunctionCallingAgent:
        """Get compiled workflow with overwritten system prompt, if provided"""

        # Create instance of compiled workflow
        return FunctionCallingAgent(
            llm=client,
            tools=TOOLS,
            system_prompt=system_prompt,
            timeout=120,
            verbose=False,
        )

    return get_agent
