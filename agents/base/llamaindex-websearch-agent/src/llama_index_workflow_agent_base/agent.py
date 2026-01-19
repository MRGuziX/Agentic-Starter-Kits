from typing import Callable

from llama_index.llms.openai_like import OpenAILike
from llama_index_workflow_agent_base import TOOLS
from llama_index_workflow_agent_base.workflow import FunctionCallingAgent
from openai import OpenAI

from apps.utils import get_env_var


def get_workflow_closure(
        client: OpenAI,
        model_id: str,
        api_base: str = None
) -> Callable:
    """Workflow generator closure."""
    api_key = get_env_var("API_KEY")
    if not api_key:
        api_key = getattr(client, 'api_key', None) or "not-needed"

    api_base = get_env_var("BASE_URL")
    if not api_base:
        api_base = api_base or getattr(client, 'base_url', None)

    default_system_prompt = "You are a helpful AI assistant, please respond to the user's query to the best of your ability!"
    context_window = 4096

    chat = OpenAILike(
        model=model_id,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,  # Bypass model name validation for custom models
        is_chat_model=True,  # Use chat completions endpoint instead of completions
        is_function_calling_model=True,  # Enable function calling/tools support
    )

    def get_agent(system_prompt: str = default_system_prompt) -> FunctionCallingAgent:
        """Get compiled workflow with overwritten system prompt, if provided"""

        # Create instance of compiled workflow
        return FunctionCallingAgent(
            llm=chat,
            tools=TOOLS,
            system_prompt=system_prompt,
            timeout=120,
            verbose=False,
        )

    return get_agent
