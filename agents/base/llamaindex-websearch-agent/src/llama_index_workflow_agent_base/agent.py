from typing import Callable

from llama_index.llms.openai_like import OpenAILike
from llama_index_workflow_agent_base import TOOLS
from llama_index_workflow_agent_base.workflow import FunctionCallingAgent
from openai import OpenAI


def get_workflow_closure(client: OpenAI, model_id: str, base_url: str = None) -> Callable:
    """Workflow generator closure."""

    # Initialise OpenAI-compatible LLM for RHOAI/LlamaStack
    # Extract API key and base URL from OpenAI client or use provided values
    api_key = getattr(client, 'api_key', None) or "not-needed"
    api_base = base_url or getattr(client, 'base_url', None)
    
    # For local/custom models, provide context_window to bypass model name validation
    # Default to 4096 for most local models, but can be overridden if needed
    context_window = 4096  # Common context window for local models
    
    chat = OpenAILike(
        model=model_id,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,  # Bypass model name validation for custom models
    )

    # Define system prompt
    default_system_prompt = "You are a helpful AI assistant, please respond to the user's query to the best of your ability!"

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
