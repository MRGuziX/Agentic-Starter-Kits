from typing import Callable

from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from llama_index_workflow_agent_base import TOOLS
from llama_index_workflow_agent_base.workflow import FunctionCallingAgent


def get_workflow_closure(client: OpenAI, model_id: str, base_url: str = None) -> Callable:
    """Workflow generator closure."""

    # Initialise OpenAI-compatible LLM for RHOAI/LlamaStack
    # Extract API key and base URL from OpenAI client or use provided values
    api_key = getattr(client, 'api_key', None) or "not-needed"
    api_base = base_url or getattr(client, 'base_url', None)
    
    chat = LlamaIndexOpenAI(
        model=model_id,
        api_key=api_key,
        api_base=api_base,
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
