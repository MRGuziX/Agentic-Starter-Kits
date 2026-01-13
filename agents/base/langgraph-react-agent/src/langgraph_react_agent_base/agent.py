from typing import Callable, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_react_agent_base import TOOLS
from openai import OpenAI


def get_graph_closure(client: OpenAI, model_id: str, base_url: str = None) -> Callable:
    """Graph generator closure."""

    # Initialise ChatOpenAI with OpenAI-compatible API (RHOAI/LlamaStack)
    # Extract API key and base URL from OpenAI client or use provided values

    # api_key = getattr(client, 'api_key', None) or "not-needed"
    api_base = base_url or getattr(client, 'base_url', None)
    
    chat = ChatOpenAI(
        model=model_id,
        temperature=0.01,
        # api_key=api_key, #not needed for local implementation
        base_url=api_base,
    )

    # Define system prompt
    default_system_prompt = "You are a helpful AI assistant, please respond to the user's query to the best of your ability!"

    def get_graph(system_prompt=default_system_prompt) -> Any:
        """Get compiled graph with overwritten system prompt, if provided"""
        
        # Create instance of compiled graph
        return create_react_agent(
            chat, tools=TOOLS, state_modifier=system_prompt
        )

    return get_graph
