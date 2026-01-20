from typing import Callable, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.base.langgraph_react_agent.src.langgraph_react_agent_base import TOOLS
from utils import get_env_var


def get_graph_closure(
        model_id: str,
        base_url: str = None
) -> Callable:
    """Graph generator closure."""

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

    def get_graph(system_prompt=default_system_prompt) -> Any:
        """Get compiled graph with overwritten system prompt, if provided"""

        chat = ChatOpenAI(
            model=model_id,
            temperature=0.01,
            api_key=api_key,
            base_url=base_url,
        )

        chat = chat.bind(system=system_prompt)

        return create_react_agent(chat, tools=TOOLS)

    return get_graph
