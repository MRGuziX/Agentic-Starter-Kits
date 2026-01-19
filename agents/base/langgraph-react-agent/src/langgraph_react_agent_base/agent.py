from typing import Callable, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_react_agent_base import TOOLS
from openai import OpenAI

from apps.utils import get_env_var


def get_graph_closure(
        client: OpenAI,
        model_id: str,
        base_url: str = None
) -> Callable:
    """Graph generator closure."""

    api_key = get_env_var("API_KEY")
    if not api_key:
        api_key = getattr(client, 'api_key', None) or "not-needed"

    base_url = get_env_var("BASE_URL")
    if not base_url:
        base_url = base_url or getattr(client, 'base_url', None)

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
