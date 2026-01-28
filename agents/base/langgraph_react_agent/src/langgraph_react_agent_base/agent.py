from typing import Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from utils import get_env_var

from ..langgraph_react_agent_base.tools import dummy_web_search, dummy_math


def get_graph_closure(
        model_id: str = None,
        base_url: str = None,
        api_key: str = None,
) -> Any:
    """

    :param model_id:
    :param base_url:
    :param api_key:
    :return:
    """

    if not api_key:
        api_key = get_env_var("API_KEY")
    if not base_url:
        base_url = get_env_var("BASE_URL")
    if not model_id:
        model_id = get_env_var("MODEL_ID")

    is_local = any(host in base_url for host in ["localhost", "127.0.0.1"])

    if not is_local and not api_key:
        raise ValueError("API_KEY is required for non-local environments.")

    tools = [dummy_web_search, dummy_math]

    chat = ChatOpenAI(
        model=model_id,
        temperature=0.01,
        api_key=api_key,
        base_url=base_url,
    )

    system_prompt = "You are a helpful assistant. When using tools, provide only the required string or number for the arguments, never the schema description."

    agent = create_agent(model=chat, tools=tools, system_prompt=system_prompt)

    return agent
