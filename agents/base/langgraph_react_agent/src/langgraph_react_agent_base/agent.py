from typing import Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from utils import get_env_var

# Import your actual tools here
from agents.base.langgraph_react_agent.src.langgraph_react_agent_base.tools import dummy_web_search, dummy_math

def get_graph_closure(
        model_id: str,
        base_url: str = None,
) -> Any:
    """Graph generator closure."""

    api_key = get_env_var("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is required.")

    if not base_url:
        base_url = get_env_var("BASE_URL")
        if not base_url:
            raise ValueError("BASE_URL is required.")

    if not model_id:
        model_id = get_env_var("MODEL_ID")
        if not model_id:
            raise ValueError("MODEL_ID is required.")

    # 1. Define the Tools List
    tools = [dummy_web_search, dummy_math]

    # 2. Initialize the Model
    # We disable parallel tool calls sometimes to force sequential thinking,
    # but it's optional.
    chat = ChatOpenAI(
        model=model_id,
        temperature=0.01,
        api_key=api_key,
        base_url=base_url,
    )

    system_prompt = "You are a helpful assistant. When using tools, provide only the required string or number for the arguments, never the schema description."

    # 3. Create the LangGraph ReAct Agent
    # This automatically binds tools and sets up the graph loop
    graph = create_agent(
        model=chat,
        tools=tools,
        system_prompt=system_prompt
    )

    return graph