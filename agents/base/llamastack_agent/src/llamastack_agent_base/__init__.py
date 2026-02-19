"""
LlamaStack Agent Base – agent built only on LlamaStack API (no LlamaIndex).

Use llama-stack-client to talk to Llama Stack (reasoning + tools). This package
provides the agent factory and placeholder tools until full LlamaStack integration.
"""

from agents.base.llamastack_agent.src.llamastack_agent_base.agent import AIAgent, get_agent_closure
from agents.base.llamastack_agent.src.llamastack_agent_base.tools import search_reviews, search_price
from utils import get_env_var

__all__ = [
    "get_agent_closure",
    "AIAgent",
    "search_reviews",
    "search_price",
    "get_env_var",
]
