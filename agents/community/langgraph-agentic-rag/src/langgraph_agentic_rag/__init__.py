"""LangGraph Agentic RAG Agent.

A retrieval-augmented generation agent using LangGraph and OpenAI-compatible APIs.
"""

from langgraph_agentic_rag.agent import get_graph_closure
from langgraph_agentic_rag.tools import create_retriever_tool
from langgraph_agentic_rag.utils import get_env_var

__version__ = "0.2.0"

__all__ = [
    "get_graph_closure",
    "create_retriever_tool",
    "get_env_var",
]
