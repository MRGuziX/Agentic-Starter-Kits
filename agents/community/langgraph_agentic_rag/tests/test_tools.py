import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_agentic_rag.tools import create_retriever_tool


class TestTools:
    def test_retriever_tool_creation(self):
        """Test that the retriever tool can be created."""
        retriever_tool = create_retriever_tool()
        assert retriever_tool is not None
        assert retriever_tool.name == "retriever"

    def test_retriever_tool_invoke(self):
        """Test that the retriever tool can be invoked with a query."""
        retriever_tool = create_retriever_tool()
        query = "What is LangGraph?"
        result = retriever_tool.invoke({"query": query})

        # Check if the result contains relevant information
        assert isinstance(result, str)
        assert len(result) > 0
        # Should return documents about LangGraph from sample data
        assert "LangGraph" in result or "library" in result