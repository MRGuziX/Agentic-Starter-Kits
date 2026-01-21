from agents.base.llamaindex_websearch_agent.src.llama_index_workflow_agent_base.tools import (
    dummy_web_search
)


class TestTools:
    def test_dummy_web_search(self):
        query = "RedHat"
        result = dummy_web_search(query)
        assert "RedHat" in result[0]
