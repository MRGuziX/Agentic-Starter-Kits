import os
from typing import Optional

from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from utils import get_env_var

# Cache the retriever to avoid re-initializing Milvus on every tool call
_retriever_cache = None


def create_retriever_tool(
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
):
    """
    Create a retriever tool that searches Milvus Lite vector store for relevant documents.

    Args:
        vector_store_path: Path to the Milvus Lite database file
        embedding_model: Name of the embedding model to use
        base_url: Base URL for the embeddings API
        api_key: API key for the embeddings API
        use_milvus: Whether to use Milvus Lite (always True, kept for compatibility)

    Returns:
        A tool function that can retrieve relevant documents
    """
    global _retriever_cache

    # Return cached retriever if it exists
    if _retriever_cache is not None:
        return _retriever_cache

    # Get configuration from environment if not provided
    if not api_key:
        api_key = get_env_var("API_KEY")
    if not vector_store_path:
        vector_store_path = get_env_var("VECTOR_STORE_PATH")
    if not embedding_model:
        embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key or "not-needed",
        base_url=base_url,
    )

    # Connect to Milvus Lite
    # Default to data folder if not provided
    if not vector_store_path:
        # Get path relative to this file's location
        # tools.py is in: agents/community/langgraph_agentic_rag/src/langgraph_agentic_rag/
        # data folder is in: agents/community/langgraph_agentic_rag/data/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to langgraph_agentic_rag root, then to data folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        vector_store_path = os.path.join(project_root, "data", "milvus_data", "milvus_lite.db")
    elif not os.path.isabs(vector_store_path):
        # If relative path provided, ensure it's relative to data folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # If path doesn't start with data/, assume it should be in data folder
        if not vector_store_path.startswith("data/"):
            vector_store_path = os.path.join(project_root, "data", vector_store_path.lstrip("./"))
        else:
            vector_store_path = os.path.join(project_root, vector_store_path)
    
    connection_args = {"uri": vector_store_path}

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name="rag_knowledge_base",
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )

    # Cache the retriever
    _retriever_cache = retriever

    return retriever


class RetrieverInput(BaseModel):
    """Schema for the retriever tool input."""
    query: str = Field(description="The search query describing what information you need to retrieve.")


@tool("retriever", args_schema=RetrieverInput)
def retriever_tool(query: str) -> str:
    """
    Search the knowledge base for information relevant to the query.

    Use this tool when you need to find specific information from the knowledge base
    to answer the user's question accurately.

    Args:
        query: The search query describing what information you need to retrieve.

    Returns:
        Retrieved documents containing relevant information.
    """
    # Handle case where query might be passed as a dict (defensive fix)
    if isinstance(query, dict):
        # Extract the actual query value from the dict
        query = query.get('value', query.get('query', str(query)))

    retriever = create_retriever_tool()
    docs = retriever.invoke(query)

    # Format the retrieved documents
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        formatted_docs.append(
            f"Document {i}:\n{doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'unknown')}"
        )

    return "\n\n".join(formatted_docs)