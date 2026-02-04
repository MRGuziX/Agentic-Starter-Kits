import os
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_retriever_tool(
    vector_store_path: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Create a retriever tool that searches a vector store for relevant documents.

    Args:
        vector_store_path: Path to the FAISS vector store. If None, creates a sample store.
        embedding_model: Name of the embedding model to use
        base_url: Base URL for the embeddings API
        api_key: API key for the embeddings API

    Returns:
        A tool function that can retrieve relevant documents
    """

    # Initialize embeddings
    is_local = base_url and any(host in base_url for host in ["localhost", "127.0.0.1"])

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key or "not-needed",
        base_url=base_url,
    )

    # Load or create vector store
    if vector_store_path and os.path.exists(vector_store_path):
        # Load existing vector store
        vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Create a sample vector store with some default documents
        sample_documents = [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models. "
                           "It enables applications that are context-aware and can reason about their actions.",
                metadata={"source": "langchain_docs", "topic": "framework"}
            ),
            Document(
                page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                           "It extends LangChain with the ability to coordinate multiple chains across multiple steps.",
                metadata={"source": "langgraph_docs", "topic": "library"}
            ),
            Document(
                page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. "
                           "It retrieves relevant documents from a knowledge base and uses them to generate informed responses.",
                metadata={"source": "rag_docs", "topic": "technique"}
            ),
            Document(
                page_content="Vector stores are databases optimized for storing and searching high-dimensional vectors. "
                           "They enable efficient similarity search, which is crucial for RAG applications.",
                metadata={"source": "vectordb_docs", "topic": "storage"}
            ),
            Document(
                page_content="FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. "
                           "It's commonly used for vector storage in RAG applications.",
                metadata={"source": "faiss_docs", "topic": "library"}
            ),
        ]

        vectorstore = FAISS.from_documents(sample_documents, embeddings)

        # Optionally save the vector store for future use
        if vector_store_path:
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            vectorstore.save_local(vector_store_path)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )

    @tool("retriever", parse_docstring=True)
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
        docs = retriever.invoke(query)

        # Format the retrieved documents
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(
                f"Document {i}:\n{doc.page_content}\n"
                f"Source: {doc.metadata.get('source', 'unknown')}"
            )

        return "\n\n".join(formatted_docs)

    return retriever_tool