"""
Script to load documents from text files into Milvus Lite vector store.

This script reads text files from the data directory, splits them into chunks,
creates embeddings, and stores them in a Milvus Lite vector database.
"""

import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_env_var


def load_and_index_documents(
    docs_to_load: str = None,
    vector_store_path: str = None,
    embedding_model: str = None,
    base_url: str = None,
    api_key: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Load documents from directory and index them in Milvus Lite.

    Args:
        docs_to_load: Directory containing text files to load
        vector_store_path: Path where Milvus data will be stored
        embedding_model: Name of the embedding model
        base_url: Base URL for embeddings API
        api_key: API key for embeddings
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    # Default to data folder for Milvus database
    if not vector_store_path:
        vector_store_path = get_env_var("VECTOR_STORE_PATH")
        if not vector_store_path:
            # Default to data folder
            data_folder = os.path.dirname(__file__)
            vector_store_path = os.path.join(data_folder, "milvus_data", "milvus_lite.db")
    else:
        # If path is provided but not absolute, ensure it's relative to data folder
        if not os.path.isabs(vector_store_path):
            data_folder = os.path.dirname(__file__)
            vector_store_path = os.path.join(data_folder, vector_store_path.lstrip("./"))

    if not embedding_model:
        embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"

    if not base_url:
        base_url = get_env_var("BASE_URL")

    if not api_key:
        api_key = get_env_var("API_KEY")
    if not docs_to_load:
        docs_to_load = get_env_var("DOCS_TO_LOAD")

    print(f"\n{'='*80}")
    print("Loading Documents into Milvus Lite Vector Store")
    print(f"{'='*80}")
    print(f"Data Directory: {docs_to_load}")
    print(f"Vector Store Path: {vector_store_path}")
    print(f"Embedding Model: {embedding_model}")
    print(f"{'='*80}\n")

    # Load documents
    print("Loading documents from directory...")
    loader = TextLoader(docs_to_load)
    documents = loader.load()

    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = text_splitter.split_documents(documents)
    chunks = [
        doc for doc in all_chunks
        if doc.page_content and doc.page_content.strip()
    ]
    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    print("\nInitializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key or "not-needed",
        base_url=base_url,
    )

    # Create vector store using Milvus Lite
    print("\nCreating Milvus Lite vector store...")

    # Ensure the parent directory exists (always in data folder)
    if vector_store_path.endswith('.db'):
        # It's a file path, create parent directory
        parent_dir = os.path.dirname(vector_store_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Created directory: {parent_dir}")
    else:
        # It's a directory path, create it
        if not os.path.exists(vector_store_path):
            os.makedirs(vector_store_path, exist_ok=True)
            print(f"Created directory: {vector_store_path}")

    # Milvus Lite connection URI
    connection_args = {
        "uri": vector_store_path  # Milvus Lite uses file path as URI
    }

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name="rag_knowledge_base",
        drop_old=True,  # Ensures a fresh start
        auto_id=True  # Milvus Lite will handle primary key generation
    )

    print("Vector store initialized. Connection established.")

    # 2. Add the chunked documents
    ids = vector_store.add_documents(chunks)

    print(f"Successfully added {len(ids)} chunks to the store.")

    print(f"\n{'='*80}")
    print("✅ Successfully loaded and indexed documents!")
    print(f"{'='*80}")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Vector store location: {vector_store_path}")
    print(f"Collection name: rag_knowledge_base")
    print(f"{'='*80}\n")

    # Test retrieval
    print("Testing retrieval...")
    test_query = "What is LangGraph?"
    results = vector_store.similarity_search(test_query, k=3)

    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"Document {i}:")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    load_and_index_documents()