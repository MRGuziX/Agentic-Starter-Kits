"""
Script to load documents from text files into Milvus Lite vector store.

This script reads text files from the data directory, splits them into chunks,
creates embeddings, and stores them in a Milvus Lite vector database.
"""

import os
import uuid

from langchain_community.document_loaders import TextLoader

from llama_stack_client import LlamaStackClient

from langchain_openai import OpenAIEmbeddings


from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_env_var

def load_and_index_documents(
    docs_to_load: str = None,
    vector_store_path: str = None,
    embedding_model: str = None,
    base_url: str = None,
    api_key: str = None,
    chunk_size: int = 64,
    chunk_overlap: int = 32,
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
    if not embedding_model:
        embedding_model = get_env_var("EMBEDDING_MODEL")

    if not base_url:
        base_url = get_env_var("BASE_URL")

    if not api_key:
        api_key = get_env_var("API_KEY")
    if not docs_to_load:
        docs_to_load = get_env_var("DOCS_TO_LOAD")
    # http://localhost:8321 -> ulr do llama server
    # http://localhost:11434 -> url do ollamy


    client = LlamaStackClient(
        base_url=base_url,
    )

    vector_store_list = client.vector_stores.list()

    if len(vector_store_list.data) == 1:
        vector_store = client.vector_stores.retrieve(vector_store_id=vector_store_list.data[0].id)
        print("Vector store registered successfully.")
    else:
        provider_id = "milvus"
        embedding_dimension = 768

        vector_store = client.vector_stores.create(
            extra_body={
                "provider_id": provider_id,
                # "provider_vector_store_id": collection_name,  # --> not working in 0.4.x
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            }
        )
        print("Vector store registered successfully.")

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
        doc.page_content for doc in all_chunks
        if doc.page_content and doc.page_content.strip()
    ]
    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    print("\nInitializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key or "not-needed",
        base_url=base_url+"/v1",
        check_embedding_ctx_length=False #that will prevent fail if embedding model is not registered in OpenAI Registry
    )

    # Create embeddings for all chunks
    print("Creating embeddings...")
    embedding_vectors = embeddings.embed_documents(texts=chunks)

    # Create properly formatted chunks for LlamaStack
    formatted_chunks = []
    for i, (text, embedding_vec) in enumerate(zip(chunks, embedding_vectors)):
        chunk = {
            "chunk_id": str(uuid.uuid4()),  # Unique ID
            "content": text,  # The actual text content
            "embedding": embedding_vec,  # The embedding vector
            "embedding_dimension": 768,  # Dimension for embeddinggemma
            "embedding_model": "ollama/embeddinggemma:latest",
            "chunk_metadata": {
                "document_id": "doc_1",
                "source": "sample_knowledge.txt",
            },
            "metadata": {
                "chunk_index": i,
            }
        }
        formatted_chunks.append(chunk)


    client.vector_io.insert(
        chunks=formatted_chunks,
        vector_store_id=vector_store.id,
    )

    print("\nEmbeddings successfully loaded.")

if __name__ == "__main__":
    load_and_index_documents()