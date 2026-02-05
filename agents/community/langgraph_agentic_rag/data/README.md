# Data Directory

This directory contains sample documents and utilities for loading data into the vector store.

## Files

- `sample_knowledge.txt` - Sample knowledge base about LangChain, LangGraph, RAG, and vector databases
- `load_documents.py` - Script to load documents into Milvus Lite vector store

## Loading Documents into Vector Store

### Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r ../requirements.txt
```

### Configuration

Set up your environment variables in the `.env` file at the project root:

```bash
BASE_URL=http://localhost:8001/v1
MODEL_ID=llama3.2:3b
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE_PATH=./milvus_data/milvus_lite.db
```

### Load Documents

Run the load script:

```bash
cd data
python load_documents.py
```

This will:
1. Load all `.txt` files from the data directory
2. Split them into chunks (default: 1000 chars with 200 char overlap)
3. Create embeddings using the configured embedding model
4. Store them in Milvus Lite vector database
5. Test retrieval with a sample query

### Adding Your Own Documents

1. Add your `.txt` files to this directory
2. Run `python load_documents.py` again
3. The script will re-index all documents

### Document Format

Text files should be plain text (`.txt` format). The loader will:
- Automatically split long documents into chunks
- Preserve metadata (filename, etc.)
- Handle multiple files in the directory

### Example Document Structure

```
# Title

Section content here...

## Subsection

More content...
```

The splitter will intelligently split on:
- Double newlines (paragraphs)
- Single newlines
- Sentences (periods)
- Words (spaces)

## Using the Vector Store

Once documents are loaded, the RAG agent will automatically use the vector store:

```bash
# Start the agent
cd ..
python main.py
```

Or use the interactive chat:

```bash
cd examples
python execute_ai_service_locally.py
```

## Customizing Chunking

You can modify chunk size and overlap by editing `load_documents.py`:

```python
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks
```

Smaller chunks:
- More precise retrieval
- Better for specific questions
- More chunks to process

Larger chunks:
- More context per chunk
- Better for broad questions
- Fewer chunks overall

## Milvus Lite vs FAISS

By default, the agent uses Milvus Lite. To switch to FAISS:

1. Set `USE_MILVUS=false` in `.env`
2. Update `VECTOR_STORE_PATH` to point to a directory (not a .db file)

Milvus Lite advantages:
- Better performance for larger datasets
- Advanced filtering capabilities
- More scalable
- Easy migration to full Milvus

FAISS advantages:
- Simpler setup
- No dependencies on Milvus
- Good for small datasets