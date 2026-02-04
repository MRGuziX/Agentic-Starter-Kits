# LangGraph Agentic RAG

A LangGraph-based Retrieval-Augmented Generation (RAG) agent that uses OpenAI-compatible APIs. This agent can retrieve information from a vector store knowledge base and generate informed responses.

## Features

- **Agentic RAG Workflow**: The agent autonomously decides when to retrieve information
- **OpenAI-Compatible**: Works with any OpenAI-compatible API (OpenAI, local models via vLLM, Ollama, etc.)
- **Flexible Vector Store**: Uses FAISS for vector storage with automatic sample data generation
- **FastAPI Service**: REST API with `/chat` and `/health` endpoints
- **Tool-based Retrieval**: LangGraph tool integration for seamless retrieval

## Architecture

The RAG workflow consists of three main steps:

1. **Agent Node**: Decides whether to retrieve information based on the user's query
2. **Retrieve Node**: If needed, retrieves relevant documents from the vector store
3. **Generate Node**: Generates a final answer based on retrieved context

```
START → Agent → [Decision] → Retrieve → Generate → END
                    ↓
                   END (if no retrieval needed)
```

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using poetry

```bash
poetry install
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure the following environment variables:

```env
# LLM Configuration
BASE_URL=http://localhost:8001/v1
MODEL_ID=gpt-4
API_KEY=your-api-key-here

# RAG Configuration
VECTOR_STORE_PATH=./data/vector_store
EMBEDDING_MODEL=text-embedding-3-small

# Server Configuration
PORT=8000
```

### Environment Variables

- `BASE_URL`: Base URL for the OpenAI-compatible API (e.g., `https://api.openai.com/v1`)
- `MODEL_ID`: Model identifier (e.g., `gpt-4`, `gpt-3.5-turbo`)
- `API_KEY`: API key for authentication (optional for local deployments)
- `VECTOR_STORE_PATH`: Path to store/load the FAISS vector store (optional, creates sample data if not provided)
- `EMBEDDING_MODEL`: Embedding model to use (default: `text-embedding-3-small`)
- `PORT`: Port to run the FastAPI server on (default: `8000`)

## Usage

### Starting the Service

```bash
python main.py
```

The service will start on `http://localhost:8000` (or the port specified in `PORT` env var).

### API Endpoints

#### POST /chat

Send a chat message to the RAG agent.

**Request:**
```json
{
  "message": "What is LangGraph?"
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is LangGraph?"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "retriever",
            "arguments": "{\"query\":\"LangGraph\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "name": "retriever",
      "content": "Document 1:\nLangGraph is a library for building stateful, multi-actor applications with LLMs..."
    },
    {
      "role": "assistant",
      "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps..."
    }
  ],
  "finish_reason": "stop"
}
```

#### GET /health

Check the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "agent_initialized": true
}
```

### Using with curl

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?"}'
```

## Vector Store

The agent uses FAISS for vector storage. If `VECTOR_STORE_PATH` is not provided or doesn't exist, the agent will create a sample vector store with documents about LangChain, LangGraph, RAG, and vector stores.

### Creating a Custom Vector Store

You can create your own vector store by:

1. Preparing your documents
2. Using the LangChain FAISS integration to create embeddings
3. Saving the vector store to `VECTOR_STORE_PATH`

Example:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create documents
documents = [
    Document(page_content="Your content here", metadata={"source": "doc1"}),
    # Add more documents...
]

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)

# Save for later use
vectorstore.save_local("./data/vector_store")
```

## Development

### Project Structure

```
langgraph-agentic-rag/
├── main.py                          # FastAPI application
├── src/
│   └── langgraph_agentic_rag/
│       ├── __init__.py
│       ├── agent.py                 # RAG agent graph definition
│       ├── tools.py                 # Retriever tool
│       └── utils.py                 # Utility functions
├── pyproject.toml                   # Poetry configuration
├── requirements.txt                 # Pip requirements
├── .env.example                     # Example environment variables
└── README.md                        # This file
```

### Running Tests

```bash
pytest tests/
```

## Differences from Base Agent

This RAG agent extends the base LangGraph agent with:

1. **Retrieval Capability**: Automatic knowledge base search
2. **Multi-step Workflow**: Agent → Retrieve → Generate pattern
3. **Vector Store Integration**: FAISS-based document storage and retrieval
4. **Context-aware Generation**: Answers based on retrieved documents

## License

MIT License