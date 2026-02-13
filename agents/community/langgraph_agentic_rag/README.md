# LangGraph Agentic RAG

A production-ready Retrieval-Augmented Generation (RAG) agent built with LangGraph that intelligently retrieves information from a Milvus Lite vector store and generates context-aware responses using Llama Stack and Ollama.

## Quick Links

- 🚀 **[Quick Start](#quick-start)** - Get running in 8 steps
- 📖 **[Detailed Guide](./QUICKSTART.md)** - Comprehensive setup with troubleshooting
- 🔧 **[Configuration](#configuration-reference)** - Environment variables and settings
- 🎯 **[API Endpoints](#api-endpoints)** - REST API documentation

## Features

- **Agentic RAG Workflow**: The agent autonomously decides when to retrieve information
- **Llama Stack Integration**: Unified model serving with Ollama for local LLM inference
- **Milvus Lite Vector Store**: High-performance vector database with easy migration to production Milvus
- **FastAPI Service**: REST API with `/chat` and `/health` endpoints
- **Tool-based Retrieval**: LangGraph tool integration for seamless retrieval
- **Document Loader**: Easy document ingestion from text files with customizable chunking

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

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running
- Llama Stack CLI installed

### Installation

**Install Ollama** (if not already installed):

```bash
# macOS/Linux
#visit ollama website

# Or via Homebrew on macOS
brew install ollama
```

**Install Llama Stack**:

```bash
pip install llama-stack llama-stack-client
```

### Setup Instructions

**Step 1: Pull Required Models**

```bash
# LLM model for text generation
ollama pull llama3.2:3b

# Embedding model for vector search
ollama pull embeddinggemma:latest
```

**Step 2: Start Ollama Service**

```bash
# Start Ollama in a separate terminal
ollama serve
```

>**Keep this terminal open** - Ollama needs to keep running.

**Step 3: Start Llama Stack Server**

From the **repository root directory**:

```bash
llama stack run run_llama_server.yaml
```

> **Keep this terminal open** - the server needs to keep running.
> You should see output indicating the server started on `http://localhost:8321`.

**Step 4: Install Agent Dependencies**

Navigate to the RAG agent directory and install dependencies:

```bash
cd agents/community/langgraph_agentic_rag
pip install -r requirements.txt
```

**Step 5: Configure Environment Variables**

Copy the example environment file:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Llama Stack Server Configuration
BASE_URL=http://localhost:8321
MODEL_ID=ollama/llama3.2:3b
API_KEY=not-needed

# RAG Configuration
VECTOR_STORE_PATH=/absolute/path/to/milvus_data/milvus_lite.db
EMBEDDING_MODEL=ollama/embeddinggemma:latest
DOCS_TO_LOAD=./data/sample_knowledge.txt

# Server Configuration
PORT=8000
```

**Important**: Update `VECTOR_STORE_PATH` to an absolute path where you want the Milvus database stored.

**Step 6: Load Documents into Vector Store**

Navigate to the data directory and run the document loader:

```bash
cd data
python load_documents.py
```

This will:
- Read documents from `sample_knowledge.txt`
- Split documents into chunks (512 characters with 128 overlap)
- Generate embeddings using the `embeddinggemma` model
- Store chunks in the Milvus Lite vector database

**Step 7: Run the Interactive Chat**

```bash
cd ../examples
python execute_ai_service_locally.py
```

You should see:
```
================================================================================
LangGraph Agentic RAG - Interactive Chat
================================================================================
Model: ollama/llama3.2:3b
Base URL: http://localhost:8321/v1
...

Choose a question or ask one of your own.
 -->
```

**Step 8: Ask Questions!**

Try asking questions about the loaded documents:
```
 --> What is LangChain?
```

📚 **For detailed troubleshooting and advanced configuration, see [QUICKSTART.md](./QUICKSTART.md)**

## Dependencies

This agent requires the following key dependencies (see `requirements.txt` for complete list):

- `langchain-core`, `langchain-openai` - LangChain framework components
- `langgraph`, `langgraph-prebuilt` - Graph-based agent orchestration
- `llama-stack-client` - Llama Stack API client
- `fastapi`, `uvicorn` - Web service framework
- `pydantic`, `python-dotenv` - Configuration and data validation

Dependencies are installed in **Step 4** of the Quick Start guide above.

## Configuration Reference

Configuration is handled through two files:

### Llama Stack Server (`run_llama_server.yaml`)

Located in the repository root, this configures:
- Server port (8321)
- Milvus Lite vector store path
- Ollama integration URL
- Registered models (LLM and embedding)

### Application Environment (`.env`)

Environment variables (configured in Step 5 of Quick Start):

- `BASE_URL` - Llama Stack server URL (default: `http://localhost:8321`)
- `MODEL_ID` - LLM model identifier (e.g., `ollama/llama3.2:3b`)
- `API_KEY` - API authentication (use `not-needed` for local setup)
- `VECTOR_STORE_PATH` - Absolute path to Milvus Lite database file
- `EMBEDDING_MODEL` - Embedding model name (e.g., `ollama/embeddinggemma:latest`)
- `DOCS_TO_LOAD` - Path to documents for vector store (e.g., `./data/sample_knowledge.txt`)
- `PORT` - FastAPI server port (default: `8000`)

## Running as a FastAPI Service

To run the agent as a REST API service instead of interactive chat:

```bash
# From agents/community/langgraph_agentic_rag directory
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

## Customizing Your Knowledge Base

### Using Your Own Documents

1. Create a text file with your content (e.g., `my_documents.txt`)
2. Update `.env` to point to your file:
   ```env
   DOCS_TO_LOAD=./data/my_documents.txt
   ```
3. Re-run the document loader:
   ```bash
   cd data
   python load_documents.py
   ```

### Adjusting Chunk Size

Edit `load_documents.py` to customize chunking parameters:

```python
load_and_index_documents(
    chunk_size=512,      # Size of text chunks (default: 512)
    chunk_overlap=128,   # Overlap between chunks (default: 128)
)
```

**Recommended chunk sizes:**
- Technical documentation: 512-1024 characters
- Narrative text: 256-512 characters
- Code snippets: 128-256 characters

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

## Troubleshooting

### Connection Errors

**Error**: `Connection refused to http://localhost:8321`

**Solution**: Ensure the Llama Stack server is running:
```bash
llama stack run run_llama_server.yaml
```

### No Vector Store Found

**Error**: `No vector store found. Please run load_documents.py first`

**Solution**: Load documents into the vector store:
```bash
cd data
python load_documents.py
```

### Empty or Irrelevant Responses

**Possible causes:**
1. **Chunk size too small** - Documents split into headers only
   - Solution: Increase `chunk_size` to 512+ in `load_documents.py`
2. **Documents not loaded** - Vector store is empty
   - Solution: Re-run `python load_documents.py`
3. **Wrong model** - Model not compatible
   - Solution: Use `llama3.2:3b` or `llama3.1:8b`

For more troubleshooting help, see [QUICKSTART.md](./QUICKSTART.md#troubleshooting).

## Differences from Base Agents

This RAG agent extends the base LangGraph agent with:

1. **Retrieval Capability**: Automatic knowledge base search via Llama Stack
2. **Multi-step Workflow**: Agent → Retrieve → Generate pattern
3. **Vector Store Integration**: Milvus Lite-based document storage and retrieval
4. **Context-aware Generation**: Answers based on retrieved documents with relevance checking
5. **Llama Stack Integration**: Unified model serving and vector operations

## Additional Resources

- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** - LangGraph framework docs
- **[Llama Stack Documentation](https://llama-stack.readthedocs.io/)** - Llama Stack API reference
- **[Ollama Documentation](https://ollama.com/docs)** - Local model serving

## License

MIT License