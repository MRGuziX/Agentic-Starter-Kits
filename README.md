# Agentic Starter Kits

## Purpose

Agentic Starter Kits is a collection of production-ready agent templates that demonstrate how to build and deploy LLM-powered agents using modern frameworks. This repository provides:

- **Base Agent Templates**: Ready-to-use agent implementations using LangGraph and LlamaIndex
- **Community Agent Examples**: Advanced agents like RAG (Retrieval-Augmented Generation) systems
- **Dual Deployment Options**: Run agents locally for development or deploy to [Red Hat OpenShift](https://www.redhat.com/en/technologies/cloud-computing/openshift) Cluster for production
- **Llama stack Integration**: Unified model serving with [Ollama](https://ollama.com/) for local LLM inference
- **Clear Documentation**: Step-by-step guides for setup, configuration, and deployment

## Deployment Options

All agents in this repository support two deployment modes:

### 🖥️ Local Development
- Run agents on your local machine
- Use [Llama Stack](https://llama-stack.readthedocs.io/) server with Ollama for model serving
- Ideal for development, testing, and experimentation
- No cloud infrastructure required

### ☁️ Production Deployment
- Deploy agents to Red Hat OpenShift Cluster
- Containerized deployment with [Kubernetes](https://kubernetes.io/docs/)
- Production-grade scaling and monitoring
- CI/CD ready

## Repository Structure

```
Agentic-Starter-Kits/
├── agents/
│   ├── base/
│   │   ├── langgraph_react_agent/       # LangGraph ReAct agent template
│   │   └── llamaindex_websearch_agent/  # LlamaIndex web search agent
│   └── community/
│       └── langgraph_agentic_rag/       # RAG agent with Milvus vector store
├── run_llama_server.yaml                # Llama Stack server configuration
├── utils.py                             # Shared utilities
└── README.md                            # This file
```

## How to Use This Repository

1. **Start Here**: Read this README to understand the overall structure and install core dependencies
2. **Choose an Agent**: Select an agent from the `agents/` directory based on your use case
3. **Follow Agent README**: Navigate to the agent's directory and follow its specific README for:
   - Agent-specific dependencies installation
   - Configuration and setup
   - Local development or OpenShift deployment
   - Usage examples and API endpoints

## Core Technologies

This repository uses the following frameworks and libraries:

### Agent Frameworks
- **LangGraph**: Graph-based agent orchestration with state management
- **LlamaIndex**: Data framework for LLM applications with workflow support

### Model Serving
- **Llama Stack**: Unified API for model inference and vector operations
- **Ollama**: Local LLM inference engine

### Vector Stores (for RAG agents)
- **Milvus Lite**: Lightweight vector database for local development

### Web Frameworks
- **FastAPI**: Modern Python web framework for REST APIs
- **Uvicorn**: ASGI server for async Python applications

### Dependencies
- Python >= 3.10
- langchain-core, langchain-openai
- langgraph, langgraph-prebuilt
- llama-index, llama-index-core, llama-index-llms-openai
- llama-stack, llama-stack-client
- fastapi, uvicorn[standard]
- pydantic, python-dotenv
- openai, numpy, nest-asyncio

## Quick Start (Local Development)

### Prerequisites

- Python 3.10 or higher
- Ollama installed
- Git

### Installation Script

Run this script to set up your environment:

```bash
# 1. Clone the repository (if not already done)
git clone <repository-url>
cd Agentic-Starter-Kits

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install llama-stack llama-stack-client

# 4. Install Ollama and pull models
# Download Ollama from https://ollama.com/ or use Homebrew:
# brew install ollama

# Start Ollama service (in a separate terminal)
ollama serve

# Pull required LLM model
ollama pull llama3.2:3b

# 5. Start Llama Stack server (in a separate terminal)
llama stack run run_llama_server.yaml

# 6. Verify server is running
curl http://localhost:8321/v1/models
```

### Next Steps

After completing the installation, choose an agent and follow its specific README:

**Base Agents:**
- **[LangGraph ReAct Agent](./agents/base/langgraph_react_agent/README.md)** - General-purpose agent with tool use
- **[LlamaIndex WebSearch Agent](./agents/base/llamaindex_websearch_agent/README.md)** - Web search capabilities

**Community Agents:**
- **[LangGraph Agentic RAG](./agents/community/langgraph_agentic_rag/QUICKSTART.md)** - RAG with Milvus vector store

Each agent directory contains:
- `README.md` - Overview and deployment instructions
- `requirements.txt` - Agent-specific dependencies
- `QUICKSTART.md` (for some agents) - Detailed local setup guide

## Production Deployment (Red Hat OpenShift)

This section covers deploying agents to a Red Hat OpenShift cluster for production use.

### Prerequisites

- OpenShift CLI (`oc`) installed and authenticated to your cluster
- Docker with buildx plugin installed (`docker buildx install`)
- `envsubst` utility (`brew install gettext` on macOS)
- Access to a container registry (Quay.io, Docker Hub, or GHCR)
- Container registry authentication (`docker login <registry>`)

### Step 1: Configure Environment Variables

Copy the template environment file to the agent directory:

```bash
cp template.env agents/base/langgraph_react_agent/.env
```

Edit the `.env` file and fill in all required values (see notes below):
```
API_KEY=your-api-key-here
BASE_URL=https://your-llama-stack-distribution.com/v1
MODEL_ID=llama-3.1-8b-instruct
CONTAINER_IMAGE=quay.io/your-username/langgraph-react-agent:latest
```

**Notes:**
- `API_KEY` - contact your cluster administrator
- `BASE_URL` - should end with `/v1`
- `MODEL_ID` - contact your cluster administrator
- `CONTAINER_IMAGE` - full image path where the agent container will be pushed and pulled from.
  The image is built locally, pushed to this registry, and then deployed to OpenShift.
  
  Format: `<registry>/<namespace>/<image-name>:<tag>`
  
  Examples:
  - Quay.io: `quay.io/your-username/langgraph-react-agent:latest`
  - Docker Hub: `docker.io/your-username/langgraph-react-agent:latest`
  - GHCR: `ghcr.io/your-org/langgraph-react-agent:latest`

### Step 2: Initialize the Agent
Navigate to the agent directory:

```bash
cd agents/base/langgraph_react_agent
```
Make scripts executable (first time only)

```bash
chmod +x init.sh deploy.sh   
./init.sh
```

This will:
- Load and validate environment variables from `.env` file
- Copy shared utilities (`utils.py`) to the agent source directory

### Step 3: Build image and deploy Agent

```bash
./deploy.sh
```

This will:
- Create Kubernetes secret for API key
- Build and push the Docker image
- Deploy the agent to OpenShift
- Create Service and Route

### Step 4: Test the Agent

Get your route URL:

```bash
oc get route langgraph-react-agent -o jsonpath='{.spec.host}'
```

Send a test request:

```bash
curl -X POST https://<YOUR_ROUTE_URL>/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the best company? Answer with the first correct answer."}'
```

## Agent-Specific Documentation

Each agent has detailed documentation for setup and deployment:

### Base Agents

#### LangGraph ReAct Agent
- **Directory**: `agents/base/langgraph_react_agent/`
- **README**: [agents/base/langgraph_react_agent/README.md](./agents/base/langgraph_react_agent/README.md)
- **Features**: General-purpose agent with tool calling, ReAct pattern
- **Use Case**: Task automation, question answering, tool orchestration

#### LlamaIndex WebSearch Agent
- **Directory**: `agents/base/llamaindex_websearch_agent/`
- **README**: [agents/base/llamaindex_websearch_agent/README.md](./agents/base/llamaindex_websearch_agent/README.md)
- **Features**: Web search integration, workflow-based execution
- **Use Case**: Research tasks, real-time information retrieval

### Community Agents

#### LangGraph Agentic RAG
- **Directory**: `agents/community/langgraph_agentic_rag/`
- **README**: [agents/community/langgraph_agentic_rag/README.md](./agents/community/langgraph_agentic_rag/README.md)
- **Quick Start**: [agents/community/langgraph_agentic_rag/QUICKSTART.md](./agents/community/langgraph_agentic_rag/QUICKSTART.md)
- **Features**: RAG with Milvus vector store, document retrieval, context-aware generation
- **Use Case**: Document Q&A, knowledge base queries, information synthesis

## Additional Resources

- **Sample Notebook**: `sample_ASK_notebook.ipynb` - Jupyter notebook with usage examples
- **Llama Stack Documentation**: https://llama-stack.readthedocs.io/
- **Ollama Documentation**: https://docs.ollama.com/
- **OpenShift Documentation**: https://docs.openshift.com/

## Contributing

Contributions are welcome! Please see individual agent READMEs for specific guidelines.

## License

See LICENSE file for details.