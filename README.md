# Agentic Starter Kits

Agentic Starter Kits is a collection of example agents that show how to use LLM
agents to support our work. It includes local and cloud-ready templates with
clear, minimal entry points.

## What's in this repo
- `agents/`: agent templates and examples
- `agents/base/langgraph_react_agent/`: LangGraph-based agent template
- `agents/base/llamaindex_websearch_agent/`: LlamaIndex-based agent template
- `ollama-config.yaml`: example Ollama configuration
- `utils.py`: shared helpers (env loading)

## Usage paths
- Local: run with a llama-stack server and an Ollama model
- Cloud: deploy on Red Hat OpenShift Cluster

## Documentation links
- Llama Stack: https://llama-stack.readthedocs.io/
- Ollama models: https://docs.ollama.com/
- OpenShift cluster docs: https://docs.openshift.com/en

## Required Libraries
- Python >= 3.10
- fastapi
- uvicorn[standard]
- pydantic
- langchain-core
- langchain-openai
- langgraph
- langgraph-prebuilt
- openai
- python-dotenv
- llama-index
- llama-index-core
- llama-index-llms-openai
- llama-index-utils-workflow
- numpy
- nest-asyncio

## Quick start (local)
```bash
# Create and activate a virtual env
python -m venv .venv
source .venv/bin/activate
```
## Install dependencies for an agent
```bash
pip install -r agents/base/langgraph_react_agent/requirements.txt
```
or
```bash
pip install -r agents/base/llamaindex_websearch_agent/requirements.txt
```
## Install Llama-Stack
```bash
pip install llama-stack llama-stack-client
```

If you want to install ollama you need to install app from [Ollama site](https://ollama.coma/)
or via
[Brew](https://brew.sh/)
```bash
pip install ollama
```
**Now you need to start Ollama app**
```bash
ollama pull llama3.2:3b
```

## Start Ollama
```bash
ollama serve
```

Update `ollama-config.yaml` with your local settings, then follow the
agent-specific README
`agents/base/<choose_your_agent>/readme.md`

## Start llama-stack (example)
```bash
llama stack run ollama-config.yaml --port 8321
```

## Check if server is running and model is there
```bash
curl http://127.0.0.1:8321/v1/models
```

## Agents
- LangGraph React agent: `agents/base/langgraph_react_agent/`
- LlamaIndex websearch agent: `agents/base/llamaindex_websearch_agent/`

## Quick Start (Deploy to OpenShift)

### Prerequisites

- `oc` CLI installed and logged in to OpenShift cluster
- `docker` (with buildx) installed (`docker buildx install`)
- `envsubst` installed (`brew install gettext` for macOS)
- Access to container registry (e.g., Quay.io)
- Logged in to your container registry (`docker login quay.io`)

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

## Notes
Each agent template has its own README with setup, configuration, and examples.

You can take a look on Our `sample_ASK_notebook.ipynb` for some guidance.