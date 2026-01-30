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

## Notes
Each agent template has its own README with setup, configuration, and examples.

You can take a look on Our `sample_ASK_notebook.ipynb` for some guidance.