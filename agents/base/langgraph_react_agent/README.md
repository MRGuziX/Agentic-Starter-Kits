
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