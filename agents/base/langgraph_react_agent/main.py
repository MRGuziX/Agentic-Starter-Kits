import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from langgraph_react_agent_base.agent import get_graph_closure

from utils import get_env_var


# Request/Response models
class ChatRequest(BaseModel):
    """Incoming chat request body for the /chat endpoint."""

    message: str


class ChatResponse(BaseModel):
    """Structured chat response (answer and optional steps)."""

    answer: str
    steps: list[str]


# Global variable for agent graph
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the ReAct agent graph on startup and clear it on shutdown.

    Reads BASE_URL and MODEL_ID from the environment, builds the graph via
    get_graph_closure, and sets the global agent_graph for the /chat endpoint.
    """
    global agent_graph

    # Get environment variables
    base_url = get_env_var("BASE_URL")
    model_id = get_env_var("MODEL_ID")

    # Ensure base_url ends with /v1 if provided
    if base_url and not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    # Get graph closure and create agent graph
    agent_graph = get_graph_closure(model_id=model_id, base_url=base_url)

    yield

    # Cleanup on shutdown (if needed)
    agent_graph = None


# Create FastAPI app
app = FastAPI(
    title="LangGraph React Agent API",
    description="FastAPI service for LangGraph React Agent",
    lifespan=lifespan,
)


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that accepts a message and streams the agent's response.

    Args:
        request: ChatRequest containing the user message

    Returns:
        StreamingResponse with Server-Sent Events (SSE) containing the agent's response
    """
    global agent_graph

    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def generate_stream():
        """Async generator that yields SSE-style chunks (JSON per line) with choices/delta or error."""
        try:
            # Convert user message to HumanMessage
            messages = [HumanMessage(content=request.message)]

            # Use invoke to get the agent's response
            result = await agent_graph.ainvoke({"messages": messages})

            # Extract the last message (agent's response)
            if "messages" in result and len(result["messages"]) > 0:
                # Find the last AIMessage
                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage):
                        content = message.content
                        if content:
                            # Send the message content as a stream chunk
                            chunk_data = {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                        "finish_reason": None,
                                    }
                                ]
                            }
                            yield f"{json.dumps(chunk_data)}\n\n"

            # Send final cfhunk with finish_reason
            final_chunk = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"{json.dumps(final_chunk)}\n\n"
            yield "[DONE]\n\n"

        except Exception as e:
            error_data = {
                "error": {
                    "message": f"Error processing request: {str(e)}",
                    "type": type(e).__name__,
                }
            }
            yield f"{json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    """Return service health and whether the agent graph has been initialized."""
    return {"status": "healthy", "agent_initialized": agent_graph is not None}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
