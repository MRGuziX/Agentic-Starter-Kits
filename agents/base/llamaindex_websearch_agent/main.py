import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.llama_index_workflow_agent_base.agent import get_workflow_closure

from utils import get_env_var


# Request/Response models
class ChatRequest(BaseModel):
    """Incoming chat request body for the /chat endpoint."""

    message: str


class ChatResponse(BaseModel):
    """Structured chat response (answer and optional steps)."""

    answer: str
    steps: list[str]


# Global variable for workflow closure (get_agent callable)
get_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the LlamaIndex workflow closure on startup and clear it on shutdown.

    Reads BASE_URL and MODEL_ID from the environment, builds the workflow via
    get_workflow_closure, and sets the global get_agent for the /chat endpoint.
    """
    global get_agent

    # Get environment variables
    base_url = get_env_var("BASE_URL")
    model_id = get_env_var("MODEL_ID")

    # Ensure base_url ends with /v1 if provided
    if base_url and not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    # Get workflow closure (returns a callable that returns an agent)
    get_agent = get_workflow_closure(model_id=model_id, base_url=base_url)

    yield

    # Cleanup on shutdown (if needed)
    get_agent = None


# Create FastAPI app
app = FastAPI(
    title="LlamaIndex Websearch Agent API",
    description="FastAPI service for LlamaIndex Websearch Agent",
    lifespan=lifespan,
)


def _get_assistant_content(last_message) -> str:
    """Extract text content from the last (assistant) ChatMessage."""
    if hasattr(last_message, "blocks") and last_message.blocks:
        return last_message.blocks[0].text or ""
    if hasattr(last_message, "content"):
        return last_message.content if isinstance(last_message.content, str) else ""
    return ""


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that accepts a message and streams the agent's response.

    Args:
        request: ChatRequest containing the user message

    Returns:
        StreamingResponse with Server-Sent Events (SSE) containing the agent's response
    """
    global get_agent

    if get_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def generate_stream():
        """Async generator that yields SSE-style chunks (JSON per line) with choices/delta or error."""
        try:
            # Create a fresh agent for this request
            agent = get_agent()
            messages = [{"role": "user", "content": request.message}]

            # Run the workflow
            result = await agent.run(input=messages)

            # Extract the last message (agent's response)
            if result and "messages" in result and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                content = _get_assistant_content(last_message)
                if content:
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

            # Send final chunk with finish_reason
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
    """Return service health and whether the workflow closure has been initialized."""
    return {"status": "healthy", "agent_initialized": get_agent is not None}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
