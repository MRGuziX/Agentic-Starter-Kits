import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage


from langgraph_react_agent_base.agent import get_graph_closure

# Load environment variables
load_dotenv()


# Request/Response models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# Global variable for agent graph
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent_graph
    
    # Get environment variables
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "").strip()
    model_id = os.getenv("MODEL_ID", "redhataillama-31-8b-instruct")
    
    # Validate required environment variables
    if not api_key:
        raise ValueError("API_KEY environment variable is required")
    
    # Ensure base_url ends with /v1 if provided
    if base_url and not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )
    
    # Get graph closure and create agent graph
    graph_closure = get_graph_closure(client, model_id, base_url=base_url if base_url else None)
    agent_graph = graph_closure()  # Use default system prompt
    
    yield
    
    # Cleanup on shutdown (if needed)
    agent_graph = None


# Create FastAPI app
app = FastAPI(
    title="LangGraph React Agent API",
    description="FastAPI service for LangGraph React Agent",
    lifespan=lifespan
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
        """Generator function that yields SSE formatted chunks"""
        try:
            # Convert user message to HumanMessage
            messages = [HumanMessage(content=request.message)]
            
            # Use invoke to get the agent's response
            result = agent_graph.invoke({"messages": messages})
            
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
                                            "content": content
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final chunk with finish_reason
            final_chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": f"Error processing request: {str(e)}",
                    "type": type(e).__name__
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "agent_initialized": agent_graph is not None}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

