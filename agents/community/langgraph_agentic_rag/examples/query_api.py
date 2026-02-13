"""Example script to query the RAG agent REST API.

This script demonstrates how to interact with the deployed FastAPI service.
Make sure the service is running (python main.py) before executing this script.
"""

import requests
import json


def query_chat_endpoint(message: str, base_url: str = "http://localhost:8000"):
    """Send a message to the chat endpoint and print the response.

    Args:
        message: The user message to send
        base_url: The base URL of the FastAPI service
    """
    url = f"{base_url}/chat"
    payload = {"message": message}

    print(f"\n{'='*80}")
    print(f"Sending: {message}")
    print(f"{'='*80}\n")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        messages = data.get("messages", [])

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                print(f"\n{' User '.center(80, '=')}")
                print(content)
            elif role == "assistant":
                if "tool_calls" in msg:
                    print(f"\n{' Assistant (Tool Call) '.center(80, '=')}")
                    for tc in msg["tool_calls"]:
                        func = tc["function"]
                        args = json.loads(func["arguments"])
                        print(f"Calling {func['name']} with args: {args}")
                elif content:
                    print(f"\n{' Assistant '.center(80, '=')}")
                    print(content)
            elif role == "tool":
                print(f"\n{' Tool Response '.center(80, '=')}")
                print(content)

        print(f"\n{'='*80}\n")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def check_health(base_url: str = "http://localhost:8000"):
    """Check the health endpoint.

    Args:
        base_url: The base URL of the FastAPI service
    """
    url = f"{base_url}/health"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        print(f"\n{' Health Check '.center(80, '=')}")
        print(f"Status: {data.get('status')}")
        print(f"Agent Initialized: {data.get('agent_initialized')}")
        print(f"{'='*80}\n")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check health
    check_health()

    # Example queries
    queries = [
        "What is LangGraph?",
        "Tell me about RAG and how it works",
        "What is Milvus used for?",
    ]

    for query in queries:
        query_chat_endpoint(query)