#!/bin/bash
#
# Initialize the LangGraph React Agent
#
# Usage:
#   ./init.sh
#
# Prerequisites:
#   - oc CLI installed and logged in to OpenShift cluster
#   - docker installed
#   - Access to container registry (e.g., Quay.io)
#

source .env && echo "Environment variables loaded from .env file"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "CONTAINER_IMAGE not set, check .env file"
    exit 1
fi
if [ -z "$API_KEY" ]; then
    echo "API_KEY not set, check .env file"
    exit 1
fi

if [ -z "$BASE_URL" ]; then
    echo "BASE_URL not set, check .env file"
    exit 1
fi

if [ -z "$MODEL_ID" ]; then
    echo "MODEL_ID not set, check .env file"
    exit 1
fi

oc delete secret langgraph-react-agent-secrets --ignore-not-found
oc create secret generic langgraph-react-agent-secrets --from-literal=api-key="${API_KEY}"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the root directory of the repository (3 levels up from script)
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Copy utils.py to the destination
cp "$ROOT_DIR/utils.py" "$SCRIPT_DIR/src/langgraph_react_agent_base/" && echo "Utils.py copied to destination"

echo "Agent initialized successfully"