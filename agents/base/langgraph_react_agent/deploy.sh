#!/bin/bash
#
# Deploy LangGraph React Agent to OpenShift
#
# Usage:
#   ./deploy.sh
#
# Prerequisites:
#   - oc CLI installed and logged in to OpenShift cluster
#   - podman or docker installed
#   - Access to container registry (e.g., Quay.io)
#

set -e  # Exit on error

source .env
export CONTAINER_IMAGE BASE_URL MODEL_ID

## ============================================
# DOCKER BUILD
## ============================================

docker buildx build --platform linux/amd64 -t "${CONTAINER_IMAGE}" -f Dockerfile --push . && echo "Docker build completed"

# ## ============================================
# # DOCKER DELETE DEPLOYMENT, SERVICE, ROUTE
# ## ============================================

oc delete deployment langgraph-react-agent && echo "Deployment deleted"
oc delete service langgraph-react-agent && echo "Service deleted"
oc delete route langgraph-react-agent && echo "Route deleted"

## ============================================
# DOCKER APPLY DEPLOYMENT, SERVICE, ROUTE
## ============================================
envsubst < k8s/deployment.yaml | oc apply -f - && echo "Deployment applied"
oc apply -f k8s/service.yaml && echo "Service applied"
oc apply -f k8s/route.yaml && echo "Route applied"

oc rollout status deployment/langgraph-react-agent --timeout=300s && echo "Deployment rolled out"

oc get deployment langgraph-react-agent && echo "Deployment exists"
oc get service langgraph-react-agent && echo "Service exists"
oc get route langgraph-react-agent && echo "Route exists"
