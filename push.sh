#!/bin/bash

# Quick single-command deployment
DOCKERHUB_USERNAME="ebuka1234"

echo "Building and pushing all images..."

# Build and push Streamlit
docker build -t $DOCKERHUB_USERNAME/churn-dashboard-streamlit:latest -f docker/Dockerfile.streamlit . && \
docker push $DOCKERHUB_USERNAME/churn-dashboard-streamlit:latest &

# Build and push FastAPI
docker build -t $DOCKERHUB_USERNAME/churn-dashboard-fastapi:latest -f docker/Dockerfile.fastapi . && \
docker push $DOCKERHUB_USERNAME/churn-dashboard-fastapi:latest &

wait

echo "✅ All images pushed to Docker Hub!"
echo "   Streamlit: $DOCKERHUB_USERNAME/churn-dashboard-streamlit:latest"
echo "   FastAPI:   $DOCKERHUB_USERNAME/churn-dashboard-fastapi:latest"