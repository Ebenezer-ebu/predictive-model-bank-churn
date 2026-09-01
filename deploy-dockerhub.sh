#!/bin/bash

# ============================================================
# DOCKER HUB DEPLOYMENT SCRIPT
# ============================================================

set -e

# Variables - CHANGE THESE
DOCKERHUB_USERNAME="ebuka1234"
IMAGE_NAME_STREAMLIT="churn-dashboard-streamlit"
IMAGE_NAME_FASTAPI="churn-dashboard-fastapi"
TAG="latest"

echo "🚀 Deploying to Docker Hub..."

# 1. Login to Docker Hub
echo "📝 Logging into Docker Hub..."
docker login

# 2. Build and Push Streamlit Image
echo "📝 Building Streamlit image..."
docker build -t $DOCKERHUB_USERNAME/$IMAGE_NAME_STREAMLIT:$TAG -f docker/Dockerfile.streamlit .

echo "📝 Pushing Streamlit image to Docker Hub..."
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME_STREAMLIT:$TAG

# 3. Build and Push FastAPI Image
echo "📝 Building FastAPI image..."
docker build -t $DOCKERHUB_USERNAME/$IMAGE_NAME_FASTAPI:$TAG -f docker/Dockerfile.fastapi .

echo "📝 Pushing FastAPI image to Docker Hub..."
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME_FASTAPI:$TAG

echo "✅ Deployment complete!"
echo ""
echo "📦 Images pushed to Docker Hub:"
echo "   - $DOCKERHUB_USERNAME/$IMAGE_NAME_STREAMLIT:$TAG"
echo "   - $DOCKERHUB_USERNAME/$IMAGE_NAME_FASTAPI:$TAG"
echo ""
echo "🌐 To run Streamlit:"
echo "   docker run -p 8501:8501 $DOCKERHUB_USERNAME/$IMAGE_NAME_STREAMLIT:$TAG"
echo ""
echo "🌐 To run FastAPI:"
echo "   docker run -p 8000:8000 $DOCKERHUB_USERNAME/$IMAGE_NAME_FASTAPI:$TAG"