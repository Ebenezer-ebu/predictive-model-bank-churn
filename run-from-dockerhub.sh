#!/bin/bash

# ============================================================
# RUN FROM DOCKER HUB
# ============================================================

DOCKERHUB_USERNAME="ebuka1234"

echo "🚀 Pulling and running from Docker Hub..."

# Run Streamlit Dashboard
echo "📊 Starting Streamlit Dashboard..."
docker run -d \
    --name churn-dashboard \
    -p 8501:8501 \
    $DOCKERHUB_USERNAME/churn-dashboard-streamlit:latest

echo "✅ Streamlit Dashboard running at http://localhost:8501"

# Optional: Run FastAPI
echo "🔧 To run FastAPI:"
echo "   docker run -d --name churn-api -p 8000:8000 $DOCKERHUB_USERNAME/churn-dashboard-fastapi:latest"