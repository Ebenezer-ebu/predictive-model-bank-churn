# tests/test_api.py

import pytest
import json
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.app.main import app
    client = TestClient(app)
except ImportError:
    # If app not found, skip API tests
    app = None
    client = None

@pytest.mark.skipif(client is None, reason="API app not found")
def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health endpoint working")

@pytest.mark.skipif(client is None, reason="API app not found")
def test_predict_endpoint():
    """Test prediction endpoint"""
    test_data = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 100000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    
    print(f"✅ Prediction endpoint: churn_probability={data['churn_probability']:.4f}")

@pytest.mark.skipif(client is None, reason="API app not found")
def test_predict_batch_endpoint():
    """Test batch prediction endpoint"""
    test_data = [
        {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 100000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 100000
        },
        {
            "CreditScore": 750,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 55,
            "Tenure": 2,
            "Balance": 50000,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 50000
        }
    ]
    
    response = client.post("/predict_batch", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    print(f"✅ Batch prediction endpoint: {len(data['predictions'])} predictions")