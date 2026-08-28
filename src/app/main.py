# src/app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
from typing import List, Optional
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn risk with explainability",
    version="1.0.0"
)

# Load model and preprocessor
MODEL_PATH = 'models/best_model.joblib'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
FEATURE_NAMES_PATH = 'data/processed/feature_names.joblib'

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"Model loaded successfully. Features: {len(feature_names)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None
    feature_names = []

# Define request model
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    
    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88
            }
        }

# Define response model
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    explanation: dict
    top_factors: List[dict]

@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict churn probability",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    if model is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "features": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Predict churn probability for a single customer"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]
        
        # Get top factors
        shap_importance = np.abs(shap_values)
        top_indices = np.argsort(shap_importance)[-5:][::-1]
        
        top_factors = []
        for idx in top_indices:
            if idx < len(feature_names):
                top_factors.append({
                    "feature": feature_names[idx],
                    "value": float(X[0, idx]),
                    "shap_value": float(shap_values[idx]),
                    "impact": "increases risk" if shap_values[idx] > 0 else "decreases risk"
                })
        
        # Create explanation summary
        explanation = {
            "prediction": "Churn" if pred == 1 else "Not Churn",
            "probability": float(proba),
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low",
            "top_factors": top_factors[:5]
        }
        
        return PredictionResponse(
            churn_probability=float(proba),
            churn_prediction=int(pred),
            explanation=explanation,
            top_factors=top_factors[:5]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: List[CustomerData]):
    """Predict churn probability for multiple customers"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([c.dict() for c in customers])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        probas = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        # Create response
        results = []
        for i in range(len(customers)):
            results.append({
                "customer_index": i,
                "churn_probability": float(probas[i]),
                "churn_prediction": int(preds[i])
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)