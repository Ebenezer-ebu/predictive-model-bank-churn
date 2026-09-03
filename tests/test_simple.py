# tests/test_simple.py

import os
import pytest
import pandas as pd
import joblib

def test_data_exists():
    """Check if data files exist"""
    files = [
        'data/raw/bank_churn_modelling.csv',
        'data/processed/X_processed.joblib',
        'data/processed/y.joblib'
    ]
    for f in files:
        assert os.path.exists(f), f"File missing: {f}"
    print("✅ Data files exist")

def test_model_exists():
    """Check if model files exist"""
    files = [
        'models/best_model.joblib',
        'models/preprocessor.joblib'
    ]
    for f in files:
        assert os.path.exists(f), f"File missing: {f}"
    print("✅ Model files exist")

def test_data_loaded():
    """Test data loading"""
    X = joblib.load('data/processed/X_processed.joblib')
    y = joblib.load('data/processed/y.joblib')
    assert len(X) == len(y)
    print(f"✅ Data loaded: {len(X)} samples")

def test_model_loaded():
    """Test model loading"""
    model = joblib.load('models/best_model.joblib')
    assert model is not None
    print(f"✅ Model loaded: {type(model).__name__}")

def test_model_predicts():
    """Test model prediction"""
    X = joblib.load('data/processed/X_processed.joblib')
    model = joblib.load('models/best_model.joblib')
    
    # Use first 5 samples
    X_sample = X[:5]
    y_pred = model.predict(X_sample)
    
    assert len(y_pred) == 5
    assert all(p in [0, 1] for p in y_pred)
    print(f"✅ Model predicts: {y_pred}")