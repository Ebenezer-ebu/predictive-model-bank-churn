import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    """
    Model training for churn prediction (without MLflow)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def get_models(self):
        """Define models with hyperparameter grids"""
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'class_weight': ['balanced', None]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', None]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, 
                                      use_label_encoder=False, 
                                      eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'scale_pos_weight': [1, 3, 5]
                }
            }
        }
        return models
    
    def train_models(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """Train multiple models with hyperparameter tuning"""
        models_config = self.get_models()
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"Best CV ROC-AUC: {cv_score:.4f}")
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Test F1: {metrics['f1']:.4f}")
            
            # Store model
            self.models[name] = {
                'model': best_model,
                'params': best_params,
                'cv_score': cv_score,
                'test_metrics': metrics
            }
            
            # Track best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = best_model
                self.best_model_name = name
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'metrics': {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                },
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        
        return results
    
    def save_best_model(self, filepath='models/best_model.joblib'):
        """Save the best model"""
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            print(f"✅ Best model saved to {filepath}")
        else:
            print("⚠️ No best model to save")

# Main execution
if __name__ == "__main__":
    import os
    import joblib
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    X = joblib.load('data/processed/X_processed.joblib')
    y = joblib.load('data/processed/y.joblib')
    feature_names = joblib.load('data/processed/feature_names.joblib')
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    trainer = ChurnModelTrainer()
    trainer.feature_names = feature_names
    models = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = trainer.evaluate_models(X_test, y_test)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Save best model
    trainer.save_best_model()
    
    # Save model metadata
    import json
    metadata = {
        'best_model_name': trainer.best_model_name,
        'best_score': trainer.best_score,
        'model_type': trainer.best_model.__class__.__name__,
        'features': feature_names,
        'test_metrics': results[trainer.best_model_name]['metrics']
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete! Best model: {trainer.best_model_name}")
    print(f"   ROC-AUC: {results[trainer.best_model_name]['metrics']['roc_auc']:.4f}")