# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing for churn prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                                   'NumOfProducts', 'EstimatedSalary']
        self.categorical_features = ['Geography', 'Gender']
        self.binary_features = ['HasCrCard', 'IsActiveMember']
        self.target = 'Exited'
        self.feature_pipeline = None
        
    def load_data(self, filepath):
        """Load raw data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        logger.info("Checking for missing values...")
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            logger.warning(f"Missing values found: {missing_cols.to_dict()}")
            for col in missing_cols.index:
                if col in self.numerical_features:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        else:
            logger.info("No missing values found")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate records"""
        initial_len = len(df)
        # Drop duplicates based on all columns except RowNumber
        cols_to_check = [col for col in df.columns if col not in ['RowNumber', 'CustomerId']]
        df = df.drop_duplicates(subset=cols_to_check)
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        return df
    
    def handle_outliers(self, df):
        """Handle outliers using IQR method"""
        logger.info("Handling outliers...")
        
        for col in self.numerical_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {len(outliers)} outliers in {col}")
        
        return df
    
    def fix_skewness(self, df):
        """Apply transformations to fix skewness"""
        logger.info("Fixing skewness...")
        
        skewness = df[self.numerical_features].skew()
        skewed_features = skewness[abs(skewness) > 1].index.tolist()
        
        for col in skewed_features:
            if col in ['Balance', 'EstimatedSalary']:
                df[col] = np.log1p(df[col].clip(lower=0))
                logger.info(f"Applied log transformation to {col}")
            elif col in ['CreditScore']:
                df[col] = np.sqrt(df[col] - df[col].min() + 1)
                logger.info(f"Applied sqrt transformation to {col}")
        
        return df
    
    def detect_data_leakage(self, df):
        """Detect and prevent data leakage"""
        logger.info("Checking for data leakage...")
        
        # Drop RowNumber (time-based leakage)
        if 'RowNumber' in df.columns:
            logger.info("Dropping RowNumber to prevent time leakage")
            df = df.drop('RowNumber', axis=1)
        
        # Drop Surname (not predictive)
        if 'Surname' in df.columns:
            logger.info("Dropping Surname (not predictive)")
            df = df.drop('Surname', axis=1)
        
        # Drop CustomerId (not predictive)
        if 'CustomerId' in df.columns:
            logger.info("Dropping CustomerId (not predictive)")
            df = df.drop('CustomerId', axis=1)
        
        return df
    
    def create_features(self, df):
        """Create derived features"""
        logger.info("Creating derived features...")
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[0, 18, 30, 40, 50, 60, 100],
                               labels=['Under 18', '18-29', '30-39', '40-49', '50-59', '60+'])
        
        # Balance to Salary Ratio (avoid division by zero)
        df['BalanceToSalary'] = df['Balance'] / (df['EstimatedSalary'] + 1)
        
        # Tenure groups
        df['TenureGroup'] = pd.cut(df['Tenure'],
                                   bins=[-1, 1, 3, 5, 7, 10],
                                   labels=['0-1', '2-3', '4-5', '6-7', '8-10'])
        
        # High value customer
        median_balance = df['Balance'].median()
        df['HighBalance'] = (df['Balance'] > median_balance).astype(int)
        
        # Activity score
        df['ActivityScore'] = df['IsActiveMember'] + df['HasCrCard'] + (df['NumOfProducts'] / 2)
        
        # Tenure * Activity
        df['TenureActivity'] = df['Tenure'] * df['IsActiveMember']
        
        logger.info(f"Created derived features")
        return df
    
    def check_class_imbalance(self, df):
        """Check class imbalance"""
        churn_rate = df[self.target].mean()
        logger.info(f"Churn rate: {churn_rate:.2%}")
        return churn_rate
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        # Drop non-predictive columns
        df = df.drop(['CustomerId', 'Surname'], axis=1, errors='ignore')
        
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        
        return X, y
    
    def build_preprocessing_pipeline(self):
        """Build sklearn preprocessing pipeline"""
        
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        binary_pipeline = Pipeline([
            ('passthrough', 'passthrough')
        ])
        
        self.feature_pipeline = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features),
            ('bin', binary_pipeline, self.binary_features)
        ], remainder='drop')
        
        return self.feature_pipeline
    
    def preprocess(self, filepath):
        """Main preprocessing pipeline"""
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Fix skewness
        df = self.fix_skewness(df)
        
        # 6. Detect leakage
        df = self.detect_data_leakage(df)
        
        # 7. Create features
        df = self.create_features(df)
        
        # 8. Check class imbalance
        self.check_class_imbalance(df)
        
        # 9. Prepare features
        X, y = self.prepare_features(df)
        
        # 10. Build preprocessing pipeline
        self.build_preprocessing_pipeline()
        
        # 11. Apply preprocessing
        X_processed = self.feature_pipeline.fit_transform(X)
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        logger.info(f"Preprocessing complete. Shape: {X_processed.shape}")
        
        return X_processed, y, feature_names
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        num_features = self.numerical_features
        cat_features = self.feature_pipeline.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features)
        bin_features = self.binary_features
        return list(num_features) + list(cat_features) + list(bin_features)
    
    def save_preprocessor(self, filepath='models/preprocessor.joblib'):
        """Save the preprocessor"""
        joblib.dump(self.feature_pipeline, filepath)
        logger.info(f"Preprocessor saved to {filepath}")


if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run preprocessing
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.preprocess('data/raw/bank_churn_modelling.csv')
    preprocessor.save_preprocessor()
    
    # Save processed data
    joblib.dump(X, 'data/processed/X_processed.joblib')
    joblib.dump(y, 'data/processed/y.joblib')
    joblib.dump(feature_names, 'data/processed/feature_names.joblib')
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target shape: {y.shape}")
    print(f"   - Number of features: {len(feature_names)}")
    print(f"   - Feature names: {feature_names[:5]}...")