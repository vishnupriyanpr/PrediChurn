import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from .utils import save_model, load_config

class DataPreprocessor:
    """Data preprocessing and cleaning class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for modeling"""
        df_clean = df.copy()
        
        # Handle TotalCharges conversion (common issue in telco dataset)
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
        
        # Remove customer ID if present
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        self.logger.info(f"Data cleaned. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                self.logger.info(f"Filled missing values in {col} with median")
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Churn' and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                self.logger.info(f"Filled missing values in {col} with mode")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method for numerical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'SeniorCitizen']  # Skip binary features
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            if len(outliers) > 0:
                self.logger.info(f"Removed {len(outliers)} outliers from {col}")
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Encode target variable
        if 'Churn' in df_encoded.columns:
            if fit:
                self.label_encoders['Churn'] = LabelEncoder()
                df_encoded['Churn'] = self.label_encoders['Churn'].fit_transform(df_encoded['Churn'])
            else:
                df_encoded['Churn'] = self.label_encoders['Churn'].transform(df_encoded['Churn'])
        
        # Get categorical features from config
        categorical_features = self.config['features']['categorical_features']
        categorical_features = [col for col in categorical_features if col in df_encoded.columns]
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_features, drop_first=True)
        
        self.logger.info(f"Features encoded. New shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        X_scaled = X.copy()
        
        # Get numerical features from config
        numerical_features = self.config['features']['numerical_features']
        numerical_features = [col for col in numerical_features if col in X_scaled.columns]
        
        if fit:
            X_scaled[numerical_features] = self.scaler.fit_transform(X_scaled[numerical_features])
        else:
            X_scaled[numerical_features] = self.scaler.transform(X_scaled[numerical_features])
        
        self.logger.info("Features scaled successfully")
        return X_scaled
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self, models_path: str = "models/") -> None:
        """Save preprocessing objects"""
        save_model(self.scaler, f"{models_path}scaler.pkl")
        save_model(self.label_encoders, f"{models_path}label_encoders.pkl")
        self.logger.info("Preprocessors saved successfully")
