import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from .utils import load_config

class FeatureEngineer:
    """Feature engineering class for creating new meaningful features"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features with robust NaN handling"""
        df_engineered = df.copy()
        
        # Financial features
        df_engineered = self._create_financial_features(df_engineered)
        
        # Service usage features
        df_engineered = self._create_service_features(df_engineered)
        
        # Behavioral features
        df_engineered = self._create_behavioral_features(df_engineered)
        
        # Interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # Final cleanup - this is the key fix
        df_engineered = self._clean_engineered_features(df_engineered)
        
        self.logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
        return df_engineered
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial-related features with safe operations"""
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            # Safe division avoiding division by zero
            df['avg_charges_per_tenure'] = np.where(
                df['tenure'] > 0,
                df['MonthlyCharges'] / df['tenure'],
                df['MonthlyCharges']
            )
            
            if 'TotalCharges' in df.columns:
                # Safe operations with multiple conditions
                df['charges_efficiency'] = np.where(
                    (df['MonthlyCharges'] > 0) & (df['tenure'] > 0),
                    df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure']),
                    1.0
                )
                
                df['charges_trend'] = np.where(
                    df['MonthlyCharges'] > 0,
                    df['TotalCharges'] / df['MonthlyCharges'],
                    1.0
                )
                
                # High value customer indicator
                total_charges_80th = df['TotalCharges'].quantile(0.8)
                df['high_value_customer'] = (df['TotalCharges'] > total_charges_80th).astype(int)
                
                # Price sensitivity with safe division
                df['price_per_month_ratio'] = np.where(
                    df['TotalCharges'] > 0,
                    df['MonthlyCharges'] / df['TotalCharges'],
                    0.0
                )
        
        return df
    
    def _create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features"""
        service_columns = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count of services used
        service_cols_present = [col for col in service_columns if col in df.columns]
        total_services = 0
        
        for col in service_cols_present:
            if col in df.columns:
                df[f'{col}_binary'] = (df[col] == 'Yes').astype(int)
                total_services += df[f'{col}_binary']
        
        df['total_services'] = total_services
        
        # Internet service type features
        if 'InternetService' in df.columns:
            df['has_internet'] = (df['InternetService'] != 'No').astype(int)
            df['fiber_optic_user'] = (df['InternetService'] == 'Fiber optic').astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features"""
        # Tenure-based features with safe binning
        if 'tenure' in df.columns:
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
            df['is_long_term_customer'] = (df['tenure'] > 48).astype(int)
        
        # Contract and payment behavior
        if 'Contract' in df.columns:
            df['month_to_month_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        if 'PaymentMethod' in df.columns:
            electronic_methods = ['Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)']
            df['electronic_payment'] = df['PaymentMethod'].isin(electronic_methods).astype(int)
            df['manual_payment'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
        
        if 'PaperlessBilling' in df.columns:
            df['paperless_billing'] = (df['PaperlessBilling'] == 'Yes').astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features with safe operations"""
        # Senior citizen and financial features
        if 'SeniorCitizen' in df.columns and 'MonthlyCharges' in df.columns:
            df['senior_monthly_charges'] = df['SeniorCitizen'] * df['MonthlyCharges']
        
        # Family status combinations
        if 'Partner' in df.columns and 'Dependents' in df.columns:
            df['family_size'] = (df['Partner'] == 'Yes').astype(int) + (df['Dependents'] == 'Yes').astype(int)
            df['single_no_dependents'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
        
        # Service and tenure interaction with safe division
        if 'total_services' in df.columns and 'tenure' in df.columns:
            df['services_per_tenure'] = np.where(
                df['tenure'] > 0,
                df['total_services'] / df['tenure'],
                df['total_services']
            )
        
        return df
    
    def _clean_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of all engineered features - KEY FIX"""
        # Replace any NaN values
        df = df.fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Ensure all values are finite
        for col in df.select_dtypes(include=[np.number]).columns:
            if not np.isfinite(df[col]).all():
                df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Log any cleanup performed
        self.logger.info("Feature engineering cleanup completed - all NaN and infinite values handled")
        
        return df
    
    def select_features(self, df: pd.DataFrame, method: str = 'correlation') -> List[str]:
        """Select most important features"""
        if method == 'correlation' and 'Churn' in df.columns:
            # Calculate correlation with target variable
            correlations = df.corr()['Churn'].abs().sort_values(ascending=False)
            
            # Select features with correlation > threshold
            threshold = 0.1
            selected_features = correlations[correlations > threshold].index.tolist()
            if 'Churn' in selected_features:
                selected_features.remove('Churn')  # Remove target variable
            
            self.logger.info(f"Selected {len(selected_features)} features based on correlation")
            return selected_features
        
        return [col for col in df.columns if col != 'Churn']
