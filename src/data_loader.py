import pandas as pd
import numpy as np
import logging
from typing import Tuple
from .utils import load_config

class DataLoader:
    """Data loading and initial validation class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        try:
            data_path = self.config['data']['raw_data_path']
            df = pd.read_csv(data_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Basic data validation
            self._validate_data(df)
            return df
            
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate loaded data"""
        # Check if required columns exist
        required_columns = ['Churn']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Log basic statistics
        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")
        self.logger.info(f"Missing values:\n{df.isnull().sum().sum()}")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get comprehensive data information"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_distribution': df['Churn'].value_counts().to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        return info
