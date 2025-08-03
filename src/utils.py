import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any, Tuple

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('churn_prediction.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model: Any, filepath: str) -> None:
    """Save model to disk"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """Load model from disk"""
    model = joblib.load(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model

def create_directories(directories: list) -> None:
    """Create project directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created: {directory}")

def calculate_business_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             avg_customer_value: float = 1000) -> Dict[str, float]:
    """Calculate business impact metrics"""
    # Assume customers with >50% churn probability will churn
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate business metrics
    total_customers = len(y_true)
    actual_churners = np.sum(y_true)
    predicted_churners = np.sum(y_pred)
    correctly_identified_churners = np.sum((y_true == 1) & (y_pred == 1))
    
    # Business impact calculations
    revenue_at_risk = actual_churners * avg_customer_value
    revenue_saved = correctly_identified_churners * avg_customer_value * 0.3  # 30% retention rate
    
    return {
        'total_customers': total_customers,
        'actual_churn_rate': actual_churners / total_customers,
        'predicted_churn_rate': predicted_churners / total_customers,
        'revenue_at_risk': revenue_at_risk,
        'potential_revenue_saved': revenue_saved,
        'intervention_efficiency': correctly_identified_churners / predicted_churners if predicted_churners > 0 else 0
    }
