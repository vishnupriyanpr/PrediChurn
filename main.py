"""
Main script to run the complete churn prediction pipeline
"""

import pandas as pd  # Add this import
import logging
from pathlib import Path
from src.utils import setup_logging, create_directories, load_config
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

def main():
    """Main execution function"""
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    directories = ['data/raw', 'data/processed', 'models', 'reports', 'visualizations']
    create_directories(directories)
    
    logger.info("Starting Churn Prediction Pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Step 1: Load Data
        logger.info("Step 1: Loading data...")
        data_loader = DataLoader()
        df = data_loader.load_data()
        data_info = data_loader.get_data_info(df)
        logger.info(f"Data loaded: {data_info['shape']}")
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_encoded = preprocessor.encode_features(df_clean)
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Engineering features...")
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.create_features(df_encoded)
        
        # Step 4: Final preprocessing and split
        logger.info("Step 4: Final preprocessing...")
        X = df_engineered.drop('Churn', axis=1)
        X_scaled = preprocessor.scale_features(X)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            pd.concat([X_scaled, df_engineered['Churn']], axis=1)
        )
        
        # Save preprocessors
        preprocessor.save_preprocessors()
        
        # Step 5: Model Training
        logger.info("Step 5: Training models...")
        trainer = ModelTrainer()
        training_results = trainer.train_models(X_train, y_train)
        trainer.save_best_model()
        
        # Step 6: Model Evaluation
        logger.info("Step 6: Evaluating model...")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(trainer.best_model, X_test, y_test)
        
        # Step 7: Generate Reports
        logger.info("Step 7: Generating reports...")
        evaluator.create_evaluation_plots(evaluation_results)
        evaluator.create_shap_plots(evaluation_results['shap_values'])
        
        business_report = evaluator.generate_business_report(evaluation_results)
        with open('reports/business_report.md', 'w') as f:
            f.write(business_report)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model ROC-AUC: {evaluation_results['metrics']['roc_auc']:.4f}")
        logger.info("Reports saved in 'reports/' directory")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
