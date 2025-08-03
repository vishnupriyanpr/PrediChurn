import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import optuna
from typing import Dict, Any, Tuple
from .utils import save_model, load_config

class ModelTrainer:
    """Model training and hyperparameter optimization class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_model = None
        self.best_params = {}
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train multiple models with hyperparameter optimization"""
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._balance_data(X_train, y_train)
        
        models_config = self.config['model']['models_to_train']
        results = {}
        
        for model_name in models_config:
            self.logger.info(f"Training {model_name}...")
            
            if self.config['model']['hyperparameter_tuning']['enabled']:
                model, params = self._optimize_hyperparameters(model_name, X_train_balanced, y_train_balanced)
            else:
                model = self._get_default_model(model_name)
                model.fit(X_train_balanced, y_train_balanced)
                params = {}
            
            # Cross-validation score
            cv_scores = self._cross_validate_model(model, X_train_balanced, y_train_balanced)
            
            self.models[model_name] = model
            results[model_name] = {
                'model': model,
                'params': params,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_params = results[best_model_name]['params']
        
        self.logger.info(f"Best model: {best_model_name}")
        return results
    
    def _balance_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance the dataset using SMOTE"""
        smote = SMOTE(random_state=self.config['data']['random_state'])
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        self.logger.info(f"Data balanced. Original: {X.shape}, Balanced: {X_balanced.shape}")
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def _get_default_model(self, model_name: str) -> Any:
        """Get default model configuration"""
        random_state = self.config['data']['random_state']
        
        models = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'xgboost': XGBClassifier(random_state=random_state, eval_metric='logloss'),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state)
        }
        
        return models[model_name]
    
    def _optimize_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = self._get_hyperparameter_space(model_name, trial)
            model = self._create_model_with_params(model_name, params)
            
            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.config['model']['hyperparameter_tuning']['cv_folds'],
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['model']['hyperparameter_tuning']['n_trials'])
        
        best_params = study.best_params
        best_model = self._create_model_with_params(model_name, best_params)
        best_model.fit(X, y)
        
        return best_model, best_params
    
    def _get_hyperparameter_space(self, model_name: str, trial) -> Dict:
        """Define hyperparameter search space for each model"""
        random_state = self.config['data']['random_state']
        
        if model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'random_state': random_state
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state,
                'eval_metric': 'logloss'
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': random_state
            }
    
    def _create_model_with_params(self, model_name: str, params: Dict) -> Any:
        """Create model instance with given parameters"""
        if model_name == 'logistic_regression':
            return LogisticRegression(**params, max_iter=1000)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            return XGBClassifier(**params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
    
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Perform cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['data']['random_state'])
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores
    
    def save_best_model(self, filepath: str = "models/best_model.pkl") -> None:
        """Save the best trained model"""
        if self.best_model is not None:
            save_model(self.best_model, filepath)
            save_model(self.best_params, filepath.replace('.pkl', '_params.pkl'))
            self.logger.info("Best model saved successfully")
        else:
            self.logger.warning("No model trained yet")
