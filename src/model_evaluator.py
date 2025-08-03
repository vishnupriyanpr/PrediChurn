import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import shap
from typing import Dict, Any, List, Tuple
from .utils import load_config, calculate_business_metrics

class ModelEvaluator:
    """Model evaluation and visualization class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Business metrics
        business_metrics = calculate_business_metrics(y_test.values, y_pred_proba)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.columns)
        
        # SHAP values (for tree-based models)
        shap_values = self._calculate_shap_values(model, X_test)
        
        evaluation_results = {
            'metrics': metrics,
            'business_metrics': business_metrics,
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_true': y_test.values
            }
        }
        
        self.logger.info("Model evaluation completed")
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                self.logger.warning("Model doesn't have feature importance")
                return pd.DataFrame()
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_shap_values(self, model: Any, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP values for model interpretability"""
        try:
            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for performance
                
                # For binary classification, take values for positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                return {
                    'values': shap_values,
                    'expected_value': explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    'feature_names': X_test.columns.tolist()
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            return {}
    
    def create_evaluation_plots(self, evaluation_results: Dict[str, Any], save_path: str = "reports/") -> None:
        """Create comprehensive evaluation visualizations"""
        
        y_true = evaluation_results['predictions']['y_true']
        y_pred = evaluation_results['predictions']['y_pred']
        y_pred_proba = evaluation_results['predictions']['y_pred_proba']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve',
                          'Feature Importance', 'Prediction Distribution', 'Business Impact'],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                      colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {auc_score:.3f})'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')),
            row=1, col=2
        )
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name='Precision-Recall'),
            row=1, col=3
        )
        
        # 4. Feature Importance
        feature_importance = evaluation_results['feature_importance']
        if not feature_importance.empty:
            top_features = feature_importance.head(10)
            fig.add_trace(
                go.Bar(x=top_features['importance'], y=top_features['feature'], orientation='h'),
                row=2, col=1
            )
        
        # 5. Prediction Distribution
        fig.add_trace(
            go.Histogram(x=y_pred_proba, nbinsx=50, name='Churn Probability'),
            row=2, col=2
        )
        
        # 6. Business Impact
        business_metrics = evaluation_results['business_metrics']
        metrics_names = ['Revenue at Risk', 'Potential Savings']
        metrics_values = [business_metrics['revenue_at_risk'], business_metrics['potential_revenue_saved']]
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=False, title_text="Model Evaluation Dashboard")
        
        # Save plot
        fig.write_html(f"{save_path}model_evaluation_dashboard.html")
        self.logger.info(f"Evaluation plots saved to {save_path}")
    
    def create_shap_plots(self, shap_values: Dict[str, Any], save_path: str = "reports/") -> None:
        """Create SHAP interpretation plots"""
        if not shap_values:
            self.logger.warning("No SHAP values available for plotting")
            return
        
        try:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values['values'], 
                            feature_names=shap_values['feature_names'], 
                            show=False)
            plt.tight_layout()
            plt.savefig(f"{save_path}shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values['values'], 
                            feature_names=shap_values['feature_names'],
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f"{save_path}shap_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP plots saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP plots: {str(e)}")
    
    def generate_business_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate business-focused evaluation report"""
        
        metrics = evaluation_results['metrics']
        business_metrics = evaluation_results['business_metrics']
        feature_importance = evaluation_results['feature_importance']
        
        report = f"""
# Churn Prediction Model - Business Report

## Executive Summary
Our churn prediction model achieves {metrics['accuracy']:.1%} accuracy with {metrics['precision']:.1%} precision 
in identifying customers likely to churn. This model can help retain {business_metrics['intervention_efficiency']:.1%} 
of identified at-risk customers.

## Key Performance Indicators
- **Accuracy**: {metrics['accuracy']:.1%}
- **Precision**: {metrics['precision']:.1%} (of predicted churners, {metrics['precision']:.1%} actually churn)
- **Recall**: {metrics['recall']:.1%} (we catch {metrics['recall']:.1%} of all churners)
- **ROC-AUC**: {metrics['roc_auc']:.3f} (excellent discrimination ability)

## Business Impact
- **Total Customers Analyzed**: {business_metrics['total_customers']:,}
- **Current Churn Rate**: {business_metrics['actual_churn_rate']:.1%}
- **Revenue at Risk**: ${business_metrics['revenue_at_risk']:,.0f}
- **Potential Revenue Saved**: ${business_metrics['potential_revenue_saved']:,.0f}
- **Intervention Efficiency**: {business_metrics['intervention_efficiency']:.1%}

## Top Churn Drivers
"""
        
        if not feature_importance.empty:
            for i, row in feature_importance.head(5).iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.3f} importance\n"
        
        report += """
## Recommendations
1. **Immediate Action**: Target high-risk customers (probability > 70%) with retention campaigns
2. **Proactive Monitoring**: Monitor medium-risk customers (30-70% probability) closely
3. **Feature Focus**: Address the top churn drivers identified above
4. **Model Deployment**: Implement monthly scoring for all active customers

## Expected ROI
Based on current performance, implementing this model could save approximately 
${:,.0f} annually through improved customer retention.
""".format(business_metrics['potential_revenue_saved'] * 12)
        
        return report
