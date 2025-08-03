
# Churn Prediction Model - Business Report

## Executive Summary
Our churn prediction model achieves 78.1% accuracy with 57.9% precision 
in identifying customers likely to churn. This model can help retain 57.2% 
of identified at-risk customers.

## Key Performance Indicators
- **Accuracy**: 78.1%
- **Precision**: 57.9% (of predicted churners, 57.9% actually churn)
- **Recall**: 65.0% (we catch 65.0% of all churners)
- **ROC-AUC**: 0.822 (excellent discrimination ability)

## Business Impact
- **Total Customers Analyzed**: 1,409
- **Current Churn Rate**: 26.5%
- **Revenue at Risk**: $374,000
- **Potential Revenue Saved**: $72,900
- **Intervention Efficiency**: 57.2%

## Top Churn Drivers
- **avg_charges_per_tenure**: 0.132 importance
- **MonthlyCharges**: 0.083 importance
- **charges_trend**: 0.076 importance
- **TotalCharges**: 0.076 importance
- **price_per_month_ratio**: 0.075 importance

## Recommendations
1. **Immediate Action**: Target high-risk customers (probability > 70%) with retention campaigns
2. **Proactive Monitoring**: Monitor medium-risk customers (30-70% probability) closely
3. **Feature Focus**: Address the top churn drivers identified above
4. **Model Deployment**: Implement monthly scoring for all active customers

## Expected ROI
Based on current performance, implementing this model could save approximately 
$874,800 annually through improved customer retention.
