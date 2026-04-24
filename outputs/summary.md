# Rossmann PPT Pack Summary

## Dataset overview
- Merged rows: 1,017,209
- Number of stores: 1,115
- Date range: 2013-01-01 to 2015-07-31
- Rows used for model after filtering closed/zero-sales days: 844,338

## EDA highlights
- Highest average sales month: Dec (8,609)
- December average sales: 8,609
- Promo uplift in mean sales: 38.77%

## Model performance
- Best RMSPE model: NaiveBaseline (21.98%)
- Best MAE model: XGBoost (1,071)

## Top XGBoost features
- 1. Promo
- 2. Promo2SinceYear
- 3. CompetitionOpenSinceYear
- 4. CompetitionDistance
- 5. Assortment

## Diagnostics
- Worst store by MAE: Store 842 with MAE 6,062

## Recommended slide order
- 1. Business context + dataset overview
- 2. Missing values / data quality
- 3. Monthly seasonality
- 4. Weekday pattern + Sunday zero-sales chart
- 5. Promo impact chart
- 6. Month x DayOfWeek heatmap
- 7. Correlation heatmap
- 8. Competition-distance effect
- 9. Model comparison (Naive vs Linear vs XGBoost)
- 10. Actual vs predicted on the test period
- 11. Feature importance
- 12. Residuals and error heatmaps
- 13. Business implications