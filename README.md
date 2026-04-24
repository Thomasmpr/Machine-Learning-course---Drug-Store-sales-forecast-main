# 📈 Rossmann Store Sales Forecasting Project

## 📝 Executive Summary
This project delivers a high-precision forecasting engine for **1,115 Rossmann drug stores**. By leveraging historical sales data (2.5 years) and store metadata, we developed a machine learning pipeline to predict daily sales, enabling optimized stock replenishment and labor scheduling.

The Core Challenge: Balancing the financial risk of stockouts during promotional peaks against the cost of over-ordering during midweek troughs.

---

## 📊 Key Insights & Visualizations

1. **Seasonal Demand & Peak Performance:** Our analysis confirms a significant **December peak** , with daily sales rising 30% above the October baseline. This supports the business requirement for aggressive Q4 inventory build-up.

2. **The Power of Promotions:** Promotions are identified as the **#1 commercial lever** , driving an average 38.8% sales uplift across the chain. Store Type 2 was identified as the most responsive to promotional activity.

3. **Feature Importance (XGBoost):** The model relies heavily on **Promotional flags** and structural store data (Competition distance, Assortment) rather than simple calendar rhythms.

---

## 🚀 Model Performance

We benchmarked our **XGBoost Champion** against a Linear Regression and a Naive Baseline (Historical Store-Mean).

| Model | RMSPE (%) | MAE (€) | RMSE (€) |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 23.0% | **1,051** | 1,700  |
| **Naive Baseline** | **21.9%** | 1,200 | 1,900 |
| **Linear Regression** | 47.0%+ | 2,030 | 2,800 |

> **Analysis:** XGBoost provides the best absolute error (MAE), making it ideal for bottom-up revenue aggregation.

---

## 📁 Methodology
* **Data Engineering:** Temporal feature extraction (MonthStr, DayOfWeek), competition open duration, and holiday mapping.
* **Validation:** 90/10 Temporal split to prevent data leakage and ensure real-world forecasting conditions.
* **Optimization:** Implementation of XGBoost with clipped outputs to ensure zero-bounded sales predictions.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Model:** XGBoost Regressor
* **Visualization:** Matplotlib (Agg backend)

---

## 🎓 Authors
This research was conducted as part of the Machine Learning course at Audencia Business School (March 2026), by:
* **Olivier Andrianoel**
* **Wassim Belabed**
* **Thomas Magallon**
