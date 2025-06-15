## Section 4: Results
# ✈️ Fuel Consumption Prediction – Model Evaluation Report

## Dataset Overview
- **Target Variable**: `actual_flight_fuel_kilograms`
- **Features Used**: Air distance, estimated takeoff weight, encoded airports, etc.
- **Train/Test Split**: 80% training – 20% testing

---

## 1. Linear Regression (Baseline Model)

| Metric | Value |
|--------|--------|
| MAE    | 1,785.23 kg |
| RMSE   | 2,773.34 kg |
| R²     | 0.9267 |

- While simple and interpretable, this model performs significantly worse than others in both error metrics and explanatory power.

---

## 2. Random Forest Regressor

| Metric | Value |
|--------|--------|
| MAE    | 596.48 kg |
| RMSE   | 1,129.82 kg |
| R²     | 0.9878 |

- Strong performance with low error and high R². Effectively captures nonlinear patterns without extensive tuning.

---

## 3. XGBoost Regressor

| Metric | Value |
|--------|--------|
| MAE    | 534.23 kg |
| RMSE   | 1,124.22 kg |
| R²     | 0.9880 |

- Best standalone performance.
- Well-suited for structured tabular data like flight logs.

---

## 4. Stacking Regressor (XGBoost + Linear Regression)

| Metric | Value |
|--------|--------|
| MAE    | 540.24 kg |
| RMSE   | 1,143.56 kg |
| R²     | 0.9875 |

- Very competitive with XGBoost, combining the strengths of linear and nonlinear models.
- Slight trade-off in RMSE vs. XGBoost, but may generalize better in some edge cases.

---

## 📊 Model Comparison Summary

| Model               | MAE (kg) | RMSE (kg) | R² Score |
|--------------------|----------|-----------|----------|
| Linear Regression  | 1,785.23 | 2,773.34  | 0.9267   |
| Random Forest      |   596.48 | 1,129.82  | 0.9878   |
| XGBoost            |   534.23 | 1,124.22  | 0.9880   |
| Stacking Regressor |   540.24 | 1,143.56  | 0.9875   |

---

## Recommendation

- **Use XGBoost** as the primary deployment model for best performance.
- **Stacking Regressor** can be considered if the goal is robustness through ensembling.
- **Random Forest** is a solid alternative with similar performance and ease of interpretation.
- **Linear Regression** is only suitable as a baseline or quick benchmark.
