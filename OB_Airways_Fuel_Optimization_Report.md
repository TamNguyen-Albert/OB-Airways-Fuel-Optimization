
# ✈️ OB Airways Fuel Optimization Report

## Section 1: Overview

### Problem Statement
Fuel consumption is one of the most significant operational costs for airlines. OB Airways aims to optimize its planned fuel estimation system to improve accuracy, reduce waste, and support sustainable operations.

### Metrics
To evaluate model performance, the following metrics are used:
- **MAE (Mean Absolute Error):** measures average prediction error.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more severely.
- **R² (R-squared):** indicates the proportion of variance explained by the model.

---

## Section 2: Analysis

### Data Exploration
Two datasets are used:
- **actual_flights:** contains true fuel usage and flight characteristics.
- **flight_plan:** contains fuel estimates and planned flight details.

Each dataset has approximately 10,000 rows.

### Data Visualization & Insights
The following insights were drawn:
- Fuel consumption is correlated with flight hours and estimated takeoff weight.
- Some flights had fuel estimates far below actual usage, indicating inefficiencies.

### Opportunities for Improvement
- Improve prediction of planned fuel for better resource planning.
- Identify patterns leading to underestimation or overestimation.

---

## Section 3: Methodology

### Data Preprocessing
- Joined actual_flights and flight_plan using `flight_id`.
- Handled missing values and formatted date-time fields.
- Scaled numeric features and one-hot encoded categorical variables.

### Implementation
Three base models were tested:
- `LinearRegression`
- `RandomForestRegressor`
- `XGBRegressor`

Then, a **stacking model** was built:
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)

stack_model = StackingRegressor(
    estimators=[('xgb', xgb)],
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=True
)
stack_model.fit(X_train, y_train)
```

### Refinement
- Used 5-fold cross-validation.
- Performed hyperparameter tuning on XGBoost.

---

## Section 4: Results

| Model                      | MAE (kg) | RMSE (kg) | R²     |
|---------------------------|----------|-----------|--------|
| Linear Regression         | 1022.19  | 1505.89   | 0.9784 |
| Random Forest Regressor   | 540.76   | 944.51    | 0.9915 |
| XGBoost Regressor         | 459.80   | 875.69    | 0.9927 |
| **Stacking (XGBoost + LR)** | **393.88** | **772.12** | **0.9943** |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

preds = stack_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
```

---

## Section 5: Conclusion

### Reflection
This project demonstrated how combining domain expertise and ensemble modeling techniques can improve planned fuel estimation. The stacking model significantly outperformed single models, suggesting strong potential for operational integration.

### Improvement Suggestions
- Integrate weather, route complexity, and aircraft load as additional features.
- Use explainable ML tools (e.g., SHAP) for interpretability.
- Deploy the model as an API to assist dispatchers in real-time.

---
