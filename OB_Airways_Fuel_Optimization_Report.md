
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

```python
import pandas as pd
actual_df = pd.read_excel('/content/drive/MyDrive/Ob_airways/ob_airways.xlsx', sheet_name=0)
plan_df = pd.read_excel('/content/drive/MyDrive/Ob_airways/ob_airways.xlsx', sheet_name=1)
merged_df = pd.merge(plan_df, actual_df, on='flight_id', how='left')
```

Each dataset has approximately 10,000 rows.

### Data Visualization & Insights
The following insights were drawn:
- Fuel consumption is correlated with flight hours and estimated takeoff weight.
- Some flights had fuel estimates far below actual usage, indicating inefficiencies.
  
![image](https://github.com/user-attachments/assets/26c7e83e-1f7e-4032-aa03-f9aee8b99718)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo một figure và một mảng 2 subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ 1 trên axes[0]
sns.histplot(actual_df['actual_flight_fuel_kilograms'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Actual Flight Fuel Consumption')
axes[0].set_xlabel('Fuel Consumption (kg)')
axes[0].set_ylabel('Frequency')

# Biểu đồ 2 trên axes[1]
sns.scatterplot(data=plan_df, x='air_distance_miles', y='planned_flight_fuel_kilograms', ax=axes[1])
axes[1].set_title('Distance vs. Planned Fuel')
axes[1].set_xlabel('Air Distance (miles)')
axes[1].set_ylabel('Planned Fuel (kg)')

plt.tight_layout()
plt.show()
```

### Opportunities for Improvement
- Improve prediction of planned fuel for better resource planning.
- Identify patterns leading to underestimation or overestimation.

---

## Section 3: Methodology

### Data Preprocessing
- Joined actual_flights and flight_plan using `flight_id`.
- Handled missing values and formatted date-time fields.
- Scaled numeric features and one-hot encoded categorical variables.
```python
from sklearn.preprocessing import LabelEncoder

df = plan_df.copy()
df['air_distance_miles'] = pd.to_numeric(df['air_distance_miles'], errors='coerce')
df.dropna(subset=['planned_flight_fuel_kilograms', 'air_distance_miles'], inplace=True)

le = LabelEncoder()
df['departure_encoded'] = le.fit_transform(df['departure_airport'])
df['arrival_encoded'] = le.fit_transform(df['arrival_airport'])
```

### Implementation
Three base models were tested:
- `LinearRegression`
- `RandomForestRegressor`
- `XGBRegressor`
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append((name, mae, rmse, r2))
```

### First results of 3 base models
First results of 3 base models  
📊 Model Performance:  
| Linear Regression | MAA: 1022.19 | RMSE: 1505.89 | R2: 0.9784 |  
| Random Forest | MAA: 540.76 | RMSE: 944.51 | R2: 0.9915 |
| XGBoost → MAA | 459.80 | RMSE: 875.69 | R2: 0.9927 |

```python
print("📊 Model Performance:")
for name, mae, rmse, r2 in results:
    print(f"{name:<20} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
```

### Refinement XGBoost
- Used 5-fold cross-validation.
- Performed hyperparameter tuning on XGBoost.

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Initialize model
xgb = XGBRegressor(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

# 5-fold GridSearchCV for MAE
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

# Fit model to training data
grid_search.fit(X_train, y_train)

# Best model and parameters
best_xgb = grid_search.best_estimator_
print("Best XGBoost params:", grid_search.best_params_)
```

### Stacking model (XGBoost + LinearRegression)
Then, a **stacking model** was built:
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

stack_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('lr', LinearRegression())
    ],
    final_estimator=LinearRegression(),
    cv=5,  # cross-validation trong stacking
    n_jobs=-1
)

stack_model.fit(X_train, y_train)
predictions = stack_model.predict(X_test)
```

#### Evaluate result of Stacking model (XGBoost + LinearRegression)
MAE: 387.16 | RMSE: 740.02 | R²: 0.9948
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
```

---

## Section 4: Results
Three models were evaluated on the test set: Linear Regression, Random Forest, and XGBoost. Subsequently, a Stacking Regressor combining XGBoost (as base) and Linear Regression (as meta-model) was developed and tested.

The results are summarized in the table below:  
| Model                      | MAE (kg) | RMSE (kg) | R²     |
|---------------------------|----------|-----------|--------|
| Linear Regression         | 1022.19  | 1505.89   | 0.9784 |
| Random Forest Regressor   | 540.76   | 944.51    | 0.9915 |
| XGBoost Regressor         | 459.80   | 875.69    | 0.9927 |
| **Stacking (XGBoost + LR)** | **393.88** | **772.12** | **0.9943** |
  
As shown, the stacking model clearly outperformed all individual models in terms of MAE, RMSE, and R². This indicates that it was able to capture both complex nonlinear relationships and overall trends more effectively.

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
In this project, I tackled the critical issue of fuel consumption optimization for OB Airways by analyzing flight operation data and developing a predictive model for planned fuel usage. Through a series of machine learning experiments, I evaluated multiple models including Linear Regression, Random Forest, XGBoost, and a Stacking Regressor.

Among these, the Stacking Model combining XGBoost and Linear Regression achieved the best performance, with:

MAE: 393.88 kg

RMSE: 772.12 kg

R²: 0.9943

This result highlights that combining a powerful tree-based model with a linear meta-model can leverage the best of both worlds—capturing complex nonlinear patterns while retaining generalization and bias correction.

One particularly interesting insight was how the ensemble stacking approach, even with only two base models, significantly outperformed strong individual models. This shows the potential of stacking even in medium-sized datasets (~10,000 rows), especially when models are sufficiently diverse in learning mechanisms.

### Improvement Suggestions
To further enhance the solution, I recommend the following:

Feature Engineering: Incorporating additional variables such as weather conditions, aircraft type categories, and airport elevation may improve model accuracy further.

Hyperparameter Tuning: Perform grid or Bayesian optimization across all base and meta-models to fine-tune their performance.

Model Explainability: Integrate SHAP or LIME to interpret which features drive fuel predictions, enabling more actionable insights for flight planners.

Deployment Readiness: The model can be deployed as a fuel estimation API or integrated into flight planning software to suggest more efficient fuel plans in real-time.

Robust Validation: In the future, applying time-series aware cross-validation (e.g., rolling forecast split) may improve generalization on sequential flight data.

🔚 Final Words:
This case study demonstrates a data-driven approach to improving operational efficiency in the airline industry. With further data enrichment and ongoing refinement, OB Airways can significantly enhance its fuel management strategy, reduce costs, and contribute to sustainable aviation.

---
