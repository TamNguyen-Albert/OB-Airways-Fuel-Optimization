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
| Model               | MAE         | RMSE         | R2           |
|---------------------|-------------|--------------|--------------|
| Linear Regression   | 1022.19     | 1505.89      | 0.9784       |
| Random Forest       | 540.76      | 944.51       | 0.9915       |
| XGBoost             | 459.80      | 875.69       | 0.9927       |

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
