
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
#### 1. Fuel Consumption vs. Flight Distance
![image](https://github.com/user-attachments/assets/2b5b10aa-ba20-45a8-b5e6-8df6ab1805d1)

- There is a clear positive correlation between **flight distance** and **fuel consumption**, confirming the intuitive relationship that longer flights require more fuel.
- Several short-distance flights appear to consume unusually high fuel amounts, suggesting **possible inefficiencies**, such as long taxiing times, reroutes, or over-fueling.
- The spread becomes wider at longer distances, indicating **variability in efficiency** that could depend on aircraft type or flight conditions.

```python
sns.scatterplot(data= merged_df, x='air_distance_miles', y='actual_flight_fuel_kilograms')
plt.title('Fuel Consumption vs. Flight Distance')
plt.xlabel('Flight Distance (miles)')
plt.ylabel('Actual Fuel (kg)')
plt.tight_layout()
plt.show()
```
---

#### 2. Uplifted Fuel vs. Estimated Takeoff Weight
![image](https://github.com/user-attachments/assets/26f30292-5edb-4296-b36b-1f39b8329fa7)

- Fuel uplift generally **increases with estimated takeoff weight**, which aligns with operational expectations.
- A number of outliers exist where heavy flights received relatively low fuel uplift or vice versa — these anomalies may point to **planning inaccuracies**, equipment constraints, or specific route considerations.
- A tighter alignment could reduce safety margins and signal **potential optimization opportunities**.

```python
sns.scatterplot(data=merged_df, x='estimated_takeoff_weight_kilograms', y='uplifted_fuel_kilograms')
plt.title('Uplifted Fuel vs. Estimated Takeoff Weight')
plt.xlabel('Estimated Takeoff Weight (kg)')
plt.ylabel('Uplifted Fuel (kg)')
plt.tight_layout()
plt.show()
```
---

#### 3. Actual vs. Planned Fuel (Boxplot)
![image](https://github.com/user-attachments/assets/8a8aba90-dbef-4479-8720-6ecc7b559b9f)

- The boxplot reveals that **planned fuel quantities tend to exceed actual consumption**, with a large portion of flights consuming **less fuel than planned**.
- This over-planning could lead to **excess operational costs** and unnecessary weight during takeoff.
- Some flights consumed **more than planned**, indicating **risk areas** that should be reviewed for route changes, unexpected delays, or misestimates in planning.
```python
fuel_df = pd.DataFrame({
    'Actual Fuel': merged_df['actual_flight_fuel_kilograms'],
    'Planned Fuel': merged_df['planned_flight_fuel_kilograms']
}).melt(var_name='Type', value_name='Fuel (kg)')

sns.boxplot(data=fuel_df, x='Type', y='Fuel (kg)', palette='pastel')
plt.title('Distribution of Actual vs. Planned Fuel')
plt.tight_layout()
plt.show()
```
---

#### 4. Fuel Efficiency per Flight Hour
![image](https://github.com/user-attachments/assets/fdbbba62-3829-4fcb-af28-ec81ef164d3c)

- The majority of flights show a **consistent fuel consumption rate per hour**, with the distribution peaking in a narrow band.
- There are significant outliers with high hourly consumption, likely tied to:
  - Short flights with long idle/taxi times,
  - Aircraft type differences,
  - Operational inefficiencies.
- Suggests a potential for creating a **fuel efficiency benchmark** to evaluate future flights or aircraft.
```python
merged_df['fuel_per_hour'] = merged_df['actual_flight_fuel_kilograms'] / merged_df['flight_hours']

sns.histplot(merged_df['fuel_per_hour'], kde=True, bins=30, color='teal')
plt.title('Fuel Consumption per Flight Hour')
plt.xlabel('Fuel per Hour (kg/hour)')
plt.tight_layout()
plt.show()
```
---

#### 5. Fuel Consumption Over Time
![image](https://github.com/user-attachments/assets/cba4bf82-9d9e-4459-a6e6-e2c0a539d061)


- Daily fuel usage shows high fluctuation, indicating varying flight activity.
- The 7-day rolling average highlights recurring weekly patterns.
- A noticeable rise in consumption mid-period suggests seasonal demand or increased operations.
- The drop at the end may reflect reduced activity or improved fuel efficiency.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load your actual_flights DataFrame
# actual_df = pd.read_csv("actual_flights.csv")  # Uncomment if loading from file

# Convert departure time to datetime
actual_df['actual_time_departure'] = pd.to_datetime(actual_df['actual_time_departure'])

# Extract date only (remove time)
actual_df['date'] = actual_df['actual_time_departure'].dt.date

# Group by date to sum total fuel per day
daily_fuel = actual_df.groupby('date')['actual_flight_fuel_kilograms'].sum()

# Convert to DataFrame and calculate 7-day rolling average
daily_fuel_df = daily_fuel.reset_index()
daily_fuel_df['rolling_avg_7d'] = daily_fuel_df['actual_flight_fuel_kilograms'].rolling(window=7).mean()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(daily_fuel_df['date'], daily_fuel_df['actual_flight_fuel_kilograms'], label='Daily Fuel Consumption', marker='o')
plt.plot(daily_fuel_df['date'], daily_fuel_df['rolling_avg_7d'], label='7-Day Rolling Average', linewidth=3, color='orange')

plt.title('Daily Total Fuel Consumption with 7-Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Total Fuel (kg)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```
---


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

## 🔍 Model Explainability

To enhance interpretability of the predictive model and provide actionable insights for flight planning teams, we integrated **SHAP (SHapley Additive exPlanations)** to explain the output of the `XGBoostRegressor` – the most performant single model in our analysis.

### ✅ Method
We applied SHAP on the trained `XGBoost` model to calculate how each feature contributed to the model’s prediction of fuel consumption (`planned_flight_fuel_kilograms`). The SHAP summary plot below provides a global view of feature importance and the direction of influence.

### 📊 SHAP Summary Plot

![SHAP Summary Plot](e83b23f4-4a71-4f6d-8d60-117e508ba57c.png)

### 🔍 Key Insights from SHAP

| Feature                              | Impact on Prediction | Interpretation |
|--------------------------------------|-----------------------|----------------|
| `air_distance_miles`                 | High                  | The longer the distance, the more fuel is expected to be consumed. Red dots on the far right indicate that high values of air distance significantly increase predictions. |
| `estimated_takeoff_weight_kilograms`| High                  | Heavier takeoff weight increases fuel needs. SHAP shows this clearly with high-weight (red) values leading to higher SHAP values. |
| `planned_flight_hours`              | Medium                | Longer planned flight durations moderately increase predicted fuel consumption. |
| `departure_encoded` / `arrival_encoded` | Low–Medium         | Certain departure or arrival airports influence fuel usage, potentially due to altitude, routing constraints, or operational environments. |

### 🎯 Business Impact

These explainability results empower OB Airways to:
- Prioritize **route optimization** for flights with long distances or heavy payloads.
- Re-evaluate **aircraft allocation** strategies based on planned flight profiles.
- Recognize specific **airport pairs** that may consistently drive higher fuel use, enabling deeper operational review.

### 🧠 Why SHAP?

SHAP offers consistent, additive explanations and is model-agnostic, making it particularly effective for tree-based models like XGBoost. Unlike traditional feature importance, it allows both **global understanding** and **individual prediction explanations** – ideal for decision support in aviation planning.

-----------------
# 📊 Fuel Consumption Insights

This section summarizes key findings from visual analyses related to fuel usage and efficiency patterns across OB Airways' actual flight data.

---

## 1. Fuel Consumption vs. Flight Distance

- There is a clear positive correlation between **flight distance** and **fuel consumption**, confirming the intuitive relationship that longer flights require more fuel.
- Several short-distance flights appear to consume unusually high fuel amounts, suggesting **possible inefficiencies**, such as long taxiing times, reroutes, or over-fueling.
- The spread becomes wider at longer distances, indicating **variability in efficiency** that could depend on aircraft type or flight conditions.

---

## 2. Uplifted Fuel vs. Estimated Takeoff Weight

- Fuel uplift generally **increases with estimated takeoff weight**, which aligns with operational expectations.
- A number of outliers exist where heavy flights received relatively low fuel uplift or vice versa — these anomalies may point to **planning inaccuracies**, equipment constraints, or specific route considerations.
- A tighter alignment could reduce safety margins and signal **potential optimization opportunities**.

---

## 3. Actual vs. Planned Fuel (Boxplot)

- The boxplot reveals that **planned fuel quantities tend to exceed actual consumption**, with a large portion of flights consuming **less fuel than planned**.
- This over-planning could lead to **excess operational costs** and unnecessary weight during takeoff.
- Some flights consumed **more than planned**, indicating **risk areas** that should be reviewed for route changes, unexpected delays, or misestimates in planning.

---

## 4. Fuel Efficiency per Flight Hour

- The majority of flights show a **consistent fuel consumption rate per hour**, with the distribution peaking in a narrow band.
- There are significant outliers with high hourly consumption, likely tied to:
  - Short flights with long idle/taxi times,
  - Aircraft type differences,
  - Operational inefficiencies.
- Suggests a potential for creating a **fuel efficiency benchmark** to evaluate future flights or aircraft.

---

## 5. Fuel Consumption Over Time

- Daily total fuel usage shows **cyclical trends** and clear variations across months.
- A sharp drop in mid-July may reflect **seasonal scheduling changes**, reduced operations, or fleet maintenance.
- The 7-day rolling average smooths short-term fluctuations and indicates an overall **stable trend** with occasional surges likely tied to peak demand or events.
- Monitoring such trends can help **forecast fuel demand** and **optimize procurement** and logistics.

---

