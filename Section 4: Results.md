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
