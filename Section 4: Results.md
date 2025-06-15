# Section 4: Results
## 📊 Model Evaluation and Validation

During the modeling phase, we experimented with four regression algorithms to predict actual flight fuel consumption (`actual_flight_fuel_kilograms`) based on features such as flight distance, estimated takeoff weight, and airport encodings.

### 🔧 Models and Parameters

1. **Linear Regression**
   - A baseline model.
   - No hyperparameters tuned.

2. **Random Forest Regressor**
   - Captures nonlinear relationships via an ensemble of decision trees.
   - Default parameters used; no Grid Search applied.

3. **XGBoost Regressor**
   - Tuned via GridSearchCV:
     ```python
     param_grid = {
         'n_estimators': [100, 200],
         'max_depth': [3, 5, 7],
         'learning_rate': [0.05, 0.1, 0.2]
     }
     ```
   - 5-fold cross-validation was used, and the best configuration was selected based on **Mean Absolute Error (MAE)**.

4. **Stacking Regressor**
   - Combines XGBoost and Linear Regression as base learners.
   - Final estimator: Linear Regression.
   - Uses 5-fold cross-validation within the stacking process.

### ⚙️ Evaluation Metrics

- **MAE (Mean Absolute Error)**: How far predictions deviate on average.
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily.
- **R² Score**: Measures the proportion of variance explained by the model.

All models were evaluated on the same test set (`X_test`, `y_test`), using the metrics above for consistency.

---

## ✅ Justification: Performance Analysis and Comparison

### 📈 Model Performance Summary

| Model               | MAE (kg) | RMSE (kg) | R² Score |
|---------------------|----------|-----------|----------|
| Linear Regression   | 1,785.23 | 2,773.34  | 0.9267   |
| Random Forest       |   596.48 | 1,129.82  | 0.9878   |
| XGBoost             |   534.23 | 1,124.22  | 0.9880   |
| Stacking Regressor  |   540.24 | 1,143.56  | 0.9875   |

> 📌 **Best Performing Model**: **XGBoost**, due to its superior accuracy across all metrics.

### 🔍 Why Did XGBoost Perform Better?

- **Gradient Boosting Framework**: Sequential learning focuses on correcting prior errors.
- **Regularization (L1/L2)**: Helps prevent overfitting.
- **Flexible Hyperparameter Tuning**: Allowed us to optimize depth, learning rate, and estimator count.

### 🧠 Interpretation:
- **Linear Regression** underfits the data due to its linear nature, missing complex patterns.
- **Random Forest** improved significantly by modeling nonlinearities and interactions but lacked the regularization benefits of boosting.
- **XGBoost** achieved the best balance between bias and variance.
- **Stacking Regressor** performed slightly worse than XGBoost alone — suggesting minimal additional benefit from combining base learners in this context.

### 📊 Visual Summary

![Model Comparison Chart](f7d6b217-3cbc-4109-a754-3c210f0c455c.png)

---

### 🏁 Conclusion

From the model validation and performance metrics, XGBoost emerges as the most reliable and accurate model for fuel consumption prediction. It is recommended for production deployment to support more accurate fuel uplift planning and cost optimization in OB Airways' operations.
