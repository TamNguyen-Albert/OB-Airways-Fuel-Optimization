## Section 3: Methodology

### Data Preprocessing
- Joined actual_flights and flight_plan using `flight_id`.
- Handled missing values and formatted date-time fields.
- Scaled numeric features and one-hot encoded categorical variables.
```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ✅ Tạo bản sao để xử lý tránh thay đổi plan_df gốc
df = plan_df.copy()

# ✅ Đảm bảo cột 'air_distance_miles' ở dạng số, lỗi sẽ bị chuyển thành NaN
df['air_distance_miles'] = pd.to_numeric(df['air_distance_miles'], errors='coerce')

# ✅ Loại bỏ các dòng có missing ở cột quan trọng
df.dropna(subset=['planned_flight_fuel_kilograms', 'air_distance_miles'], inplace=True)

# ✅ Kiểm tra xem 2 cột cần label encode có tồn tại không
if 'departure_airport' in df.columns and 'arrival_airport' in df.columns:
    # Tạo 2 encoder riêng biệt cho từng cột (tránh encode nhầm nghĩa)
    le_departure = LabelEncoder()
    le_arrival = LabelEncoder()
    
    df['departure_encoded'] = le_departure.fit_transform(df['departure_airport'])
    df['arrival_encoded'] = le_arrival.fit_transform(df['arrival_airport'])
else:
    print("⚠️ Một trong hai cột 'departure_airport' hoặc 'arrival_airport' không tồn tại.")

```

### Implementation
Three base models were tested:
- `LinearRegression`
- `RandomForestRegressor`
- `XGBRegressor`
  
![image](https://github.com/user-attachments/assets/ef618f14-7af6-4742-8637-92a0d88fc2da)

```pythonimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🧠 Bước 1: Xác định features (X) và target (y)
# Ví dụ: dự đoán lượng nhiên liệu theo khoảng cách và sân bay mã hóa
X = df[['air_distance_miles', 'departure_encoded', 'arrival_encoded']]
y = df['planned_flight_fuel_kilograms']

# 🧪 Bước 2: Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Bước 3: Khởi tạo mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

# 📊 Bước 4: Huấn luyện và đánh giá
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 📄 Bước 5: Hiển thị kết quả
results_df = pd.DataFrame(results)
print(results_df)
```

### Refinement XGBoost
- Used 5-fold cross-validation.
- Performed hyperparameter tuning on XGBoost.
![image](https://github.com/user-attachments/assets/e37e6694-d693-4e39-aa18-cce3f2c5ac7b)

```from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# ✅ Khởi tạo mô hình gốc
xgb = XGBRegressor(random_state=42, verbosity=0)

# ✅ Định nghĩa các tham số để Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

# ✅ Thiết lập GridSearchCV với 5-fold cross-validation và MAE làm tiêu chí đánh giá
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1  # Cho biết quá trình chạy (có thể bỏ nếu muốn yên lặng)
)

# ✅ Huấn luyện mô hình trên tập huấn luyện
grid_search.fit(X_train, y_train)

# ✅ Lấy ra mô hình tốt nhất và các tham số tương ứng
best_xgb = grid_search.best_estimator_
print("✅ Best XGBoost Parameters:", grid_search.best_params_)

# ✅ Dự đoán trên tập kiểm tra và đánh giá
y_pred = best_xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ MAE of Best Model on Test Set: {mae:.2f}")

```

### Stacking model (XGBoost + LinearRegression)
Then, a **stacking model** was built:
![image](https://github.com/user-attachments/assets/36659f1c-1878-493c-85bc-771a283595b4)

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ✅ Đảm bảo best_xgb đã được huấn luyện (từ GridSearchCV)
# ✅ Đảm bảo X_train, y_train, X_test, y_test đã được chuẩn bị từ trước

# Khởi tạo mô hình Stacking Regressor
stack_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('lr', LinearRegression())
    ],
    final_estimator=LinearRegression(),  # meta-model
    cv=5,
    n_jobs=-1,
    passthrough=False  # Đặt True nếu muốn truyền X gốc vào final_estimator cùng với predictions
)

# Huấn luyện mô hình
stack_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions = stack_model.predict(X_test)

# Đánh giá hiệu suất
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# In kết quả
print("📊 Stacking Regressor Performance:")
print(f"✅ MAE : {mae:.2f} kg")
print(f"✅ RMSE: {rmse:.2f} kg")
print(f"✅ R²   : {r2:.4f}")
```
