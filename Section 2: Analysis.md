## Section 2: Analysis

### 1. Data Exploration

This case study is based on two core datasets from OB Airways:

#### a. `actual_flights` Dataset
This dataset captures real-world operational data from completed flights.

**Key Features:**
- `flight_id`: Unique identifier of each flight
- `actual_time_departure`: Actual departure timestamp
- `actual_flight_fuel_kilograms`: Actual fuel consumed during the flight
- `uplifted_fuel_kilograms`: Fuel quantity loaded before the flight
- `estimated_takeoff_weight_kilograms`: Estimated weight of the aircraft at takeoff

---

#### b. `flight_plan` Dataset
This dataset contains planned flight information prepared prior to departure.

**Key Features:**
- `flight_id`: Identifier to join with the actual data
- `departure_airport`: IATA code of departure location
- `arrival_airport`: IATA code of arrival location
- `air_distance_miles`: Distance of the planned flight
- `planned_flight_fuel_kilograms`: Forecasted fuel requirement

---

#### c. 📊 Descriptive Statistics

A summary of key numeric features (after data cleaning) is shown below:

| Feature                             | Count  | Mean     | Std Dev  | Min   | 25%    | 50%    | 75%    | Max     | IQR    |
|-------------------------------------|--------|----------|----------|-------|--------|--------|--------|---------|--------|
| `air_distance_miles`                | 8734   | 982.57   | 756.59   | 34.0  | 637.0  | 815.0  | 918.0  | 5484.0  | 281.0  |
| `actual_flight_fuel_kilograms`      | 8734   | 7610.62  | 9842.83  | 740.0 | 4440.0 | 5440.0 | 6200.0 | 71840.0 | 1760.0 |
| `planned_flight_fuel_kilograms`     | 8734   | 7430.16  | 10129.89 | 544.0 | 4184.6 | 5152.0 | 5970.0 | 78164.9 | 1786.4 |
| `uplifted_fuel_kilograms`           | 8734   | 7810.20  | 10566.95 | 0.0   | 3290.0 | 6192.0 | 8679.0 | 74630.4 | 5388.8 |
| `estimated_takeoff_weight_kg`       | 8734   | 71946.02 | 32203.22 | 45501 | 59646  | 66849  | 68970  | 233000  | 9324.0 |

**Insights:**
- There is **high variance** in fuel-related variables, indicating a wide operational range.
- **Outliers** may exist in `uplifted_fuel_kilograms` and `estimated_takeoff_weight_kilograms` based on their large max values.
- All features show **non-normal, right-skewed distributions**, suggesting the need for robust models or feature scaling.

---

#### d. 📈 Data Relationships

- **Fuel vs. Distance**: Scatterplots show a clear linear trend between fuel usage and air distance.
- **Fuel vs. Takeoff Weight**: Uplifted fuel increases with estimated takeoff weight, highlighting the role of payload.
- **Boxplot** comparisons show that **actual fuel is often lower than planned**, implying room for optimization in flight planning.

These findings shaped the selection of predictive models and performance metrics in later stages.


```python
import pandas as pd
actual_df = pd.read_excel('/content/drive/MyDrive/Ob_airways/ob_airways.xlsx', sheet_name=0)
plan_df = pd.read_excel('/content/drive/MyDrive/Ob_airways/ob_airways.xlsx', sheet_name=1)
merged_df = pd.merge(plan_df, actual_df, on='flight_id', how='left')
```

```python
import pandas as pd

# Chọn các cột số liên quan đến phân tích mô hình
columns_to_describe = [
    'air_distance_miles',
    'actual_flight_fuel_kilograms',
    'planned_flight_fuel_kilograms',
    'uplifted_fuel_kilograms',
    'estimated_takeoff_weight_kilograms'
]

# Tạo bảng mô tả thống kê
desc_stats = merged_df[columns_to_describe].describe().T

# Thêm cột IQR (interquartile range)
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']

# Làm tròn số cho dễ đọc
desc_stats = desc_stats.round(2)

# Hiển thị bảng thống kê mô tả
print(desc_stats)
```

### 2. Data Cleaning
#### a. Column Normalization
- Standardized all column names: 
  - Stripped whitespace
  - Converted to lowercase
  - Replaced spaces with underscores

#### b. Duplicate Removal
- Dropped any exact duplicate rows

#### c. Handling Missing Values
- Checked for missing values
- Dropped rows where essential fields like `flight_id`, `actual_flight_fuel_kilograms`, or `planned_flight_fuel_kilograms` were missing.

#### d. Data Type Conversion
- Converted date/time columns (e.g., `departure_time`, `arrival_time`) to `datetime` objects

#### e. Filtering Invalid Values
- Removed rows with invalid fuel values (e.g., negative or zero)

#### f. Feature Engineering
- Created new columns for analysis:
  - `fuel_diff`: Difference between actual and planned fuel
  - `fuel_ratio`: Ratio of actual to planned fuel

#### g. Final Checks
- Used `.info()` and `.describe()` to inspect the cleaned dataset structure and summary statistics.

```python
import pandas as pd

# 1. Đổi tên cột cho đồng nhất và dễ xử lý
merged_df.columns = merged_df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Kiểm tra và loại bỏ các dòng bị duplicate (nếu có)
merged_df = merged_df.drop_duplicates()

# 3. Kiểm tra và xử lý missing values
missing_summary = merged_df.isnull().sum()
print("Missing values:\n", missing_summary[missing_summary > 0])

merged_df = merged_df.dropna(subset=['flight_id', 'actual_flight_fuel_kilograms', 'planned_flight_fuel_kilograms'])

# 4. Chuyển đổi kiểu dữ liệu
if 'departure_time' in merged_df.columns:
    merged_df['departure_time'] = pd.to_datetime(merged_df['departure_time'], errors='coerce')
if 'arrival_time' in merged_df.columns:
    merged_df['arrival_time'] = pd.to_datetime(merged_df['arrival_time'], errors='coerce')

# 5. Loại bỏ các giá trị không hợp lệ (âm hoặc quá lớn)
merged_df = merged_df[(merged_df['actual_flight_fuel_kilograms'] > 0) & 
                      (merged_df['planned_flight_fuel_kilograms'] > 0)]

# 6. Tạo cột mới 
merged_df['fuel_diff'] = merged_df['actual_flight_fuel_kilograms'] - merged_df['planned_flight_fuel_kilograms']
merged_df['fuel_ratio'] = merged_df['actual_flight_fuel_kilograms'] / merged_df['planned_flight_fuel_kilograms']

# Kiểm tra lại kết quả
print(merged_df.info())
print(merged_df.describe())
```


### 3. Data Visualization & Insights
#### a. Fuel Consumption vs. Flight Distance
![image](https://github.com/user-attachments/assets/b856c464-735b-4e81-bacb-1f5187e465fb)
**Insights:**
- There is a clear positive correlation between **flight distance** and **fuel consumption**, confirming the intuitive relationship that longer flights require more fuel.
- Several short-distance flights appear to consume unusually high fuel amounts, suggesting **possible inefficiencies**, such as long taxiing times, reroutes, or over-fueling.
- The spread becomes wider at longer distances, indicating **variability in efficiency** that could depend on aircraft type or flight conditions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot: Flight Distance vs Actual Fuel Consumption
plt.figure(figsize=(10, 6))  # Optional: set figure size
sns.scatterplot(data=merged_df, x='air_distance_miles', y='actual_flight_fuel_kilograms')

plt.title('Fuel Consumption vs. Flight Distance')
plt.xlabel('Flight Distance (miles)')
plt.ylabel('Actual Fuel Consumption (kg)')
plt.tight_layout()
plt.show()
```
---

#### b. Uplifted Fuel vs. Estimated Takeoff Weight
![image](https://github.com/user-attachments/assets/b5513631-65e7-4ece-b37a-8ac0a55137b2)
**Insights:**
- Fuel uplift generally **increases with estimated takeoff weight**, which aligns with operational expectations.
- A number of outliers exist where heavy flights received relatively low fuel uplift or vice versa — these anomalies may point to **planning inaccuracies**, equipment constraints, or specific route considerations.
- A tighter alignment could reduce safety margins and signal **potential optimization opportunities**.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot: Estimated Takeoff Weight vs Uplifted Fuel
plt.figure(figsize=(10, 6))  # Tùy chọn: thiết lập kích thước biểu đồ
sns.scatterplot(
    data=merged_df,
    x='estimated_takeoff_weight_kilograms',
    y='uplifted_fuel_kilograms'
)

plt.title('Uplifted Fuel vs. Estimated Takeoff Weight')
plt.xlabel('Estimated Takeoff Weight (kg)')
plt.ylabel('Uplifted Fuel (kg)')
plt.tight_layout()
plt.show()

```
---

#### c. Actual vs. Planned Fuel
![image](https://github.com/user-attachments/assets/639a2c0b-9f49-41a6-9c36-4f74f9fcb60f)
**Insights:**
- The boxplot reveals that **planned fuel quantities tend to exceed actual consumption**, with a large portion of flights consuming **less fuel than planned**.
- This over-planning could lead to **excess operational costs** and unnecessary weight during takeoff.
- Some flights consumed **more than planned**, indicating **risk areas** that should be reviewed for route changes, unexpected delays, or misestimates in planning.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bước 1: Tạo DataFrame dạng long format để vẽ biểu đồ
fuel_df = merged_df[['actual_flight_fuel_kilograms', 'planned_flight_fuel_kilograms']].rename(
    columns={
        'actual_flight_fuel_kilograms': 'Actual Fuel',
        'planned_flight_fuel_kilograms': 'Planned Fuel'
    }
).melt(var_name='Type', value_name='Fuel (kg)')

# Bước 2: Vẽ biểu đồ boxplot + stripplot
plt.figure(figsize=(8, 6))

# Vẽ boxplot
sns.boxplot(data=fuel_df, x='Type', y='Fuel (kg)', palette='pastel', width=0.4)

# Vẽ stripplot để thấy phân phối dữ liệu
sns.stripplot(data=fuel_df, x='Type', y='Fuel (kg)', color='gray', alpha=0.4, jitter=0.25, size=2)

# Thêm tiêu đề và nhãn
plt.title('Distribution of Actual vs. Planned Fuel')
plt.xlabel('Fuel Type')
plt.ylabel('Fuel (kg)')
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
```
---

#### d. Fuel Consumption Over Time
![image](https://github.com/user-attachments/assets/3808e527-b09f-46fa-adf8-d3ad9d56bc97)
**Insights:**
- Daily fuel usage shows high fluctuation, indicating varying flight activity.
- The 7-day rolling average highlights recurring weekly patterns.
- A noticeable rise in consumption mid-period suggests seasonal demand or increased operations.
- The drop at the end may reflect reduced activity or improved fuel efficiency.

```python
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Chuyển cột thời gian cất cánh sang định dạng datetime
actual_df['actual_time_departure'] = pd.to_datetime(actual_df['actual_time_departure'])

# ✅ Trích xuất phần ngày (loại bỏ giờ)
actual_df['date'] = actual_df['actual_time_departure'].dt.date

# ✅ Tính tổng nhiên liệu theo từng ngày
daily_fuel = actual_df.groupby('date')['actual_flight_fuel_kilograms'].sum().reset_index()

# ✅ Đổi tên cột cho rõ ràng (nếu cần)
daily_fuel.columns = ['date', 'total_fuel_kg']

# ✅ Tính trung bình trượt 7 ngày
daily_fuel['rolling_avg_7d'] = daily_fuel['total_fuel_kg'].rolling(window=7).mean()

# ✅ Vẽ biểu đồ
plt.figure(figsize=(14, 7))

# Biểu đồ đường cho tổng nhiên liệu mỗi ngày
plt.plot(daily_fuel['date'], daily_fuel['total_fuel_kg'], label='Daily Fuel Consumption', marker='o')

# Biểu đồ đường cho trung bình trượt
plt.plot(daily_fuel['date'], daily_fuel['rolling_avg_7d'], label='7-Day Rolling Average', linewidth=3, color='orange')

# Thiết lập biểu đồ
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

### 4. Opportunities for Improvement
- Improve prediction of planned fuel for better resource planning.
- Identify patterns leading to underestimation or overestimation.
