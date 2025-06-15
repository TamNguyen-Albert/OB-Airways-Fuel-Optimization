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
