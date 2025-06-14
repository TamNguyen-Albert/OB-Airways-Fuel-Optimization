
# ✈️ OB Airways Fuel Optimization

This project analyzes and predicts planned flight fuel consumption using machine learning models based on OB Airways flight data.

## 📂 Structure
- `report/`: Project report (Markdown format)
- `notebooks/`: Step-by-step implementation in Jupyter
- `data/`: Input datasets (CSV)
- `models/`: Trained models (optional)

## 🔧 Setup
```bash
pip install -r requirements.txt
```

## 💻 Run the notebooks
Use `notebooks/` in order: 1 → 4 for full pipeline.

## 📊 Best Model
**StackingRegressor (XGBoost + Linear)**  
- MAE: 393.88 kg  
- RMSE: 772.12 kg  
- R²: 0.9943
