# House Price Prediction — Regression Model Comparison & Web App

— Course: Intelligent AI and Data Engineering
Dataset: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Ames Housing) | Target: `SalePrice`

A full machine learning workflow that cleans and explores the Ames Housing dataset, trains and compares 7+ regression algorithms under one shared preprocessing pipeline, evaluates them with MAE / MSE / RMSE / R², and deploys the best model in a Streamlit web app.

## Features

- Full EDA: distributions, boxplots, scatter plots, correlation heatmap, outlier detection
- Leak-safe preprocessing pipeline (imputation, scaling, one-hot encoding) shared across all models
- 7 required algorithms — Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR
- Bonus algorithms — XGBoost, Extra Trees, AdaBoost, LightGBM
- Model comparison table + charts, feature importance, cross-validation, residual analysis
- Saved best model (`.joblib`) ready for inference
- Streamlit app with Home, Project Description, Dataset Info, Model Info, Prediction, Model Comparison, and Team pages

## Before You Run This for Submission

## Project Structure

```
.
├── House_Price_Regression_Project.ipynb   # Full ML workflow (EDA -> models -> save)
├── app/
│   └── streamlit_app.py                   # Web application (7 pages)
├── data/
│   └── train.csv                          # Dataset (replace with real Kaggle file)
├── model/
│   ├── best_house_price_model.joblib      # Saved best pipeline (created by notebook)
│   ├── metadata.joblib                    # Feature lists + best model metrics
│   └── results_comparison.csv             # Model comparison table (for the app)
├── requirements.txt
└── README.md
```
## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook House_Price_Prediction_Project.ipynb
```

Run all cells top to bottom. This will:
- Load and clean the data
- Perform EDA
- Train & compare Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, plus bonus models (XGBoost, Extra Trees, AdaBoost, LightGBM)
- Evaluate every model with MAE, MSE, RMSE, R²
- Tune and save the best model to `model/best_house_price_model.joblib`

## Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Pages: Home, Project Description, Dataset Information, Model Information, Prediction, Model Comparison, Team Members.

## Model Comparison

*(Fill in after running on real data — pull these numbers from `model/results_comparison.csv`)*

| Algorithm | Training Time (s) | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| | | | | | |

## Best Model Selection

*State which model won and why, referencing MAE/MSE/RMSE/R² and training time — not R² alone.*

## Team Members

| Name | Role |
|---|---|
| | |

## Dataset Citation

Kaggle, *House Prices: Advanced Regression Techniques* (Ames Housing dataset, compiled by Dean De Cock).

## License

*Add a license if required by your course (e.g. MIT).*
