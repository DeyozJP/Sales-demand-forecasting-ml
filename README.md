#### Sales Forecasting Using Machine Learning (Recursive Forecasting)
##### Project Overview

This project focuses on forecasting daily unit sales for multiple SKUs over a 7-day horizon using machine learning–based recursive forecasting.

A global model is trained across multiple time series (SKUs), leveraging carefully engineered lag, rolling, temporal, and domain-specific features. The project emphasizes:

- Preventing data leakage
- Proper walk-forward validation
- Recursive inference logic
- Robust model evaluation
- Production-ready inference workflow
The final model achieves low MAE (~9 units) and high R² (~0.88) on the test set, demonstrating strong generalization performance.

##### Project Structure
```sales-forecasting/
│
├── data/
│   ├── sales_dataset.csv (for EDA, feature engineering, training, validations and testing)
│   └── sales_forecasting_input.csv (for inference)
│
├── models/
│   └── lgbm_model.pkl
│
├── src/
│   ├── helper_functions.py
│   ├── models.py
│   └── config.py
│
├── notebooks/
│   └── sales_forecasting.ipynb
│
├── README.md
└── requirements.txt


##### Problem Statement
Given historical sales data for multiple SKUs, the objective is to:
- Forecast daily units sold for the next 7 days
- Handle multiple time series simultaneously
- Use recursive forecasting, where predictions feed future steps
- Build a model that generalizes well across SKUs

##### Feature Engineering

The following features are created are used:
- Previous day sales (lag_1)
- 7-day moving average of sales
- 30-day moving average of sales
- Sell-through rate (sales / stock available)
- Day of week
- Day of month
- Week number
- Month
All features are created without future information, ensuring no data leakage.

##### Forecasting Strategy
This project uses **recursive (iterative) forecasting**:
1. Train model using historical data
2. Predict day +1
3. Append prediction to history
4. Recompute features
5. Predict day +2
6. Repeat until horizon = 7 days
This approach mirrors **real-world deployment conditions**.

##### Model Training & Validation
**Walk-Forward (Rolling Origin) Validation**
- 5-fold walk-forward validation
- Each fold simulates real forecasting conditions
- Metrics computed per horizon day

**Models Evaluated**
- Linear Regression
- Decision Tree
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- LightGBM
- Seasonal Naive (baseline)

##### Evaluation Metrics
- MAE (Mean Absolute Error)
- R² Score

*Key Results*
- Ensemble models achieve MAE ≈ 9–12
- R² consistently above 0.85
- Seasonal naive performs poorly, confirming learned patterns
- Linear regression performs surprisingly well, indicating strong feature quality

##### Final Model Selection
`LightGBM` was selected as the final model due to:
- Strong performance
- Stability across horizons
- Lightweight and production-friendly architecture
The model was retrained on the full training set and evaluated on a held-out test set with **consistent results**.

##### Inference Pipeline

For inference:
- The trained model is saved as a pickle file
- Required artifacts are stored:
- - Feature creation functions
- - Scaler
- - Feature column list
- - Last available history per SKU
- Batch inference uses recursive forecasting logic identical to validation
This ensures **training–inference consistency**.

##### Reliability & Best Practices
- No data leakage
- Strict time-based splits
- Feature alignment using reindex
- Recursive inference tested on unseen data
- Residual analysis confirms near-normal error distribution

##### How to Run

Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook notebooks/sales_forecasting.ipynb

Follow the notebook sections for:
- Training
- Validation
- Testing
- Inference


