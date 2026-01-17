# Model Card: Sales Forecasting Model (LightGBM)

## 1. Model Overview
This model forecasts daily units sold for multiple SKUs over a 7-day horizon using a recursive forecasting approach.  
It is trained as a **global model**, meaning a single model is used to learn patterns across all SKUs.

The model is designed to support short-term sales forecasting and operational planning.

---

## 2. Intended Use
**Intended uses:**
- Short-term (up to 7 days) sales forecasting
- Inventory planning and replenishment decisions
- Demand trend analysis across multiple SKUs

**Not intended for:**
- Long-term forecasting beyond the trained horizon
- SKUs with no historical sales data
- One-time or highly irregular demand events (e.g., promotions, stockouts not seen in training)

---

## 3. Training Data
- Historical daily sales data
- Approximately 30 SKUs
- Target variable: `units_sold`
- Typical sales range: 30 to 300 units (occasionally up to 450–500)
- Data is sorted by SKU and date to preserve temporal order

Strict care was taken to prevent **data leakage** during feature creation and validation.

---

## 4. Feature Engineering
The model uses only information available up to the prediction date.

Key feature groups include:
- Lag features (previous day sales)
- Rolling statistics (e.g., rolling means)
- Custom features derived from historical sales patterns
- Temporal features (day of week, month, etc.)

All features are generated consistently during training, validation, and inference.

---

## 5. Model Architecture
- Algorithm: LightGBM Regressor
- Learning type: Supervised regression
- Training strategy: Global model across all SKUs
- Forecasting strategy: Recursive (predictions are fed back to generate future features)

LightGBM was chosen for its strong performance, efficiency, and suitability for tabular time series data.

---

## 6. Evaluation Methodology
- Validation strategy: 5-fold walk-forward (time-based) validation
- Metrics used:
  - Mean Absolute Error (MAE)
  - R² score
- Evaluation performed separately for each forecast horizon day

This approach closely simulates real-world forecasting scenarios.

---

## 7. Performance Summary
- Validation MAE: ~9 units
- Test MAE: ~9 units
- R² score: ~0.88

Given the sales scale and number of SKUs, these results indicate strong and consistent predictive performance.

---

## 8. Limitations
- Recursive forecasting can cause error accumulation across horizon days
- Model performance may degrade if sales patterns change significantly (data drift)
- Less reliable for SKUs with very limited historical data

Regular retraining is recommended to maintain performance.

---

## 9. Ethical and Practical Considerations
- Forecasts should be used as decision support, not as the sole decision-making tool
- Human review is recommended for high-impact business decisions
- The model does not explicitly account for external factors such as promotions or holidays unless present in the data

---

## 10. Model Artifacts
The following artifacts are saved and reused during inference:
- Trained LightGBM model
- Feature engineering functions
- Feature column list
- Last available historical data used for recursive forecasting

---

## 11. Model Version
- Version: 1.0
- Author: <Deyoz Rayamajhi>
- Date: <2026-01-16>
