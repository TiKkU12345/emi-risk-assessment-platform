# EMI Risk Assessment Platform ||  [Live](https://emi-risk-assessment-platform.streamlit.app/%E2%89%A1%C6%92%C3%B6%C2%AB_Predict)


A Streamlit web application for predicting loan/EMI eligibility and maximum affordable monthly EMI, using models trained on 404,800 historical applications.

## Features

- **🔮 Predict** — real-time eligibility (Eligible / High_Risk / Not_Eligible) and max EMI prediction for a new applicant
- **📋 Data Explorer** — full CRUD (Create, Read, Update, Delete) on applicant records, backed by SQLite
- **📊 Model Insights** — model comparison charts and feature importance, explaining what drives each prediction

## Results

| Task | Best Model | Score |
|---|---|---|
| Eligibility Classification | XGBoost | Macro F1 = 0.835 |
| Max EMI Regression | XGBoost | R² = 0.993, MAE = Rs. 259 |

Full data cleaning, EDA, and model comparison (including honest reporting of a hyperparameter tuning attempt that didn't help) are in the companion notebooks: `EMI_EDA.ipynb`, `EMI_Classification_Modeling.ipynb`, `EMI_Regression_Modeling.ipynb`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project structure

```
├── app.py                          # Home page
├── pages/
│   ├── 1_🔮_Predict.py             # Real-time prediction form
│   ├── 2_📋_Data_Explorer.py        # CRUD on applicant records (SQLite)
│   └── 3_📊_Model_Insights.py       # Model comparison & feature importance
├── utils.py                        # Shared model loading & prediction logic
├── best_classification_model.pkl   # Trained XGBoost classifier
├── best_regression_model.pkl       # Trained XGBoost regressor
├── classification_feature_columns.pkl
├── regression_feature_columns.pkl
├── model_insights.json             # Precomputed feature importances & comparison metrics
└── requirements.txt
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected to this GitHub repository. Any push to `main` auto-redeploys.

## Tech stack

Python, scikit-learn, XGBoost, MLflow (experiment tracking during model development), Streamlit.
