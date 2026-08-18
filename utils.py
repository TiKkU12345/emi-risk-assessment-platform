"""
Shared utilities for loading models and running predictions.
Used across all pages of the EMI Risk Assessment app.
"""
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAT_COLS = ['gender', 'marital_status', 'education', 'employment_type', 'company_type',
            'house_type', 'existing_loans', 'emi_scenario']
NUM_COLS = ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
            'dependents', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
            'other_monthly_expenses', 'current_emi_amount', 'credit_score', 'bank_balance',
            'emergency_fund', 'requested_amount', 'requested_tenure', 'total_expenses',
            'disposable_income', 'expense_to_income', 'debt_to_income']

CATEGORY_OPTIONS = {
    'gender': ['Male', 'Female'],
    'marital_status': ['Married', 'Single'],
    'education': ['Graduate', 'High School', 'Post Graduate', 'Professional', 'Unknown'],
    'employment_type': ['Government', 'Private', 'Self-employed'],
    'company_type': ['Large Indian', 'MNC', 'Mid-size', 'Small', 'Startup'],
    'house_type': ['Family', 'Own', 'Rented'],
    'existing_loans': ['No', 'Yes'],
    'emi_scenario': ['E-commerce Shopping EMI', 'Education EMI', 'Home Appliances EMI',
                      'Personal Loan EMI', 'Vehicle EMI'],
}


def load_models():
    """Load both trained models and their expected feature column order."""
    clf_model = joblib.load(os.path.join(BASE_DIR, 'best_classification_model.pkl'))
    reg_model = joblib.load(os.path.join(BASE_DIR, 'best_regression_model.pkl'))
    clf_cols = joblib.load(os.path.join(BASE_DIR, 'classification_feature_columns.pkl'))
    reg_cols = joblib.load(os.path.join(BASE_DIR, 'regression_feature_columns.pkl'))
    return clf_model, reg_model, clf_cols, reg_cols


def build_feature_row(raw_input: dict) -> pd.DataFrame:
    """
    Take raw user-entered fields (matching original dataset columns) and produce
    the engineered + one-hot-encoded feature row the models expect.
    """
    row = raw_input.copy()

    total_expenses = (row['monthly_rent'] + row['school_fees'] + row['college_fees'] +
                       row['travel_expenses'] + row['groceries_utilities'] +
                       row['other_monthly_expenses'] + row['current_emi_amount'])
    row['total_expenses'] = total_expenses
    row['disposable_income'] = row['monthly_salary'] - total_expenses
    row['debt_to_income'] = row['current_emi_amount'] / row['monthly_salary'] if row['monthly_salary'] else 0
    row['expense_to_income'] = total_expenses / row['monthly_salary'] if row['monthly_salary'] else 0

    df_row = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df_row, columns=CAT_COLS)
    return df_encoded


def align_columns(df_encoded: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """Reindex the encoded row to match the exact column set/order the model was trained on."""
    return df_encoded.reindex(columns=expected_cols, fill_value=0)


def predict(raw_input: dict, clf_model, reg_model, clf_cols, reg_cols):
    """Run both predictions on a single raw applicant input dict."""
    df_encoded = build_feature_row(raw_input)

    X_clf = align_columns(df_encoded, clf_cols)
    X_reg = align_columns(df_encoded, reg_cols)

    eligibility_idx = clf_model.predict(X_clf)[0]
    eligibility_proba = clf_model.predict_proba(X_clf)[0]
    label_map = {0: 'Not_Eligible', 1: 'High_Risk', 2: 'Eligible'}
    eligibility = label_map[eligibility_idx]

    max_emi = reg_model.predict(X_reg)[0]

    return {
        'eligibility': eligibility,
        'eligibility_confidence': float(eligibility_proba[eligibility_idx]),
        'probabilities': {label_map[i]: float(p) for i, p in enumerate(eligibility_proba)},
        'max_monthly_emi': float(max_emi),
    }
