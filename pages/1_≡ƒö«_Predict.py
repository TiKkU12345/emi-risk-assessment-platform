import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_models, predict, CATEGORY_OPTIONS

st.set_page_config(page_title="Predict - EMI Risk Assessment", page_icon="🔮", layout="wide")
st.title("🔮 Real-Time EMI Prediction")
st.markdown("Enter an applicant's details below to get an instant eligibility assessment and maximum affordable EMI estimate.")

@st.cache_resource
def get_models():
    return load_models()

clf_model, reg_model, clf_cols, reg_cols = get_models()

with st.form("prediction_form"):
    st.subheader("Personal Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=70, value=35)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS['gender'])
    with c2:
        marital_status = st.selectbox("Marital Status", CATEGORY_OPTIONS['marital_status'])
        education = st.selectbox("Education", CATEGORY_OPTIONS['education'])
    with c3:
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=8, value=1)

    st.subheader("Employment & Income")
    c1, c2, c3 = st.columns(3)
    with c1:
        employment_type = st.selectbox("Employment Type", CATEGORY_OPTIONS['employment_type'])
        company_type = st.selectbox("Company Type", CATEGORY_OPTIONS['company_type'])
    with c2:
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        monthly_salary = st.number_input("Monthly Salary (Rs.)", min_value=0, max_value=1000000, value=50000, step=1000)
    with c3:
        house_type = st.selectbox("House Type", CATEGORY_OPTIONS['house_type'])
        monthly_rent = st.number_input("Monthly Rent (Rs.)", min_value=0, max_value=100000, value=0, step=500)

    st.subheader("Monthly Expenses")
    c1, c2, c3 = st.columns(3)
    with c1:
        school_fees = st.number_input("School Fees (Rs.)", min_value=0, max_value=50000, value=0, step=500)
        college_fees = st.number_input("College Fees (Rs.)", min_value=0, max_value=50000, value=0, step=500)
    with c2:
        travel_expenses = st.number_input("Travel Expenses (Rs.)", min_value=0, max_value=50000, value=3000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (Rs.)", min_value=0, max_value=100000, value=12000, step=500)
    with c3:
        other_monthly_expenses = st.number_input("Other Monthly Expenses (Rs.)", min_value=0, max_value=50000, value=5000, step=500)

    st.subheader("Financial Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        existing_loans = st.selectbox("Existing Loans?", CATEGORY_OPTIONS['existing_loans'])
        current_emi_amount = st.number_input("Current EMI Amount (Rs.)", min_value=0, max_value=100000, value=0, step=500)
    with c2:
        credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900, value=700)
        bank_balance = st.number_input("Bank Balance (Rs.)", min_value=0, max_value=5000000, value=100000, step=5000)
    with c3:
        emergency_fund = st.number_input("Emergency Fund (Rs.)", min_value=0, max_value=2000000, value=50000, step=5000)

    st.subheader("Loan Application")
    c1, c2, c3 = st.columns(3)
    with c1:
        emi_scenario = st.selectbox("Loan Type", CATEGORY_OPTIONS['emi_scenario'])
    with c2:
        requested_amount = st.number_input("Requested Amount (Rs.)", min_value=1000, max_value=2000000, value=200000, step=5000)
    with c3:
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=24)

    submitted = st.form_submit_button("🔍 Get Prediction", use_container_width=True)

if submitted:
    raw_input = {
        'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
        'monthly_salary': monthly_salary, 'employment_type': employment_type,
        'years_of_employment': years_of_employment, 'company_type': company_type,
        'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
        'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
        'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
        'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
        'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
    }

    result = predict(raw_input, clf_model, reg_model, clf_cols, reg_cols)

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        color_map = {'Eligible': 'green', 'High_Risk': 'orange', 'Not_Eligible': 'red'}
        st.markdown(f"**Eligibility Status**")
        st.markdown(f"## :{color_map[result['eligibility']]}[{result['eligibility'].replace('_', ' ')}]")
        st.caption(f"Confidence: {result['eligibility_confidence']*100:.1f}%")

        st.markdown("**Class Probabilities**")
        for label, prob in result['probabilities'].items():
            st.progress(prob, text=f"{label.replace('_', ' ')}: {prob*100:.1f}%")

    with col2:
        st.markdown(f"**Maximum Affordable Monthly EMI**")
        st.markdown(f"## Rs. {result['max_monthly_emi']:,.0f}")
        if requested_tenure > 0:
            implied_loan_capacity = result['max_monthly_emi'] * requested_tenure
            st.caption(f"Over the requested {requested_tenure}-month tenure, this implies a total repayment capacity of roughly Rs. {implied_loan_capacity:,.0f}")

    if result['eligibility'] == 'Not_Eligible':
        st.warning("⚠️ This applicant is predicted **Not Eligible**. Consider a smaller loan amount, a co-applicant, or improving the expense-to-income ratio before reapplying.")
    elif result['eligibility'] == 'High_Risk':
        st.warning("⚠️ This applicant falls in the **High Risk** category — recommend manual underwriter review before approval.")
    else:
        st.success("✅ This applicant is predicted **Eligible**.")
