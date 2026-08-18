import streamlit as st

st.set_page_config(
    page_title="EMI Risk Assessment Platform",
    page_icon="💳",
    layout="wide"
)

st.title("💳 EMI Risk Assessment Platform")
st.markdown("### Financial Risk Assessment for Loan/EMI Applications")

st.markdown("""
This platform uses machine learning to assess loan applicants across two dimensions:

1. **EMI Eligibility** — classifies an applicant as `Eligible`, `High_Risk`, or `Not_Eligible`
2. **Maximum Affordable EMI** — estimates the maximum monthly EMI an applicant can realistically sustain

Trained on 404,800 historical applications across 5 loan scenarios (Personal Loan, Vehicle EMI, Education EMI, Home Appliances EMI, and E-commerce Shopping EMI).
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Records", "404,800")
col2.metric("Classification Macro F1", "0.835")
col3.metric("Regression R²", "0.993")
col4.metric("Loan Scenarios", "5")

st.divider()

st.markdown("""
### How to use this app

Use the sidebar to navigate:

- **🔮 Predict** — enter a new applicant's details and get real-time eligibility + max EMI predictions
- **📋 Data Explorer** — view, add, edit, and delete applicant records (CRUD)
- **📊 Model Insights** — explore what drives the model's predictions and how the models compare

---

**Built with:** scikit-learn, XGBoost, MLflow (experiment tracking), Streamlit
""")

st.info("💡 This tool supports lending decisions with a data-driven estimate — it does not replace human underwriting judgment.", icon="💡")
