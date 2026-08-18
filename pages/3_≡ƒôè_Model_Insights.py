import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Insights - EMI Risk Assessment", page_icon="📊", layout="wide")
st.title("📊 Model Insights")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, 'model_insights.json')) as f:
    insights = json.load(f)

st.subheader("Model Comparison")
tab1, tab2 = st.tabs(["Classification (Eligibility)", "Regression (Max EMI)"])

with tab1:
    clf_df = pd.DataFrame(insights['model_comparison']['classification'])
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(clf_df, x='model', y='macro_f1', color='model',
                     title='Macro F1 Score by Model', text_auto='.3f')
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(clf_df.set_index('model'), use_container_width=True)
    st.info("**XGBoost was selected as the final model** — it raised `High_Risk` precision from 14% to 43% while keeping recall at 92%, and trained faster than Random Forest despite the better result. Hyperparameter tuning was attempted but did not beat this baseline, and that negative result is reported rather than hidden.")

with tab2:
    reg_df = pd.DataFrame(insights['model_comparison']['regression'])
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(reg_df, x='model', y='r2', color='model',
                     title='R² Score by Model', text_auto='.3f')
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(reg_df.set_index('model'), use_container_width=True)
    st.info("**XGBoost was selected as the final model** — R² = 0.993 with an average error of just ~Rs.259, while training roughly 7x faster than Random Forest.")

st.divider()

st.subheader("What Drives Each Prediction?")
tab3, tab4 = st.tabs(["Eligibility Drivers", "Max EMI Drivers"])

with tab3:
    clf_feat = pd.Series(insights['classification_top_features']).sort_values()
    fig = px.bar(x=clf_feat.values, y=clf_feat.index, orientation='h',
                 title='Top 15 Features - Eligibility Classification', labels={'x': 'Importance', 'y': ''})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("`disposable_income` is the single strongest driver — an applicant's income left over after expenses matters more than income or expenses alone, confirming the ratio-based feature engineering done during EDA.")

with tab4:
    reg_feat = pd.Series(insights['regression_top_features']).sort_values()
    fig = px.bar(x=reg_feat.values, y=reg_feat.index, orientation='h',
                 title='Top 15 Features - Max EMI Regression', labels={'x': 'Importance', 'y': ''})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Dataset Overview")
stats = insights['dataset_stats']
c1, c2 = st.columns(2)
with c1:
    st.metric("Total Training Records", f"{stats['total_records']:,}")
with c2:
    elig_df = pd.DataFrame(list(stats['eligibility_distribution'].items()), columns=['Class', 'Percent'])
    fig = px.pie(elig_df, names='Class', values='Percent', title='Eligibility Class Distribution (Training Data)',
                 color='Class', color_discrete_map={'Eligible': '#55a868', 'High_Risk': '#dd8452', 'Not_Eligible': '#c44e52'})
    st.plotly_chart(fig, use_container_width=True)

st.warning("**Known limitation:** the `High_Risk` class makes up only 4.3% of training data. Despite class-weighted training, predictions for this class should be treated as a *flag for review*, not a final decision — the same caution applied throughout this project's evaluation.")
