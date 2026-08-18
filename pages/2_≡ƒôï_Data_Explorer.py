import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(page_title="Data Explorer - EMI Risk Assessment", page_icon="📋", layout="wide")
st.title("📋 Applicant Data Explorer")
st.markdown("Add, view, edit, and delete applicant records. This is a separate working database for new applications — not the original 400K-row training dataset.")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'applicants.db')


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applicants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            monthly_salary REAL,
            credit_score REAL,
            requested_amount REAL,
            emi_scenario TEXT,
            predicted_eligibility TEXT,
            predicted_max_emi REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_all_records():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM applicants ORDER BY id DESC", conn)
    conn.close()
    return df


def add_record(record: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO applicants (name, age, monthly_salary, credit_score, requested_amount,
                                 emi_scenario, predicted_eligibility, predicted_max_emi, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (record['name'], record['age'], record['monthly_salary'], record['credit_score'],
          record['requested_amount'], record['emi_scenario'], record['predicted_eligibility'],
          record['predicted_max_emi'], record['notes']))
    conn.commit()
    conn.close()


def update_record(record_id: int, record: dict):
    conn = get_connection()
    conn.execute("""
        UPDATE applicants SET name=?, age=?, monthly_salary=?, credit_score=?, requested_amount=?,
                               emi_scenario=?, notes=?
        WHERE id=?
    """, (record['name'], record['age'], record['monthly_salary'], record['credit_score'],
          record['requested_amount'], record['emi_scenario'], record['notes'], record_id))
    conn.commit()
    conn.close()


def delete_record(record_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM applicants WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


init_db()

tab1, tab2, tab3 = st.tabs(["📄 View Records", "➕ Add Record", "✏️ Edit / Delete"])

with tab1:
    df = get_all_records()
    if df.empty:
        st.info("No records yet. Add one from the 'Add Record' tab, or run a prediction on the Predict page and save it here.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} record(s) total")

with tab2:
    with st.form("add_record_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Applicant Name")
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            monthly_salary = st.number_input("Monthly Salary (Rs.)", min_value=0, value=50000, step=1000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        with c2:
            requested_amount = st.number_input("Requested Amount (Rs.)", min_value=1000, value=200000, step=5000)
            emi_scenario = st.selectbox("Loan Type", ['E-commerce Shopping EMI', 'Education EMI',
                                                        'Home Appliances EMI', 'Personal Loan EMI', 'Vehicle EMI'])
            notes = st.text_area("Notes (optional)")

        add_submitted = st.form_submit_button("Add Record")
        if add_submitted:
            if not name.strip():
                st.error("Please enter a name.")
            else:
                add_record({
                    'name': name, 'age': age, 'monthly_salary': monthly_salary,
                    'credit_score': credit_score, 'requested_amount': requested_amount,
                    'emi_scenario': emi_scenario, 'predicted_eligibility': None,
                    'predicted_max_emi': None, 'notes': notes
                })
                st.success(f"Added record for {name}.")
                st.rerun()

with tab3:
    df = get_all_records()
    if df.empty:
        st.info("No records to edit yet.")
    else:
        record_id = st.selectbox("Select a record to edit or delete", df['id'].tolist(),
                                   format_func=lambda x: f"#{x} - {df[df['id']==x]['name'].values[0]}")
        selected = df[df['id'] == record_id].iloc[0]

        with st.form("edit_record_form"):
            c1, c2 = st.columns(2)
            with c1:
                e_name = st.text_input("Applicant Name", value=selected['name'])
                e_age = st.number_input("Age", min_value=18, max_value=70, value=int(selected['age']))
                e_salary = st.number_input("Monthly Salary (Rs.)", min_value=0, value=int(selected['monthly_salary']), step=1000)
                e_credit = st.number_input("Credit Score", min_value=300, max_value=900, value=int(selected['credit_score']))
            with c2:
                e_amount = st.number_input("Requested Amount (Rs.)", min_value=1000, value=int(selected['requested_amount']), step=5000)
                e_scenario = st.selectbox("Loan Type", ['E-commerce Shopping EMI', 'Education EMI',
                                                          'Home Appliances EMI', 'Personal Loan EMI', 'Vehicle EMI'],
                                            index=['E-commerce Shopping EMI', 'Education EMI', 'Home Appliances EMI',
                                                   'Personal Loan EMI', 'Vehicle EMI'].index(selected['emi_scenario']))
                e_notes = st.text_area("Notes", value=selected['notes'] if selected['notes'] else "")

            col1, col2 = st.columns(2)
            update_submitted = col1.form_submit_button("💾 Save Changes", use_container_width=True)
            delete_submitted = col2.form_submit_button("🗑️ Delete Record", use_container_width=True, type="secondary")

            if update_submitted:
                update_record(record_id, {
                    'name': e_name, 'age': e_age, 'monthly_salary': e_salary,
                    'credit_score': e_credit, 'requested_amount': e_amount,
                    'emi_scenario': e_scenario, 'notes': e_notes
                })
                st.success("Record updated.")
                st.rerun()

            if delete_submitted:
                delete_record(record_id)
                st.success("Record deleted.")
                st.rerun()
