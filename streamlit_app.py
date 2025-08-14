import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px

# -------------------------
# Config & paths
# -------------------------
st.set_page_config(page_title="Employee Mental Health Dashboard", page_icon="🧠", layout="wide")
BUNDLE_FILE = "mh_bundle.joblib"
LEGACY_MODEL_FILE = "mental_health_model.pkl"
RECORDS_FILE = "employee_records.csv"
MANAGER_PASSWORD = "admin123"

# -------------------------
# Load model bundle
# -------------------------
def load_model_bundle():
    if os.path.exists(BUNDLE_FILE):
        try:
            bundle = joblib.load(BUNDLE_FILE)
            if isinstance(bundle, dict) and 'model' in bundle:
                return bundle['model'], bundle.get('encoders', {}), bundle.get('features', None)
            return bundle, {}, getattr(bundle, "feature_names_in_", None)
        except Exception as e:
            st.warning(f"Failed to load {BUNDLE_FILE}: {e}")
    if os.path.exists(LEGACY_MODEL_FILE):
        try:
            mdl = joblib.load(LEGACY_MODEL_FILE)
            return mdl, {}, getattr(mdl, "feature_names_in_", None)
        except Exception as e:
            st.error(f"Failed to load fallback model file: {e}")
    return None, {}, None

model, encoders, feature_names = load_model_bundle()
if model is None:
    st.error("Model bundle not found. Place 'mh_bundle.joblib' or 'mental_health_model.pkl' in the app folder.")
    st.stop()

if feature_names is None:
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        st.error("Feature names missing. Please recreate bundle including 'features'.")
        st.stop()

# -------------------------
# Records handling
# -------------------------
def load_records():
    if os.path.exists(RECORDS_FILE):
        df = pd.read_csv(RECORDS_FILE, parse_dates=["date"])
        if 'department' in df.columns:
            df['department'] = df['department'].astype(str).str.strip().str.title()
        if 'employee_id' not in df.columns:
            df['employee_id'] = ""
        df['employee_id'] = df['employee_id'].astype(str).str.strip()
        return df
    else:
        return pd.DataFrame()

def append_record(record):
    try:
        df = pd.read_csv(RECORDS_FILE, dtype=str)
    except FileNotFoundError:
        df = pd.DataFrame(columns=record.keys())
    if 'employee_id' not in df.columns:
        df['employee_id'] = ""
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(RECORDS_FILE, index=False)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Access")
role = st.sidebar.selectbox("I'm a", ["Employee", "Manager"])
if role == "Manager":
    password = st.sidebar.text_input("Manager Password", type="password")
    manager_authenticated = password == MANAGER_PASSWORD
    if not manager_authenticated:
        st.sidebar.warning("Enter manager password to access dashboard.")
else:
    manager_authenticated = False

st.sidebar.markdown("---")
st.sidebar.write("💾 Submissions saved to `employee_records.csv`.")
st.sidebar.write("🔒 Change `MANAGER_PASSWORD` for production.")

# -------------------------
# Title
# -------------------------
st.title("🧠 Employee Mental Health - Prediction & Manager Dashboard")
st.markdown("Employees submit anonymous survey answers. Managers view aggregated company insights.")
st.markdown("---")

# -------------------------
# Employee form
# -------------------------
if role == "Employee":
    st.header("Quick Questionnaire")
    st.write("Answer the questions below. **Employee ID is required** so you can view past submissions.")

    form = st.form("employee_form")

    # Employee ID always shown first
    default_emp_id = st.session_state.get("employee_id", "")
    employee_id = form.text_input("Employee ID (Required)", value=default_emp_id,
                                  help="Enter your unique Employee ID").strip()

    department = form.text_input("Department / Team (optional)", value="General")

    inputs = {}
    cols = form.columns(2)

    def _norm(s: str) -> str:
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    # Render other feature fields
    for i, feat in enumerate(feature_names):
        if feat.lower() == "employee_id":
            continue  # We already have a dedicated Employee ID field

        col = cols[i % 2]
        label = feat.replace("_", " ").title()
        norm_feat = _norm(feat)
        is_no_employees = (
            norm_feat in {
                "noemployees", "noofemployees", "numberofemployees", "numemployees",
                "companysize", "employeesize"
            }
            or ("employees" in norm_feat and ("no" in norm_feat or "num" in norm_feat or "size" in norm_feat))
        )

        if feat in encoders:
            if feat.lower() == "gender":
                options = ["Male", "Female", "Other"]
                inputs[feat] = col.selectbox(label, options, key=f"emp_{feat}")
            elif is_no_employees:
                options = ["1-5", "6-25", "26-100", "101-500", "501-1000", "More than 1000"]
                inputs[feat] = col.selectbox("No. of Employees", options, key=f"emp_{feat}")
            else:
                options = list(encoders[feat].classes_)
                inputs[feat] = col.selectbox(label, options, key=f"emp_{feat}")
        else:
            if "age" in feat.lower():
                inputs[feat] = col.slider(label, 18, 65, 30, key=f"emp_{feat}")
            else:
                inputs[feat] = col.number_input(label, value=0, key=f"emp_{feat}")

    submitted = form.form_submit_button("Submit & Predict")

    if submitted:
        if not employee_id:
            st.warning("⚠️ Please enter your Employee ID before submitting.")
        else:
            try:
                st.session_state.employee_id = employee_id

                row = []
                decoded_row = {}
                for feat in feature_names:
                    if feat.lower() == "employee_id":
                        continue
                    val = inputs[feat]
                    decoded_row[feat] = val
                    if feat in encoders:
                        enc = encoders[feat]
                        import numpy as np
                        if val not in enc.classes_:
                            enc.classes_ = np.append(enc.classes_, val)
                        encoded = int(enc.transform([val])[0])
                        row.append(encoded)
                    else:
                        row.append(val)

                X_input = pd.DataFrame([row], columns=[f for f in feature_names if f.lower() != "employee_id"])
                pred = int(model.predict(X_input)[0])
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_input)[0][pred])

                record = decoded_row.copy()
                record.update({
                    "employee_id": employee_id,
                    "department": (department if department else "Unknown").strip().title(),
                    "risk": pred,
                    "probability": prob if prob is not None else "",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                })
                append_record(record)

                st.success("✅ Your submission has been saved.")

                all_records = load_records()
                my_records = all_records[all_records["employee_id"].str.strip() == employee_id]
                st.subheader("Your Past Submissions")
                if my_records.empty:
                    st.info("No past submissions found for this Employee ID.")
                else:
                    st.dataframe(my_records.sort_values("date", ascending=False).reset_index(drop=True))

                if pred == 1:
                    if prob is not None:
                        st.error(f"⚠️ Risk detected (probability {prob:.2f}). Consider seeking support.")
                    else:
                        st.error("⚠️ Risk detected. Consider seeking support.")
                else:
                    if prob is not None:
                        st.success(f"✅ Low risk (probability {prob:.2f}).")
                    else:
                        st.success("✅ Low risk.")

                st.caption("Your responses are saved for aggregated insights. Managers cannot see your ID here.")
            except Exception as e:
                st.exception(e)

# -------------------------
# Manager dashboard
# -------------------------
if role == "Manager" and manager_authenticated:
    st.header("Manager Dashboard — Company-wide Insights")
    df = load_records()
    if df.empty:
        st.info("No submissions yet.")
    else:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['risk'] = pd.to_numeric(df['risk'], errors='coerce').fillna(0).astype(int)

        st.sidebar.markdown("### Dashboard Filters")
        depts = sorted(df['department'].dropna().unique().tolist())
        sel_depts = st.sidebar.multiselect("Departments", options=depts, default=depts)
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )

        mask = df['department'].isin(sel_depts) & (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        filtered = df[mask]

        total = len(filtered)
        at_risk = filtered['risk'].sum()
        pct_risk = (at_risk / total * 100) if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Submissions", total)
        col2.metric("No. Employees At Risk", at_risk)
        col3.metric("% At Risk", f"{pct_risk:.1f}%")

        age_col = next((c for c in ["Age", "age"] if c in filtered.columns), None)
        if age_col:
            avg_age_risk = round(filtered[filtered['risk'] == 1][age_col].mean(), 1)
            col4.metric("Avg Age (At Risk)", avg_age_risk if not pd.isna(avg_age_risk) else "—")
        else:
            col4.metric("Avg Age (At Risk)", "N/A")

        st.markdown("---")

        st.subheader("Risk by Department")
        risk_by_dept = filtered.groupby('department').agg(total=('risk', 'count'), at_risk=('risk', 'sum')).reset_index()
        if not risk_by_dept.empty:
            risk_by_dept['pct_risk'] = risk_by_dept['at_risk'] / risk_by_dept['total'] * 100
            fig1 = px.bar(
                risk_by_dept.sort_values('pct_risk', ascending=False),
                x='department', y='pct_risk', text='pct_risk',
                labels={'pct_risk': 'Risk %', 'department': 'Department'}
            )
            fig1.update_layout(yaxis_ticksuffix="%", xaxis_title=None)
            st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Overall Risk Distribution")
        dist = filtered['risk'].value_counts().rename({0: 'Low Risk', 1: 'At Risk'}).reset_index()
        dist.columns = ['label', 'count']
        fig2 = px.pie(dist, values='count', names='label', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Trend: Submissions & Risk Over Time")
        trend = filtered.groupby(filtered['date'].dt.to_period('M')).agg(total=('risk', 'count'), at_risk=('risk', 'sum')).reset_index()
        if not trend.empty:
            trend['date'] = trend['date'].dt.to_timestamp()
            fig3 = px.line(trend, x='date', y=['total', 'at_risk'], labels={'value': 'Count', 'date': 'Month'})
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Cross Insights")
        if 'benefits' in filtered.columns:
            tmp = filtered.groupby('benefits').agg(total=('risk', 'count'), at_risk=('risk', 'sum')).reset_index()
            tmp['pct_risk'] = tmp['at_risk'] / tmp['total'] * 100
            fig4 = px.bar(tmp, x='benefits', y='pct_risk', labels={'pct_risk': 'Risk %', 'benefits': 'Benefits'})
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.subheader("Sample of Aggregated Records (with Employee ID)")
        # Now include 'employee_id' for managers
        show_cols = [c for c in filtered.columns if c != 'Unnamed: 0']
        st.dataframe(filtered[show_cols].sort_values('date', ascending=False).reset_index(drop=True).head(200))
