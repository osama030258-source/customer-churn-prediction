import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ── Load Models & Preprocessing Objects ─────────────────
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost_model.pkl"),
    }
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return models, scaler, feature_columns

models, scaler, feature_columns = load_artifacts()

@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)
    return df

raw_df = load_data()

# ── Sidebar Navigation ───────────────────────────────────
page = st.sidebar.radio("Navigate", ["Churn Prediction", "Customer Insights", "Model Performance"])

# ══════════════════════════════════════════════════════════
# PAGE 1: CHURN PREDICTION
# ══════════════════════════════════════════════════════════
if page == "Churn Prediction":
    st.title("📊 Customer Churn Prediction")
    st.write("Enter customer details to predict churn risk.")

    model_choice = st.selectbox("Select Model", list(models.keys()))

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure))

    if st.button("Predict Churn", type="primary"):
        # Build raw input row
        input_dict = {
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0, 'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure, 'PhoneService': 1 if phone_service == "Yes" else 0,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_dict])

        # Feature engineering (must match notebook exactly)
        def tenure_segment(t):
            if t <= 12: return 'New Customer'
            elif t <= 48: return 'Regular Customer'
            else: return 'Loyal Customer'

        def spend_category(c):
            if c <= 35: return 'Low Spend'
            elif c <= 70: return 'Medium Spend'
            else: return 'High Spend'

        def contract_risk(c):
            return {'Month-to-month': 3, 'One year': 2, 'Two year': 1}[c]

        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        input_df['TenureSegment'] = input_df['tenure'].apply(tenure_segment)
        input_df['SpendCategory'] = input_df['MonthlyCharges'].apply(spend_category)
        input_df['ContractRiskScore'] = input_df['Contract'].apply(contract_risk)

        service_count = sum(1 for col in service_cols if input_dict[col] == 'Yes')
        if input_dict['InternetService'] != 'No':
            service_count += 1
        input_df['ServiceCount'] = service_count

        # One-hot encode
        nominal_cols = ['gender', 'InternetService', 'PaymentMethod', 'Contract',
                         'TenureSegment', 'SpendCategory', 'MultipleLines',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
        input_encoded = pd.get_dummies(input_df, columns=nominal_cols)

        # Align to training columns (add missing dummy cols as 0, drop extras, order correctly)
        input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scale numeric columns
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_final[scale_cols] = scaler.transform(input_final[scale_cols])

        # Predict
        model = models[model_choice]
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk — Probability: {probability:.1%}")
        else:
            st.success(f"✅ Low Churn Risk — Probability: {probability:.1%}")

        st.progress(float(probability))

# ══════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "Customer Insights":
    st.title("🔍 Customer Insights Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(raw_df):,}")
    col2.metric("Churn Rate", f"{(raw_df['Churn']=='Yes').mean():.1%}")
    col3.metric("Avg Monthly Charges", f"${raw_df['MonthlyCharges'].mean():.2f}")
    col4.metric("Avg Tenure", f"{raw_df['tenure'].mean():.1f} months")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=raw_df, x='Contract', hue='Churn', ax=ax, palette='Set2')
        st.pyplot(fig)

    with c2:
        st.subheader("Churn by Internet Service")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=raw_df, x='InternetService', hue='Churn', ax=ax, palette='Set2')
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=raw_df, x='Churn', y='tenure', ax=ax, palette='Set2')
        st.pyplot(fig)

    with c4:
        st.subheader("Monthly Charges by Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=raw_df, x='Churn', y='MonthlyCharges', ax=ax, palette='Set2')
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("🎯 Model Performance Comparison")

    performance_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.80, 0.76, 0.76],
        "Precision (Churn)": [0.65, 0.53, 0.53],
        "Recall (Churn)": [0.53, 0.78, 0.76],
        "F1 (Churn)": [0.58, 0.63, 0.62],
        "ROC-AUC": [0.8414, 0.8406, 0.8362]
    }
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.info("💡 **Business takeaway:** Random Forest catches the most actual churners (78% recall), "
            "making it the better choice for proactive retention campaigns, despite slightly lower overall accuracy.")