import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("credit_model.json")
    return model

model = load_model()

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# Feature names (STRICT ORDER - DO NOT CHANGE)
feature_names = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'MonthlyDebtPayment',
    'UtilizationPerAge',
    'TotalPastDue'
]

# -------------------------------
# UI
# -------------------------------
st.title("Credit Risk Predictor")
st.markdown("Predict the probability of a borrower experiencing serious delinquency within 2 years.")

st.sidebar.header("Borrower Profile")

# Inputs
utilization = st.sidebar.number_input(
    "Revolving Utilization (0-1+)",
    0.0, 2.0, 0.3,
    help="Percentage of your available credit that you are currently using. Example: 0.3 means 30% of your credit limit is used. Higher values indicate higher risk."
)
age = st.sidebar.slider("Age", 18, 100, 45)
monthly_income = st.sidebar.number_input(
    "Monthly Income",
    0, 100000, 5000,
    help="Your total monthly income in currency units. Used to assess repayment capacity."
)
debt_ratio = st.sidebar.number_input(
    "Debt Ratio (0-1+)",
    0.0, 2.0, 0.3,
    help="Ratio of your total monthly debt payments to your income. Example: 0.3 means 30% of income goes to debt. Higher values indicate higher financial stress."
)
open_lines = st.sidebar.slider(
    "Open Credit Lines/Loans",
    0, 50, 10,
    help="Total number of active credit accounts such as credit cards or loans."
)
real_estate = st.sidebar.slider(
    "Real Estate Loans",
    0, 10, 1,
    help="Number of loans related to real estate such as home loans or mortgages."
)
dependents = st.sidebar.slider(
    "Number of Dependents",
    0, 10, 1,
    help="Number of people financially dependent on you (e.g., children, family members)."
)

st.sidebar.subheader("Delinquency History")
past_due_30 = st.sidebar.number_input(
    "Times 30-59 Days Past Due",
    0, 50, 0,
    help="Number of times payments were delayed by 30–59 days."
)
past_due_60 = st.sidebar.number_input(
    "Times 60-89 Days Past Due",
    0, 50, 0,
    help="Number of times payments were delayed by 60–89 days."
)
past_due_90 = st.sidebar.number_input(
    "Times 90+ Days Late",
    0, 50, 0,
    help="Number of times payments were delayed by more than 90 days. This is a strong indicator of high risk."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Run Credit Analysis"):

    # Feature Engineering (must match training EXACTLY)
    monthly_debt = debt_ratio * monthly_income
    util_per_age = utilization / (age + 1)
    total_past_due = past_due_30 + past_due_60 + past_due_90

    # Build input in STRICT ORDER
    input_data = pd.DataFrame([[
        utilization,
        age,
        past_due_30,
        debt_ratio,
        monthly_income,
        open_lines,
        past_due_90,
        real_estate,
        past_due_60,
        dependents,
        monthly_debt,
        util_per_age,
        total_past_due
    ]], columns=feature_names)

    # Prediction
    prob = model.predict_proba(input_data)[0][1]

    # -------------------------------
    # UI Output
    # -------------------------------
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Delinquency Risk", f"{prob:.2%}")

    with col2:
        if prob < 0.2:
            st.success("Category: Low Risk")
        elif prob < 0.5:
            st.warning("Category: Moderate Risk")
        else:
            st.error("Category: High Risk")

    st.progress(float(prob))

    # Show engineered features (important for debugging)
    st.subheader("Derived Features (Model Inputs)")
    st.write({
        "MonthlyDebtPayment": monthly_debt,
        "UtilizationPerAge": util_per_age,
        "TotalPastDue": total_past_due
    })

    # -------------------------------
    # SHAP Explainability
    # -------------------------------
    st.subheader("Why this prediction? (SHAP Explanation)")

    shap_values = explainer.shap_values(input_data)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.dataframe(shap_df)

    st.info("Positive impact = increases risk, Negative impact = reduces risk")

    st.info("Note: Tree models (XGBoost) are NOT monotonic. Increasing a feature does not always increase risk due to feature interactions.")
