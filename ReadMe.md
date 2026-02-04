import streamlit as st
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt

# -----------------------------------
# App Config
# -----------------------------------
st.set_page_config(
    page_title="Retail 30-Day CLV Dashboard",
    layout="wide"
)

st.title("🛒 Retail Short-Term CLV Dashboard")
st.markdown("Predicting **30-day customer spend** using transaction history")

# -----------------------------------
# Load SQLite data
# -----------------------------------
@st.cache_data
def load_tables():
    conn = sqlite3.connect("retail.db")
    tables = {
        "customers": pd.read_sql("SELECT * FROM customer_details", conn),
        "stores": pd.read_sql("SELECT * FROM stores", conn),
        "products": pd.read_sql("SELECT * FROM products", conn),
        "sales_header": pd.read_sql("SELECT * FROM store_sales_header", conn),
        "line_items": pd.read_sql("SELECT * FROM store_sales_line_items", conn),
    }
    conn.close()
    return tables

tables = load_tables()

# -----------------------------------
# Sidebar navigation
# -----------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "🤖 Model Comparison", "🔍 Model Details", "💰 Spend Estimator"]
)

# -----------------------------------
# PAGE 1: DATA OVERVIEW
# -----------------------------------
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", len(tables["customers"]))
    col2.metric("Stores", len(tables["stores"]))
    col3.metric("Products", len(tables["products"]))
    col4.metric("Transactions", len(tables["sales_header"]))

    st.subheader("Table Relationships")
    st.markdown("""
    - **Customers → Transactions** (1-to-many)  
    - **Stores → Transactions** (1-to-many)  
    - **Transactions → Line Items** (1-to-many)  
    - **Products → Line Items** (1-to-many)  
    """)

    st.subheader("Sample Transactions")
    st.dataframe(tables["sales_header"].head())

# -----------------------------------
# PAGE 2: MODEL COMPARISON
# -----------------------------------
elif page == "🤖 Model Comparison":
    st.header("🤖 Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
        "MAE": [638.09, 591.91, 599.51],
        "R2": [0.4406, 0.4361, 0.4455]
    })

    st.dataframe(results)

    st.subheader("R² Comparison")
    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["R2"])
    ax.set_ylabel("R² Score")
    st.pyplot(fig)

    st.success("🏆 Best Model: Gradient Boosting (Highest R²)")

# -----------------------------------
# PAGE 3: MODEL DETAILS
# -----------------------------------
elif page == "🔍 Model Details":
    st.header("🔍 Model Deep Dive")

    model_choice = st.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting"]
    )

    if model_choice == "Linear Regression":
        st.markdown("""
        **Why Linear Regression?**
        - Simple & interpretable
        - Strong baseline
        - Performs well with RFM features
        """)
        st.metric("R²", "0.44")
        st.metric("MAE", "638")

    elif model_choice == "Random Forest":
        st.markdown("""
        **Why Random Forest?**
        - Handles non-linearity
        - Lowest MAE
        - Robust to noise
        """)
        st.metric("R²", "0.43")
        st.metric("MAE", "592")

    else:
        st.markdown("""
        **Why Gradient Boosting?**
        - Best overall balance
        - Captures subtle customer behavior
        - Final production model
        """)
        st.metric("R²", "0.45")
        st.metric("MAE", "599")

# -----------------------------------
# PAGE 4: SPEND ESTIMATOR
# -----------------------------------
elif page == "💰 Spend Estimator":
    st.header("💰 30-Day Spend Estimator")

    st.markdown("Estimate customer spend using the **selected ML model**")

    model_name = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting"]
    )

    # Load model
    model_map = {
        "Linear Regression": "models/lr_model.pkl",
        "Random Forest": "models/rf_model.pkl",
        "Gradient Boosting": "models/gb_model.pkl"
    }

    model = joblib.load(model_map[model_name])

    st.subheader("Enter Customer Features")

    recency = st.number_input("Recency (days since last purchase)", 0, 365, 30)
    frequency = st.number_input("Transactions in last 90 days", 0, 50, 5)
    monetary = st.number_input("Total spend last 90 days", 0.0, 50000.0, 5000.0)

    input_df = pd.DataFrame([{
        "recency_days": recency,
        "frequency_90d": frequency,
        "monetary_90d": monetary
    }])

    if st.button("Predict Spend"):
        prediction = model.predict(input_df)[0]
        st.success(f"💸 Estimated 30-Day Spend: ₹ {prediction:,.2f}")
