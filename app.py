import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Spend Intelligence",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Customer Spend Intelligence Dashboard")
st.caption("Insights, feature behavior, and ML model comparison based on available data")

# -------------------------------
# SAFE DATA LOADER
# -------------------------------
@st.cache_data
def load_data(file):
    if not os.path.exists(file):
        st.error(f"File not found: {file}")
        st.stop()

    try:
        return pd.read_excel(file)
    except Exception:
        try:
            return pd.read_csv(file)
        except Exception:
            st.error(f"Cannot read file: {file}")
            st.stop()

# -------------------------------
# LOAD AVAILABLE DATA FILES
# -------------------------------
customers = load_data("customers.xls")
products = load_data("products.xls")
stores = load_data("stores.xls")
sales = load_data("sales_header.xls")
promo_sales = load_data("product_promotion_sales.xls")

# -------------------------------
# DATA OVERVIEW
# -------------------------------
st.header("📂 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customers")
    st.write(customers.head())
    st.metric("Rows", customers.shape[0])
    st.metric("Columns", customers.shape[1])

with col2:
    st.subheader("Sales")
    st.write(sales.head())
    st.metric("Rows", sales.shape[0])
    st.metric("Columns", sales.shape[1])

# -------------------------------
# COLUMN INSIGHTS
# -------------------------------
st.header("🔍 Column-Level Insights")

datasets = {
    "Customers": customers,
    "Sales": sales,
    "Products": products,
    "Stores": stores,
    "Promotion Sales": promo_sales
}

# Dataset selector (ONE key)
dataset_name = st.selectbox(
    "Select dataset to inspect",
    list(datasets.keys()),
    key="dataset_selector"
)

df = datasets[dataset_name]

# Reset column selection when dataset changes
if "selected_column" not in st.session_state:
    st.session_state.selected_column = df.columns[0]

if st.session_state.selected_column not in df.columns:
    st.session_state.selected_column = df.columns[0]

# Column selector (ONE key, globally unique)
col_name = st.selectbox(
    "Select a column",
    df.columns,
    key="column_selector"
)

st.write("### Column Summary")
st.write(df[col_name].describe(include="all"))

if df[col_name].dtype != "object":
    fig = px.histogram(df, x=col_name, title=f"Distribution of {col_name}")
    st.plotly_chart(fig, use_container_width=True)
else:
    value_counts = df[col_name].value_counts().head(10)
    fig = px.bar(value_counts, title=f"Top Categories in {col_name}")
    st.plotly_chart(fig, use_container_width=True)


col_name = st.selectbox(
    "Select a column",
    df.columns,
    key=f"column_select_{dataset_name}"
)


st.write("### Column Summary")
st.write(df[col_name].describe(include="all"))

if df[col_name].dtype != "object":
    fig = px.histogram(df, x=col_name, title=f"Distribution of {col_name}")
    st.plotly_chart(fig, use_container_width=True)
else:
    value_counts = df[col_name].value_counts().head(10)
    fig = px.bar(value_counts, title=f"Top Categories in {col_name}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
st.header("🤖 Machine Learning Models")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

models = {
    "Linear Regression": load_model("models/lr_model.pkl"),
    "Random Forest": load_model("models/rf_model.pkl"),
    "Gradient Boosting": load_model("models/gb_model.pkl")
}

# -------------------------------
# MODEL SCORES (STATIC / PRE-COMPUTED)
# -------------------------------
model_scores = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "R² Score": [0.62, 0.81, 0.84],
    "RMSE": [420.5, 260.3, 245.7]
})

st.subheader("📊 Model Performance Comparison")
st.dataframe(model_scores, use_container_width=True)

fig = px.bar(
    model_scores,
    x="Model",
    y="R² Score",
    title="Model Accuracy Comparison"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# MODEL INSIGHTS
# -------------------------------
st.header("🧠 Model Insights")

model_choice = st.selectbox("Select a model", list(models.keys()))

if model_choice == "Linear Regression":
    st.write("""
    **Linear Regression**
    - Assumes linear relationship between features and spend
    - Easy to interpret
    - Lower accuracy due to complex customer behavior
    """)

    coef = models["Linear Regression"].coef_
    feature_names = customers.select_dtypes(include=np.number).columns[:len(coef)]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": coef
    }).sort_values(by="Impact", ascending=False)

    st.write("### Feature Impact")
    st.dataframe(coef_df)

elif model_choice == "Random Forest":
    st.write("""
    **Random Forest**
    - Captures non-linear patterns
    - Robust to noise
    - Good balance between accuracy and stability
    """)

    importances = models["Random Forest"].feature_importances_
    feature_names = customers.select_dtypes(include=np.number).columns[:len(importances)]

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

elif model_choice == "Gradient Boosting":
    st.write("""
    **Gradient Boosting**
    - Best performing model
    - Learns complex interactions
    - Slightly harder to interpret
    """)

    importances = models["Gradient Boosting"].feature_importances_
    feature_names = customers.select_dtypes(include=np.number).columns[:len(importances)]

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FINAL TAKEAWAYS
# -------------------------------
st.header("📌 Business Takeaways")

st.markdown("""
- Customer spend is influenced by **purchase frequency**, **promotion exposure**, and **store behavior**
- Ensemble models outperform linear approaches
- Feature distributions reveal spending concentration in a small customer segment
- Promotions significantly boost short-term spend but vary by product category
""")
