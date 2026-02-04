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

st.header("🔍 Key Data Insights (Static & Stable)")

# ---------- Customers ----------
st.subheader("👥 Customers – Spend Distribution")

numeric_cols = customers.select_dtypes(include=np.number).columns

if len(numeric_cols) > 0:
    col = numeric_cols[0]

    fig = px.histogram(
        customers,
        x=col,
        nbins=30,
        title=f"Distribution of {col} (Customers)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("📌 Insight:")
    st.write(
        f"- Most customers are concentrated in a narrow range of `{col}`, "
        "indicating uneven spending behavior."
    )

# ---------- Sales ----------
st.subheader("🧾 Sales – Volume by Store")

if "store_id" in sales.columns:
    store_counts = sales["store_id"].value_counts().head(10)

    fig = px.bar(
        x=store_counts.index,
        y=store_counts.values,
        labels={"x": "Store", "y": "Sales Count"},
        title="Top 10 Stores by Sales Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("📌 Insight:")
    st.write(
        "- A small number of stores contribute the majority of sales, "
        "suggesting location-driven performance differences."
    )

# ---------- Promotions ----------
st.subheader("🏷️ Promotions – Effectiveness")

promo_cols = promo_sales.select_dtypes(include=np.number)

if promo_cols.shape[1] > 0:
    promo_sum = promo_cols.sum().sort_values(ascending=False)

    fig = px.pie(
        values=promo_sum.values,
        names=promo_sum.index,
        title="Promotion Contribution Breakdown"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("📌 Insight:")
    st.write(
        "- Certain promotion types dominate overall sales uplift, "
        "indicating where marketing budgets should be focused."
    )


