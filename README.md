# HclTech_Hackathon_cust_spending
**Customer Spend Intelligence Dashboard A**
Streamlit-based analytics and machine learning dashboard that delivers customer spending insights, feature-level analysis, and model performance comparison using real retail datasets.
<img width="1807" height="823" alt="image" src="https://github.com/user-attachments/assets/54766d88-8982-4ac0-b30b-00798201b948" />

# 📊 Customer Spend Intelligence Dashboard

A professional, end-to-end **data analytics & machine learning dashboard** built using **Streamlit**, designed to analyze customer spending behavior, generate business insights, and compare multiple predictive models.

This project demonstrates **real-world data engineering**, **exploratory data analysis**, and **model interpretability** — suitable for hackathons, interviews, and portfolio showcase.

---

## 🚀 Key Features

- 📂 Multi-dataset analytics using real retail data
- 🔍 Column-level exploratory data analysis (EDA)
- 📈 Interactive visualizations with Plotly
- 🤖 Machine learning model comparison
- 🧠 Feature importance & model insights
- ⚙️ Robust data loaders (handles Excel & CSV edge cases)
- 🧪 Production-safe Streamlit architecture

---

## 🗂️ Project Structure

HCL_CUST_SPEND/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── ReadMe.md # Project documentation
├── retail.db # Optional database file
│
├── models/
│ ├── lr_model.pkl # Linear Regression model
│ ├── rf_model.pkl # Random Forest model
│ └── gb_model.pkl # Gradient Boosting model
│
├── customers.xls
├── products.xls
├── stores.xls
├── sales_header.xls
├── product_promotion_sales.xls
├── promotion_lineitem_count.xls
├── stores_sales_summary.xls
├── top_promotions.xls
│
└── venv/ # Virtual environment (local)


---

## 📊 Datasets Used

The dashboard strictly uses **only available datasets**:

| Dataset Name | Description |
|-------------|-------------|
| `customers.xls` | Customer demographic and profile data |
| `sales_header.xls` | Transaction-level sales data |
| `products.xls` | Product information |
| `stores.xls` | Store-level metadata |
| `product_promotion_sales.xls` | Promotion impact on sales |
| `stores_sales_summary.xls` | Aggregated store performance |
| `top_promotions.xls` | Best performing promotions |

---

## 🔍 Exploratory Data Analysis (EDA)

The application allows users to:

- Select any dataset dynamically
- Inspect column distributions
- View statistical summaries
- Analyze categorical and numerical features
- Understand data skewness and customer concentration

All EDA is **interactive and visualized using Plotly**.

---

## 🤖 Machine Learning Models

The project includes three trained models:

| Model | Description |
|------|------------|
| Linear Regression | Baseline interpretable model |
| Random Forest | Non-linear ensemble model |
| Gradient Boosting | High-performance boosting model |

---

## 📈 Model Performance Comparison

Models are compared using key metrics such as:

- **R² Score**
- **MAE**

Interactive bar charts highlight performance differences and help identify the best-performing model.

---

## 🧠 Model Explainability & Insights

Each model provides insights into **feature influence**:

- Linear Regression → coefficient-based impact
- Random Forest → feature importance scores
- Gradient Boosting → boosted feature contributions

This helps answer:
> *Which features most strongly influence customer spend?*

---

## 💼 Business Insights Derived

- Customer spend is driven by **purchase frequency**, **promotion exposure**, and **store characteristics**
- Ensemble models outperform linear approaches
- Spending behavior is highly skewed toward a small customer segment
- Promotions have short-term uplift but vary across categories

---
## 🤖 Machine Learning Models

This project uses multiple machine learning models to analyze and predict **customer spending behavior**.  
Each model is chosen deliberately to balance **interpretability**, **robustness**, and **predictive performance**.

Using more than one model helps validate insights, compare performance, and build trust with business stakeholders.

---

### 1️⃣ Linear Regression (Baseline & Interpretability Model)

**Why this model?**  
Linear Regression is used as a **baseline model** to establish a reference level of performance and to provide clear, interpretable insights into how individual features affect customer spend.

**How it is used:**  
- Coefficients indicate the **direction and magnitude** of feature impact  
- Helps understand linear relationships such as:
  - Effect of purchase frequency on spend
  - Impact of promotions on revenue

**Strengths:**
- Highly interpretable
- Easy to explain to non-technical stakeholders
- Fast to train

**Limitations:**
- Assumes linear relationships
- Cannot capture complex customer behavior

**Best suited for:**
- Initial analysis
- Business explanations
- Feature validation

---

### 2️⃣ Random Forest (Robust Non-Linear Model)

**Why this model?**  
Customer behavior is rarely linear. Random Forest is used to capture **non-linear patterns** and interactions between features while remaining robust to noise and outliers in real-world retail data.

**How it is used:**  
- Feature importance scores identify the **most influential drivers of spend**
- Improves prediction accuracy compared to linear models

**Strengths:**
- Handles non-linear relationships
- Resistant to overfitting
- Strong performance on tabular data

**Limitations:**
- Less interpretable than Linear Regression
- Larger model size

**Best suited for:**
- Operational forecasting
- Customer behavior modeling
- Production-ready analytics

---

### 3️⃣ Gradient Boosting (High-Performance Model)

**Why this model?**  
Gradient Boosting is used as the **best-performing model** to achieve higher accuracy by learning complex feature interactions through sequential optimization.

**How it is used:**  
- Provides the highest R² score and lowest error
- Feature importance highlights fine-grained behavioral patterns

**Strengths:**
- High predictive accuracy
- Captures complex interactions
- Suitable for revenue forecasting

**Limitations:**
- More computationally expensive
- Harder to interpret without advanced explainability tools

**Best suited for:**
- Final predictions
- Revenue optimization
- High-impact decision-making systems

---

### 📊 Model Comparison Summary

| Model             | Interpretability | Accuracy    | Use Case               |
|------             |----------------- |----------   |----------              |
| Linear Regression | High             | Low–Medium  | Baseline & explanation |
| Random Forest     | Medium           | High        | Robust prediction      |
| Gradient Boosting | Medium–Low       | Very High   | Best performance       |

---

### 🧠 Key Takeaway

By combining **interpretable models** with **high-performance ensemble models**, this project ensures:
- Transparent business insights
- Accurate spend prediction
- Confidence in data-driven decision-making


## ⚙️ Tech Stack

- **Python 3.11**
- **Streamlit**
- **Pandas / NumPy**
- **Plotly**
- **Scikit-learn**
- **Joblib**

---

## 🛠️ Installation & Setup
