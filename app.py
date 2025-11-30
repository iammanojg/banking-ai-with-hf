# app.py — TOP OF FILE — REPLACE EVERYTHING ABOVE YOUR IMPORTS WITH THIS

import streamlit as st
import os

# ==================== CRITICAL: LOAD SECRETS FIRST ====================
# This MUST be the very first thing after importing streamlit
# (Streamlit loads secrets only when st.secrets is accessed)
try:
    # This line forces Streamlit Cloud to load .streamlit/secrets.toml
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]
except Exception as e:
    st.error(
        "HF_TOKEN is missing or secrets.toml not loaded!\n\n"
        "1. Make sure the file exists exactly at:\n"
        "   https://github.com/iammanojg/Banking-AI-with-HF/blob/main/.streamlit/secrets.toml\n\n"
        "2. Content must be exactly:\n"
        "```toml\n"
        "HF_TOKEN = \"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\"\n"
        "HF_MODEL = \"HuggingFaceH4/zephyr-7b-beta\"\n"
        "```"
    )
    st.stop()

# Optional: show success (remove later if you want)
# st.success("Hugging Face token loaded successfully!")

# NOW safe to import everything else
from dotenv import load_dotenv
load_dotenv()  # only helps locally
from llm_local import safe_generate_tip
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# ==================== 2. PAGE CONFIG ====================
st.set_page_config(page_title="Smart Spending Advisor", page_icon="💳", layout="centered")
st.title("Smart Spending Advisor")
st.write("Customer-level model + Hugging Face LLM recommendations")

DATA_PATH = os.path.join("data", "spending_patterns_REALISTIC_97percent.csv")

# ==================== 3. LOAD DATA ====================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        return df, "uploaded file"
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, sep=None, engine="python")
        return df, DATA_PATH
    else:
        df = pd.DataFrame({
            "Customer ID": ["CUST_001", "CUST_002", "CUST_002"],
            "Transaction Date": ["2025-01-10", "2025-02-15", "2025-03-20"],
            "Category": ["Groceries", "Travel", "Electronics"],
            "Total Spent": [85.50, 450.00, 1200.00],
            "Payment Method": ["Debit Card", "Credit Card", "Credit Card"]
        })
        return df, "mock data"

# ==================== 4. FEATURE ENGINEERING ====================
def featurize_tx(df):
    df = df.copy()
    
    # Fix column names — handle common variations
    column_mapping = {
        'total spent': 'Total_Spent',
        'Total Spent': 'Total_Spent',
        'total_spent': 'Total_Spent',
        'Amount': 'Total_Spent',
        'amount': 'Total_Spent',
        'Spent': 'Total_Spent',
        'spent': 'Total_Spent',
        'Total Amount': 'Total_Spent',
        'Transaction Amount': 'Total_Spent'
    }
    df = df.rename(columns=lambda x: column_mapping.get(x.strip(), x))
    
    # Now safely create Total_Spent
    if 'Total_Spent' not in df.columns:
        # Try to find any column that looks like money
        money_cols = [c for c in df.columns if any(word in c.lower() for word in ['spent', 'amount', 'total', 'price'])]
        if money_cols:
            df['Total_Spent'] = pd.to_numeric(df[money_cols[0]], errors='coerce').fillna(0)
        else:
            df['Total_Spent'] = 0.0
    else:
        df['Total_Spent'] = pd.to_numeric(df['Total_Spent'], errors='coerce').fillna(0)
    
    # Date
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    elif 'Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df
# ==================== 5. TRAIN MODEL ====================
@st.cache_resource(show_spinner="Training model...")
def train_model(tx_df):
    cust_df = build_customer_features(tx_df)
    if len(cust_df) < 2:
        st.error("Need at least 2 customers to train.")
        st.stop()
    X = cust_df.drop(columns=["Customer ID", "primary_payment"])
    y = cust_df["primary_payment"]
    model = CatBoostClassifier(iterations=300, depth=5, verbose=False, random_seed=42)
    model.fit(X, y)
    return {"model": model, "features": X.columns.tolist(), "cust_df": cust_df}

# ==================== 6. UI ====================
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv", "txt"])
tx_df, source = load_csv(uploaded)
st.sidebar.write(f"Data source: {source} | Rows: {len(tx_df)}")

bundle = train_model(tx_df)
cust_df = bundle["cust_df"]
customers = sorted(cust_df["Customer ID"].astype(str).unique())
selected = st.sidebar.selectbox("Select Customer", [""] + customers)

if selected:
    row = cust_df[cust_df["Customer ID"].astype(str) == selected].iloc[0]

    st.subheader(f"Customer {selected}")
    c1, c2 = st.columns(2)
    c1.metric("Total Spent", f"€{row['total_spent']:,.2f}")
    c2.metric("Transactions", int(row['total_tx']))

    # Model prediction
    X_in = pd.DataFrame([row]).drop(columns=["Customer ID", "primary_payment"])
    X_in = X_in.reindex(columns=bundle["features"], fill_value=0)
    pred = bundle["model"].predict(X_in)[0]
    prob = bundle["model"].predict_proba(X_in)[0].max()
    st.success(f"Predicted payment method: **{pred}** ({prob:.1%} confidence)")

    # LLM tip
    summary = {
        "total_spent": float(row["total_spent"]),
        "avg_tx": float(row["avg_spent"]),
        "top_category": str(row["top_category"]),
        "primary_payment": str(row["primary_payment"])
    }
    with st.spinner("Generating personalized tip..."):
        tip = safe_generate_tip(summary, fallback_label=str(row["primary_payment"]))

    st.markdown("### Personalized Recommendation")
    st.info(tip)

    with st.expander("Recent transactions"):
        recent = tx_df[tx_df["Customer ID"].astype(str) == selected].head(10)
        st.dataframe(recent[["Transaction Date", "Category", "Total Spent", "Payment Method"]])

st.caption("Only aggregated, non-PII data is sent to the LLM.")
