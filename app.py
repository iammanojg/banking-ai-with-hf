# Final app.py — customer-level model + HuggingFace LLM integration (uses llm_local.safe_generate_tip)
# IMPORTANT: load_dotenv() is called before importing the helper so HF_TOKEN from .env is read.
import os
from dotenv import load_dotenv

# Load local .env (for local development). Do NOT commit .env to git.
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Import the HF helper (will read HF_TOKEN via os.getenv inside the helper)
from llm_local import safe_generate_tip

# ---------- Page config ----------
st.set_page_config(page_title="Smart Spending Advisor (Customer-level)", page_icon="💳", layout="centered")
st.title("💳 Smart Spending Advisor — Customer-targeted Recommendations")
st.write("Aggregates transactions per customer and generates a tailored recommendation (LLM-backed when HF_TOKEN is set).")

DATA_PATH = os.path.join("data", "spending_patterns_REALISTIC_97percent.csv")

SUGGESTIONS_STATIC = {
    'Cash': "Tip: Consider using a Credit Card for larger purchases to earn rewards and build credit. Cards also offer purchase protection.",
    'Debit Card': "Insight: Debit is practical for daily spending. A Credit Card can offer cashback, rewards, and better buyer protections when used responsibly.",
    'Credit Card': "Excellent Choice! You're optimizing for rewards and protection. Continue paying in full each month.",
    'Digital Wallet': "Smart move! Link your digital wallet to a rewards Credit Card to earn points while keeping convenience."
}

# ---------- Load CSV ----------
@st.cache_data(show_spinner=False)
def load_csv(maybe_uploaded):
    if maybe_uploaded is not None:
        df = pd.read_csv(maybe_uploaded, sep=None, engine='python')
        source = "uploaded file"
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, sep=None, engine='python')
        source = DATA_PATH
    else:
        df = pd.DataFrame({
            "Customer ID": ["CUST_0159","CUST_0245","CUST_0312","CUST_0312"],
            "Transaction Date": ["2025-01-05","2025-02-10","2025-03-20","2025-04-01"],
            "Location": ["app","store","web","app"],
            "Category": ["Groceries","Travel","Fitness","Groceries"],
            "Item": ["Milk","Flight","Gym","Eggs"],
            "Quantity": [1,1,1,12],
            "Total Spent": [45.5,280.0,130.0,12.0],
            "Payment Method": ["Debit Card","Credit Card","Cash","Cash"]
        })
        source = "mock data"
    return df, source

# ---------- Tx-level featurize (used to build aggregates) ----------
def featurize_tx(df):
    df = df.copy()
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], dayfirst=True, errors='coerce')
    if 'Total Spent' in df.columns:
        df['Total_Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce').fillna(0)
    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    if 'Price Per Unit' in df.columns:
        df['Price_Per_Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce').fillna(np.nan)
    if 'Price_Per_Unit' not in df.columns or df['Price_Per_Unit'].isna().all():
        if 'Quantity' in df.columns and 'Total_Spent' in df.columns:
            df['Price_Per_Unit'] = df['Total_Spent'] / df['Quantity'].replace(0,1)
            df['Price_Per_Unit'] = df['Price_Per_Unit'].fillna(0)
    def get_channel(loc):
        loc = str(loc).lower()
        if any(x in loc for x in ['app','mobile','ios','android']): return 'Mobile App'
        if any(x in loc for x in ['web','online','site']): return 'Online'
        if any(x in loc for x in ['store','pos','shop','in-store']): return 'In-store'
        return 'Other'
    if 'Location' in df.columns:
        df['Channel'] = df['Location'].apply(get_channel)
    for c in ['Customer ID','Category','Item','Payment Method','Channel','Total_Spent','Price_Per_Unit','Quantity']:
        if c not in df.columns:
            df[c] = np.nan if c in ['Total_Spent','Price_Per_Unit','Quantity'] else "Unknown"
    return df

# ---------- Build per-customer aggregate features ----------
def build_customer_features(tx_df):
    tx = featurize_tx(tx_df)
    group = tx.groupby('Customer ID', dropna=False)
    rows = []
    for cust, sub in group:
        if pd.isna(cust) or str(cust).strip() == '':
            continue
        total_tx = len(sub)
        total_spent = sub['Total_Spent'].sum()
        avg_spent = sub['Total_Spent'].mean() if total_tx>0 else 0.0
        pct_mobile = (sub['Channel']=='Mobile App').mean() if 'Channel' in sub else 0.0
        unique_categories = sub['Category'].nunique()
        top_category = sub['Category'].mode().iat[0] if sub['Category'].nunique()>0 else "Unknown"
        pm_counts = sub['Payment Method'].value_counts()
        primary_pm = pm_counts.idxmax() if len(pm_counts)>0 else "Unknown"
        rows.append({
            "Customer ID": cust,
            "total_tx": total_tx,
            "total_spent": total_spent,
            "avg_spent": avg_spent,
            "pct_mobile": pct_mobile,
            "unique_categories": unique_categories,
            "top_category": top_category,
            "primary_payment": primary_pm,
            "pct_cash": pm_counts.get("Cash", 0)/pm_counts.sum() if pm_counts.sum()>0 else 0,
            "pct_debit": pm_counts.get("Debit Card", 0)/pm_counts.sum() if pm_counts.sum()>0 else 0,
            "pct_credit": pm_counts.get("Credit Card", 0)/pm_counts.sum() if pm_counts.sum()>0 else 0,
            "pct_wallet": pm_counts.get("Digital Wallet", 0)/pm_counts.sum() if pm_counts.sum()>0 else 0,
        })
    return pd.DataFrame(rows)

# ---------- Train customer-level model ----------
@st.cache_resource(show_spinner=False)
def train_customer_model(tx_df, iterations=300, depth=6, random_seed=42):
    cust_df = build_customer_features(tx_df)
    if cust_df.shape[0] < 2:
        raise ValueError("Need at least 2 unique customers to train a customer-level model.")
    cust_df = cust_df.dropna(subset=['primary_payment']).reset_index(drop=True)
    X = cust_df.drop(columns=['Customer ID','primary_payment'])
    y = cust_df['primary_payment']
    pm_counts = y.value_counts()
    if pm_counts.min() < 2 or pm_counts.shape[0] < 2:
        stratify = None
    else:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=stratify)
    cat_features = [c for c in ['top_category'] if c in X.columns]
    model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=0.05, random_seed=random_seed, verbose=False)
    model.fit(X_train, y_train, cat_features=cat_features)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {
        "model": model,
        "cat_features": cat_features,
        "accuracy": acc,
        "X_columns": X.columns.tolist(),
        "cust_df": cust_df,
        "payment_counts": y.value_counts().to_dict()
    }

# ---------- Sidebar / data load ----------
st.sidebar.header("Data & model")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv","tsv","txt"])
tx_df, source = load_csv(uploaded)
st.sidebar.write("Data source:", source, "| rows:", len(tx_df))

st.sidebar.markdown("### Training options")
iterations = st.sidebar.number_input("Iterations", min_value=50, max_value=2000, value=300, step=50)
depth = st.sidebar.slider("Depth", 3, 12, 6)

try:
    with st.spinner("Training customer-level model (cached)..."):
        bundle = train_customer_model(tx_df, iterations=int(iterations), depth=int(depth))
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

model = bundle['model']
cust_df = bundle['cust_df']

unique_customers = sorted(cust_df['Customer ID'].astype(str).unique())
st.sidebar.markdown("### Customers")
st.sidebar.write(f"{len(unique_customers)} customers in dataset")
cust_choice = st.sidebar.selectbox("Select Customer ID to analyze", [""] + unique_customers)

# ---------- When customer selected ----------
if cust_choice:
    row = cust_df[cust_df['Customer ID'].astype(str) == str(cust_choice)]
    row0 = row.iloc[0]
    st.subheader(f"Customer-level summary — {cust_choice}")
    st.write(f"Total transactions: {int(row0['total_tx'])}")
    st.write(f"Total spent: €{row0['total_spent']:.2f}")
    st.write(f"Avg transaction: €{row0['avg_spent']:.2f}")
    st.write(f"Top category: {row0['top_category']}")
    st.write(f"Primary payment (from data): {row0['primary_payment']}")

    # Prepare model input
    Xcols = bundle['X_columns']
    X_input = row0.drop(labels=['Customer ID','primary_payment']).reindex(index=Xcols).to_frame().T.fillna(0)

    pred_raw = model.predict(X_input)
    pred_label = str(pred_raw[0])
    proba = model.predict_proba(X_input)[0]
    class_order = list(model.classes_)
    pred_index = class_order.index(pred_label) if pred_label in class_order else None
    confidence = float(proba[pred_index]) if pred_index is not None else float(np.max(proba))
    st.success(f"🧠 Model prediction for this customer: **{pred_label}** (Confidence: {confidence*100:.2f}%)")

    # Analytical bullets
    analysis_msgs = []
    if row0['total_spent'] > 1000:
        analysis_msgs.append("High cumulative spend — rewards and protections from credit cards could be valuable.")
    if row0['pct_mobile'] > 0.6:
        analysis_msgs.append("This customer uses mobile channels frequently — mobile-friendly offers may work well.")
    if row0['unique_categories'] > 5:
        analysis_msgs.append("Diverse spending categories — a rewards credit card with broad category coverage could help.")
    if analysis_msgs:
        st.markdown("### 🔎 Analytical AI")
        for m in analysis_msgs:
            st.write("- " + m)

    # ---------- LLM call (Hugging Face via llm_local.safe_generate_tip) ----------
    # Build a small non-PII summary dict for the LLM
    customer_summary = {
        "total_spent": float(row0['total_spent']),
        "avg_tx": float(row0['avg_spent']),
        "pct_mobile": float(row0['pct_mobile']),
        "top_category": str(row0['top_category']),
        "primary_payment": str(row0['primary_payment'])
    }

    # safe_generate_tip will try HF (if HF_TOKEN present), otherwise return a static fallback
    gen_tip = safe_generate_tip(customer_summary, fallback_label=str(row0['primary_payment']))
    st.markdown("### 💬 AI Recommendation")
    st.info(gen_tip)

    # show customer's raw transactions for context
    st.markdown("### Recent transactions (for context)")
    cust_tx = featurize_tx(tx_df)
    cust_tx = cust_tx[cust_tx['Customer ID'].astype(str) == str(cust_choice)].sort_values(by='Transaction Date', ascending=False)
    if not cust_tx.empty:
        st.dataframe(cust_tx[['Transaction Date','Category','Item','Total_Spent','Channel','Payment Method']].head(20))

    # diagnostics
    with st.expander("Model diagnostics & details"):
        st.write("Per-customer counts used for training (payment method distribution):")
        st.write(bundle['payment_counts'])
        st.write(f"Model accuracy on holdout (debug): {bundle['accuracy']:.3f}")
        try:
            fi = model.get_feature_importance(type='FeatureImportance')
            names = model.feature_names_
            imp = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
            st.write("Top features:")
            for n, v in imp[:20]:
                st.write(f"- {n}: {v:.3f}")
        except Exception as e:
            st.write("Feature importance not available:", e)

st.markdown("---")
st.caption("Note: For privacy, the LLM receives only aggregated non-PII customer summaries. Do not commit secrets (.env) to the repository.")
