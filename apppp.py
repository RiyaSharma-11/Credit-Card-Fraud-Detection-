import streamlit as st
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def save_transaction(username, prob, prediction):
    file = f"{username}_history.csv"

    new_data = pd.DataFrame([{
        "prob": prob,
        "prediction": prediction
    }])

    if os.path.exists(file):
        old = pd.read_csv(file)
        updated = pd.concat([old, new_data], ignore_index=True)
    else:
        updated = new_data

    updated.to_csv(file, index=False)


def load_history(username):
    file = f"{username}_history.csv"

    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        return pd.DataFrame(columns=["prob", "prediction"])

# -------------------------------
# CONFIG & MODEL LOAD
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
st.set_page_config(page_title="SafeGuard AI", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

try:
    model = load_model()
except:
    st.error("Model file 'fraud_model.pkl' not found. Please ensure it is in the directory.")
    st.stop()

# -------------------------------
# SESSION STATE
# -------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prob" not in st.session_state:
    st.session_state.prob = None

# -------------------------------
# CUSTOM CSS
# -------------------------------
# -------------------------------
# UPDATED CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Inter:wght@300;400;700&display=swap');

    /* 1. SIDEBAR BACKGROUND & TEXT */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #161b2e 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* 2. PROMINENT WHITE SIDEBAR ARROW */
    button[kind="headerNoPadding"] {
        color: white !important;
        background-color: rgba(99, 102, 241, 0.2) !important;
        border-radius: 50% !important;
        transform: scale(1.2);
        margin: 10px;
    }

    /* 3. BOXED DESCRIPTION WITH NEW FONT */
    .desc-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 15px;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #a5b4fc !important;
        text-align: center;
        margin-bottom: 20px;
    }

    /* 4. NAVIGATION ICON & TEXT */
    .nav-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 30px;
    }

    /* 5. BRAND TITLE (CURSIVE) */
    .brand-text {
        font-family: 'Dancing Script', cursive;
        font-size: 1.8rem; /* Increased from 2.5rem */
        color: #fbbf24; 
        text-shadow: 0 0 20px rgba(251, 191, 36, 0.6); /* Stronger glow */
        margin-bottom: 10px;
        line-height: 1.2;
        display: block;
    }
    
    /* Ensuring the sidebar container allows for the larger text */
    [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    /* Tabs & Radio Styling */
    div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR
# -------------------------------
# -------------------------------
# UPDATED SIDEBAR
# -------------------------------
with st.sidebar:
    # Cursive Title
    st.markdown('<div class="brand-text">💳 SafeGuard AI</div>', unsafe_allow_html=True)
    
    # Boxed Description
    st.markdown("""
        <div class="desc-box">
            ML-based Credit Card Fraud Detection System
        </div>
    """, unsafe_allow_html=True)
    username = st.text_input("Enter Username")
    
    st.markdown("---")
    
    # Icon + Navigate Header
    st.markdown("""
        <div class="nav-header">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
            Navigate
        </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio("", ["Check Transaction", "Analytics", "About Model"], label_visibility="collapsed")

    if username:
        if "history" not in st.session_state:
            st.session_state.history = []
    # load previous data from CSV
        df_old = load_history(username)
        st.session_state.history = df_old.to_dict("records")

# -------------------------------
# MAIN: CHECK TRANSACTION
# -------------------------------
if app_mode == "Check Transaction":
    st.header("Transaction Analysis")

    tab1, tab2 = st.tabs(["Manual Entry", "Bulk Input"])

    user_input_list = None  # IMPORTANT FIX

    # ---------------- MANUAL INPUT ----------------
    with tab1:
        st.subheader("Enter 30 Features")

        col1, col2, col3 = st.columns(3)
        inputs = []

        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        for i in range(30):
            target_col = [col1, col2, col3][i % 3]
            val = target_col.number_input(
                feature_names[i],
                value=0.0,
                format="%.4f",
                key=f"v{i}"
                )
            inputs.append(val)

        

        user_input_list = inputs

    # ---------------- BULK INPUT ----------------
    with tab2:
        bulk_input = st.text_area("Paste 30 comma-separated values")

        if bulk_input:
            try:
                user_input_list = [float(x.strip()) for x in bulk_input.split(",")]
            except:
                st.error("Invalid bulk input format")

    values = np.array(user_input_list).reshape(1, -1)



    # ---------------- PREDICTION ----------------
    if st.button("Run Diagnostic"):

        if user_input_list is None:
            st.warning("Please enter input values first.")
            st.stop()

        try:
            data = np.array(user_input_list).reshape(1, -1)

            if user_input_list is None or len(user_input_list) == 0:
                st.error("⚠️ No input provided.")
                st.stop()

            if np.all(data == 0):
                st.error("⚠️ Invalid Transaction: All values are zero.")
                st.stop()
            if np.count_nonzero(data) < 5:
                st.warning("⚠️ Input too weak. Please enter realistic values.")
                st.stop()

            expected_features = getattr(model, "n_features_in_", 30)

            if data.shape[1] != expected_features:
                st.error(f"Feature mismatch: model expects {expected_features} features.")
                st.stop()

            prob = model.predict_proba(data)[0][1]
            

            threshold = 0.3
            prediction = 1 if prob > threshold else 0
            if username:
                save_transaction(username, prob, prediction)
            st.session_state.history.append({
            "prob": prob,
            "prediction": prediction
})
            st.session_state.prob = prob
            st.session_state.prediction = prediction

            st.success("Analysis Complete")

            if prediction == 1:
                st.error(f"⚠️ FRAUD DETECTED (Risk: {prob:.2%})")
            else:
                st.success(f"✅ LEGITIMATE TRANSACTION (Risk: {prob:.2%})")

        except Exception as e:
            st.error(f"Error: {e}")
elif app_mode == "Analytics":

    st.header("📊 Fraud Detection Analytics Dashboard")

    history = st.session_state.history

    if len(history) == 0:
        st.info("No transactions yet. Run a prediction first.")
        st.stop()

    df = pd.DataFrame(history)

    # -------------------------------
    # KPI METRICS
    # -------------------------------
    total = len(df)
    fraud = int(df["prediction"].sum())
    safe = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", fraud)
    col3.metric("Safe Transactions", safe)

    # -------------------------------
    # 🥧 PIE CHART
    # -------------------------------
    st.subheader("🥧 Transaction Distribution")

    fig1, ax1 = plt.subplots()
    ax1.pie(
        [safe, fraud],
        labels=["Safe", "Fraud"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    # -------------------------------
    # ⚠️ RISK LEVELS
    # -------------------------------
    st.subheader("⚠️ Risk Level Breakdown")

    def risk_level(prob):
        if prob < 0.3:
            return "Low"
        elif prob < 0.7:
            return "Medium"
        else:
            return "High"

    df["risk"] = df["prob"].apply(risk_level)

    risk_counts = df["risk"].value_counts()

    st.bar_chart(risk_counts)

    # -------------------------------
    # 📈 TREND LINE
    # -------------------------------
    st.subheader("📈 Risk Probability Trend")

    fig2, ax2 = plt.subplots()
    ax2.plot(df["prob"], marker="o")
    ax2.set_xlabel("Transaction Number")
    ax2.set_ylabel("Fraud Probability")
    ax2.set_title("Risk Over Time")
    st.pyplot(fig2)

    # -------------------------------
    # RAW DATA
    # -------------------------------
    st.subheader("📄 Transaction History")
    st.dataframe(df)
# -------------------------------
# ABOUT PAGE
# -------------------------------
elif app_mode == "About Model":
    st.header("About SafeGuard AI")
    st.markdown("""
    ### 🧠 Model Info
    - PCA-based dataset (V1–V28 + Time + Amount)
    - Trained for fraud detection

    ### ⚙️ Tech Stack
    - Streamlit
    - Scikit-learn
    - Joblib
    """)

