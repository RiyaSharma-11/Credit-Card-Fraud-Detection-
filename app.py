import streamlit as st
import numpy as np
import joblib

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("fraud_model.pkl")

# -------------------------------
# SESSION STATE (PAGE CONTROL)
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------------------
# PASTEL THEME 💖
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffd1dc, #ffe4e1);
    }
    h1, h2 {
        color: #8b3a62;
        text-align: center;
    }
    .stButton button {
        background-color: #ffb6c1;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🟣 PAGE 1 (HOME)
# -------------------------------
if st.session_state.page == "home":

    st.title("💳 Credit Card Fraud Detection")
    st.image("https://user-images.githubusercontent.com/28294942/190845223-1daf8cbd-5dfb-4071-9885-aa1883d64bdc.png")

    st.markdown("### Enter 30 feature values (comma separated):")

    user_input = st.text_area("Example: 0.1, -1.2, ...")

    if st.button("🔍 Predict"):

        try:
            values = [float(x.strip()) for x in user_input.split(',')]

            if len(values) != 30:
                st.error("⚠️ Enter exactly 30 values")
            else:
                data = np.array(values).reshape(1, -1)

                prob = model.predict_proba(data)[0][1]
                threshold = 0.3

                prediction = 1 if prob > threshold else 0

                # Save results in session
                st.session_state.prediction = prediction
                st.session_state.prob = prob

                # Move to result page
                st.session_state.page = "result"
                st.rerun()

        except:
            st.error("❌ Invalid input format")

# -------------------------------
# 🟣 PAGE 2 (RESULT)
# -------------------------------
elif st.session_state.page == "result":

    st.title("📊 Prediction Result")

    if st.session_state.prediction == 0:
        st.success("✅ Normal Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")

    st.info(f"💡 Fraud Probability: {st.session_state.prob:.4f}")

    # 🔁 BACK BUTTON
    if st.button("🔄 Predict Again"):
        st.session_state.page = "home"
        st.rerun()