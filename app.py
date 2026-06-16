import streamlit as st
import pandas as pd
import joblib
import numpy as np

from agents.llm_agent import generate_ai_report

# -----------------------------
# LOAD MODEL & FEATURES
# -----------------------------
model = joblib.load("model/interaction_model.pkl")
features = joblib.load("model/features.pkl")

# Clinical feature names (must match train.py)
CLINICAL_FEATURES = [
    "glucose", "bp", "cholesterol", "kidney",
    "hba1c", "bmi", "hdl", "ldl", "ulcer"
]

# Drug features = remaining columns
DRUG_FEATURES = [f for f in features if f not in CLINICAL_FEATURES]

# -----------------------------
# UI TITLE
# -----------------------------
st.title("💊 Agentic Drug AI")

# -----------------------------
# DRUG INPUT
# -----------------------------
selected_drugs = st.multiselect("Select Drugs", DRUG_FEATURES)

# -----------------------------
# CLINICAL INPUTS
# -----------------------------
glucose = st.number_input("Glucose", min_value=0.0)
bp = st.number_input("BP", min_value=0.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0)

kidney = st.selectbox("Kidney Function", ["Normal", "Impaired"])
kidney_val = 1 if kidney == "Impaired" else 0

hba1c = st.number_input("HbA1c", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
hdl = st.number_input("HDL", min_value=0.0)
ldl = st.number_input("LDL", min_value=0.0)

ulcer = st.selectbox("Gastric Ulcer", ["No", "Yes"])
ulcer_val = 1 if ulcer == "Yes" else 0

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🔍 Analyze"):

    st.write("🔄 Running analysis...")

    # -----------------------------
    # CREATE INPUT VECTOR
    # -----------------------------
    input_dict = {f: 0 for f in features}

    # Encode selected drugs
    for drug in selected_drugs:
        if drug in input_dict:
            input_dict[drug] = 1

    # Add clinical data
    input_dict.update({
        "glucose": glucose,
        "bp": bp,
        "cholesterol": cholesterol,
        "kidney": kidney_val,
        "hba1c": hba1c,
        "bmi": bmi,
        "hdl": hdl,
        "ldl": ldl,
        "ulcer": ulcer_val
    })

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])[features]

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    confidence = round(float(np.max(prob)), 2)

    risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.write(f"### Prediction: {risk_label}")
    st.write(f"### Confidence: {confidence}")

    # -----------------------------
    # PREPARE LLM DATA
    # -----------------------------
    llm_data = {
        "drugs": selected_drugs,
        "glucose": glucose,
        "bp": bp,
        "cholesterol": cholesterol,
        "kidney": kidney,
        "hba1c": hba1c,
        "bmi": bmi,
        "hdl": hdl,
        "ldl": ldl,
        "ulcer": ulcer
    }

    # -----------------------------
    # GENERATE AI REPORT
    # -----------------------------
    st.write("## 🧠 AI Report:")

    try:
        report = generate_ai_report(llm_data, pred, confidence)
        st.write(report)
    except Exception as e:
        st.error(f"LLM Error: {e}")