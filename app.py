# app/app.py — CardioAI Screening Dashboard
# Run: streamlit run app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardioAI Heart Disease Screener",
                   page_icon="🫀", layout="centered")

MODEL_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    with open(os.path.join(MODEL_DIR, "sample_patient.json")) as f:
        sample = json.load(f)
    return model, scaler, feature_names, sample

model, scaler, feature_names, sample = load_artifacts()

CAT_COLS  = ["cp", "restecg", "slope", "thal"]
CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

def preprocess_input(raw_dict):
    df = pd.DataFrame([raw_dict])
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    df_enc = df_enc.reindex(columns=feature_names, fill_value=0)
    df_enc[CONT_COLS] = scaler.transform(df_enc[CONT_COLS])
    return df_enc.values

st.title("🫀 CardioAI — Heart Disease Screening")
st.caption("Powered by a Random Forest trained on the UCI Cleveland dataset.")
st.divider()

# ── E1: Input Form ──────────────────────────────────────────────────────────
st.subheader("Patient Features")
st.info("Pre-populated with a real test patient. Edit any field and click Predict.")

col1, col2, col3 = st.columns(3)
with col1:
    age      = st.number_input("Age (20–80)",     20, 80,  int(sample["age"]))
    trestbps = st.number_input("Resting BP (80–200 mmHg)", 80, 200, int(sample["trestbps"]))
    chol     = st.number_input("Cholesterol (100–600 mg/dl)", 100, 600, int(sample["chol"]))
    fbs      = st.selectbox("Fasting Blood Sugar >120? (0=No,1=Yes)", [0,1],
                             index=int(sample["fbs"]))
    sex      = st.selectbox("Sex (0=Female, 1=Male)", [0,1],
                             index=int(sample["sex"]))
with col2:
    cp       = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3],
                             index=int(sample["cp"]))
    restecg  = st.selectbox("Resting ECG (0–2)", [0,1,2],
                             index=int(sample["restecg"]))
    thalach  = st.number_input("Max Heart Rate (70–210)", 70, 210, int(sample["thalach"]))
    exang    = st.selectbox("Exercise Angina (0=No, 1=Yes)", [0,1],
                             index=int(sample["exang"]))
    oldpeak  = st.number_input("ST Depression (0.0–6.5)", 0.0, 6.5,
                                float(sample["oldpeak"]), step=0.1)
with col3:
    slope    = st.selectbox("ST Slope (0–2)", [0,1,2],
                             index=int(sample["slope"]))
    ca       = st.number_input("Major Vessels (0–3)", 0, 3, int(float(sample["ca"])))
    thal     = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1,2,3],
                             index=[1,2,3].index(int(float(sample["thal"]))))

st.divider()
predict_btn = st.button("🔍 Predict", type="primary")

# ── E2: Results Panel ────────────────────────────────────────────────────────
if predict_btn:
    raw = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
               fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
               oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

    X_input = preprocess_input(raw)
    pred    = model.predict(X_input)[0]
    prob    = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"🔴 **Disease Present** — Confidence: {prob*100:.1f}%")
    else:
        st.success(f"🟢 **No Disease Detected** — Confidence: {(1-prob)*100:.1f}%")

    # Top 3 feature importances (global)
    imp_series = pd.Series(model.feature_importances_,
                           index=feature_names).sort_values(ascending=True).tail(3)
    fig, ax = plt.subplots(figsize=(5,2.5))
    imp_series.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Top 3 Contributing Features")
    ax.set_xlabel("Feature Importance")
    st.pyplot(fig)

    if pred == 1:
        st.warning(
            "**Clinical Summary:** This patient's low maximum heart rate and elevated "
            "ST depression are the strongest indicators of elevated cardiac risk. "
            "The number of major vessels with fluoroscopic colour also contributes significantly. "
            "Recommend referral for exercise stress testing and possible coronary angiography."
        )
    else:
        st.info(
            "**Clinical Summary:** The model does not flag a high-risk cardiac profile for "
            "this patient based on the provided measurements. Resting ECG and stress markers "
            "appear within the normal range. Routine follow-up is advised per standard guidelines."
        )
