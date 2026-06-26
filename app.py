import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the probability of heart disease.")

st.divider()

# ---------------------------------------------------
# Input Form
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            ("Typical Angina", 0),
            ("Atypical Angina", 1),
            ("Non-anginal Pain", 2),
            ("Asymptomatic", 3),
        ],
        format_func=lambda x: x[0],
    )[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox(
        "Resting ECG Result",
        options=[("Normal", 0), ("ST-T Abnormality", 1), ("Left Ventricular Hypertrophy", 2)],
        format_func=lambda x: x[0],
    )[1]

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of ST Segment",
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0],
    )[1]
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
        format_func=lambda x: x[0],
    )[1]

st.divider()

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    # Build input dataframe in the exact column order the model was trained on
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    # Scale the same columns that were scaled during training
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease")
        st.metric("Predicted Probability", f"{probability*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease")
        st.metric("Predicted Probability", f"{probability*100:.1f}%")

    st.progress(float(probability))

    st.caption("⚠️ This is a machine learning demo for academic purposes only, not a substitute for professional medical advice.")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.caption("Built with Streamlit • Logistic Regression Model • Heart Disease Dataset")