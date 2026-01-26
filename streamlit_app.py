import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="AI Cyber Threat Detection", layout="centered")

st.title("🛡️ AI Cyber Threat Detection System")
st.write("Predict cyber attack type using CICIDS2017 trained model.")

# Load model files
model = joblib.load("model/cyber_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
FEATURES = joblib.load("model/features.pkl")

st.subheader("Enter Feature Values")

# input box for each feature
input_data = {}
for feat in FEATURES[:10]:
    input_data[feat] = st.number_input(feat, value=0.0)

st.info("For demo: we are showing only first 10 features. Remaining features will be set to 0 automatically.")

if st.button("Predict Attack Type"):
    full_features = []
    for feat in FEATURES:
        full_features.append(input_data.get(feat, 0.0))

    X = np.array(full_features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = np.max(model.predict_proba(X_scaled))

    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Prediction: {label}")
    st.write(f"Confidence: **{prob*100:.2f}%**")
