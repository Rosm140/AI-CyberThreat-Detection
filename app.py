import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="AI Cyber Threat Detection", layout="wide")

st.title("🛡️ AI Cyber Threat Detection System")
st.write("Upload CICIDS2017 CSV row or use manual inputs to predict attack type.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/cyber_model_small.pkl")
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    features = joblib.load("model/features.pkl")
    return model, scaler, label_encoder, features

model, scaler, label_encoder, FEATURES = load_artifacts()

mode = st.sidebar.radio("Choose Prediction Mode", ["Manual Input", "CSV Upload"])

if mode == "Manual Input":
    st.subheader("Enter Feature Values (First 10 Features Only)")

    user_values = {}
    cols = st.columns(2)

    for i, feature in enumerate(FEATURES[:10]):
        with cols[i % 2]:
            user_values[feature] = st.number_input(feature, value=0.0, format="%.6f")

    if st.button("🚀 Predict from Manual Input"):
        full_input = np.zeros(len(FEATURES))

        for i, feature in enumerate(FEATURES[:10]):
            full_input[i] = float(user_values[feature])

        full_input_scaled = scaler.transform(full_input.reshape(1, -1))

        prediction = model.predict(full_input_scaled)[0]
        probabilities = model.predict_proba(full_input_scaled)[0]

        attack_label = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities) * 100)

        st.success(f"Prediction: **{attack_label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

elif mode == "CSV Upload":
    st.subheader("Upload CICIDS2017 CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded CSV is empty.")
        else:
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head())

            if "Attack Type" in df.columns:
                df_features = df.drop(columns=["Attack Type"])
            else:
                df_features = df.copy()

            df_features = df_features.reindex(columns=FEATURES, fill_value=0)

            max_index = len(df_features) - 1

            if max_index >= 0:
                row_index = st.number_input("Select Row Index", min_value=0, max_value=max_index, value=0, step=1)

                if st.button("🚀 Predict Selected Row"):
                    row = df_features.iloc[int(row_index)].values.reshape(1, -1)

                    row_scaled = scaler.transform(row)

                    prediction = model.predict(row_scaled)[0]
                    probabilities = model.predict_proba(row_scaled)[0]

                    attack_label = label_encoder.inverse_transform([prediction])[0]
                    confidence = float(np.max(probabilities) * 100)

                    st.success(f"Prediction: **{attack_label}**")
                    st.info(f"Confidence: **{confidence:.2f}%**")

                    prob_df = pd.DataFrame({
                        "Attack Type": label_encoder.classes_,
                        "Probability": probabilities
                    })

                    st.write("### Prediction Probabilities")
                    st.bar_chart(prob_df.set_index("Attack Type"))
            else:
                st.error("No rows available for prediction.")

st.markdown("---")
st.caption("Developed by Rohit Mahadane • AI Cyber Threat Detection Project")

