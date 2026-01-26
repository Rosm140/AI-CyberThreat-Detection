from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model + preprocessing tools
model = joblib.load("model/cyber_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Load feature names from dataset (to keep correct order)
df_sample = pd.read_csv("data/cicids2017_cleaned.csv", nrows=1)
FEATURES = joblib.load("model/features.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Cyber Threat Detection API is running ✅",
        "endpoints": ["/predict"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Expect dictionary of feature:value
        input_features = [data.get(feat, 0) for feat in FEATURES]

        X = np.array(input_features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        prob = max(model.predict_proba(X_scaled)[0])

        attack_type = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "prediction": attack_type,
            "confidence": round(float(prob) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
