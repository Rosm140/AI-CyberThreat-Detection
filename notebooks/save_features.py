import pandas as pd
import joblib
import os

df = pd.read_csv("data/cicids2017_cleaned.csv", nrows=1)
FEATURES = [c for c in df.columns if c != "Attack Type"]

os.makedirs("model", exist_ok=True)
joblib.dump(FEATURES, "model/features.pkl")

print("✅ Saved features list:", len(FEATURES))
print("✅ File saved: model/features.pkl")

