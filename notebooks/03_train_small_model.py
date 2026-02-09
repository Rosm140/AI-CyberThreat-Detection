import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

print("🚀 Loading preprocessed data...")
X_train, X_test, y_train, y_test = joblib.load("model/split_data.pkl")

# smaller sample
SAMPLE_TRAIN = 100000
SAMPLE_TEST  = 50000

X_train_s = X_train[:SAMPLE_TRAIN]
y_train_s = y_train[:SAMPLE_TRAIN]

X_test_s = X_test[:SAMPLE_TEST]
y_test_s = y_test[:SAMPLE_TEST]

print("✅ Training small RandomForest...")

small_model = RandomForestClassifier(
    n_estimators=50,      # smaller
    max_depth=20,         # limit depth
    random_state=42,
    n_jobs=-1
)

small_model.fit(X_train_s, y_train_s)

pred = small_model.predict(X_test_s)
acc = accuracy_score(y_test_s, pred)
print("✅ Small model accuracy:", acc)

os.makedirs("model", exist_ok=True)

# ✅ compress heavily
joblib.dump(small_model, "model/cyber_model_small.pkl", compress=3)

print("✅ Saved: model/cyber_model_small.pkl")
