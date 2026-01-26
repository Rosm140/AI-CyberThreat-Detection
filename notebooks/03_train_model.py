import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

print("🚀 Loading preprocessed data...")
X_train, X_test, y_train, y_test = joblib.load("model/split_data.pkl")

print("✅ Loaded Train:", X_train.shape, " Test:", X_test.shape)

# ✅ Use sample for faster training (change this if you want full training)
SAMPLE_TRAIN = 300000   # you can increase gradually
SAMPLE_TEST  = 100000

X_train_s = X_train[:SAMPLE_TRAIN]
y_train_s = y_train[:SAMPLE_TRAIN]

X_test_s = X_test[:SAMPLE_TEST]
y_test_s = y_test[:SAMPLE_TEST]

print("\n✅ Using sample:")
print("Train sample:", X_train_s.shape)
print("Test sample:", X_test_s.shape)

print("\n🚀 Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=None
)

model.fit(X_train_s, y_train_s)

print("✅ Training complete!")

print("\n🚀 Evaluating model...")
y_pred = model.predict(X_test_s)

acc = accuracy_score(y_test_s, y_pred)
print("✅ Accuracy:", acc)

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test_s, y_pred))

print("\n✅ Classification Report:")
print(classification_report(y_test_s, y_pred))

# Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/cyber_model.pkl")
print("\n✅ Saved trained model: model/cyber_model.pkl")
