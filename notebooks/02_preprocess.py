import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = "data/cicids2017_cleaned.csv"
LABEL_COL = "Attack Type"

print("🚀 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("✅ Loaded shape:", df.shape)

# Separate X and y
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

print("\n✅ Label distribution (Top 15):")
print(y.value_counts().head(15))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n✅ Classes found:")
print(list(le.classes_))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Save split data (optional, for faster training later)
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "model/split_data.pkl")

print("\n✅ Preprocessing complete!")
print("✅ Saved: model/label_encoder.pkl")
print("✅ Saved: model/scaler.pkl")
print("✅ Saved: model/split_data.pkl")
