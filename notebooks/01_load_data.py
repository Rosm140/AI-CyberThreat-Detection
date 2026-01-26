import pandas as pd

DATA_PATH = "data/cicids2017_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("\n✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)

print("\n✅ Column names:")
print(df.columns)

print("\n✅ First 5 rows:")
print(df.head())

print("\n✅ Missing values (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))
