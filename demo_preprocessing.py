import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("\n=== STEP 1: LOAD RAW DATA ===")
df = pd.read_csv("BMW sales data (2010-2024).csv")
print(f"Original Dataset Shape: {df.shape}")
print(df[['Price_USD', 'Model']].head(3))

# 2. Outlier Handling
print("\n=== STEP 2: HANDLING OUTLIERS (IQR METHOD) ===")
Q1 = df['Price_USD'].quantile(0.25)
Q3 = df['Price_USD'].quantile(0.75)
IQR = Q3 - Q1
print(f"IQR Value: {IQR}")
print(f"Lower Bound: {Q1 - 1.5 * IQR}")
print(f"Upper Bound: {Q3 + 1.5 * IQR}")

df_filtered = df[(df['Price_USD'] >= Q1 - 1.5 * IQR) & (df['Price_USD'] <= Q3 + 1.5 * IQR)]
print(f"Rows Before: {df.shape[0]}")
print(f"Rows After:  {df_filtered.shape[0]}")
print(f"Removed {df.shape[0] - df_filtered.shape[0]} outliers.")

# 3. Encoding/Scaling
print("\n=== STEP 3: NORMALIZING & ENCODING ===")
df_ml = df_filtered.copy()
le = LabelEncoder()
print("Before Encoding (Fuel_Type):")
print(df_ml['Fuel_Type'].head(4).values)

df_ml['Fuel_Type'] = le.fit_transform(df_ml['Fuel_Type'].astype(str))
print("After Encoding (Fuel_Type):")
print(df_ml['Fuel_Type'].head(4).values)

print("\n=== PREPROCESSING DEMO COMPLETE for SCREENSHOT ===")
