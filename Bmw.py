# ============================================================
# BMW Sales Data Analysis & Machine Learning Pipeline
# Assignments 01, 02, & 03
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib 

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
pd.set_option('display.float_format', '{:,.2f}'.format)

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

# ============================================================
# PART 1: Data Cleaning & Preprocessing (Assignments 1 & 2)
# ============================================================
print_header("PART 1: Data Loading and Cleaning")

df = pd.read_csv("BMW sales data (2010-2024).csv")
print(f"Original Dataset Shape: {df.shape}")

# Handling Missing Values
df = df.fillna(df.median(numeric_only=True))
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled.")

# Removing Duplicates
before_dupes = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before_dupes - df.shape[0]} duplicate rows.")

# Handling Outliers (IQR Method) - Applied to Price_USD
Q1 = df['Price_USD'].quantile(0.25)
Q3 = df['Price_USD'].quantile(0.75)
IQR = Q3 - Q1
before_outliers = df.shape[0]
df = df[(df['Price_USD'] >= Q1 - 1.5 * IQR) & (df['Price_USD'] <= Q3 + 1.5 * IQR)]
print(f"Removed {before_outliers - df.shape[0]} outlier rows (Price_USD).")
df = df.reset_index(drop=True)

# Save Cleaned Data (for EDA)
df.to_csv("bmw_sales_cleaned.csv", index=False)
print("Saved 'bmw_sales_cleaned.csv' (Human-readable).")

# ============================================================
# PART 2: Exploratory Data Analysis (EDA) & Key Questions (Assignments 2 & 3)
# ============================================================
print_header("PART 2: Exploratory Data Analysis (Q1-Q5)")

print("Dataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe().T)

# --- Q1. Which BMW models have the highest sales? ---
print("\n[Q1] Analyzing Top BMW Models by Sales...")
model_sales = df.groupby('Model')['Sales_Volume'].sum().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=model_sales.head(15).index, y=model_sales.head(15).values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 15 BMW Models by Total Sales Volume", fontsize=14, fontweight='bold')
plt.ylabel("Total Sales Volume")
plt.tight_layout()
plt.savefig('q1_top_models.png')
plt.close()
print("-> Insight: Top selling models identified. Chart saved as 'q1_top_models.png'.")

# --- Q2. How does the price affect sales volume? ---
print("\n[Q2] Analyzing Price vs Sales Volume...")
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Price_USD', y='Sales_Volume', alpha=0.5)
plt.title("Price vs Sales Volume", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_price_vs_sales.png')
plt.close()

corr_price_sales = df['Price_USD'].corr(df['Sales_Volume'])
print(f"-> Correlation (Price vs Sales): {corr_price_sales:.4f}")
print("-> Insight: Negative correlation implies cheaper cars sell slightly more, but price isn't the only factor.")

# --- Q3. Does mileage or engine size influence customer choices? ---
print("\n[Q3] Analyzing Mileage & Engine Size Influence...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(ax=axes[0], data=df, x='Mileage_KM', y='Sales_Volume', alpha=0.4)
axes[0].set_title("Mileage vs Sales Volume")

sns.scatterplot(ax=axes[1], data=df, x='Engine_Size_L', y='Sales_Volume', alpha=0.5)
axes[1].set_title("Engine Size vs Sales Volume")

plt.tight_layout()
plt.savefig('q3_mileage_engine_influence.png')
plt.close()

corr_mileage = df['Mileage_KM'].corr(df['Sales_Volume'])
corr_engine = df['Engine_Size_L'].corr(df['Sales_Volume'])
print(f"-> Correlation (Mileage vs Sales): {corr_mileage:.4f}")
print(f"-> Correlation (Engine vs Sales): {corr_engine:.4f}")

# --- Q4. What are the most popular car colors or fuel types? ---
print("\n[Q4] Analyzing Popular Colors & Fuel Types...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

color_counts = df['Color'].value_counts()
sns.barplot(ax=axes[0], x=color_counts.head(10).index, y=color_counts.head(10).values, palette="viridis")
axes[0].set_title("Top 10 Colors")
axes[0].tick_params(axis='x', rotation=45)

fuel_counts = df['Fuel_Type'].value_counts()
axes[1].pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set2(range(len(fuel_counts))))
axes[1].set_title("Fuel Type Distribution")

plt.tight_layout()
plt.savefig('q4_colors_fuel.png')
plt.close()
print(f"-> Most Popular Color: {color_counts.index[0]}")
print(f"-> Most Popular Fuel: {fuel_counts.index[0]}")

# --- Q5. What are the BMW sales trends over the years? ---
print("\n[Q5] Analyzing Sales Trends (2010-2024)...")
yearly_sales = df.groupby('Year')['Sales_Volume'].sum()

plt.figure(figsize=(12,6))
sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o')
plt.title("BMW Sales Volume Trend", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q5_sales_trend.png')
plt.close()

growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[0]) / yearly_sales.iloc[0]) * 100
print(f"-> Total Sales Growth (2010-2024): {growth:.2f}%")

# ============================================================
# PART 3: Machine Learning Pipeline (Assignment 3)
# ============================================================
print_header("PART 3: Machine Learning (Price Prediction)")

# 3.1 Data Preparation
# We use the 'df' (cleaned) and must encode ALL categorical columns for ML.
print("Preparing data for training...")
df_ml = df.copy()

# Label Encoding for ALL categorical features
target_col = 'Price_USD'
categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# --- CRITICAL FIX: Store encoders for later use in Deployment ---
encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le # Save the specific encoder for this column

# Create (Processsed) CSV for reference
df_ml.to_csv("bmw_sales_processed.csv", index=False)
print("Saved 'bmw_sales_processed.csv' (Fully Encoded for ML).")

# 3.2 Feature Selection
# DROP 'Sales_Volume' because it's a target-dependent or future metric not available for a single car prediction.
X = df_ml.drop([target_col, 'Sales_Volume', 'Sales_Classification'], axis=1, errors='ignore')
y = df_ml[target_col]

# 3.3 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")

# 3.4 Model 1: Linear Regression
print("\n--- Training Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print(f"MAE:  {lr_mae:,.2f}")
print(f"RMSE: {lr_rmse:,.2f}")
print(f"R2:   {lr_r2:.4f}")

# 3.5 Model 2: Random Forest Regressor
print("\n--- Training Random Forest (this may take a moment) ---")
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) 
# Reduced estimators slightly for speed, n_jobs=-1 uses all cores
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"MAE:  {rf_mae:,.2f}")
print(f"RMSE: {rf_rmse:,.2f}")
print(f"R2:   {rf_r2:.4f}")

# 3.6 Comparison
print("\n--- Model Comparison Results ---")
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2 Score": [lr_r2, rf_r2]
})
print(results)
with open("metrics.txt", "w") as f:
    f.write(results.to_string())


best_model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
best_model = rf if rf_r2 > lr_r2 else lr
print(f"\n[CONCLUSION] The best performing model is: {best_model_name}")

# ============================================================
# PART 4: Deployment Preparation (Saving Model) (Required for Assgn 4)
# ============================================================
print_header("PART 4: Saving Model and Encoders for Deployment")

deployment_artifacts = {
    "model": best_model,
    "encoders": encoders,
    "features": X.columns.tolist(),
    "model_name": best_model_name,
    "r2_score": max(rf_r2, lr_r2)
}

joblib.dump(deployment_artifacts, "bmw_price_model.pkl")
print("[SUCCESS] Model and artifacts saved to 'bmw_price_model.pkl'")

