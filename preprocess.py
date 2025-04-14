import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "parkinsons_updrs.data"  # Ensure correct path
df = pd.read_csv(file_path)

# Check for missing values
print("Missing values before imputation:\n", df.isnull().sum())

# Step 1: Handle Missing Values (replace NaNs with column mean)
imputer = SimpleImputer(strategy="mean")
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# Step 2: Normalize All Features using Min-Max Scaling (excluding first column if it's an ID)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Save the scaler for inverse transformation later
import joblib
joblib.dump(scaler, "feature_scaler.pkl")

# Save the preprocessed dataset
df.to_csv("preprocessed_parkinsons_full.csv", index=False)
print("Preprocessing complete! Saved as 'preprocessed_parkinsons_full.csv'.")
