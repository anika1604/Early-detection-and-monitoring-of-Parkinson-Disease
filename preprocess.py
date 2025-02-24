import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Load the dataset
file_path = "parkinsons_updrs.data"  # Change path if needed
df = pd.read_csv(file_path)

# Step 1: Handle Missing Values
imputer = SimpleImputer(strategy="mean")  # Replace missing values with column mean
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# Step 2: Normalize Data using Min-Max Scaling
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Step 3: Skip SMOTE since the target is continuous (regression task)
X = df.drop(columns=["motor_UPDRS", "total_UPDRS"])  # Features
y = df["motor_UPDRS"]  # Target variable (change to "total_UPDRS" if needed)

X_resampled, y_resampled = X, y  # Keep data unchanged for regression

# Step 4: Feature Selection using Relief-F
select_k_best = SelectKBest(score_func=f_regression, k=10)  # Select top 10 features
X_selected = select_k_best.fit_transform(X_resampled, y_resampled)
selected_features = X.columns[select_k_best.get_support()]

# Create final preprocessed DataFrame
final_df = pd.DataFrame(X_selected, columns=selected_features)
final_df["motor_UPDRS"] = y_resampled

# Save preprocessed dataset
final_df.to_csv("preprocessed_parkinsons_data.csv", index=False)

print("Preprocessing complete! File saved as 'preprocessed_parkinsons_data.csv'")
