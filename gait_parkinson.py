# -*- coding: utf-8 -*-
"""Gait_Parkinson.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qdXtPyCXPw86cmI0FFdJ5_vLwpUlxTb1
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import glob
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Start timing
start_time = time.time()

# Step 1: Load all TXT files
directory_path = "/content/drive/My Drive/gait-in-parkinsons-disease"  # Update path
all_files = glob.glob(os.path.join(directory_path, "*.txt"))

df_list = []
for file in all_files:
    try:
        temp_df = pd.read_csv(file, sep="\t", header=None, engine="python", on_bad_lines="skip")
        df_list.append(temp_df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all files
df = pd.concat(df_list, ignore_index=True)

# Step 2: Preprocess Data
df = df.dropna()  # Remove missing values

# Assign column names if needed
num_columns = df.shape[1]  # Get the number of columns
column_names = [f"Feature_{i}" for i in range(num_columns - 1)] + ["Label"]
df.columns = column_names

# Convert all values to float
df = df.astype(float)

# If dataset is huge, sample only 50,000 rows for faster training
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

# Check label values
print("Unique Label Values:", df["Label"].unique())

# Convert labels to binary classification
df["Label"] = df["Label"].apply(lambda x: 1 if x > df["Label"].median() else 0)

# Split data into features and labels
X = df.drop(columns=["Label"])  # Features
y = df["Label"].astype(int)  # Convert to integer labels

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use fewer trees for faster training
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 4: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
execution_time = end_time - start_time

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"⚡ Execution Time: {execution_time:.2f} seconds")

print(df["Label"].value_counts(normalize=True))  # To see % of each class

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")

import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

# Correct way:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform test data, don't fit again!

# Split data into features and labels
X = df.drop(columns=["Label"])  # Features
y = df["Label"].astype(int)  # Convert to integer labels

# Split into train and test BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Step 3: Train Model
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()