import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed dataset
file_path = "preprocessed_parkinsons_full.csv"
df = pd.read_csv(file_path)

# Define Features (X) and Target (y)
X = df.drop(columns=["motor_UPDRS"])
y = df["motor_UPDRS"].values.reshape(-1, 1)

# Print min and max UPDRS before scaling (Debugging Step)
print(f"Min UPDRS: {y.min()}, Max UPDRS: {y.max()}")

# Save feature names before converting to NumPy
feature_names = list(X.columns)

# Normalize Features (0-1 scaling)
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

# Normalize Target (UPDRS) (MinMax scaling based on actual range)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)

# Print scaler parameters (To verify correct scaling)
print(f"Feature Scaler Min: {feature_scaler.data_min_}, Max: {feature_scaler.data_max_}")
print(f"Target Scaler Min: {target_scaler.data_min_}, Max: {target_scaler.data_max_}")

# Save scalers and feature names
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

# Reshape Data for LSTM (samples, time steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(200, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.3),
    LSTM(100, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1, activation='linear')  # Regression output
])

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Save Trained Model in Both Formats
model.save("trained_lstm_model.keras")  # TensorFlow's recommended format
model.save("trained_lstm_model.h5")  # HDF5 format for compatibility

# Make Predictions
y_pred_scaled = model.predict(X_test)

# Convert Predictions Back to Original Scale (Fixed)
y_pred = target_scaler.inverse_transform(y_pred_scaled)  # Correct inverse transform
y_test = target_scaler.inverse_transform(y_test)  # Convert test values back for evaluation

# Evaluate Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R²): {r2}")
