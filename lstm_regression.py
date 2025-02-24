import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed dataset
file_path = "preprocessed_parkinsons_data.csv"
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=["motor_UPDRS"])  # Features
y = df["motor_UPDRS"].values.reshape(-1, 1)  # Target for regression

# Normalize target variable
target_scaler = MinMaxScaler()
y = target_scaler.fit_transform(y)

# Reshape data for LSTM (samples, time steps, features)
X = np.expand_dims(X, axis=1)  # Add time step dimension

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(150, activation='tanh', return_sequences=True, input_shape=(1, X.shape[2])),
    Dropout(0.2),
    LSTM(100, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1, activation='linear')  # Regression output
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred)  # Convert back to original scale
y_test = target_scaler.inverse_transform(y_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R²): {r2}")
