import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed dataset
file_path = "preprocessed_parkinsons_data.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=["motor_UPDRS"])  # Features
y = (df["motor_UPDRS"] > df["motor_UPDRS"].median()).astype(int)  # Convert regression target to binary

# Apply PCA for Dimensionality Reduction
pca = PCA(n_components=None)  # Choose 10 principal components
X_pca = pca.fit_transform(X)

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after PCA:", accuracy)
print("Classification Report after PCA:")
print(classification_report(y_test, y_pred))
