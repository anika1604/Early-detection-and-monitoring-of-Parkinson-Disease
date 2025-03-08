import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv(r"C:\Users\CVO ECL\Downloads\Early-detection-and-monitoring-of-Parkinson-Disease-main\Early-detection-and-monitoring-of-Parkinson-Disease-main\preprocessed_parkinsons_data.csv")

# Print the columns to check the names and ensure that 'total_UPDRS' exists
print(df.columns)

# Step 1: Handle Missing Values using SimpleImputer
imputer = SimpleImputer(strategy="mean")  # Replace missing values with column mean
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# Step 2: Normalize Data using Min-Max Scaling
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Step 3: Convert motor_UPDRS into binary classification target
# Convert 'motor_UPDRS' to binary: 1 for Parkinson's disease, 0 for no Parkinson's disease
median_updrs = df['motor_UPDRS'].median()
df['has_parkinsons'] = (df['motor_UPDRS'] > median_updrs).astype(int)  # 1 if motor_UPDRS > median, else 0

# Step 4: Check the columns and drop the correct ones
# If 'total_UPDRS' is not present, drop only 'motor_UPDRS' and 'has_parkinsons'
columns_to_drop = ["motor_UPDRS", "has_parkinsons"]
if "total_UPDRS" in df.columns:
    columns_to_drop.append("total_UPDRS")

X = df.drop(columns=columns_to_drop)  # Features
y = df["has_parkinsons"]  # Target variable (0 or 1 indicating Parkinson's presence)

# Step 5: Feature Selection using SelectKBest (using regression scoring function)
select_k_best = SelectKBest(score_func=f_regression, k=10)  # Select top 10 features
X_selected = select_k_best.fit_transform(X, y)
selected_features = X.columns[select_k_best.get_support()]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Step 6: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.2, random_state=42)

# Step 7: Apply PCA for Dimensionality Reduction (Optional, for further optimization)
pca = PCA(n_components=5)  # Reduce to 5 components for simplicity
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 8: Model 1 - Random Forest Classifier (For classification tasks)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train_pca, y_train)

# Step 9: Model 2 - Logistic Regression (Optional for comparison)
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_pca, y_train)

# Step 10: Combine models using VotingClassifier
voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('logreg', logreg_model)], voting='hard')
voting_clf.fit(X_train_pca, y_train)

# Evaluate Voting Classifier Model
y_pred = voting_clf.predict(X_test_pca)
voting_accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {voting_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Print raw predictions (0 or 1) for the test set
print("Predictions for the test set:", y_pred)

# Step 11: Feature Importance Visualization (For Random Forest)
feature_importances = rf_model.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

# Final Conclusion
print("Model training and evaluation complete!")
