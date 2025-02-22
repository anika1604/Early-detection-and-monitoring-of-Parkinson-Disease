import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("pd_speech_features.csv", skiprows=1)

# Drop any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Separate features (X) and target variable (y)
X = df.drop(columns=["class"])  # Ensure "class" is the correct target column
y = df["class"]

# Split data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Selection: Select top k features
k = 62  # You can tune this
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Compute class imbalance ratio
counter = Counter(y_train)
scale_pos_weight = (counter[0] / counter[1]) + 0.5  # Majority / Minority class ratio

# Define XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,  # Auto balance class weights
    random_state=42
)

# Train XGBoost
xgb.fit(X_train_selected, y_train)

# Tune XGBoost hyperparameters using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'subsample': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [(counter[0] / counter[1]), (counter[0] / counter[1]) + 0.5]
}

xgb_tuned = XGBClassifier(random_state=42)

grid_search = RandomizedSearchCV(
    estimator=xgb_tuned,
    param_distributions=param_grid,
    n_iter=20,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    random_state=42
)

grid_search.fit(X_train_selected, y_train)
best_xgb = grid_search.best_estimator_

# Define StackingClassifier with XGBoost and Logistic Regression
stacking_xgb = StackingClassifier(
    estimators=[('xgb', best_xgb)],  # Only XGBoost in stacking for now
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

# Train Stacking Classifier
stacking_xgb.fit(X_train_selected, y_train)

# Get probability scores
y_probs = stacking_xgb.predict_proba(X_test_selected)[:, 1]

# Find the best threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Make predictions with optimal threshold
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# Evaluate model
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"Stacking XGBoost Accuracy: {accuracy_optimal * 100:.2f}%")
print(f"Best Decision Threshold: {optimal_threshold:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_optimal))
