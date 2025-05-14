import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nModel: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ------------------------
# Part 1: Wine Classification
# ------------------------

# Load and prepare the Wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_wine = scaler.fit_transform(X_train_wine)
X_test_wine = scaler.transform(X_test_wine)

# Train models
models_wine = [
    LogisticRegression(max_iter=1000),
    SVC(),
    DecisionTreeClassifier(random_state=0)
]

for model in models_wine:
    model.fit(X_train_wine, y_train_wine)
    evaluate_model(model, X_test_wine, y_test_wine)

# Effect of changing random state in Decision Tree
print("\n--- Decision Tree with different random states ---")
for state in [0, 10, 42, 99]:
    dt = DecisionTreeClassifier(random_state=state)
    dt.fit(X_train_wine, y_train_wine)
    print(f"Random State: {state}")
    evaluate_model(dt, X_test_wine, y_test_wine)


# ------------------------
# Part 2: Breast Cancer Classification
# ------------------------

# Load and prepare the Breast Cancer dataset
print("Breast cancer section")
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42)

X_train_cancer = scaler.fit_transform(X_train_cancer)
X_test_cancer = scaler.transform(X_test_cancer)

# Train models on Breast Cancer dataset
models_cancer = [
    LogisticRegression(max_iter=1000),
    SVC(),
    DecisionTreeClassifier(random_state=0)
]

print("\n--- Breast Cancer Classification ---")
for model in models_cancer:
    model.fit(X_train_cancer, y_train_cancer)
    evaluate_model(model, X_test_cancer, y_test_cancer)

# Evaluate Random Forest on Breast Cancer
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_cancer, y_train_cancer)
print("\n--- Random Forest on Breast Cancer Dataset ---")
evaluate_model(rf, X_test_cancer, y_test_cancer)
