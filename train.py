# train_model.py
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data, target = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Save scaler and model
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and scaler saved.")