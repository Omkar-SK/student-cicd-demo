import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/student_placement.csv")

# Check for missing values
if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values!")

# Features and target
X = df.drop("Placement", axis=1)
y = df["Placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=1,
    stratify=y
)

# Train model
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")

if accuracy < 0.80:
    raise ValueError("Accuracy below 80%!")

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")