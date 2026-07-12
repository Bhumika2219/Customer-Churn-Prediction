import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------
# Load Saved Models
# -------------------------

lr = joblib.load("saved_model/logistic_regression.pkl")
dt = joblib.load("saved_model/decision_tree.pkl")
rf = joblib.load("saved_model/random_forest.pkl")

# -------------------------
# Load Test Data
# -------------------------

X_test = joblib.load("saved_model/X_test.pkl")
y_test = joblib.load("saved_model/y_test.pkl")

# -------------------------
# Function to Evaluate
# -------------------------

def evaluate(model, name):

    prediction = model.predict(X_test)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction))
    print("Recall   :", recall_score(y_test, prediction))
    print("F1 Score :", f1_score(y_test, prediction))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))


# -------------------------
# Evaluate All Models
# -------------------------

evaluate(lr, "Logistic Regression")

evaluate(dt, "Decision Tree")

evaluate(rf, "Random Forest")