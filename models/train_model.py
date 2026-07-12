import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/clean_customer_churn.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------
# Encode Categorical Columns
# -------------------------

label_encoders = {}

for column in df.select_dtypes(include=["object"]).columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column].astype(str))

    label_encoders[column] = encoder

print("\nEncoding Completed!")

# -------------------------
# Check Missing Values
# -------------------------

print("\nMissing Values")

print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

# -------------------------
# Features and Target
# -------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------
# Logistic Regression
# -------------------------

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

print("Logistic Regression Trained")

# -------------------------
# Decision Tree
# -------------------------

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

print("Decision Tree Trained")

# -------------------------
# Random Forest
# -------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("Random Forest Trained")

# -------------------------
# Save Models
# -------------------------

joblib.dump(
    lr,
    "saved_model/logistic_regression.pkl"
)

joblib.dump(
    dt,
    "saved_model/decision_tree.pkl"
)

joblib.dump(
    rf,
    "saved_model/random_forest.pkl"
)

joblib.dump(
    X_test,
    "saved_model/X_test.pkl"
)

joblib.dump(
    y_test,
    "saved_model/y_test.pkl"
)

joblib.dump(
    label_encoders,
    "saved_model/label_encoders.pkl"
)

print("\nModels Saved Successfully!")

print("\nProject Completed Successfully!")