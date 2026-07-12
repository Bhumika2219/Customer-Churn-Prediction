import os
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/clean_customer_churn.csv")

# Create reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

print("="*50)
print("Dataset Shape:", df.shape)
print("="*50)

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
print(df.describe())

print("\nChurn Distribution")
print(df["Churn"].value_counts())

# -----------------------------
# Churn Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/churn_distribution.png")
plt.close()

print("EDA completed successfully!")