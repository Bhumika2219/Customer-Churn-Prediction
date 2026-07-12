import pandas as pd

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/customer_churn.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# -------------------------
# Remove Customer ID
# -------------------------

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# -------------------------
# Convert TotalCharges
# -------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# -------------------------
# Fill Missing Values
# -------------------------

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# -------------------------
# Remove Duplicate Rows
# -------------------------

df.drop_duplicates(inplace=True)

# -------------------------
# Final Information
# -------------------------

print("\nCleaned Dataset Shape")
print(df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# -------------------------
# Save Clean Dataset
# -------------------------

df.to_csv(
    "data/clean_customer_churn.csv",
    index=False
)

print("\nClean dataset saved successfully!")