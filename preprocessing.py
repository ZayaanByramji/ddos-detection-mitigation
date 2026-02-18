import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load datasets (update paths if needed)
benign_path = "Benign-Monday-no-metadata.parquet"
ddos_path = "DDoS-Friday-no-metadata.parquet"

df_benign = pd.read_parquet(benign_path)
df_ddos = pd.read_parquet(ddos_path)


# Combine datasets
df = pd.concat([df_benign, df_ddos], ignore_index=True)

print("Combined dataset shape:", df.shape)
print(df["Label"].value_counts())


# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("After cleaning:", df.shape)


# Encode labels: Benign -> 0, DDoS -> 1
df["Label"] = df["Label"].map({"Benign": 0, "DDoS": 1})

print(df["Label"].value_counts())


# Feature / target split
X = df.drop(columns=["Label"])
y = df["Label"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final check
print("Preprocessing completed successfully.")
print("Ready for model training.")

# Save preprocessed data
X_train.to_parquet("X_train.parquet")
X_test.to_parquet("X_test.parquet")

y_train.to_frame("Label").to_parquet("y_train.parquet")
y_test.to_frame("Label").to_parquet("y_test.parquet")

print("Preprocessed data saved to disk.")
