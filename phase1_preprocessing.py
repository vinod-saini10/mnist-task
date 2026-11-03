
# ==========================================
# PHASE 1: DATA ACQUISITION & PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 1. LOAD MNIST CSV FILE
df = pd.read_csv("mnist.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# 2. DISPLAY FIRST 5 ROWS
print(df.head())

# 3. CHECK DATA INFORMATION
print("\nDataset Information:")
print(df.info())

# 4. SEPARATE LABEL AND PIXEL DATA
# Label = actual digit (0-9)
# Pixels = image information

if "label" not in df.columns:
    raise ValueError(
        "CSV must contain a 'label' column for supervised training."
    )

y = df["label"].values
X = df.drop(columns=["label"]).values

print("\nFeatures Shape:", X.shape)
print("Labels Shape:", y.shape)

# Check that each image has 784 pixels
if X.shape[1] != 784:
    raise ValueError("Each MNIST image must contain 784 pixel columns.")

# 5. CONVERT PIXELS INTO IMAGE FORMAT
# 784 pixels -> 28 x 28 image

X = X.astype(np.float32).reshape(-1, 1, 28, 28)

# 6. NORMALIZE PIXEL VALUES
# Convert pixel range 0-255 into 0-1

X = X / 255.0

# Standard MNIST normalization
mean = 0.1307
std = 0.3081

X = (X - mean) / std

print("\nNormalized Image Shape:", X.shape)

# 7. DISPLAY SAMPLE IMAGE

plt.imshow(
    X[0][0] * std + mean,
    cmap="gray"
)

plt.title(f"Actual Digit: {y[0]}")
plt.axis("off")
plt.show()

# 8. SPLIT DATA INTO TRAINING AND VALIDATION

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Images:", X_train.shape)
print("Validation Images:", X_val.shape)

# 9. SAVE PROCESSED DATA

np.savez_compressed(
    "mnist_processed.npz",
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val
)

print("\nPreprocessing Completed!")
print("Processed data saved successfully!")
