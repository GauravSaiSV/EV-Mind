import os
import pandas as pd
import numpy as np


# ============================================================
# EV-MIND RUL DATASET CHECK
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "EV_IoT_PredictiveMaintenance_15min.csv"
)

print("=" * 70)
print("EV-MIND RUL DATASET ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nTimestamp range:")
print("Start:", df["timestamp"].min())
print("End  :", df["timestamp"].max())


# ============================================================
# RUL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("RUL ANALYSIS")
print("=" * 70)

print("\nRUL minimum:", df["RUL_hours"].min())
print("RUL maximum:", df["RUL_hours"].max())
print("RUL mean   :", df["RUL_hours"].mean())
print("RUL median :", df["RUL_hours"].median())


# ============================================================
# HEALTH CLASSES
# ============================================================

print("\nHealth class distribution:")

print(
    df["health_class"].value_counts()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end]
test = df.iloc[val_end:]


print("\n" + "=" * 70)
print("RUL RANGE BY SPLIT")
print("=" * 70)


for name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    print(f"\n{name}")

    print(
        "RUL min    :",
        data["RUL_hours"].min()
    )

    print(
        "RUL max    :",
        data["RUL_hours"].max()
    )

    print(
        "RUL mean   :",
        data["RUL_hours"].mean()
    )

    print(
        "RUL median :",
        data["RUL_hours"].median()
    )


# ============================================================
# RUL MONOTONICITY
# ============================================================

rul_diff = df["RUL_hours"].diff().dropna()

increases = (
    rul_diff > 0
).sum()

decreases = (
    rul_diff < 0
).sum()

unchanged = (
    rul_diff == 0
).sum()


print("\n" + "=" * 70)
print("RUL TREND")
print("=" * 70)

print(
    "RUL decreases:",
    decreases
)

print(
    "RUL increases:",
    increases
)

print(
    "RUL unchanged:",
    unchanged
)


# ============================================================
# SAMPLE RUL VALUES
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 RUL VALUES")
print("=" * 70)

print(
    df[
        [
            "timestamp",
            "RUL_hours",
            "health_class"
        ]
    ].head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("LAST 10 RUL VALUES")
print("=" * 70)

print(
    df[
        [
            "timestamp",
            "RUL_hours",
            "health_class"
        ]
    ].tail(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("RUL DATASET CHECK COMPLETED")
print("=" * 70)