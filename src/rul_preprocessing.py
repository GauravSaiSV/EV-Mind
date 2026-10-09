import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# EV-MIND - RUL PREPROCESSING
# Degradation-stage-aware sequence splitting
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "EV_IoT_PredictiveMaintenance_15min.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "processed_rul")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 32
RANDOM_STATE = 42


print("=" * 70)
print("EV-MIND - EV POWERTRAIN RUL PREPROCESSING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("\nTimestamp range:")
print("Start:", df["timestamp"].min())
print("End  :", df["timestamp"].max())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

df = df.dropna().reset_index(drop=True)

print("\nMissing values after cleaning:", df.isnull().sum().sum())


# ============================================================
# 3. KEEP POSITIVE RUL
# ============================================================

print("\nOriginal RUL statistics:")
print(df["RUL_hours"].describe())
df = df[df["RUL_hours"] > 0].copy()
df = df.reset_index(drop=True)

print("\nAfter removing zero-RUL plateau:")
print("Rows:", len(df))
print("RUL minimum:", df["RUL_hours"].min())
print("RUL maximum:", df["RUL_hours"].max())


# ============================================================
# 4. REMOVE NON-PREDICTIVE / TARGET-DERIVED COLUMNS
# ============================================================

DROP_COLUMNS = [
    "timestamp",
    "RUL_hours",
    "health_class",
    "charge_cycle_count",
    "overheat_event_flag",
    "overheat_event_count_7d",
    "maintenance_history_flag"
]

DROP_COLUMNS = [
    col for col in DROP_COLUMNS
    if col in df.columns
]

features_df = df.drop(columns=DROP_COLUMNS)


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

CATEGORICAL_COLUMNS = [
    "operating_mode",
    "load_condition"
]

CATEGORICAL_COLUMNS = [
    col for col in CATEGORICAL_COLUMNS
    if col in features_df.columns
]

features_df = pd.get_dummies(
    features_df,
    columns=CATEGORICAL_COLUMNS,
    dtype=float
)


# ============================================================
# 6. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    col
    for col in features_df.columns
    if features_df[col].nunique() <= 1
]

if constant_columns:
    print("\nRemoving constant features:")
    print(constant_columns)

    features_df = features_df.drop(columns=constant_columns)


print("\nFinal number of features:", features_df.shape[1])

print("\nFeatures:")
for feature in features_df.columns:
    print(" -", feature)


# ============================================================
# 7. RAW ARRAYS
# ============================================================

X_raw = features_df.values.astype(np.float32)

y_raw = df["RUL_hours"].values.astype(np.float32)


# ============================================================
# 8. CREATE NON-OVERLAPPING SEQUENCES
# ============================================================

print("\nCreating non-overlapping sequences...")

X_sequences = []
y_sequences = []
sequence_rul = []

# Step by SEQUENCE_LENGTH so sequences do not overlap.
for start in range(
    0,
    len(X_raw) - SEQUENCE_LENGTH + 1,
    SEQUENCE_LENGTH
):

    end = start + SEQUENCE_LENGTH
    X_seq = X_raw[start:end]
    # Target = RUL at the final timestep
    y_target = y_raw[end - 1]
    X_sequences.append(X_seq)
    y_sequences.append(y_target)
    sequence_rul.append(y_target)
X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.float32)
sequence_rul = np.array(sequence_rul, dtype=np.float32)


print("Total sequences:", len(X_sequences))

print("Sequence shape:", X_sequences.shape)


# ============================================================
# 9. CREATE RUL STAGES
# ============================================================

# Divide RUL into 10 degradation stages.
# This allows train/validation/test to contain
# early, middle and late degradation examples.

num_bins = 10

rul_bins = pd.qcut(
    sequence_rul,
    q=num_bins,
    labels=False,
    duplicates="drop"
)

rul_bins = np.asarray(rul_bins)


print("\nRUL stage distribution:")

for stage in sorted(np.unique(rul_bins)):
    count = np.sum(rul_bins == stage)

    print(
        f"Stage {stage + 1}: "
        f"{count} sequences"
    )


# ============================================================
# 10. STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================

indices = np.arange(len(X_sequences))


# First: 70% train, 30% temporary
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=rul_bins
)


# Split temporary into 15% validation + 15% test
temp_bins = rul_bins[temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_bins
)


print("\nSplit sizes:")

print("Training  :", len(train_idx))
print("Validation:", len(val_idx))
print("Testing   :", len(test_idx))


# ============================================================
# 11. SORT EACH SPLIT BY TIME/ORIGINAL ORDER
# ============================================================

train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)
test_idx = np.sort(test_idx)


X_train_raw = X_sequences[train_idx]
X_val_raw = X_sequences[val_idx]
X_test_raw = X_sequences[test_idx]

y_train_raw = y_sequences[train_idx]
y_val_raw = y_sequences[val_idx]
y_test_raw = y_sequences[test_idx]


# ============================================================
# 12. DISPLAY RUL DISTRIBUTION
# ============================================================

print("\nRUL distribution by split:")

for name, values in [
    ("TRAIN", y_train_raw),
    ("VALIDATION", y_val_raw),
    ("TEST", y_test_raw)
]:

    print(f"\n{name}")

    print(
        f"Min  : {values.min():.2f} hours"
    )

    print(
        f"Max  : {values.max():.2f} hours"
    )

    print(
        f"Mean : {values.mean():.2f} hours"
    )

    print(
        f"Median: {np.median(values):.2f} hours"
    )


# ============================================================
# 13. SCALE FEATURES
# ============================================================

print("\nScaling features...")

feature_scaler = MinMaxScaler()

# Fit ONLY on training data.
feature_scaler.fit(
    X_train_raw.reshape(
        -1,
        X_train_raw.shape[-1]
    )
)


def scale_sequences(X):
    original_shape = X.shape

    X_2d = X.reshape(
        -1,
        original_shape[-1]
    )

    X_scaled = feature_scaler.transform(X_2d)

    return X_scaled.reshape(original_shape).astype(
        np.float32
    )

X_train = scale_sequences(X_train_raw)
X_val = scale_sequences(X_val_raw)
X_test = scale_sequences(X_test_raw)


# ============================================================
# 14. SCALE TARGET
# ============================================================

target_scaler = MinMaxScaler()

target_scaler.fit(
    y_train_raw.reshape(-1, 1)
)


y_train = target_scaler.transform(
    y_train_raw.reshape(-1, 1)
).astype(np.float32)


y_val = target_scaler.transform(
    y_val_raw.reshape(-1, 1)
).astype(np.float32)


y_test = target_scaler.transform(
    y_test_raw.reshape(-1, 1)
).astype(np.float32)


# ============================================================
# 15. SAVE PROCESSED DATA
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_val.npy"),
    X_val
)

np.save(
    os.path.join(OUTPUT_DIR, "y_val.npy"),
    y_val
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)


# ============================================================
# 16. SAVE SCALERS
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        feature_scaler,
        f
    )


with open(
    os.path.join(
        OUTPUT_DIR,
        "target_scaler.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        target_scaler,
        f
    )


# ============================================================
# 17. SAVE CONFIGURATION
# ============================================================

config = {

    "sequence_length":
        SEQUENCE_LENGTH,

    "num_features":
        X_train.shape[-1],

    "feature_names":
        list(features_df.columns),

    "target":
        "RUL_hours",

    "split_method":
        "RUL-stage stratified non-overlapping sequences",

    "random_state":
        RANDOM_STATE
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "config.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        config,
        f
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print("\nFinal dataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nNumber of features:", X_train.shape[-1])

print("Sequence length:", SEQUENCE_LENGTH)

print("\nProcessed data saved to:")

print(OUTPUT_DIR)

print("\nFiles created:")

print(" - X_train.npy")
print(" - y_train.npy")
print(" - X_val.npy")
print(" - y_val.npy")
print(" - X_test.npy")
print(" - y_test.npy")
print(" - feature_scaler.pkl")
print(" - target_scaler.pkl")
print(" - config.pkl")

print("\nReady for deep learning training!")