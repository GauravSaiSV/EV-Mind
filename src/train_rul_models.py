import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# EV-MIND
# EV POWERTRAIN RUL DEEP LEARNING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed_rul"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models_rul"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results_rul"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 70)
print("EV-MIND - EV POWERTRAIN RUL DEEP LEARNING")
print("=" * 70)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("\nLoading processed RUL data...")

X_train = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        PROCESSED_DIR,
        "y_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X_val.npy"
    )
)

y_val = np.load(
    os.path.join(
        PROCESSED_DIR,
        "y_val.npy"
    )
)

X_test = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        PROCESSED_DIR,
        "y_test.npy"
    )
)


# Load target scaler
target_scaler = joblib.load(
    os.path.join(
        PROCESSED_DIR,
        "target_scaler.pkl"
    )
)


print("\nData loaded successfully!")

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = X_train.shape[1]

NUM_FEATURES = X_train.shape[2]

EPOCHS = 30

BATCH_SIZE = 256


print("\nSequence length:", SEQUENCE_LENGTH)

print("Number of features:", NUM_FEATURES)

print("Maximum epochs:", EPOCHS)

print("Batch size:", BATCH_SIZE)


# ============================================================
# 3. CALLBACKS
# ============================================================

def create_callbacks():

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    return [
        early_stopping,
        reduce_lr
    ]


# ============================================================
# 4. LSTM MODEL
# ============================================================

def build_lstm_model():
    model = Sequential([
        Input(
            shape=(
                SEQUENCE_LENGTH,
                NUM_FEATURES
            )
        ),
        LSTM(64),
        Dropout(0.2),
        Dense(
            32,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# 5. GRU MODEL
# ============================================================

def build_gru_model():

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                NUM_FEATURES
            )
        ),

        GRU(64),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 6. LSTM + ATTENTION MODEL
# ============================================================

def build_attention_model():

    inputs = Input(
        shape=(
            SEQUENCE_LENGTH,
            NUM_FEATURES
        )
    )

    # LSTM returns the hidden representation
    # for every time step.
    lstm_output = LSTM(
        64,
        return_sequences=True
    )(inputs)

    # Self-attention
    attention_output = Attention()(
        [
            lstm_output,
            lstm_output
        ]
    )

    # Convert sequence representation
    # into a single feature vector.
    pooled = GlobalAveragePooling1D()(
        attention_output
    )

    dense = Dense(
        32,
        activation="relu"
    )(pooled)

    dropout = Dropout(
        0.2
    )(dense)

    output = Dense(1)(
        dropout
    )

    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name
):

    print(
        f"\nEvaluating {model_name}..."
    )

    predictions_scaled = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Convert normalized predictions
    # back to actual RUL hours.
    predictions = target_scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = target_scaler.inverse_transform(
        y_test
    ).flatten()

    mae = mean_absolute_error(
        actual,
        predictions
    )

    mse = mean_squared_error(
        actual,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predictions
    )

    print(
        f"\n{model_name} Results:"
    )

    print(
        f"MAE  : {mae:.4f} hours"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f} hours"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 8. TRAIN LSTM
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1: LSTM")
print("=" * 70)
lstm_model = build_lstm_model()
lstm_model.summary()
history_lstm = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=create_callbacks(),
    verbose=1
)
lstm_model.save(
    os.path.join(
        MODELS_DIR,
        "rul_lstm_model.keras"
    )
)
lstm_results = evaluate_model(
    lstm_model,
    "LSTM"
)


# ============================================================
# 9. TRAIN GRU
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2: GRU")
print("=" * 70)


gru_model = build_gru_model()

gru_model.summary()


history_gru = gru_model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=create_callbacks(),

    verbose=1
)


gru_model.save(
    os.path.join(
        MODELS_DIR,
        "rul_gru_model.keras"
    )
)


gru_results = evaluate_model(
    gru_model,
    "GRU"
)


# ============================================================
# 10. TRAIN LSTM + ATTENTION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3: LSTM + ATTENTION")
print("=" * 70)


attention_model = build_attention_model()

attention_model.summary()


history_attention = attention_model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=create_callbacks(),

    verbose=1
)


attention_model.save(
    os.path.join(
        MODELS_DIR,
        "rul_lstm_attention_model.keras"
    )
)


attention_results = evaluate_model(
    attention_model,
    "LSTM + Attention"
)


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([

    lstm_results,

    gru_results,

    attention_results

])


# Lower RMSE is better
results = results.sort_values(
    by="RMSE"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("FINAL RUL MODEL COMPARISON")
print("=" * 70)


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "rul_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 13. IDENTIFY BEST MODEL
# ============================================================

best_model_name = (
    results.iloc[0]["Model"]
)


if best_model_name == "LSTM":

    best_model = lstm_model

    best_model_file = (
        "rul_lstm_model.keras"
    )

elif best_model_name == "GRU":

    best_model = gru_model

    best_model_file = (
        "rul_gru_model.keras"
    )

else:

    best_model = attention_model

    best_model_file = (
        "rul_lstm_attention_model.keras"
    )


print("\n" + "=" * 70)
print("BEST RUL MODEL")
print("=" * 70)


print(
    "Model:",
    best_model_name
)

print(
    "MAE:",
    results.iloc[0]["MAE"],
    "hours"
)

print(
    "RMSE:",
    results.iloc[0]["RMSE"],
    "hours"
)

print(
    "R²:",
    results.iloc[0]["R2"]
)

print(
    "\nSaved model:",
    best_model_file
)


print("\n" + "=" * 70)
print("RUL MODEL TRAINING COMPLETED!")
print("=" * 70)