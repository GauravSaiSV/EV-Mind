import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# EV-MIND - RUL RESULTS GENERATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed_rul"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_rul"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results_rul"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


print("=" * 70)
print("EV-MIND - RUL RESULTS GENERATION")
print("=" * 70)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

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


# ============================================================
# 2. LOAD TARGET SCALER
# ============================================================

with open(
    os.path.join(
        PROCESSED_DIR,
        "target_scaler.pkl"
    ),
    "rb"
) as f:

    target_scaler = pickle.load(f)


# ============================================================
# 3. LOAD BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rul_lstm_attention_model.keras"
)

print("\nLoading best model:")
print(MODEL_PATH)

model = load_model(
    MODEL_PATH
)


# ============================================================
# 4. PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 5. INVERSE TRANSFORM
# ============================================================

y_actual = target_scaler.inverse_transform(
    y_test
).flatten()

y_pred = target_scaler.inverse_transform(
    y_pred_scaled
).flatten()


# ============================================================
# 6. METRICS
# ============================================================

mae = mean_absolute_error(
    y_actual,
    y_pred
)

mse = mean_squared_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_actual,
    y_pred
)


print("\n" + "=" * 70)
print("BEST MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.2f} hours")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f} hours")
print(f"R²   : {r2:.4f}")


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_RUL_hours":
        y_actual,

    "Predicted_RUL_hours":
        y_pred,

    "Absolute_Error_hours":
        np.abs(
            y_actual - y_pred
        )

})


prediction_path = os.path.join(
    RESULT_DIR,
    "rul_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


print("\nPrediction file saved:")
print(prediction_path)


# ============================================================
# 8. ACTUAL VS PREDICTED RUL
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    y_actual,
    label="Actual RUL"
)

plt.plot(
    y_pred,
    label="Predicted RUL"
)

plt.xlabel(
    "Test Sequence"
)

plt.ylabel(
    "RUL (hours)"
)

plt.title(
    "Actual vs Predicted EV Powertrain RUL"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "actual_vs_predicted_rul.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 9. SCATTER PLOT
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.6
)

min_value = min(
    y_actual.min(),
    y_pred.min()
)

max_value = max(
    y_actual.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual RUL (hours)"
)

plt.ylabel(
    "Predicted RUL (hours)"
)

plt.title(
    "Actual vs Predicted RUL"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "rul_prediction_scatter.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 10. ERROR DISTRIBUTION
# ============================================================

errors = (
    y_pred - y_actual
)


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    errors,
    bins=30
)

plt.xlabel(
    "Prediction Error (hours)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "RUL Prediction Error Distribution"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "rul_error_distribution.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    RESULT_DIR,
    "rul_model_comparison.csv"
)

comparison_df = pd.read_csv(
    comparison_path
)

print("\nModel comparison:")

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 12. MAE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    comparison_df["Model"],
    comparison_df["MAE"]
)

plt.ylabel(
    "MAE (hours)"
)

plt.title(
    "RUL Model Comparison - MAE"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "model_comparison_mae.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 13. R² COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    comparison_df["Model"],
    comparison_df["R2"]
)

plt.ylabel(
    "R² Score"
)

plt.title(
    "RUL Model Comparison - R²"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "model_comparison_r2.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESULT GENERATION COMPLETED")
print("=" * 70)

print("\nBest model:")
print("LSTM + Attention")

print(
    f"MAE  : {mae:.2f} hours"
)

print(
    f"RMSE : {rmse:.2f} hours"
)

print(
    f"R²   : {r2:.4f}"
)

print("\nResults saved in:")

print(RESULT_DIR)

print("\nGenerated files:")

print(" - rul_predictions.csv")
print(" - actual_vs_predicted_rul.png")
print(" - rul_prediction_scatter.png")
print(" - rul_error_distribution.png")
print(" - model_comparison_mae.png")
print(" - model_comparison_r2.png")

print("\nEV-Mind results are ready!")