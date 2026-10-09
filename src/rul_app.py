import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model


# ============================================================
# EV-MIND - PROFESSIONAL STREAMLIT DASHBOARD
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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EV-Mind | Powertrain RUL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD PROJECT
# ============================================================

@st.cache_resource
def load_project():
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
    with open(
        os.path.join(
            PROCESSED_DIR,
            "target_scaler.pkl"
        ),
        "rb"
    ) as f:
        target_scaler = pickle.load(f)
    with open(
        os.path.join(
            PROCESSED_DIR,
            "feature_scaler.pkl"
        ),
        "rb"
    ) as f:
        feature_scaler = pickle.load(f)
    with open(
        os.path.join(
            PROCESSED_DIR,
            "config.pkl"
        ),
        "rb"
    ) as f:
        config = pickle.load(f)

    model = load_model(
        os.path.join(
            MODEL_DIR,
            "rul_lstm_attention_model.keras"
        )
    )

    predictions_df = pd.read_csv(
        os.path.join(
            RESULT_DIR,
            "rul_predictions.csv"
        )
    )

    comparison_df = pd.read_csv(
        os.path.join(
            RESULT_DIR,
            "rul_model_comparison.csv"
        )
    )

    return (
        X_test,
        y_test,
        target_scaler,
        feature_scaler,
        config,
        model,
        predictions_df,
        comparison_df
    )


(
    X_test,
    y_test,
    target_scaler,
    feature_scaler,
    config,
    model,
    predictions_df,
    comparison_df
) = load_project()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚡ EV-Mind")

    st.markdown(
        """
        **EV Powertrain Predictive Maintenance**

        Deep learning system for estimating
        Remaining Useful Life (RUL) from
        multivariate EV telemetry.
        """
    )

    st.divider()

    st.markdown("### 🧠 Model")

    st.write("**LSTM + Attention**")

    st.markdown("### ⚙️ Configuration")

    st.write(
        f"Sequence length: **{config['sequence_length']}**"
    )

    st.write(
        f"Input features: **{config['num_features']}**"
    )

    st.write(
        "Sampling interval: **15 minutes**"
    )

    st.write(
        "History window: **8 hours**"
    )

    st.divider()

    st.markdown("### 📊 Test Set")

    st.write(
        f"Test sequences: **{len(X_test)}**"
    )

    st.write(
        "Evaluation: **RUL regression**"
    )

    st.divider()

    st.caption(
        "EV-Mind | Deep Learning Project"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ EV-Mind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered EV Powertrain Remaining Useful Life Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    EV-Mind analyzes battery, motor, inverter, thermal,
    driving and charging telemetry using temporal deep
    learning models to estimate the remaining useful life
    of an electric vehicle powertrain.
    """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🏆 Deep Learning Model</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best Model",
    "LSTM + Attention"
)

col2.metric(
    "MAE",
    "119.86 h"
)

col3.metric(
    "RMSE",
    "151.04 h"
)

col4.metric(
    "R² Score",
    "0.9922"
)


st.divider()


# ============================================================
# RUL PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Live RUL Prediction Demo</div>',
    unsafe_allow_html=True
)

st.caption(
    "Select a test sequence to simulate a powertrain health assessment."
)

sequence_index = st.slider(
    "Test sequence",
    min_value=0,
    max_value=len(X_test) - 1,
    value=100
)


# ============================================================
# PREDICT
# ============================================================

selected_sequence = X_test[
    sequence_index:sequence_index + 1
]

prediction_scaled = model.predict(
    selected_sequence,
    verbose=0
)

predicted_rul = float(
    target_scaler.inverse_transform(
        prediction_scaled
    )[0][0]
)

actual_rul = float(
    predictions_df.iloc[
        sequence_index
    ]["Actual_RUL_hours"]
)

error = abs(
    actual_rul - predicted_rul
)

error_percentage = (
    error / max(actual_rul, 1)
) * 100

predicted_days = predicted_rul / 24


# ============================================================
# HEALTH CLASSIFICATION
# ============================================================

if predicted_rul >= 3000:

    health_status = "🟢 Healthy"
    health_message = (
        "Powertrain shows a high remaining useful life."
    )

elif predicted_rul >= 1500:

    health_status = "🟡 Degrading"
    health_message = (
        "Powertrain is showing moderate degradation."
    )

elif predicted_rul >= 500:

    health_status = "🟠 Warning"
    health_message = (
        "Powertrain has reduced remaining useful life."
    )

else:

    health_status = "🔴 Critical"
    health_message = (
        "Powertrain is approaching end-of-life."
    )


# ============================================================
# RUL HEALTH SUMMARY
# ============================================================

st.markdown("### 🔋 Powertrain Health Assessment")

health_col1, health_col2, health_col3 = st.columns(3)

health_col1.metric(
    "Predicted Remaining Useful Life",
    f"{predicted_rul:,.1f} hours"
)

health_col2.metric(
    "Equivalent Time",
    f"{predicted_days:,.1f} days"
)

health_col3.metric(
    "Health Status",
    health_status
)

st.info(
    health_message
)


# ============================================================
# PREDICTION METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted RUL",
    f"{predicted_rul:,.1f} h"
)

col2.metric(
    "Actual RUL",
    f"{actual_rul:,.1f} h"
)

col3.metric(
    "Absolute Error",
    f"{error:,.1f} h"
)

col4.metric(
    "Relative Error",
    f"{error_percentage:.1f}%"
)


st.divider()


# ============================================================
# TELEMETRY
# ============================================================

st.markdown(
    '<div class="section-title">📡 Powertrain Telemetry</div>',
    unsafe_allow_html=True
)

last_scaled = X_test[
    sequence_index,
    -1,
    :
].reshape(1, -1)

last_original = feature_scaler.inverse_transform(
    last_scaled
)[0]

feature_names = config[
    "feature_names"
]

sensor_data = dict(
    zip(
        feature_names,
        last_original
    )
)


def get_value(name, default=0):

    return sensor_data.get(
        name,
        default
    )


row1 = st.columns(4)

row1[0].metric(
    "Battery Voltage",
    f"{get_value('battery_voltage_V'):.2f} V"
)

row1[1].metric(
    "Battery Current",
    f"{get_value('battery_current_A'):.2f} A"
)

row1[2].metric(
    "State of Charge",
    f"{get_value('soc_pct'):.1f}%"
)

row1[3].metric(
    "Battery SOH",
    f"{get_value('soh_pct'):.1f}%"
)


row2 = st.columns(4)

row2[0].metric(
    "Motor RPM",
    f"{get_value('motor_rpm'):.0f}"
)

row2[1].metric(
    "Motor Torque",
    f"{get_value('motor_torque_Nm'):.1f} Nm"
)

row2[2].metric(
    "Motor Temperature",
    f"{get_value('motor_temp_C'):.1f} °C"
)

row2[3].metric(
    "Vibration Index",
    f"{get_value('vibration_index'):.3f}"
)


row3 = st.columns(4)

row3[0].metric(
    "Inverter Efficiency",
    f"{get_value('inverter_efficiency'):.2f}"
)

row3[1].metric(
    "Battery Temperature",
    f"{get_value('battery_temp_C'):.1f} °C"
)

row3[2].metric(
    "Vehicle Speed",
    f"{get_value('vehicle_speed_kmh'):.1f} km/h"
)

row3[3].metric(
    "Cooling Load",
    f"{get_value('cooling_system_load'):.2f}"
)


st.divider()


# ============================================================
# RUL CURVE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Actual vs Predicted RUL</div>',
    unsafe_allow_html=True
)

chart_df = predictions_df[
    [
        "Actual_RUL_hours",
        "Predicted_RUL_hours"
    ]
].copy()

chart_df.columns = [
    "Actual RUL",
    "Predicted RUL"
]

st.line_chart(
    chart_df,
    height=450
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Deep Learning Model Comparison</div>',
    unsafe_allow_html=True
)

comparison_display = comparison_df.copy()

comparison_display["MAE"] = (
    comparison_display["MAE"]
    .round(2)
)

comparison_display["RMSE"] = (
    comparison_display["RMSE"]
    .round(2)
)

comparison_display["R2"] = (
    comparison_display["R2"]
    .round(4)
)

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# KEY FINDING
# ============================================================

st.success(
    """
    **Key Finding:** LSTM + Attention achieved the best
    performance with an MAE of 119.86 hours, RMSE of
    151.04 hours and R² of 0.9922.
    """
)


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">ℹ️ About EV-Mind</div>',
    unsafe_allow_html=True
)

st.write(
    """
    EV-Mind is a deep learning-based predictive maintenance
    system designed for EV powertrain health monitoring.

    The system processes multivariate telemetry and uses a
    32-timestep history window representing approximately
    8 hours of vehicle operation.

    Three deep learning architectures were evaluated:

    • LSTM  
    • GRU  
    • LSTM + Attention

    The LSTM + Attention architecture produced the strongest
    RUL prediction performance on the evaluation dataset.
    """
)

st.caption(
    "EV-Mind • Deep Learning-Based EV Powertrain RUL Prediction"
)