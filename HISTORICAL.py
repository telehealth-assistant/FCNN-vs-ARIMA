import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from statsmodels.tsa.arima.model import ARIMA

# =========================================================
# IEEE LEVEL TEMPERATURE FORECASTING
# HISTORICAL HOURLY WEATHER DATASET
# FCNN vs ARIMA
# =========================================================

FILE_NAME = r"C:\Users\Sandeep Kumar k\Downloads\archive (7)\temperature.csv"

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading Dataset...")

data = pd.read_csv(FILE_NAME)

print(data.columns)

# =========================================================
# SELECT CITY
# =========================================================

CITY_NAME = "Vancouver"

data = data[['datetime', CITY_NAME]]

data.columns = ['DATE', 'TAVG']

# =========================================================
# PREPROCESSING
# =========================================================

data.dropna(inplace=True)

data['DATE'] = pd.to_datetime(data['DATE'])

data = data.sort_values('DATE')

# Kelvin to Celsius
data['TAVG'] = data['TAVG'] - 273.15

# Last 20000 samples
data = data.iloc[-20000:]

print("\nDataset Shape:", data.shape)

print(data.head())

# =========================================================
# NORMALIZATION
# =========================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    data[['TAVG']]
)

# =========================================================
# CREATE SEQUENCES
# =========================================================

TIME_STEP = 30

def create_dataset(dataset, time_step=30):

    X = []
    y = []

    for i in range(len(dataset) - time_step):

        X.append(dataset[i:i+time_step, 0])

        y.append(dataset[i+time_step, 0])

    return np.array(X), np.array(y)

X, y = create_dataset(
    scaled_data,
    TIME_STEP
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================================================
# FCNN MODEL
# =========================================================

model = Sequential([

    Input(shape=(TIME_STEP,)),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(32, activation='relu'),

    Dense(1)

])

model.compile(
    optimizer='adam',
    loss='mse'
)

print("\nTraining FCNN...\n")

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.1,

    verbose=1
)

# =========================================================
# FCNN PREDICTIONS
# =========================================================

print("\nGenerating FCNN predictions...\n")

y_pred_fcnn = model.predict(X_test)

# Inverse scaling

y_test_inv = scaler.inverse_transform(
    y_test.reshape(-1,1)
)

y_pred_fcnn_inv = scaler.inverse_transform(
    y_pred_fcnn
)

# =========================================================
# ARIMA
# =========================================================

print("\nTraining ARIMA...\n")

train_arima = data['TAVG'][:split]

test_arima = data['TAVG'][split:split+len(y_test_inv)]

arima_model = ARIMA(
    train_arima,
    order=(5,1,0)
)

arima_fit = arima_model.fit()

forecast = arima_fit.forecast(
    steps=len(test_arima)
)

forecast = np.array(forecast)

# =========================================================
# METRICS
# =========================================================

def evaluate_all(y_true, y_pred):

    y_true = np.array(y_true).flatten()

    y_pred = np.array(y_pred).flatten()

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    epsilon = 1e-10

    mape = np.mean(
        np.abs(
            (y_true - y_pred) /
            (np.abs(y_true) + epsilon)
        )
    ) * 100

    r2 = r2_score(
        y_true,
        y_pred
    )

    exp_var = explained_variance_score(
        y_true,
        y_pred
    )

    median_ae = np.median(
        np.abs(y_true - y_pred)
    )

    max_err = max_error(
        y_true,
        y_pred
    )

    return (

        mse,
        rmse,
        mae,
        mape,
        r2,
        exp_var,
        median_ae,
        max_err
    )

# =========================================================
# EVALUATION
# =========================================================

fcnn_metrics = evaluate_all(
    y_test_inv,
    y_pred_fcnn_inv
)

arima_metrics = evaluate_all(
    test_arima.values,
    forecast
)

# =========================================================
# RESULTS
# =========================================================

print("\n========== FCNN ==========")

print(f"MSE: {fcnn_metrics[0]:.4f}")
print(f"RMSE: {fcnn_metrics[1]:.4f}")
print(f"MAE: {fcnn_metrics[2]:.4f}")
print(f"MAPE: {fcnn_metrics[3]:.4f}")
print(f"R2 Score: {fcnn_metrics[4]:.4f}")
print(f"Explained Variance: {fcnn_metrics[5]:.4f}")
print(f"Median AE: {fcnn_metrics[6]:.4f}")
print(f"Max Error: {fcnn_metrics[7]:.4f}")

print("\n========== ARIMA ==========")

print(f"MSE: {arima_metrics[0]:.4f}")
print(f"RMSE: {arima_metrics[1]:.4f}")
print(f"MAE: {arima_metrics[2]:.4f}")
print(f"MAPE: {arima_metrics[3]:.4f}")
print(f"R2 Score: {arima_metrics[4]:.4f}")
print(f"Explained Variance: {arima_metrics[5]:.4f}")
print(f"Median AE: {arima_metrics[6]:.4f}")
print(f"Max Error: {arima_metrics[7]:.4f}")

# =========================================================
# RESIDUAL ANALYSIS
# =========================================================

fcnn_residuals = (
    y_test_inv.flatten()
    -
    y_pred_fcnn_inv.flatten()
)

arima_residuals = (
    np.array(test_arima.values)
    -
    np.array(forecast)
)

print("\n========== RESIDUAL ANALYSIS ==========")

print(
    f"FCNN Residual Mean: "
    f"{np.mean(fcnn_residuals):.4f}"
)

print(
    f"FCNN Residual Std: "
    f"{np.std(fcnn_residuals):.4f}"
)

print(
    f"ARIMA Residual Mean: "
    f"{np.mean(arima_residuals):.4f}"
)

print(
    f"ARIMA Residual Std: "
    f"{np.std(arima_residuals):.4f}"
)

# =========================================================
# IMPORTANT ALIGNMENT FIX
# =========================================================

plot_length = 1000

actual_plot = y_test_inv[:plot_length].flatten()

fcnn_plot = y_pred_fcnn_inv[:plot_length].flatten()

arima_plot = forecast[:plot_length]

# =========================================================
# PLOT 1
# =========================================================

plt.figure(figsize=(16,8))

plt.plot(
    actual_plot,
    label="Actual",
    linewidth=2
)

plt.plot(
    fcnn_plot,
    label="FCNN Prediction",
    linewidth=2
)

plt.plot(
    arima_plot,
    label="ARIMA Prediction",
    linewidth=2
)

plt.title(
    "Temperature Forecasting: FCNN vs ARIMA",
    fontsize=22,
    fontweight='bold'
)

plt.xlabel(
    "Time Step",
    fontsize=15
)

plt.ylabel(
    "Temperature (°C)",
    fontsize=15
)

plt.legend(fontsize=13)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "TEMPERATURE_FORECAST_RESULTS.png",
    dpi=300
)

# =========================================================
# PLOT 2
# =========================================================

plt.figure(figsize=(10,6))

models = ['FCNN', 'ARIMA']

rmse_values = [
    fcnn_metrics[1],
    arima_metrics[1]
]

plt.bar(models, rmse_values)

plt.title(
    "RMSE Comparison",
    fontsize=18,
    fontweight='bold'
)

plt.ylabel(
    "RMSE",
    fontsize=14
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "RMSE_COMPARISON.png",
    dpi=300
)

# =========================================================
# PLOT 3
# =========================================================

plt.figure(figsize=(12,6))

plt.hist(
    fcnn_residuals,
    bins=30,
    alpha=0.7,
    label='FCNN Residuals'
)

plt.hist(
    arima_residuals,
    bins=30,
    alpha=0.7,
    label='ARIMA Residuals'
)

plt.title(
    "Residual Error Distribution",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    "Residual Error",
    fontsize=14
)

plt.ylabel(
    "Frequency",
    fontsize=14
)

plt.legend(fontsize=12)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "RESIDUAL_ANALYSIS.png",
    dpi=300
)

# =========================================================
# PLOT 4
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(
    history.history['loss'],
    linewidth=2
)

plt.title(
    "FCNN Training Loss Curve",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    "Epoch",
    fontsize=14
)

plt.ylabel(
    "Loss",
    fontsize=14
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "TRAINING_LOSS_CURVE.png",
    dpi=300
)

# =========================================================
# SAVE RESULTS
# =========================================================

results = pd.DataFrame({

    'Model': ['FCNN', 'ARIMA'],

    'RMSE': [
        fcnn_metrics[1],
        arima_metrics[1]
    ],

    'MAE': [
        fcnn_metrics[2],
        arima_metrics[2]
    ],

    'MAPE': [
        fcnn_metrics[3],
        arima_metrics[3]
    ],

    'R2': [
        fcnn_metrics[4],
        arima_metrics[4]
    ],

    'ExplainedVariance': [
        fcnn_metrics[5],
        arima_metrics[5]
    ]
})

results.to_csv(
    "TEMPERATURE_MODEL_RESULTS.csv",
    index=False
)

print("\nResults saved successfully!")

print("\nGenerated Files:")
print("1. TEMPERATURE_FORECAST_RESULTS.png")
print("2. RMSE_COMPARISON.png")
print("3. RESIDUAL_ANALYSIS.png")
print("4. TRAINING_LOSS_CURVE.png")
print("5. TEMPERATURE_MODEL_RESULTS.csv")

plt.show()