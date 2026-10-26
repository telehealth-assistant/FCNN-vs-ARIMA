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

# ======================================================
# CONFIGURATION
# ======================================================

FILE_NAME = "weatherHistory.csv"

DATE_COLUMN = "Formatted Date"

TEMP_COLUMN = "Temperature (C)"

PLOT_TITLE = "Historical Weather Forecasting"

OUTPUT_IMAGE = "WEATHER_HISTORY_RESULTS.png"

# ======================================================
# LOAD DATA
# ======================================================

print("\nLoading Dataset...")

data = pd.read_csv(FILE_NAME)

print(data.columns)

# ======================================================
# SELECT REQUIRED COLUMNS
# ======================================================

data = data[[DATE_COLUMN, TEMP_COLUMN]]

data.columns = ['DATE', 'TAVG']

# ======================================================
# CLEANING
# ======================================================

data.dropna(inplace=True)

data['DATE'] = pd.to_datetime(
    data['DATE'],
    utc=True
)

data = data.sort_values('DATE')

# Use latest samples
data = data.iloc[-20000:]

print("\nDataset Shape:", data.shape)

print(data.head())

# ======================================================
# NORMALIZATION
# ======================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    data[['TAVG']]
)

# ======================================================
# CREATE TIME SERIES
# ======================================================

def create_dataset(dataset, time_step=30):

    X, y = [], []

    for i in range(len(dataset) - time_step):

        X.append(dataset[i:i+time_step, 0])

        y.append(dataset[i+time_step, 0])

    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# ======================================================
# SPLIT
# ======================================================

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]

# ======================================================
# FCNN MODEL
# ======================================================

model = Sequential([

    Input(shape=(30,)),

    Dense(64, activation='relu'),

    Dense(32, activation='relu'),

    Dense(16, activation='relu'),

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

    verbose=1

)

# ======================================================
# FCNN PREDICTIONS
# ======================================================

y_pred_fcnn = model.predict(X_test)

# ======================================================
# INVERSE TRANSFORM
# ======================================================

y_test_inv = scaler.inverse_transform(
    y_test.reshape(-1,1)
)

y_pred_fcnn_inv = scaler.inverse_transform(
    y_pred_fcnn
)

# ======================================================
# ARIMA
# ======================================================

print("\nTraining ARIMA...\n")

train_arima = data['TAVG'].values[:split]

test_arima = data['TAVG'].values[
    split:split+len(y_test_inv)
]

model_arima = ARIMA(
    train_arima,
    order=(5,1,0)
)

fit_arima = model_arima.fit()

forecast = fit_arima.forecast(
    steps=len(test_arima)
)

forecast = np.array(forecast).flatten()

# ======================================================
# METRICS
# ======================================================

def evaluate_all(y_true, y_pred):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred)/(y_true + 1e-8))
    ) * 100

    r2 = r2_score(y_true, y_pred)

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

# ======================================================
# RESULTS
# ======================================================

fcnn_metrics = evaluate_all(
    y_test_inv,
    y_pred_fcnn_inv
)

arima_metrics = evaluate_all(
    test_arima,
    forecast
)

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

# ======================================================
# RESIDUAL ANALYSIS
# ======================================================

residuals_fcnn = (
    y_test_inv.flatten()
    - y_pred_fcnn_inv.flatten()
)

residuals_arima = (
    test_arima.flatten()
    - forecast.flatten()
)

print("\n========== RESIDUAL ANALYSIS ==========")

print(f"FCNN Residual Mean: {np.mean(residuals_fcnn):.4f}")
print(f"FCNN Residual Std: {np.std(residuals_fcnn):.4f}")

print(f"ARIMA Residual Mean: {np.mean(residuals_arima):.4f}")
print(f"ARIMA Residual Std: {np.std(residuals_arima):.4f}")

# ======================================================
# PLOT
# ======================================================

plt.figure(figsize=(14,6))

x_axis = np.arange(len(y_test_inv))

plt.plot(
    x_axis,
    y_test_inv.flatten(),
    label="Actual",
    linewidth=2
)

plt.plot(
    x_axis,
    y_pred_fcnn_inv.flatten(),
    label="FCNN",
    linewidth=2
)

plt.plot(
    x_axis,
    forecast.flatten(),
    label="ARIMA",
    linewidth=2
)

plt.title(
    PLOT_TITLE,
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Time Step")

plt.ylabel("Temperature (°C)")

plt.legend()

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches='tight'
)

plt.show()