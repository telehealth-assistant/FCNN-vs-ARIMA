import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm
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

warnings.filterwarnings("ignore")

# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

DATASETS = [

    {
        "name": "Global",
        "file": "GlobalTemperatures.csv",
        "date_col": "dt",
        "temp_col": "LandAverageTemperature",
        "filter": None
    },

    {
        "name": "Country_India",
        "file": "GlobalLandTemperaturesByCountry.csv",
        "date_col": "dt",
        "temp_col": "AverageTemperature",
        "filter_col": "Country",
        "filter_value": "India"
    },

    {
        "name": "City_Chennai",
        "file": "GlobalLandTemperaturesByCity.csv",
        "date_col": "dt",
        "temp_col": "AverageTemperature",
        "filter_col": "City",
        "filter_value": "Madras"
    },

    {
        "name": "MajorCity_Delhi",
        "file": "GlobalLandTemperaturesByMajorCity.csv",
        "date_col": "dt",
        "temp_col": "AverageTemperature",
        "filter_col": "City",
        "filter_value": "Delhi"
    },

    {
        "name": "State_TamilNadu",
        "file": "GlobalLandTemperaturesByState.csv",
        "date_col": "dt",
        "temp_col": "AverageTemperature",
        "filter_col": "State",
        "filter_value": "Tamil Nadu"
    }

]

# ==========================================================
# RESULT STORAGE
# ==========================================================

final_results = []

# ==========================================================
# PROCESS EACH DATASET
# ==========================================================

for dataset_info in DATASETS:

    print("\n================================================")
    print("PROCESSING:", dataset_info["name"])
    print("================================================")

    # ======================================================
    # LOAD DATA
    # ======================================================

    data = pd.read_csv(
        dataset_info["file"]
    )

    # ======================================================
    # FILTER IF NEEDED
    # ======================================================

    if "filter_col" in dataset_info:

        data = data[
            data[
                dataset_info["filter_col"]
            ] == dataset_info["filter_value"]
        ]

    # ======================================================
    # SELECT REQUIRED COLUMNS
    # ======================================================

    data = data[[
        dataset_info["date_col"],
        dataset_info["temp_col"]
    ]]

    data.columns = [
        "DATE",
        "TAVG"
    ]

    # ======================================================
    # REMOVE MISSING VALUES
    # ======================================================

    data.dropna(inplace=True)

    # ======================================================
    # DATE CONVERSION
    # ======================================================

    data["DATE"] = pd.to_datetime(
        data["DATE"]
    )

    # ======================================================
    # SORT
    # ======================================================

    data = data.sort_values(
        "DATE"
    )

    # ======================================================
    # LIMIT SIZE
    # ======================================================

    if len(data) > 20000:

        data = data.iloc[-20000:]

    print("\nDataset Shape:", data.shape)

    # ======================================================
    # NORMALIZATION
    # ======================================================

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        data[["TAVG"]]
    )

    # ======================================================
    # CREATE SEQUENCES
    # ======================================================

    TIME_STEP = 30

    X = []
    y = []

    for i in range(
        len(scaled_data) - TIME_STEP
    ):

        X.append(
            scaled_data[
                i:i+TIME_STEP, 0
            ]
        )

        y.append(
            scaled_data[
                i+TIME_STEP, 0
            ]
        )

    X = np.array(X)
    y = np.array(y)

    # ======================================================
    # SPLIT
    # ======================================================

    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # ======================================================
    # FCNN MODEL
    # ======================================================

    model = Sequential([

        Input(shape=(TIME_STEP,)),

        Dense(128, activation='relu'),

        Dense(64, activation='relu'),

        Dense(32, activation='relu'),

        Dense(16, activation='relu'),

        Dense(1)

    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # ======================================================
    # TRAIN FCNN
    # ======================================================

    print("\nTraining FCNN...")

    history = model.fit(

        X_train,
        y_train,

        epochs=50,

        batch_size=32,

        validation_data=(
            X_test,
            y_test
        ),

        verbose=0
    )

    # ======================================================
    # FCNN PREDICTION
    # ======================================================

    y_pred_fcnn = model.predict(
        X_test,
        verbose=0
    )

    y_test_inv = scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    y_pred_fcnn_inv = scaler.inverse_transform(
        y_pred_fcnn
    )

    # ======================================================
    # ARIMA
    # ======================================================

    print("Training ARIMA...")

    train_arima = data["TAVG"][:split]

    test_arima = data["TAVG"][
        split:split+len(y_test_inv)
    ]

    arima_model = ARIMA(
        train_arima,
        order=(5,1,0)
    )

    arima_fit = arima_model.fit()

    forecast = arima_fit.forecast(
        steps=len(test_arima)
    )

    # ======================================================
    # METRICS FUNCTION
    # ======================================================

    def evaluate(y_true, y_pred):

        y_true = np.array(
            y_true
        ).flatten()

        y_pred = np.array(
            y_pred
        ).flatten()

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        mape = np.mean(
            np.abs(
                (y_true - y_pred)
                /
                (y_true + 1e-8)
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

        return [
            mse,
            rmse,
            mae,
            mape,
            r2,
            exp_var,
            median_ae,
            max_err
        ]

    # ======================================================
    # CALCULATE METRICS
    # ======================================================

    fcnn_metrics = evaluate(
        y_test_inv,
        y_pred_fcnn_inv
    )

    arima_metrics = evaluate(
        test_arima.values,
        forecast
    )
    # ======================================================
    # DIEBOLD-MARIANO TEST
    # ======================================================

    error_fcnn = (
        y_test_inv.flatten()
        -
        y_pred_fcnn_inv.flatten()
    )

    error_arima = (
        test_arima.values.flatten()
        -
        np.array(forecast).flatten()
    )

    # Squared loss differential
    d = (
        error_arima ** 2
    ) - (
        error_fcnn ** 2
    )

    # Mean differential
    d_mean = np.mean(d)

    # Variance
    d_var = np.var(
        d,
        ddof=1
    )

    # Number of observations
    T = len(d)

    # DM statistic
    dm_stat = d_mean / np.sqrt(
        d_var / T
    )

    # Two-tailed p-value
    p_value = 2 * (
        1 - norm.cdf(abs(dm_stat))
    )

    print("\n========== DIEBOLD-MARIANO TEST ==========")

    print(
        f"DM Statistic : {dm_stat:.4f}"
    )

    print(
        f"P-Value      : {p_value:.10f}"
    )

    if p_value < 0.05:

        print(
            "Forecasting improvement is statistically significant."
        )

    else:

        print(
            "No statistically significant difference."
        )
    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print("\n========== FCNN ==========")

    print(f"RMSE: {fcnn_metrics[1]:.4f}")
    print(f"MAE: {fcnn_metrics[2]:.4f}")
    print(f"MAPE: {fcnn_metrics[3]:.4f}")
    print(f"R2: {fcnn_metrics[4]:.4f}")

    print("\n========== ARIMA ==========")

    print(f"RMSE: {arima_metrics[1]:.4f}")
    print(f"MAE: {arima_metrics[2]:.4f}")
    print(f"MAPE: {arima_metrics[3]:.4f}")
    print(f"R2: {arima_metrics[4]:.4f}")

    # ======================================================
    # STORE RESULTS
    # ======================================================

    final_results.append({

        "Dataset": dataset_info["name"],

        "FCNN_RMSE": round(
            fcnn_metrics[1], 4
        ),

        "ARIMA_RMSE": round(
            arima_metrics[1], 4
        ),

        "FCNN_MAE": round(
            fcnn_metrics[2], 4
        ),

        "ARIMA_MAE": round(
            arima_metrics[2], 4
        ),

        "FCNN_MAPE": round(
            fcnn_metrics[3], 4
        ),

        "ARIMA_MAPE": round(
            arima_metrics[3], 4
        ),

        "FCNN_R2": round(
            fcnn_metrics[4], 4
        ),

        "ARIMA_R2": round(
            arima_metrics[4], 4
        )

    })

    # ======================================================
    # PLOT
    # ======================================================

    plt.figure(figsize=(14,6))

    plt.plot(
        y_test_inv,
        label="Actual"
    )

    plt.plot(
        y_pred_fcnn_inv,
        label="FCNN"
    )

    plt.plot(
        forecast,
        label="ARIMA"
    )

    plt.title(
        f"{dataset_info['name']} Forecasting"
    )

    plt.xlabel("Time Step")

    plt.ylabel("Temperature")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        f"{dataset_info['name']}_RESULT.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# FINAL COMPARISON TABLE
# ==========================================================

results_df = pd.DataFrame(
    final_results
)

print("\n================================================")
print("FINAL COMPARISON")
print("================================================")

print(results_df)

results_df.to_csv(
    "FINAL_ALL_RESULTS.csv",
    index=False
)

# ==========================================================
# RMSE BAR GRAPH
# ==========================================================

plt.figure(figsize=(12,6))

x = np.arange(len(results_df))

width = 0.35

plt.bar(

    x - width/2,

    results_df["FCNN_RMSE"],

    width,

    label="FCNN"

)

plt.bar(

    x + width/2,

    results_df["ARIMA_RMSE"],

    width,

    label="ARIMA"

)

plt.xticks(
    x,
    results_df["Dataset"],
    rotation=20
)

plt.ylabel("RMSE")

plt.title(
    "FCNN vs ARIMA RMSE Comparison"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "FINAL_RMSE_COMPARISON.png",
    dpi=300
)

plt.show()

print("\nAll Results Saved Successfully!")