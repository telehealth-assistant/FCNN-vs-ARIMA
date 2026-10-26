import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from statsmodels.tsa.arima.model import ARIMA

# ======================================================
# LOAD CLEAN DATASET
# ======================================================

print("\nLoading cleaned dataset...\n")

data = pd.read_csv(
    "clean_chennai_temperature.csv"
)

print(data.head())

# ======================================================
# DATE CONVERSION
# ======================================================

data['DATE'] = pd.to_datetime(
    data['DATE']
)

# ======================================================
# SORT DATA
# ======================================================

data = data.sort_values(
    'DATE'
)

# ======================================================
# SELECT TEMPERATURE COLUMN
# ======================================================

temperature_data = data[['TAVG']]

# ======================================================
# NORMALIZATION
# ======================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    temperature_data
)

# ======================================================
# CREATE TIME SERIES SEQUENCES
# ======================================================

TIME_STEP = 30

def create_dataset(dataset, time_step):

    X = []
    y = []

    for i in range(
        len(dataset) - time_step
    ):

        X.append(
            dataset[i:i+time_step, 0]
        )

        y.append(
            dataset[i+time_step, 0]
        )

    return np.array(X), np.array(y)

X, y = create_dataset(
    scaled_data,
    TIME_STEP
)

print("\nDataset Shapes:")
print("X:", X.shape)
print("y:", y.shape)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

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

# ======================================================
# COMPILE MODEL
# ======================================================

model.compile(
    optimizer='adam',
    loss='mse'
)

# ======================================================
# TRAIN MODEL
# ======================================================

print("\nTraining FCNN...\n")

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_data=(
        X_test,
        y_test
    ),

    verbose=1
)

# ======================================================
# FCNN PREDICTIONS
# ======================================================

print("\nGenerating FCNN predictions...\n")

y_pred_fcnn = model.predict(
    X_test
)

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
# ARIMA MODEL
# ======================================================

print("\nTraining ARIMA...\n")

train_arima = (
    temperature_data.values[:split]
    .flatten()
)

test_arima = (
    temperature_data.values[
        split:split+len(y_test_inv)
    ]
    .flatten()
)

arima_model = ARIMA(

    train_arima,

    order=(5,1,0)

)

arima_fit = arima_model.fit()

forecast = arima_fit.forecast(
    steps=len(test_arima)
)

forecast = np.array(
    forecast
).flatten()

# ======================================================
# EVALUATION FUNCTION
# ======================================================

def evaluate_model(
    y_true,
    y_pred,
    model_name
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n{model_name} RESULTS")
    print("="*40)

    print("RMSE :", rmse)
    print("MAE  :", mae)
    print("R²   :", r2)

    return rmse

# ======================================================
# EVALUATE FCNN
# ======================================================

rmse_fcnn = evaluate_model(

    y_test_inv,

    y_pred_fcnn_inv,

    "FCNN"

)

# ======================================================
# EVALUATE ARIMA
# ======================================================

rmse_arima = evaluate_model(

    test_arima,

    forecast,

    "ARIMA"

)

# ======================================================
# PLOT RESULTS
# ======================================================

plt.figure(figsize=(16,7))

plt.plot(
    y_test_inv,
    label='Actual'
)

plt.plot(
    y_pred_fcnn_inv,
    label='FCNN Prediction'
)

plt.plot(
    forecast,
    label='ARIMA Prediction'
)

plt.title(
    "FCNN vs ARIMA Temperature Forecasting"
)

plt.xlabel("Time")

plt.ylabel("Temperature °C")

plt.legend()

plt.grid(True)

plt.savefig(
    "FCNN_ARIMA_RESULTS.png",
    dpi=300
)

plt.show()

# ======================================================
# RMSE COMPARISON GRAPH
# ======================================================

plt.figure(figsize=(8,5))

models = ['FCNN', 'ARIMA']

rmse_values = [
    rmse_fcnn,
    rmse_arima
]

plt.bar(
    models,
    rmse_values
)

plt.ylabel("RMSE")

plt.title(
    "RMSE Comparison"
)

plt.grid(True)

plt.savefig(
    "RMSE_COMPARISON.png",
    dpi=300
)

plt.show()

# ======================================================
# SAVE RESULTS
# ======================================================

results = pd.DataFrame({

    'Model': ['FCNN', 'ARIMA'],

    'RMSE': [
        rmse_fcnn,
        rmse_arima
    ]

})

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nResults saved successfully!")

print("\nGenerated Files:")
print("1. FCNN_ARIMA_RESULTS.png")
print("2. RMSE_COMPARISON.png")
print("3. model_results.csv")