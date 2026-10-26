import matplotlib.pyplot as plt
import numpy as np

# ===============================
# DATASET NAMES
# ===============================

datasets = [
    "NOAA",
    "Delhi",
    "Jena",
    "Daily Min Temp",
    "Chennai",
    "Historical",
    "Daily Weather",
    "Berkeley Global",
    "Berkeley India",
    "Hourly Weather"
]

# ===============================
# RMSE VALUES
# ===============================

fcnn_rmse = [
    1.85,
    1.72,
    0.3442,
    2.2492,
    0.5529,
    0.8454,
    0.8338,
    0.4538,
    0.6103,
    0.6934
]

arima_rmse = [
    19.66,
    7.34,
    10.4224,
    4.4192,
    2.8011,
    9.6044,
    2.7115,
    4.0916,
    4.2511,
    9.0917
]

# ===============================
# PLOT SETTINGS
# ===============================

x = np.arange(len(datasets))

plt.figure(figsize=(16,7))

# FCNN Plot
plt.plot(
    x,
    fcnn_rmse,
    marker='o',
    linewidth=3,
    markersize=8,
    label='FCNN RMSE'
)

# ARIMA Plot
plt.plot(
    x,
    arima_rmse,
    marker='s',
    linewidth=3,
    markersize=8,
    label='ARIMA RMSE'
)

# ===============================
# LABELS
# ===============================

plt.xticks(
    x,
    datasets,
    rotation=20,
    fontsize=11
)

plt.ylabel(
    "RMSE",
    fontsize=15,
    fontweight='bold'
)

plt.xlabel(
    "Datasets",
    fontsize=15,
    fontweight='bold'
)

plt.title(
    "RMSE Stability Comparison Across Multiple Climate Datasets",
    fontsize=20,
    fontweight='bold'
)

# ===============================
# GRID + LEGEND
# ===============================

plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)

plt.legend(
    fontsize=13
)

# ===============================
# SAVE FIGURE
# ===============================

plt.tight_layout()

plt.savefig(
    "RMSE_STABILITY_COMPARISON.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()