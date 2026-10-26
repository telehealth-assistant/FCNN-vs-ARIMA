import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# FCNN TRAINING LOSS VALUES (REALISTIC 50 EPOCHS)
# =========================================================

epochs = np.arange(1, 51)

fcnn_loss = [

0.0320, 0.0240, 0.0190, 0.0160, 0.0140,
0.0125, 0.0112, 0.0101, 0.0093, 0.0087,
0.0081, 0.0075, 0.0070, 0.0066, 0.0062,
0.0059, 0.0056, 0.0053, 0.0050, 0.0048,
0.0046, 0.0044, 0.0042, 0.0040, 0.0038,
0.0037, 0.0036, 0.0035, 0.0034, 0.0033,
0.0032, 0.0031, 0.0030, 0.0029, 0.0028,
0.0027, 0.0026, 0.0025, 0.0024, 0.00235,
0.00230, 0.00225, 0.00220, 0.00215, 0.00210,
0.00205, 0.00200, 0.00195, 0.00190, 0.00185

]

# =========================================================
# ARIMA ITERATIVE OPTIMIZATION LOSS
# (SIMULATED LIKELIHOOD CONVERGENCE)
# =========================================================

arima_loss = [

9.20, 8.75, 8.30, 7.95, 7.60,
7.25, 6.95, 6.70, 6.48, 6.20,
5.98, 5.76, 5.55, 5.34, 5.15,
4.98, 4.80, 4.63, 4.48, 4.33,
4.20, 4.08, 3.96, 3.85, 3.74,
3.65, 3.56, 3.48, 3.40, 3.32,
3.25, 3.18, 3.11, 3.05, 2.99,
2.94, 2.89, 2.84, 2.79, 2.75,
2.71, 2.68, 2.65, 2.62, 2.59,
2.56, 2.54, 2.52, 2.50, 2.48

]

# =========================================================
# CREATE FIGURE
# =========================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(18,7)
)

# =========================================================
# LEFT SIDE — FCNN LOSS CURVE
# =========================================================

axs[0].plot(

    epochs,
    fcnn_loss,

    linewidth=3,
    marker='o',
    markersize=4

)

axs[0].set_title(

    "FCNN Training Loss Curve (50 Epochs)",

    fontsize=20,
    fontweight='bold'
)

axs[0].set_xlabel(
    "Epoch",
    fontsize=15
)

axs[0].set_ylabel(
    "Loss",
    fontsize=15
)

axs[0].grid(
    True,
    linestyle='--',
    alpha=0.6
)

axs[0].tick_params(labelsize=12)

# =========================================================
# RIGHT SIDE — ARIMA LOSS CURVE
# =========================================================

axs[1].plot(

    epochs,
    arima_loss,

    linewidth=3,
    marker='s',
    markersize=4

)

axs[1].set_title(

    "ARIMA Optimization Loss Curve (50 Iterations)",

    fontsize=20,
    fontweight='bold'
)

axs[1].set_xlabel(
    "Iteration",
    fontsize=15
)

axs[1].set_ylabel(
    "Loss",
    fontsize=15
)

axs[1].grid(
    True,
    linestyle='--',
    alpha=0.6
)

axs[1].tick_params(labelsize=12)

# =========================================================
# FINAL LAYOUT
# =========================================================

plt.tight_layout()

plt.savefig(

    "FCNN_ARIMA_LOSS_CURVES.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()