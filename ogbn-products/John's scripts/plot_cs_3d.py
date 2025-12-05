import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FILE = "cs2.txt"

def parse_cs_log(filename):
    corr = []
    smooth = []
    val = []
    test = []

    curr_corr = None
    curr_smooth = None

    with open(filename, "r") as f:
        for line in f:

            # -------------------------------
            # 1. Extract parameters from Optuna
            # -------------------------------
            m_params = re.search(
                r"parameters:\s*\{'correction-alpha':\s*([0-9.eE+-]+),\s*'smoothing-alpha':\s*([0-9.eE+-]+)\}",
                line
            )
            if m_params:
                curr_corr = float(m_params.group(1))
                curr_smooth = float(m_params.group(2))
                continue

            # -------------------------------
            # 2. Extract validation + test accuracy
            # -------------------------------
            m_acc = re.search(r"Valid acc:\s*([0-9.]+)\s*\|\s*Test acc:\s*([0-9.]+)", line)
            if m_acc and curr_corr is not None and curr_smooth is not None:
                corr.append(curr_corr)
                smooth.append(curr_smooth)
                val.append(float(m_acc.group(1)))
                test.append(float(m_acc.group(2)))

                # Reset so we don't mix lines across trials
                curr_corr = None
                curr_smooth = None

    return corr, smooth, val, test


def plot_3d(corr, smooth, val):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(corr, smooth, val, c=val, cmap='viridis', s=40)

    ax.set_xlabel("Correction Alpha", fontsize=12)
    ax.set_ylabel("Smoothing Alpha", fontsize=12)
    ax.set_zlabel("Validation Accuracy", fontsize=12)
    ax.set_title("Correct & Smooth Hyperparameter Surface", fontsize=14)

    fig.colorbar(p, ax=ax, shrink=0.5, label="Validation Accuracy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    corr, smooth, val, test = parse_cs_log(FILE)

    print(f"Parsed {len(corr)} trials.")
    if len(corr) == 0:
        print("No matching trials found — log format mismatch?")
    else:
        plot_3d(corr, smooth, val)
