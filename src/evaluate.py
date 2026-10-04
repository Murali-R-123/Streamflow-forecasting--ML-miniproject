"""Metrics and plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def regression_metrics(y_true, y_pred, peak_threshold=1500.0):
    """All inputs in original units (cfs).

    R2 here is 1 - SSE/SST around the observed mean, which is the same formula
    as the Nash-Sutcliffe Efficiency (NSE) used in hydrology.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = mse ** 0.5
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    peak = y_true > peak_threshold

    def _rmse(mask):
        return float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("nan")

    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": rmse,
        "R2_NSE": 1.0 - float(np.sum(err ** 2)) / sst,
        "RMSE_pct_of_max": 100.0 * rmse / float(y_true.max()),
        "RMSE_peak": _rmse(peak),
        "RMSE_nonpeak": _rmse(~peak),
        "n_peak_days": int(peak.sum()),
        "n_days": int(len(y_true)),
    }


def plot_hydrograph(dates, obs, preds: dict, path, zoom_days=60):
    """Observed vs predicted: full test period + zoom around the largest observed peak."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax in axes:
        ax.plot(dates, obs, color="black", lw=1.2, label="Observed")
        for name, (p, style) in preds.items():
            ax.plot(dates, p, lw=style.get("lw", 1.0), color=style["color"], label=name, alpha=style.get("alpha", 0.9))
        ax.set_ylabel("Streamflow (cfs)")
    axes[0].set_title("Observed vs predicted streamflow - test period")
    i = int(np.argmax(obs))
    lo, hi = max(0, i - zoom_days), min(len(obs), i + zoom_days)
    axes[1].set_xlim(dates[lo], dates[hi - 1])
    axes[1].set_title("Zoom: around the largest observed peak in the test period")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_mse"], label="Training MSE")
    ax.plot(history["val_mse"], label="Validation MSE")
    ax.axvline(history["best_epoch"] - 1, ls="--", color="gray", label=f"Best epoch ({history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled units)")
    ax.set_yscale("log")
    ax.set_title("ANN training vs validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(obs, pred, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(obs, pred, s=6, alpha=0.5)
    lim = max(obs.max(), pred.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Observed (cfs)")
    ax.set_ylabel("Predicted (cfs)")
    ax.set_title("Observed vs predicted - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eda(df, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].bar(df.index, df["rainfall_mm"], width=1.0, color="steelblue")
    axes[0].set_ylabel("Rainfall (mm/day)")
    axes[1].plot(df.index, df["discharge_cfs"], color="darkgreen", lw=0.8)
    axes[1].set_ylabel("Discharge (cfs)")
    axes[0].set_title("Daily rainfall and streamflow")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
