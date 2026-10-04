"""End-to-end pipeline:  data -> clean -> lag features -> split -> scale -> ANN
-> hyper-parameter search (validation years only) -> final test evaluation -> plots.

Usage:
    python pipeline.py                 # full run (grid search, several seeds)
    python pipeline.py --quick         # tiny grid, for a fast demo / debugging
    python pipeline.py --peak-threshold 1500
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import chronological_split, load_master
from src.evaluate import (plot_eda, plot_hydrograph, plot_loss, plot_scatter,
                          regression_metrics)
from src.features import MaxScaler, make_lagged
from src.model import train_ann


def prepare(df, m, n):
    """lag features -> chronological split -> scaler fitted on TRAIN only."""
    lagged = make_lagged(df, m, n)
    tr, va, te = chronological_split(lagged)
    sc = MaxScaler().fit(tr)
    return lagged, (tr, va, te), sc


def run(data_path="data/streamflow_data.csv", out_dir=".", quick=False, n_seeds=5,
        peak_threshold=1500.0, activation="logistic", lr=0.005, verbose=True):
    out = Path(out_dir)
    for sub in ["results/figures", "results/metrics", "data/processed"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # 1-2. load + clean + EDA -------------------------------------------------
    df, rep = load_master(data_path)
    log("Data report:", json.dumps(rep, indent=2))
    (out / "results/metrics/data_report.json").write_text(json.dumps(rep, indent=2))
    plot_eda(df, out / "results/figures/eda_rainfall_streamflow.png")

    # 3-4. hyper-parameter search on TRAIN (fit) / VALIDATION (score) only ----
    if quick:
        grid_m, grid_n, grid_h, grid_seeds = [3], [4], [(8,), (12,)], 2
    else:
        grid_m, grid_n = [2, 3, 5], [2, 4, 6]
        grid_h, grid_seeds = [(8,), (12,), (16,), (16, 8)], 3
    rows = []
    for m, n, h in itertools.product(grid_m, grid_n, grid_h):
        lagged, (tr, va, te), sc = prepare(df, m, n)
        Xtr, ytr = sc.transform(tr)
        Xva, yva = sc.transform(va)
        vals = []
        for s in range(grid_seeds):
            model, hist = train_ann(Xtr, ytr, Xva, yva, hidden=h, activation=activation, lr=lr, seed=s)
            pred = sc.inverse_target(model.predict(Xva))
            vals.append(regression_metrics(sc.inverse_target(yva), pred, peak_threshold)["RMSE"])
        rows.append({"m_rain_lags": m, "n_flow_lags": n, "hidden": str(h),
                     "val_RMSE_mean": float(np.mean(vals)), "val_RMSE_std": float(np.std(vals))})
        log(f"  m={m} n={n} hidden={h}: val RMSE = {np.mean(vals):.2f} +/- {np.std(vals):.2f}")
    grid = pd.DataFrame(rows).sort_values("val_RMSE_mean")
    grid.to_csv(out / "results/metrics/grid_results.csv", index=False)
    best = grid.iloc[0]
    m, n, h = int(best.m_rain_lags), int(best.n_flow_lags), eval(best.hidden)
    log(f"\nBest by VALIDATION RMSE: m={m}, n={n}, hidden={h}")

    # 5. final models: several seeds, evaluated ONCE on the untouched test years
    lagged, (tr, va, te), sc = prepare(df, m, n)
    Xtr, ytr = sc.transform(tr); Xva, yva = sc.transform(va); Xte, yte = sc.transform(te)
    obs = te["target"].values
    tr.to_csv(out / "data/processed/train_lagged.csv")
    va.to_csv(out / "data/processed/val_lagged.csv")
    te.to_csv(out / "data/processed/test_lagged.csv")
    log(f"Rows -> train {len(tr)}, val {len(va)}, test {len(te)}  "
        f"(days dropped because a lag window hit missing data: "
        f"{len(df.loc[lagged.index.min():]) - len(lagged)})")

    runs = []
    for s in range(n_seeds):
        model, hist = train_ann(Xtr, ytr, Xva, yva, hidden=h, activation=activation, lr=lr, seed=s)
        runs.append((hist["best_val_mse"], model, hist))
    runs.sort(key=lambda r: r[0])
    _, best_model, best_hist = runs[0]                   # best seed chosen by VALIDATION loss
    ann_pred = np.clip(sc.inverse_target(best_model.predict(Xte)), 0, None)
    seed_metrics = [regression_metrics(obs, np.clip(sc.inverse_target(r[1].predict(Xte)), 0, None),
                                       peak_threshold) for r in runs]

    # baselines ---------------------------------------------------------------
    persistence = te["Q_lag1"].values                    # tomorrow = today
    lin = LinearRegression().fit(Xtr, ytr)
    lin_pred = np.clip(sc.inverse_target(lin.predict(Xte)), 0, None)

    table = pd.DataFrame({
        "Persistence Q(t)=Q(t-1)": regression_metrics(obs, persistence, peak_threshold),
        "Linear regression (same features)": regression_metrics(obs, lin_pred, peak_threshold),
        f"ANN best-val seed (m={m}, n={n}, hidden={h})": regression_metrics(obs, ann_pred, peak_threshold),
    }).T
    keys = ["MAE", "RMSE", "R2_NSE", "RMSE_pct_of_max", "RMSE_peak", "RMSE_nonpeak"]
    ms = pd.DataFrame(seed_metrics)[keys]
    table.loc[f"ANN mean over {n_seeds} seeds"] = {**ms.mean().to_dict(),
                                                   "n_peak_days": seed_metrics[0]["n_peak_days"],
                                                   "n_days": seed_metrics[0]["n_days"], "MSE": np.nan}
    table.loc[f"ANN std over {n_seeds} seeds"] = {**ms.std().to_dict(), "MSE": np.nan}
    table = table[["MAE", "MSE", "RMSE", "R2_NSE", "RMSE_pct_of_max", "RMSE_peak", "RMSE_nonpeak",
                   "n_peak_days", "n_days"]]
    table.to_csv(out / "results/metrics/metrics_summary.csv")
    log("\nTEST-set metrics (cfs; peak threshold = %.0f cfs):" % peak_threshold)
    log(table.round(3).to_string())

    # 6. plots + saved artefacts ---------------------------------------------
    dates = te.index
    plot_hydrograph(dates, obs, {
        "ANN": (ann_pred, {"color": "tab:red"}),
        "Linear regression": (lin_pred, {"color": "tab:blue", "lw": 0.8, "alpha": 0.7}),
    }, out / "results/figures/hydrograph_comparison.png")
    plot_loss(best_hist, out / "results/figures/training_loss.png")
    plot_scatter(obs, ann_pred, out / "results/figures/scatter_obs_vs_pred.png")
    pd.DataFrame({"observed_cfs": obs, "ann_cfs": ann_pred, "linreg_cfs": lin_pred,
                  "persistence_cfs": persistence}, index=dates).to_csv(out / "results/metrics/test_predictions.csv")
    cfg = {"m_rain_lags": m, "n_flow_lags": n, "hidden": list(h), "activation": activation,
           "learning_rate": lr, "peak_threshold_cfs": peak_threshold, "best_epoch": best_hist["best_epoch"],
           "train_q_max_cfs": sc.q_max, "train_r_max_mm": sc.r_max}
    (out / "results/metrics/best_config.json").write_text(json.dumps(cfg, indent=2))
    joblib.dump({"model": best_model, "scaler": sc, "config": cfg}, out / "results/ann_model.joblib")
    log("\nSaved figures, metrics and model under", out / "results")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/streamflow_data.csv")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--peak-threshold", type=float, default=1500.0,
                    help="cfs; days above this count as 'peak' (reference paper used 1500)")
    ap.add_argument("--activation", default="logistic", choices=["logistic", "relu"])
    ap.add_argument("--lr", type=float, default=0.005)
    a = ap.parse_args()
    run(a.data, ".", a.quick, a.seeds, a.peak_threshold, a.activation, a.lr)
