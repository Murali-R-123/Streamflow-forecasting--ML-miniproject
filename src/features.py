"""Lag-feature generation and (leak-free) scaling.

Stanford reference model:  Q(t) = f(R(t-1..t-m), Q(t-1..t-n))
m = number of rainfall lags, n = number of runoff lags.
"""
from __future__ import annotations

import pandas as pd


def make_lagged(df: pd.DataFrame, m: int, n: int, include_current_rain: bool = False) -> pd.DataFrame:
    """Build the supervised table from a continuous daily dataframe.

    Because df has one row per calendar day, shift(k) is exactly 'k days ago'.
    Rows with any NaN (missing data inside the window) are dropped.
    include_current_rain=True adds R(t) (same-day rain). Default False follows
    the reference formula (pure forecast from information up to t-1).
    """
    out = pd.DataFrame(index=df.index)
    if include_current_rain:
        out["R_t"] = df["rainfall_mm"]
    for k in range(1, m + 1):
        out[f"R_lag{k}"] = df["rainfall_mm"].shift(k)
    for k in range(1, n + 1):
        out[f"Q_lag{k}"] = df["discharge_cfs"].shift(k)
    out["target"] = df["discharge_cfs"]
    return out.dropna()


class MaxScaler:
    """Divide rainfall columns by the TRAIN max rainfall and discharge columns
    (Q_lag*, target) by the TRAIN max discharge (same idea as the reference
    paper's divide-by-max normalisation). Fit on training data only (no leakage)."""

    def fit(self, train_lagged: pd.DataFrame):
        self.q_max = float(train_lagged["target"].max())
        r_cols = [c for c in train_lagged.columns if c.startswith("R_")]
        self.r_max = float(train_lagged[r_cols].max().max()) if r_cols else 1.0
        return self

    def transform(self, lagged: pd.DataFrame):
        X = lagged[[c for c in lagged.columns if c != "target"]].copy()
        for c in X.columns:
            X[c] = X[c] / (self.r_max if c.startswith("R_") else self.q_max)
        y = lagged["target"] / self.q_max
        return X.values, y.values

    def inverse_target(self, y_scaled):
        return y_scaled * self.q_max
