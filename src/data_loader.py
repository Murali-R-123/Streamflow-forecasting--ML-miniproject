"""Data ingestion, cleaning and chronological splitting.

Master CSV format (data/streamflow_data.csv):
    date,rainfall_mm,discharge_cfs
Daily rows. Missing values are left as NaN (never invented).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

# Chronological split. The Stanford reference uses 2003-2007 train / 2008-2012
# validation. We keep 2008-2012 as an untouched TEST set and carve the last
# training year (2007) out as a VALIDATION set for tuning / early stopping,
# so no model choice is made by looking at the test years.
TRAIN_END = "2006-12-31"
VAL_END = "2007-12-31"


# ----------------------------------------------------------------- raw parsing
def parse_usgs_rdb(text: str) -> pd.DataFrame:
    """Parse USGS NWIS daily-values 'rdb' text (parameter 00060, mean daily)."""
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str)
    df = df.iloc[1:]  # row 0 is the rdb column-format row (e.g. '5s 15s 20d ...')
    value_cols = [c for c in df.columns if "00060" in c and not c.endswith("_cd")]
    if not value_cols:
        raise ValueError(f"No discharge (00060) column found. Columns: {list(df.columns)}")
    return pd.DataFrame({
        "date": pd.to_datetime(df["datetime"]),
        "discharge_cfs": pd.to_numeric(df[value_cols[0]], errors="coerce"),
    })


def parse_daymet_csv(text: str) -> pd.DataFrame:
    """Parse a Daymet single-pixel CSV (year, yday, prcp (mm/day))."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("year,yday"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    prcp_col = [c for c in df.columns if c.lower().startswith("prcp")][0]
    date = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(df["yday"] - 1, unit="D")
    return pd.DataFrame({"date": date, "rainfall_mm": pd.to_numeric(df[prcp_col], errors="coerce")})


def build_master(discharge: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    """Merge discharge and rainfall on date (outer join; gaps stay NaN)."""
    return pd.merge(rain, discharge, on="date", how="outer").sort_values("date").reset_index(drop=True)


# -------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame):
    """Return (clean_df, report). clean_df has one row per calendar day.

    Steps: parse dates -> drop duplicate dates -> sort -> reindex to full daily
    calendar -> mark impossible values (negative) as NaN. Missing values are NOT
    interpolated; rows whose lag window touches a NaN are dropped later.
    """
    report = {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    report["rows_raw"] = len(df)
    report["duplicate_dates"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date", keep="first").sort_values("date").set_index("date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    report["calendar_days_missing_rows"] = int(len(full) - len(df))
    df = df.reindex(full)
    df.index.name = "date"
    for col in ["rainfall_mm", "discharge_cfs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        report[f"{col}_negative"] = int((df[col] < 0).sum())
        df.loc[df[col] < 0, col] = np.nan
        report[f"{col}_missing"] = int(df[col].isna().sum())
    report["date_min"] = str(df.index.min().date())
    report["date_max"] = str(df.index.max().date())
    report["rows_clean"] = len(df)
    return df, report


def load_master(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first (see README)."
        )
    return clean(pd.read_csv(path))


# ------------------------------------------------------------------- splitting
def chronological_split(lagged: pd.DataFrame, train_end=TRAIN_END, val_end=VAL_END):
    """Split a (lagged) table by TARGET date: past -> train, then val, then test."""
    train = lagged.loc[:train_end]
    val = lagged.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):val_end]
    test = lagged.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):]
    return train, val, test
