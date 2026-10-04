"""SMOKE TEST ONLY - uses SYNTHETIC (made-up) data to check that the code runs.
Outputs go to a temp folder. Never use these numbers in the report or slides."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pipeline import run


def synthetic(path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2003-01-01", "2012-12-31", freq="D")
    rain = rng.gamma(0.4, 12, len(dates)) * (rng.random(len(dates)) < 0.3)
    q, s = np.zeros(len(dates)), 100.0
    for i, r in enumerate(rain):
        s = 0.9 * s + 8 * r ** 1.1
        q[i] = s * 0.1 + 80
    df = pd.DataFrame({"date": dates, "rainfall_mm": rain, "discharge_cfs": q})
    df.loc[500:503, "discharge_cfs"] = np.nan          # simulate a data gap
    df.to_csv(path, index=False)


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    synthetic(tmp / "synthetic.csv")
    t = run(tmp / "synthetic.csv", tmp, quick=True, n_seeds=2, peak_threshold=400)
    print("\nSMOKE TEST PASSED (synthetic data, outputs in", tmp, ")")
