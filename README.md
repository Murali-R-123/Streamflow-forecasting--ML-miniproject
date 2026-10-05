# Application of Artificial Neural Network in Streamflow Forecasting

ML Mini-Project (UE24CS352A) - Problem 11 - team of two.

**Team:**

_<R Muralidharan, PES2UG24CS387>_
_<Nikhil K R, PES2UG24CS815>_

Predicts daily streamflow Q(t) at **USGS 02472000, Leaf River near Collins, MS** from past rainfall and past
streamflow with a feed-forward ANN, compared against a persistence baseline and linear regression.

Reference: Xiao & You, *Application of Artificial Neural Network in Streamflow Forecasting*, Stanford CS229 (Spring 2020),
<https://cs229.stanford.edu/proj2020spr/report/Xiao_You.pdf>. Model form: `Q(t) = f(R(t-1..t-m), Q(t-1..t-n))`.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python download_data.py          # 1. downloads raw data, builds data/streamflow_data.csv (needs internet)
python pipeline.py               # 2. full run: search on validation years, test evaluation, figures, saved model
python pipeline.py --quick       #    (optional) tiny grid for a fast check
jupyter notebook notebooks/streamflow_demo.ipynb   # 3. step-by-step live demo
python tests/smoke_test.py       #    code check on SYNTHETIC data only (never quote its numbers)
```
Outputs: `results/figures/*.png`, `results/metrics/*.csv|json`, `results/ann_model.joblib`, `data/processed/*.csv`.

## Data
| Item | Source |
|---|---|
| Streamflow | USGS NWIS daily mean discharge (param 00060, cfs), site 02472000 |
| Rainfall | Daymet v4 daily precipitation (mm/day), grid cell at the gauge's lat/lon (31.7069, -89.4069) |
| Period | 2003-01-01 to 2012-12-31 (same years as the reference project) |

Differences from the reference: the paper used rainfall "at Collins" (source not stated); we use gridded Daymet.
Daymet has no 31 Dec in leap years (2004, 2008, 2012), so those days become missing and are handled as gaps.
The paper's text says "Collins, LA", but USGS 02472000 is in Covington County, **Mississippi**.

## Method (short)
1. Clean: parse dates, sort, drop duplicates, reindex to full daily calendar, negatives -> NaN. **No interpolation.**
2. Lag features R(t-1..t-m), Q(t-1..t-n); rows whose window touches a gap are dropped.
3. Chronological split by target date: **train 2003-2006, validation 2007, test 2008-2012** (no shuffling).
4. Scaling: divide by train maximum (rain and flow separately), fitted on train only.
5. ANN: MLP (sigmoid hidden layer(s), linear output, Adam, MSE), early stopping on validation MSE.
6. Grid over m, n and hidden size scored on validation RMSE only; final model evaluated once on test years, 5 seeds.
7. Metrics: MAE, MSE, RMSE, R2 (= NSE), RMSE as % of max flow, RMSE on peak (> threshold) and non-peak days.

The reference used 2003-2007 train / 2008-2012 "validation" and picked the best model on those same years; we
keep 2008-2012 as an untouched test set instead.
##Peak-Flow Results
Model	Peak RMSE	Non-Peak RMSE
Persistence	2366.2	213.8
Linear Regression	1451.5	339.5
ANN	1528.2	253.8

## Manual download (if `download_data.py` fails)
* USGS: site page -> Daily data -> select discharge, 2003-01-01..2012-12-31, tab-separated -> save as
  `data/raw/usgs_02472000_daily_discharge_raw.tsv`.
* Daymet: <https://daymet.ornl.gov/> single-pixel tool, lat 31.7069 lon -89.4069, variable prcp, CSV -> `data/raw/daymet_precip_raw.csv`.
* Then in Python: `from src.data_loader import *` and build the CSV with `parse_usgs_rdb`, `parse_daymet_csv`, `build_master`,
  or create `data/streamflow_data.csv` by hand with columns `date,rainfall_mm,discharge_cfs`.

## Layout
```
pipeline.py  download_data.py  requirements.txt
src/        data_loader.py  features.py  model.py  evaluate.py
notebooks/  streamflow_demo.ipynb
data/       raw/  processed/  streamflow_data.csv (created by download_data.py)
results/    figures/  metrics/  ann_model.joblib (created by pipeline.py)
deliverables/  report outline, slide outline  (put final PDFs here)
tests/      smoke_test.py (synthetic data)
```
