# Two-page report outline (fill ONLY from your real outputs, then export to PDF)

1. **Problem statement** - why streamflow forecasting matters (flood warning, water planning); regression: predict Q(t)
   from R(t-1..t-m), Q(t-1..t-n). Reference: Xiao & You, Stanford CS229 2020.
2. **Dataset** - USGS 02472000 Leaf River near Collins, MS (discharge, cfs) + Daymet precipitation (mm/day), 2003-2012.
   Fill from `results/metrics/data_report.json`: rows, missing values. Split: train 2003-2006, val 2007, test 2008-2012
   (rows per split are printed by pipeline.py).
3. **Methodology** - cleaning, lag features, max-scaling fitted on train, ANN (hidden layers, sigmoid, linear output, Adam, MSE),
   early stopping, validation-only grid search (`grid_results.csv`), no shuffling / no leakage.
4. **Implementation** - Python, pandas, scikit-learn (MLPRegressor), matplotlib; repo layout; how to run.
5. **Results** - table from `metrics_summary.csv` (persistence vs linear regression vs ANN), figures
   `hydrograph_comparison.png`, `training_loss.png`. Report peak vs non-peak RMSE and the threshold used.
6. **Conclusions & limitations** - what the ANN did compared with baselines (state it whichever way it came out),
   peak-flow behaviour, single station, gridded rainfall, no same-day rainfall input, possible extensions.
