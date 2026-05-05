# TsSHAP: Feature-based Explainability for Time Series Forecasting

Implementation and evaluation of the TsSHAP algorithm from:

> Raykar et al. (2023). *TsSHAP: Robust model agnostic feature-based explainability for univariate time series forecasting.* IBM Research.

---

## Project Structure

```
final_project/
├── tsshap/                    # Core library
│   ├── features/              # Interpretable feature extractors
│   │   ├── stl_features.py    # STL decomposition (trend, seasonal, residual)
│   │   ├── lag_features.py    # Lag & seasonal lag features
│   │   ├── window_features.py # Rolling & expanding window statistics
│   │   ├── trend_features.py  # Polynomial trend features
│   │   └── date_features.py   # Calendar date/time encodings
│   ├── forecasters/           # Forecaster wrappers (fit/predict interface)
│   │   ├── naive.py           # Naive, SeasonalNaive, MovingAverage
│   │   ├── statistical.py     # ExponentialSmoothing (statsmodels)
│   │   └── ml.py              # XGBoost, Prophet
│   ├── backtesting.py         # Expanding window backtesting
│   ├── surrogate.py           # Tree surrogate model (XGBoost/LightGBM/CatBoost)
│   ├── explainer.py           # TsSHAPExplainer (main API)
│   └── explanations.py        # Plotting: importance, PDP, SDP, SHAP summary
├── evaluation/
│   ├── synthetic_data.py      # Synthetic time series generators
│   ├── metrics.py             # Faithfulness, sensitivity, complexity, surrogate accuracy
│   └── tsice_comparison.py    # TSICE wrapper + TsSHAP vs TSICE comparison
├── notebooks/
│   ├── 01_features_demo.ipynb          # Feature extraction showcase
│   ├── 02_synthetic_evaluation.ipynb   # End-to-end TsSHAP on synthetic data
│   └── 03_tsice_comparison.ipynb       # TsSHAP vs TSICE side-by-side
├── pyproject.toml           # Packaging metadata (pip / PyPI)
├── LICENSE
└── requirements.txt         # Full stack for local notebooks (optional extras below)
```

---

## Quick Start

### Install as a library (PyPI)

Distribution name on PyPI: **`tsshap-timeseries`** (import name remains **`tsshap`**).

```bash
pip install tsshap-timeseries          # core deps + sklearn surrogate fallback
pip install "tsshap-timeseries[all]"   # XGBoost, LightGBM, CatBoost, Prophet, seaborn
```

After you publish: `python -m build` then `twine upload dist/*` from this directory.

### Install from source (editable)

```bash
cd final_project
pip install -e ".[all]"
```

### Requirements file (notebooks / evaluation)

```bash
pip install -r requirements.txt
```

```python
from tsshap import TsSHAPExplainer
from tsshap.features import STLFeatures, LagFeatures, SeasonalLagFeatures
from tsshap.forecasters import SeasonalNaiveForecaster

forecaster = SeasonalNaiveForecaster(period=12)

explainer = TsSHAPExplainer(
    forecaster=forecaster,
    feature_extractors=[
        STLFeatures(period=12),
        LagFeatures(lags=3),
        SeasonalLagFeatures(lags=2, period=12),
    ],
    horizon=12,
)

explainer.fit(y_train)

global_result   = explainer.explain(scope='global')
local_result    = explainer.explain(scope='local', t=-1)
semilocal_result = explainer.explain(scope='semi-local', t_start=-12, t_end=None)

global_result.plot_importance()   # via explanations module
print(global_result.top_features())
```

---

## Algorithm Overview

TsSHAP works in three stages:

```
1. Backtesting      black-box forecaster  →  backtested forecast series
2. Feature extraction  original series    →  interpretable feature matrix X
3. Surrogate + SHAP    XGBoost on (X, backtested forecasts)  →  SHAP values
```

**Feature families (in order of complexity):**

| Family | Description |
|---|---|
| `STLFeatures` | STL decomposition: trend, seasonal, residual (expanding window, no look-ahead) |
| `LagFeatures` | y(t-1), y(t-2), ... |
| `SeasonalLagFeatures` | y(t-m), y(t-2m), ... |
| `RollingWindowFeatures` | mean/max/min/std over last k observations |
| `ExpandingWindowFeatures` | mean/max/min/std over all past observations |
| `TrendFeatures` | Polynomial time index t, t², ... |
| `DateFeatures` | Month, quarter, season, weekday, etc. (requires DatetimeIndex) |

**Explanation scopes:**
- **Local** — SHAP values at a single time step
- **Semi-local** — mean |SHAP| over a time interval
- **Global** — mean |SHAP| over the entire series

**Evaluation metrics:**
- **Faithfulness** — correlation between prediction change and SHAP change under perturbation
- **Sensitivity** — Euclidean distance between explanations on perturbed inputs
- **Complexity** — entropy of fractional feature importance distribution

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_features_demo.ipynb` | Visualise each feature family on the CO2 dataset |
| `02_synthetic_evaluation.ipynb` | End-to-end TsSHAP + correctness validation against ground truth |
| `03_tsice_comparison.ipynb` | TsSHAP vs TSICE rank correlation and visual comparison |
