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
│   ├── surrogate.py           # Tree surrogate (XGBoost / LightGBM / CatBoost / sklearn GB)
│   ├── explainer.py           # TsSHAPExplainer (main API)
│   └── explanations.py        # Plotting helpers: importance + forecast comparison
├── evaluation/
│   ├── synthetic_data.py      # Synthetic trend+seasonal generator + component metadata
│   ├── metrics.py             # surrogate_accuracy (MAPE vs black-box forecasts)
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
from tsshap.explanations import plot_importance

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
n = len(explainer.feature_matrix)
semilocal_result = explainer.explain(scope='semi-local', t_start=n - 12, t_end=n)

plot_importance(global_result)
print(global_result.top_features())
```

---

## User manual

### Workflow

1. Choose a **black-box forecaster** (any class from `tsshap.forecasters`).
2. Build a list of **feature extractors** (`tsshap.features`). Their outputs are concatenated **column-wise** into one design matrix.
3. Instantiate **`TsSHAPExplainer`**, call **`fit(y)`** on your observed series (a **`pd.Series`**; use a **`pd.DatetimeIndex`** if you use `DateFeatures` or `ProphetForecaster`).
4. Call **`explain(...)`** with scope **`global`**, **`local`**, or **`semi-local`**.
5. Inspect **`ExplanationResult`** (importances, SHAP matrix, surrogate vs black-box series) or plot via **`tsshap.explanations`**.

### `TsSHAPExplainer` constructor

| Argument | Meaning |
|---|---|
| `forecaster` | Instance implementing `fit(y)` / `predict(horizon)`. |
| `feature_extractors` | List of `FeatureExtractor` instances (can mix families). |
| `horizon` | Forecast horizon for backtesting; if omitted, defaults to ~10% of series length. |
| `surrogate_backend` | `"xgboost"`, `"lightgbm"`, `"catboost"`, or `"sklearn"`; omitted → auto-pick (often `sklearn` if tree libs missing). |
| `surrogate_params` | Extra kwargs for the underlying surrogate model. |
| `min_train_size` | Minimum history length before the first backtest point. |
| `backtest_step` | Step between expanding-window refits (default 1). |
| `refit_forecaster` | If `True`, refit the black-box each window; set `False` for heavy models you fit once. |

After **`fit(y)`**:

- **`explainer.feature_matrix`** — full feature DataFrame (with NaNs where lags/STL are not yet defined).
- **`explainer.backtest_result`** — actuals and black-box backtested forecasts.
- **`explainer.surrogate`** — fitted `SurrogateModel` (for R² with `surrogate.score(X, y_bt)` if needed).

### After `explain(scope=...)`

`explain` returns **`ExplanationResult`** with:

- **`feature_importance`** — `pd.Series`: for `global` / `semi-local`, mean `|SHAP|` per feature; for `local`, signed SHAP at time `t`.
- **`shap_matrix`** — full SHAP values over time (`DataFrame`).
- **`surrogate_preds`**, **`backtested_forecasts`**, **`actuals`** — aligned series for comparison plots.

| Scope | Call | Notes |
|---|---|---|
| Global | `explainer.explain(scope='global')` | Mean `|SHAP|` over valid rows. |
| Local | `explainer.explain(scope='local', t=-1)` | Integer index `t` (negative allowed). |
| Semi-local | `n = len(explainer.feature_matrix)` then `explainer.explain(scope='semi-local', t_start=n - 12, t_end=n)` | Slice `[t_start:t_end)` over row indices; both must be set. |

Plot helpers:

```python
from tsshap.explanations import plot_importance, plot_forecasts

plot_importance(result_global, top_n=15, title="Global importance")
plot_forecasts(result_global, title="Surrogate vs black-box")
```

### Feature extractors (`from tsshap.features import ...`)

All extractors implement **`fit(y)`** then **`transform(y)`** (or use them through `TsSHAPExplainer`, which fits them for you).

| Class | Role | Example |
|---|---|---|
| **`STLFeatures`** | Expanding-window STL: `stl_trend`, `stl_seasonal`, `stl_residual`, optional strength columns. | `STLFeatures(period=12, robust=True, include_strength=True)` — `period` can be `None` if frequency is inferable from index. |
| **`LagFeatures`** | Short lags `y(t-1)…y(t-k)`. | `LagFeatures(lags=6)` or `LagFeatures(lags=[1, 3, 7])`. |
| **`SeasonalLagFeatures`** | Lags at multiples of season length: `y(t-m)`, `y(t-2m)`, …. | `SeasonalLagFeatures(lags=2, period=12)`. |
| **`RollingWindowFeatures`** | Causal rolling stats on `y(t-window)…y(t-1)`: `mean`, `max`, `min`, `std`, `median`. | `RollingWindowFeatures(window=6, stats=('mean', 'std'))`. |
| **`ExpandingWindowFeatures`** | Expanding stats on all past values up to `t-1`. | `ExpandingWindowFeatures(stats=('mean', 'max'))`. |
| **`TrendFeatures`** | Polynomial time index `t1, t2, …` (0-based integer position). | `TrendFeatures(degree=2)` → columns `t1`, `t2`. |
| **`DateFeatures`** | Calendar encodings from **`DatetimeIndex`** only. | `DateFeatures(features=['month', 'quarter', 'day_of_week'])` — see class docstring for full name list. |

**Combining extractors:** pass a list in declaration order; duplicate column names should be avoided across extractors.

### Black-box forecasters (`from tsshap.forecasters import ...`)

| Class | Constructor example | Notes |
|---|---|---|
| **`NaiveForecaster`** | `NaiveForecaster()` | Last value repeated. |
| **`SeasonalNaiveForecaster`** | `SeasonalNaiveForecaster(period=12)` | Value from `period` steps ago. |
| **`MovingAverageForecaster`** | `MovingAverageForecaster(k=6)` | Trailing mean. |
| **`ExponentialSmoothingForecaster`** | `ExponentialSmoothingForecaster(alpha=0.5)` | Statsmodels simple exponential smoothing. |
| **`XGBoostForecaster`** | `XGBoostForecaster(lags=12)` | Lag regression + recursive horizon; needs working **XGBoost**. |
| **`ProphetForecaster`** | `ProphetForecaster()` or `ProphetForecaster(prophet_params={...})` | Needs **`prophet`** and **`y`** with **`DatetimeIndex`**. |

### Surrogate backend quick reference

Pass explicitly if you want reproducibility across machines:

```python
explainer = TsSHAPExplainer(
    forecaster=forecaster,
    feature_extractors=[...],
    horizon=12,
    surrogate_backend="sklearn",  # or "xgboost", "lightgbm", "catboost"
)
```

---

## Algorithm Overview

TsSHAP works in three stages:

```
1. Backtesting      black-box forecaster  →  backtested forecast series
2. Feature extraction  original series    →  interpretable feature matrix X
3. Surrogate + SHAP    Tree ensemble on (X, backtested forecasts)  →  SHAP values
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

**Bundled black-box forecasters (`tsshap.forecasters`):**

| Class | Notes |
|---|---|
| `NaiveForecaster` | Repeat last value |
| `SeasonalNaiveForecaster` | Repeat value from one seasonal period ago |
| `MovingAverageForecaster` | Mean of trailing window |
| `ExponentialSmoothingForecaster` | Simple exponential smoothing (statsmodels) |
| `XGBoostForecaster` | Recursive lag regression; optional deps / OpenMP on some platforms |
| `ProphetForecaster` | Prophet model; **`DatetimeIndex` required on `y`** |

**Evaluation (`evaluation/metrics.py`):**
- **`surrogate_accuracy`** — MAPE between surrogate predictions and black-box backtested forecasts (how well the surrogate mimics the forecaster outputs).

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_features_demo.ipynb` | Visualise **every** feature family on the CO₂ series (STL, lag, seasonal lag, rolling / expanding windows, trend, calendar). |
| `02_synthetic_evaluation.ipynb` | Six experiments: baseline TsSHAP, local / semi-local scopes, **surrogate fidelity across all six forecasters** (MAPE table + two-panel plots with **forecaster suptitles**), STL vs ground-truth variance shares, efficiency (surrogate backends + TreeSHAP vs KernelSHAP). |
| `03_tsice_comparison.ipynb` | TsSHAP vs TSICE on synthetic data: side-by-side lag-importance bars (with **forecaster suptitle** on the Naive walkthrough), then the same comparison **looped over all six forecasters**. |

**Environment tip:** XGBoost and Prophet need working installs (see `requirements.txt` or `pip install -e ".[all]"`). If `tsicebox` is installed, TSICE’s native path builds `Series` without the original time index; forecasters that require a **`DatetimeIndex`** (e.g. Prophet) are safer with the library’s **manual ICE fallback** (triggered when `tsicebox` is not importable) or after aligning that behavior in code.
