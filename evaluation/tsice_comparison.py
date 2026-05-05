"""
TSICE comparison utilities.

TSICE (Time Series Individual Conditional Expectation) computes ICE curves
for time series forecasts to show how changing the value at each time step
affects the forecast.  This module wraps the `tsicebox` library and provides
utilities to align TSICE feature rankings with TsSHAP importances.

Reference: tsicebox — https://github.com/davide-burba/tsicebox

Installation: pip install tsicebox
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# TSICE wrapper
# ---------------------------------------------------------------------------

def run_tsice(
    y: pd.Series,
    forecaster,
    horizon: int = 1,
    n_samples: int = 20,
    perturbation_std: float | None = None,
) -> dict[str, float]:
    """
    Compute TSICE feature importances for a given forecaster.

    Each lag position's importance is measured as the mean absolute change
    in the forecast when that position is perturbed.

    Parameters
    ----------
    y : pd.Series
        Full observed time series.
    forecaster : BaseForecaster
        A fitted (or re-fittable) forecaster with a predict(horizon) method.
    horizon : int
        Forecast step to explain (1-indexed).
    n_samples : int
        Number of perturbation samples per time step.
    perturbation_std : float or None
        Standard deviation of Gaussian perturbations applied at each lag
        position.  Defaults to the series standard deviation.

    Returns
    -------
    dict[str, float]
        Mapping from lag label (e.g. "y(t-1)") to mean absolute ICE
        importance.
    """
    try:
        from tsicebox import TSICE
        return _run_tsice_native(y, forecaster, horizon, n_samples, perturbation_std)
    except ImportError:
        warnings.warn(
            "tsicebox is not installed.  Falling back to a manual ICE implementation. "
            "Install with: pip install tsicebox",
            ImportWarning,
            stacklevel=2,
        )
        return _run_tsice_manual(y, forecaster, horizon, n_samples, perturbation_std)


def _run_tsice_native(
    y: pd.Series,
    forecaster,
    horizon: int,
    n_samples: int,
    perturbation_std: float | None,
) -> dict[str, float]:
    """Use the tsicebox library if available."""
    from tsicebox import TSICE

    sigma = perturbation_std or float(y.std())

    def predict_fn(series: np.ndarray) -> float:
        s = pd.Series(series, dtype=float)
        forecaster.fit(s)
        return float(forecaster.predict(horizon)[horizon - 1])

    tsice = TSICE(predict_fn=predict_fn, n_samples=n_samples, perturbation_std=sigma)
    importances = tsice.fit_transform(y.values)

    # importances is a 1-D array of length len(y); index 0 = oldest lag
    n = len(importances)
    return {f"y(t-{n - i})": float(importances[i]) for i in range(n)}


def _run_tsice_manual(
    y: pd.Series,
    forecaster,
    horizon: int,
    n_samples: int,
    perturbation_std: float | None,
) -> dict[str, float]:
    """
    Fallback manual ICE implementation.

    For each lag position k (from 1 to len(y)), perturbs y(T-k) with
    n_samples Gaussian draws and measures the average absolute change in
    forecast at step `horizon`.
    """
    rng = np.random.default_rng(0)
    sigma = perturbation_std or float(y.std())

    # Baseline forecast
    forecaster.fit(y)
    baseline = float(forecaster.predict(horizon)[horizon - 1])

    values = y.values.astype(float).copy()
    n = len(values)
    importances: dict[str, float] = {}

    for k in range(1, n + 1):
        idx = n - k  # position of y(t-k) in the array
        original_val = values[idx]
        deltas = []
        for _ in range(n_samples):
            perturbed = values.copy()
            perturbed[idx] = original_val + rng.normal(0, sigma)
            s = pd.Series(perturbed, index=y.index, dtype=float)
            forecaster.fit(s)
            p = float(forecaster.predict(horizon)[horizon - 1])
            deltas.append(abs(p - baseline))
        importances[f"y(t-{k})"] = float(np.mean(deltas))

    return importances


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def align_importances(
    tsshap_importance: pd.Series,
    tsice_importance: dict[str, float],
) -> pd.DataFrame:
    """
    Align TsSHAP and TSICE importance vectors on common lag feature names.

    Both sides are normalised to [0, 1] for visual comparison.

    Returns
    -------
    pd.DataFrame with columns ["tsshap", "tsice"] and lag labels as index.
    """
    tsice_series = pd.Series(tsice_importance)

    # Only keep lag features that appear in TsSHAP (those starting with "y(t-")
    tsshap_lags = tsshap_importance[
        tsshap_importance.index.str.startswith("y(t-")
    ]

    common = tsshap_lags.index.intersection(tsice_series.index)
    if len(common) == 0:
        raise ValueError(
            "No common lag features found between TsSHAP and TSICE. "
            "Ensure LagFeatures are included in the TsSHAP feature extractors."
        )

    df = pd.DataFrame(
        {
            "tsshap": tsshap_lags.loc[common].abs(),
            "tsice": tsice_series.loc[common].abs(),
        }
    )
    # Normalise each column to [0, 1]
    for col in df.columns:
        col_max = df[col].max()
        if col_max > 0:
            df[col] = df[col] / col_max

    return df.sort_values("tsshap", ascending=False)


def compare_rankings(
    tsshap_importance: pd.Series,
    tsice_importance: dict[str, float],
) -> dict[str, float]:
    """
    Compute Spearman rank correlation between TsSHAP and TSICE importances.

    Returns
    -------
    dict with keys "spearman_rho" and "n_common_features".
    """
    df = align_importances(tsshap_importance, tsice_importance)
    if len(df) < 2:
        return {"spearman_rho": float("nan"), "n_common_features": len(df)}
    rho, pval = spearmanr(df["tsshap"], df["tsice"])
    return {
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "n_common_features": len(df),
    }


def plot_comparison(
    tsshap_importance: pd.Series,
    tsice_importance: dict[str, float],
    top_n: int = 15,
    ax=None,
    title: str | None = None,
):
    """
    Side-by-side horizontal bar chart comparing TsSHAP and TSICE importances.
    """
    import matplotlib.pyplot as plt

    df = align_importances(tsshap_importance, tsice_importance).head(top_n)
    ax = ax or plt.gca()

    y_pos = np.arange(len(df))
    bar_height = 0.35

    ax.barh(y_pos + bar_height / 2, df["tsshap"].values[::-1],
            height=bar_height, label="TsSHAP", color="#4C72B0")
    ax.barh(y_pos - bar_height / 2, df["tsice"].values[::-1],
            height=bar_height, label="TSICE", color="#DD8452")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index[::-1])
    ax.set_xlabel("Normalised importance")
    ax.set_title(title or "TsSHAP vs TSICE Feature Importance")
    ax.legend()
    plt.tight_layout()
    return ax
