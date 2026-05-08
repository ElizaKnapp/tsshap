"""
Visualization helpers for TsSHAP explanations.

Provides functions for:
    - Bar chart of global / semi-local feature importance
    - Forecast vs surrogate comparison plot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tsshap.explainer import ExplanationResult


# ---------------------------------------------------------------------------
# Importance bar chart
# ---------------------------------------------------------------------------

def plot_importance(
    result: ExplanationResult,
    top_n: int = 15,
    ax: plt.Axes | None = None,
    title: str | None = None,
    color: str = "#4C72B0",
) -> plt.Axes:
    """
    Horizontal bar chart of mean |SHAP| feature importance.

    Works for global and semi-local scopes.  For local scope, raw signed
    SHAP values are shown (positive = pushes prediction up, negative = down).
    """
    ax = ax or plt.gca()
    imp = result.feature_importance
    top = imp.abs().sort_values(ascending=False).head(top_n)
    vals = imp.loc[top.index]

    colors = [color if v >= 0 else "#C44E52" for v in vals]
    ax.barh(range(len(vals)), vals.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index[::-1])
    ax.axvline(0, color="black", linewidth=0.8)

    scope_label = result.scope.capitalize()
    ax.set_xlabel("SHAP value" if result.scope == "local" else "Mean |SHAP|")
    ax.set_title(title or f"{scope_label} Feature Importance (TsSHAP)")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Forecast comparison
# ---------------------------------------------------------------------------

def plot_forecasts(
    result: ExplanationResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot original series, backtested forecasts, and surrogate predictions.
    """
    ax = ax or plt.gca()
    ax.plot(result.actuals, label="Actual", color="black", linewidth=1.2)
    bt = result.backtested_forecasts.dropna()
    ax.plot(bt, label="Black-box forecast", color="#4C72B0", linestyle="--", linewidth=1)
    sp = result.surrogate_preds
    ax.plot(sp, label="Surrogate", color="#DD8452", linestyle=":", linewidth=1.2)
    ax.legend()
    ax.set_title(title or "Forecaster vs Surrogate")
    ax.set_xlabel("Time")
    plt.tight_layout()
    return ax