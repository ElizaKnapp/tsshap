"""
Visualization helpers for TsSHAP explanations.

Provides functions for:
    - Bar chart of global / semi-local feature importance
    - SHAP waterfall / force plot for local explanations
    - Partial Dependence Plot (PDP)
    - SHAP Dependence Plot (SDP)
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


# ---------------------------------------------------------------------------
# Partial Dependence Plot
# ---------------------------------------------------------------------------

def plot_pdp(
    result: ExplanationResult,
    feature: str,
    n_points: int = 50,
    ax: plt.Axes | None = None,
    surrogate=None,
    feature_matrix: pd.DataFrame | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Partial Dependence Plot: how the surrogate's average prediction changes
    as ``feature`` varies over its range, while other features are held at
    their observed values (Monte Carlo PDP).

    Parameters
    ----------
    result : ExplanationResult
    feature : str
        Name of the feature to vary.
    n_points : int
        Number of grid points for the feature range.
    surrogate : SurrogateModel or None
        Must be passed if not available via result.
    feature_matrix : pd.DataFrame or None
        Full feature matrix; must be provided alongside ``surrogate``.
    """
    if surrogate is None or feature_matrix is None:
        raise ValueError(
            "Pass `surrogate` and `feature_matrix` (from explainer.surrogate and "
            "explainer.feature_matrix)."
        )

    ax = ax or plt.gca()
    X = feature_matrix.fillna(0)
    col_idx = list(X.columns).index(feature)
    grid = np.linspace(X[feature].min(), X[feature].max(), n_points)

    means = []
    for val in grid:
        X_copy = X.copy()
        X_copy.iloc[:, col_idx] = val
        means.append(surrogate.predict(X_copy).mean())

    ax.plot(grid, means, color="#4C72B0", linewidth=1.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("Average surrogate prediction")
    ax.set_title(title or f"PDP: {feature}")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# SHAP Dependence Plot
# ---------------------------------------------------------------------------

def plot_sdp(
    result: ExplanationResult,
    feature: str,
    interaction_feature: str | None = None,
    ax: plt.Axes | None = None,
    feature_matrix: pd.DataFrame | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    SHAP Dependence Plot: scatter of feature value vs. SHAP value for that
    feature across the dataset, optionally colored by an interaction feature.

    Parameters
    ----------
    result : ExplanationResult
    feature : str
        The primary feature to plot on the x-axis.
    interaction_feature : str or None
        A second feature used for point coloring.
    feature_matrix : pd.DataFrame or None
        Full feature matrix (needed to extract feature values by name).
    """
    if feature not in result.shap_matrix.columns:
        raise ValueError(f"Feature {feature!r} not in SHAP matrix.")

    ax = ax or plt.gca()
    shap_col = result.shap_matrix[feature]

    if feature_matrix is not None and feature in feature_matrix.columns:
        feat_vals = feature_matrix[feature].reindex(result.shap_matrix.index)
    else:
        feat_vals = result.shap_matrix.index.to_series().reset_index(drop=True)

    scatter_kwargs: dict = {"s": 15, "alpha": 0.7, "edgecolors": "none"}
    if interaction_feature and feature_matrix is not None and interaction_feature in feature_matrix.columns:
        c = feature_matrix[interaction_feature].reindex(result.shap_matrix.index)
        sc = ax.scatter(feat_vals, shap_col, c=c, cmap="coolwarm", **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label=interaction_feature)
    else:
        ax.scatter(feat_vals, shap_col, color="#4C72B0", **scatter_kwargs)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(title or f"SDP: {feature}")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Summary plot (beeswarm-style) — delegates to shap library
# ---------------------------------------------------------------------------

def plot_shap_summary(
    result: ExplanationResult,
    feature_matrix: pd.DataFrame | None = None,
    max_display: int = 15,
) -> None:
    """
    Render a SHAP summary (beeswarm) plot using the `shap` library.
    """
    import shap as shap_lib

    X_display = (
        feature_matrix.reindex(result.shap_matrix.index).fillna(0)
        if feature_matrix is not None
        else result.shap_matrix
    )
    shap_lib.summary_plot(
        result.shap_matrix.values,
        X_display,
        feature_names=result.feature_names,
        max_display=max_display,
        show=True,
    )
