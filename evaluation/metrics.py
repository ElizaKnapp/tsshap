"""
Evaluation metrics for TsSHAP explanations.

Implements the three metrics from §6.4 of the TsSHAP paper:
    - Faithfulness (Eq. 11): correlation between prediction change and
      explanation change under perturbation.
    - Sensitivity  (Eq. 12): average Euclidean distance between SHAP values
      on original vs. perturbed series.
    - Complexity   (Eq. 13): entropy of the fractional feature importance
      distribution.

Additionally provides:
    - surrogate_accuracy: MAPE between surrogate and black-box.
    - rank_correlation: Spearman rank correlation between two importance vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from evaluation.synthetic_data import block_bootstrap
from tsshap.explainer import TsSHAPExplainer, ExplanationResult


# ---------------------------------------------------------------------------
# Faithfulness  (Eq. 11)
# ---------------------------------------------------------------------------

def faithfulness(
    explainer: TsSHAPExplainer,
    y: pd.Series,
    scope: str = "global",
    n_bootstrap: int = 10,
    block_length: int | None = None,
    seed: int = 0,
) -> float:
    """
    Faithfulness metric: Pearson correlation between prediction change and
    total SHAP change across block-bootstrap perturbations.

    A higher value indicates that the explanation accurately reflects how
    the black-box forecaster's output changes with the input.

    Parameters
    ----------
    explainer : TsSHAPExplainer
        A *fitted* TsSHAPExplainer.
    y : pd.Series
        The training series (same one used to fit the explainer).
    scope : str
        Explanation scope ("global", "local", "semi-local").
    n_bootstrap : int
        Number of bootstrap perturbations.
    block_length : int or None
        Block length for bootstrap.  Defaults to sqrt(n).
    seed : int

    Returns
    -------
    float
        Pearson correlation μ_F.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    block_length = block_length or max(2, int(np.sqrt(n)))

    # Surrogate predictions on original
    X_orig = explainer.feature_matrix.fillna(0)
    preds_orig = explainer.surrogate.predict(X_orig)
    shap_orig = explainer.surrogate.shap_values(X_orig)

    delta_f_list, delta_phi_list = [], []

    for b in range(n_bootstrap):
        y_perturbed = block_bootstrap(y, block_length=block_length, seed=int(rng.integers(1e6)))

        # Refit extractors and recompute feature matrix on perturbed series
        frames = []
        for extractor in explainer.feature_extractors:
            extractor.fit(y_perturbed)
            frames.append(extractor.transform(y_perturbed))
        X_pert = pd.concat(frames, axis=1).fillna(0)

        preds_pert = explainer.surrogate.predict(X_pert)
        shap_pert = explainer.surrogate.shap_values(X_pert)

        # Per-time-step deltas
        df = preds_orig - preds_pert
        dphi = np.sum(np.abs(shap_orig) - np.abs(shap_pert), axis=1)

        delta_f_list.append(df)
        delta_phi_list.append(dphi)

    delta_f = np.concatenate(delta_f_list)
    delta_phi = np.concatenate(delta_phi_list)

    # Remove NaNs / infs
    mask = np.isfinite(delta_f) & np.isfinite(delta_phi)
    if mask.sum() < 2:
        return float("nan")
    r, _ = pearsonr(delta_f[mask], delta_phi[mask])
    return float(r)


# ---------------------------------------------------------------------------
# Sensitivity  (Eq. 12)
# ---------------------------------------------------------------------------

def sensitivity(
    explainer: TsSHAPExplainer,
    y: pd.Series,
    n_bootstrap: int = 10,
    block_length: int | None = None,
    seed: int = 0,
) -> float:
    """
    Sensitivity metric: average Euclidean distance between SHAP explanations
    on original and perturbed series.

    A lower value indicates more robust explanations.

    Returns
    -------
    float
        Median sensitivity μ_S.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    block_length = block_length or max(2, int(np.sqrt(n)))

    X_orig = explainer.feature_matrix.fillna(0)
    shap_orig = explainer.surrogate.shap_values(X_orig)

    distances = []
    for b in range(n_bootstrap):
        y_pert = block_bootstrap(y, block_length=block_length, seed=int(rng.integers(1e6)))

        frames = []
        for extractor in explainer.feature_extractors:
            extractor.fit(y_pert)
            frames.append(extractor.transform(y_pert))
        X_pert = pd.concat(frames, axis=1).fillna(0)

        shap_pert = explainer.surrogate.shap_values(X_pert)
        dist = np.linalg.norm(shap_orig - shap_pert, axis=1)
        distances.extend(dist.tolist())

    return float(np.nanmedian(distances))


# ---------------------------------------------------------------------------
# Complexity  (Eq. 13)
# ---------------------------------------------------------------------------

def complexity(result: ExplanationResult) -> float:
    """
    Complexity metric: entropy of the fractional feature importance distribution.

    Lower complexity means the explanation is concentrated on fewer features
    (easier to interpret).

    Uses mean |SHAP| values from the result's feature_importance attribute.

    Returns
    -------
    float
        Entropy μ_C.
    """
    imp = result.feature_importance.abs()
    total = imp.sum()
    if total == 0:
        return 0.0
    p = imp / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Surrogate accuracy
# ---------------------------------------------------------------------------

def surrogate_accuracy(
    backtested_forecasts: pd.Series,
    surrogate_preds: pd.Series,
    y: pd.Series,
) -> dict[str, float]:
    """
    Compute MAPE between surrogate and black-box forecasts.

    Parameters
    ----------
    backtested_forecasts : pd.Series
        Black-box backtested forecasts (surrogate training targets).
    surrogate_preds : pd.Series
        Surrogate model predictions on the same time steps.
    y : pd.Series
        Original observed series (used for MASE scaling).

    Returns
    -------
    dict with keys "MAPE"
    """
    valid = ~(backtested_forecasts.isna() | surrogate_preds.isna())
    actual = backtested_forecasts.loc[valid].values
    pred = surrogate_preds.loc[valid].values
    mape = float(np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-10))))

    return {"MAPE": mape}


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def rank_correlation(
    importance_a: pd.Series,
    importance_b: pd.Series,
) -> float:
    """
    Spearman rank correlation between two feature importance vectors.

    Features not present in both vectors are dropped.

    Returns
    -------
    float
        Spearman ρ in [-1, 1].
    """
    common = importance_a.index.intersection(importance_b.index)
    if len(common) < 2:
        return float("nan")
    a = importance_a.loc[common].abs()
    b = importance_b.loc[common].abs()
    rho, _ = spearmanr(a, b)
    return float(rho)
