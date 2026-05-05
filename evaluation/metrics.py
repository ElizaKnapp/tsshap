"""
Evaluation metrics for TsSHAP surrogate quality.

Provides surrogate_accuracy: error between surrogate predictions and black-box
backtested forecasts (training targets for the surrogate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        Original observed series (reserved for future metrics such as MASE;
        currently unused).

    Returns
    -------
    dict with keys "MAPE"
    """
    valid = ~(backtested_forecasts.isna() | surrogate_preds.isna())
    actual = backtested_forecasts.loc[valid].values
    pred = surrogate_preds.loc[valid].values
    mape = float(np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-10))))

    return {"MAPE": mape}
