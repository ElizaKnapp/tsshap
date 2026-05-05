"""
Expanding window backtesting to generate surrogate training data.

The procedure (§4.2 of TsSHAP paper):
    1. Partition the series into expanding train splits, each test window
       is exactly `horizon` steps.
    2. For each split, train the forecaster on the train portion and produce
       a forecast over the test portion.
    3. Concatenate all test-window predictions to obtain a backtested
       forecast series aligned with the original index.

This backtested series becomes the regression target y_surrogate for the
surrogate model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsshap.forecasters.base import BaseForecaster


@dataclass
class BacktestResult:
    """Container for backtesting outputs."""

    forecasts: pd.Series
    """Backtested forecasts aligned to the original series index."""

    actuals: pd.Series
    """Original series values aligned to the same index as forecasts."""

    horizon: int
    """Forecast horizon used."""

    n_splits: int
    """Number of expanding window splits used."""


def expanding_window_backtest(
    y: pd.Series,
    forecaster: BaseForecaster,
    horizon: int,
    min_train_size: int | None = None,
    step: int = 1,
    refit: bool = True,
) -> BacktestResult:
    """
    Run expanding-window backtesting and return aligned forecasts.

    Parameters
    ----------
    y : pd.Series
        The full univariate time series.
    forecaster : BaseForecaster
        Forecaster to be backtested.  Its fit() method is called for each
        window when ``refit=True``.
    horizon : int
        Forecast horizon H.  Each test window is exactly H steps.
    min_train_size : int or None
        Minimum number of training observations for the first split.
        Defaults to max(2 * horizon, 10% of series length).
    step : int
        Step size between consecutive splits (1 = one new observation per split).
        Larger values reduce computation at the cost of fewer training samples.
    refit : bool
        Whether to call fit() for each window.  Set to False for pre-trained
        ML models that do not need to be retrained per window (§4.2).

    Returns
    -------
    BacktestResult
        Backtested forecasts with the same DatetimeIndex / RangeIndex as y.
    """
    n = len(y)
    if min_train_size is None:
        min_train_size = max(2 * horizon, max(int(0.1 * n), 10))

    forecast_values = np.full(n, np.nan)
    n_splits = 0

    for train_end in range(min_train_size, n - horizon + 1, step):
        y_train = y.iloc[:train_end]
        test_start = train_end
        test_end = train_end + horizon

        if refit:
            forecaster.fit(y_train)

        preds = forecaster.predict(horizon)
        forecast_values[test_start:test_end] = preds[: test_end - test_start]
        n_splits += 1

    mask = ~np.isnan(forecast_values)
    return BacktestResult(
        forecasts=pd.Series(forecast_values, index=y.index, name="backtested_forecast"),
        actuals=y,
        horizon=horizon,
        n_splits=n_splits,
    )
