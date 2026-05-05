"""Simple baseline forecasters: Naive, SeasonalNaive, MovingAverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsshap.forecasters.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Forecast = last observed value, repeated for all horizon steps."""

    def __init__(self):
        self._last_value: float | None = None

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self._last_value = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("Call fit() first.")
        return np.full(horizon, self._last_value)

    @property
    def name(self) -> str:
        return "Naive"


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Forecast = value from the same position in the previous season.

    f_hat(T+h) = y(T + h - m)  where m is the seasonal period.
    """

    def __init__(self, period: int = 12):
        self.period = period
        self._y: pd.Series | None = None

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        self._y = y.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("Call fit() first.")
        preds = np.empty(horizon)
        n = len(self._y)
        for h in range(1, horizon + 1):
            idx = n - self.period + ((h - 1) % self.period)
            preds[h - 1] = float(self._y.iloc[idx])
        return preds

    @property
    def name(self) -> str:
        return f"SeasonalNaive(m={self.period})"


class MovingAverageForecaster(BaseForecaster):
    """
    Forecast = mean of the last k observations.

    All horizon steps receive the same value (fixed moving average).
    """

    def __init__(self, k: int = 6):
        self.k = k
        self._mean: float | None = None

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        self._mean = float(y.iloc[-self.k:].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() first.")
        return np.full(horizon, self._mean)

    @property
    def name(self) -> str:
        return f"MovingAverage(k={self.k})"
