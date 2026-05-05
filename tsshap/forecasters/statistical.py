"""Statistical forecasters wrapping statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from tsshap.forecasters.base import BaseForecaster


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Simple Exponential Smoothing forecaster (Holt-Winters single).

    Parameters
    ----------
    alpha : float or None
        Smoothing parameter.  If None, it is optimized by MLE.
    """

    def __init__(self, alpha: float | None = 0.5):
        self.alpha = alpha
        self._fit_result = None
        self._last_level: float | None = None

    def fit(self, y: pd.Series) -> "ExponentialSmoothingForecaster":
        model = SimpleExpSmoothing(y, initialization_method="estimated")
        if self.alpha is not None:
            self._fit_result = model.fit(smoothing_level=self.alpha, optimized=False)
        else:
            self._fit_result = model.fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Call fit() first.")
        forecast = self._fit_result.forecast(horizon)
        return np.asarray(forecast)

    @property
    def name(self) -> str:
        alpha_str = f"{self.alpha}" if self.alpha is not None else "opt"
        return f"ExponentialSmoothing(alpha={alpha_str})"
