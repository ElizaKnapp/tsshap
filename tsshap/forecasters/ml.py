"""ML-based forecasters: XGBoost and Prophet."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsshap.forecasters.base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based time series forecaster using lag features.

    Reduces forecasting to regression: at each training step t the input is
    [y(t-1), y(t-2), ..., y(t-lags)] and the target is y(t).  Multi-step
    forecasting is done recursively (MIMO-style using the previous forecast).

    Parameters
    ----------
    lags : int
        Number of lag features.
    xgb_params : dict or None
        Parameters passed to xgboost.XGBRegressor.
    """

    def __init__(self, lags: int = 12, xgb_params: dict | None = None):
        self.lags = lags
        self.xgb_params = xgb_params or {}
        self._model = None
        self._history: np.ndarray | None = None

    def fit(self, y: pd.Series) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        values = y.values.astype(float)
        X, Y = self._make_lags(values)
        self._model = XGBRegressor(n_estimators=100, **self.xgb_params)
        self._model.fit(X, Y)
        # Keep the last `lags` values for recursive prediction
        self._history = values[-self.lags:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None or self._history is None:
            raise RuntimeError("Call fit() first.")
        buf = list(self._history)
        preds = []
        for _ in range(horizon):
            x = np.array(buf[-self.lags:]).reshape(1, -1)
            p = float(self._model.predict(x)[0])
            preds.append(p)
            buf.append(p)
        return np.array(preds)

    def _make_lags(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for i in range(self.lags, len(values)):
            X.append(values[i - self.lags: i])
            Y.append(values[i])
        return np.array(X), np.array(Y)

    @property
    def name(self) -> str:
        return f"XGBoost(lags={self.lags})"


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecaster.

    Parameters
    ----------
    prophet_params : dict or None
        Parameters passed to prophet.Prophet (e.g. yearly_seasonality, etc.).
    """

    def __init__(self, prophet_params: dict | None = None):
        self.prophet_params = prophet_params or {}
        self._model = None
        self._last_date: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, y: pd.Series) -> "ProphetForecaster":
        from prophet import Prophet

        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("ProphetForecaster requires a DatetimeIndex.")

        self._freq = pd.infer_freq(y.index) or "D"
        self._last_date = y.index[-1]

        df = pd.DataFrame({"ds": y.index, "y": y.values})
        self._model = Prophet(**self.prophet_params)
        self._model.fit(df)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        future = self._model.make_future_dataframe(
            periods=horizon, freq=self._freq, include_history=False
        )
        forecast = self._model.predict(future)
        return forecast["yhat"].values[:horizon]

    @property
    def name(self) -> str:
        return "Prophet"
