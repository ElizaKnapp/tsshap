"""
STL-based interpretable features.

Decomposes the time series using Seasonal-Trend decomposition via LOESS (STL)
and exposes the trend, seasonal, and residual components as features.  These
are the most human-interpretable features and the natural starting point for
TsSHAP applied to seasonal time series.

Reference: Cleveland et al. (1990) STL: A Seasonal-Trend Decomposition
Procedure Based on Loess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from tsshap.features.base import FeatureExtractor


class STLFeatures(FeatureExtractor):
    """
    Extract trend, seasonal, and residual components via STL decomposition.

    For each time point t, the features are the STL components computed on
    the series up to t using an expanding window.  This ensures no look-ahead
    leakage: the component value at t is re-estimated using only y(1..t).

    Parameters
    ----------
    period : int or None
        Seasonal period (e.g. 12 for monthly, 52 for weekly, 7 for daily).
        If None, inferred from the pandas DatetimeIndex frequency.
    robust : bool
        Use robust STL fitting (down-weights outliers).  Default True.
    min_obs : int or None
        Minimum number of observations required before computing STL.
        Must be >= 2 * period + 1.  If None, defaults to 2 * period + 1.
    include_trend : bool
        Include the trend component as a feature.
    include_seasonal : bool
        Include the seasonal component as a feature.
    include_residual : bool
        Include the residual component as a feature.
    include_strength : bool
        Include trend-strength and seasonal-strength scalar features
        (computed globally, not per-step — added as constant columns).
    """

    _FREQ_TO_PERIOD = {
        "A": 1, "Y": 1,
        "Q": 4, "QS": 4,
        "M": 12, "MS": 12,
        "W": 52, "W-SUN": 52, "W-MON": 52,
        "D": 7,
        "H": 24,
        "T": 60, "min": 60,
    }

    def __init__(
        self,
        period: int | None = None,
        robust: bool = True,
        min_obs: int | None = None,
        include_trend: bool = True,
        include_seasonal: bool = True,
        include_residual: bool = True,
        include_strength: bool = True,
    ):
        self.period = period
        self.robust = robust
        self.min_obs = min_obs
        self.include_trend = include_trend
        self.include_seasonal = include_seasonal
        self.include_residual = include_residual
        self.include_strength = include_strength

        self._period: int | None = None
        self._trend_strength: float | None = None
        self._seasonal_strength: float | None = None

    # ------------------------------------------------------------------
    # FeatureExtractor interface
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series) -> "STLFeatures":
        self._period = self._resolve_period(y)
        # Compute global strength metrics from the full series
        result = STL(y, period=self._period, robust=self.robust).fit()
        var_resid = np.var(result.resid)
        var_trend_resid = np.var(result.trend + result.resid)
        var_seasonal_resid = np.var(result.seasonal + result.resid)
        self._trend_strength = max(0.0, 1.0 - var_resid / (var_trend_resid + 1e-10))
        self._seasonal_strength = max(0.0, 1.0 - var_resid / (var_seasonal_resid + 1e-10))
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        if self._period is None:
            raise RuntimeError("Call fit() before transform().")

        period = self._period
        min_obs = self.min_obs if self.min_obs is not None else 2 * period + 1

        n = len(y)
        trends = np.full(n, np.nan)
        seasonals = np.full(n, np.nan)
        residuals = np.full(n, np.nan)

        for t in range(min_obs - 1, n):
            sub = y.iloc[: t + 1]
            try:
                res = STL(sub, period=period, robust=self.robust).fit()
                trends[t] = res.trend.iloc[-1]
                seasonals[t] = res.seasonal.iloc[-1]
                residuals[t] = res.resid.iloc[-1]
            except Exception:
                pass

        cols: dict[str, np.ndarray] = {}
        if self.include_trend:
            cols["stl_trend"] = trends
        if self.include_seasonal:
            cols["stl_seasonal"] = seasonals
        if self.include_residual:
            cols["stl_residual"] = residuals
        if self.include_strength:
            cols["stl_trend_strength"] = np.where(
                np.isnan(trends), np.nan, self._trend_strength
            )
            cols["stl_seasonal_strength"] = np.where(
                np.isnan(seasonals), np.nan, self._seasonal_strength
            )

        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        names = []
        if self.include_trend:
            names.append("stl_trend")
        if self.include_seasonal:
            names.append("stl_seasonal")
        if self.include_residual:
            names.append("stl_residual")
        if self.include_strength:
            names += ["stl_trend_strength", "stl_seasonal_strength"]
        return names

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_period(self, y: pd.Series) -> int:
        if self.period is not None:
            return int(self.period)
        if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None:
            freq_str = y.index.freqstr.split("-")[0]
            if freq_str in self._FREQ_TO_PERIOD:
                return self._FREQ_TO_PERIOD[freq_str]
        # Fallback: try to detect from index
        if isinstance(y.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(y.index)
            if inferred:
                freq_str = inferred.split("-")[0]
                if freq_str in self._FREQ_TO_PERIOD:
                    return self._FREQ_TO_PERIOD[freq_str]
        raise ValueError(
            "Cannot infer seasonal period. Pass `period` explicitly to STLFeatures."
        )
