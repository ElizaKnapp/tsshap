"""Lag and seasonal-lag feature extractors."""

from __future__ import annotations

import pandas as pd

from tsshap.features.base import FeatureExtractor


class LagFeatures(FeatureExtractor):
    """
    Value of the time series at previous time steps.

    Feature ``y(t-k)`` is the observation k steps before t.

    Parameters
    ----------
    lags : int or list[int]
        Lag indices to include.  E.g. ``lags=3`` produces lags 1, 2, 3.
        A list like ``[1, 3, 7]`` includes only those specific lags.
    name : str
        Base name prefix for the series (used in column names).
    """

    def __init__(self, lags: int | list[int] = 3, name: str = "y"):
        if isinstance(lags, int):
            self.lags = list(range(1, lags + 1))
        else:
            self.lags = sorted(lags)
        self.name = name

    def fit(self, y: pd.Series) -> "LagFeatures":
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        cols = {f"{self.name}(t-{k})": y.shift(k) for k in self.lags}
        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        return [f"{self.name}(t-{k})" for k in self.lags]


class SeasonalLagFeatures(FeatureExtractor):
    """
    Value of the time series at previous seasonal time steps.

    Feature ``y(t - s*m)`` is the value s seasons ago.

    Parameters
    ----------
    lags : int or list[int]
        Number of seasonal lags (i.e. multiples of the period).
    period : int
        The seasonal period m (e.g. 52 for weekly, 12 for monthly).
    name : str
        Base name prefix for the series.
    """

    def __init__(self, lags: int | list[int] = 2, period: int = 12, name: str = "y"):
        if isinstance(lags, int):
            self.lags = list(range(1, lags + 1))
        else:
            self.lags = sorted(lags)
        self.period = period
        self.name = name

    def fit(self, y: pd.Series) -> "SeasonalLagFeatures":
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        cols = {
            f"{self.name}(t-{s}*{self.period})": y.shift(s * self.period)
            for s in self.lags
        }
        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        return [f"{self.name}(t-{s}*{self.period})" for s in self.lags]
