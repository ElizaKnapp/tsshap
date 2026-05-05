"""Rolling and expanding window feature extractors."""

from __future__ import annotations

import pandas as pd

from tsshap.features.base import FeatureExtractor


_DEFAULT_STATS = ("mean", "max", "min", "std")


class RollingWindowFeatures(FeatureExtractor):
    """
    Rolling window statistics over the most recent `window` observations.

    All statistics are computed on y(t-window), ..., y(t-1) so they are
    strictly causal (no look-ahead).

    Parameters
    ----------
    window : int
        Number of lagged observations in the rolling window.
    stats : tuple[str, ...]
        Statistics to compute.  Supported: "mean", "max", "min", "std", "median".
    name : str
        Base name prefix for column names.
    """

    def __init__(
        self,
        window: int = 3,
        stats: tuple[str, ...] = _DEFAULT_STATS,
        name: str = "y",
    ):
        self.window = window
        self.stats = stats
        self.name = name

    def fit(self, y: pd.Series) -> "RollingWindowFeatures":
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        # shift(1) makes the window end at t-1 (strictly causal)
        shifted = y.shift(1)
        cols: dict[str, pd.Series] = {}
        roller = shifted.rolling(window=self.window, min_periods=self.window)
        for stat in self.stats:
            col = f"{self.name}-{stat}(t-1,t-{self.window})"
            if stat == "mean":
                cols[col] = roller.mean()
            elif stat == "max":
                cols[col] = roller.max()
            elif stat == "min":
                cols[col] = roller.min()
            elif stat == "std":
                cols[col] = roller.std()
            elif stat == "median":
                cols[col] = roller.median()
            else:
                raise ValueError(f"Unsupported stat: {stat}")
        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        return [f"{self.name}-{s}(t-1,t-{self.window})" for s in self.stats]


class ExpandingWindowFeatures(FeatureExtractor):
    """
    Expanding window statistics over all observations up to t-1.

    Parameters
    ----------
    stats : tuple[str, ...]
        Statistics to compute.  Supported: "mean", "max", "min", "std".
    name : str
        Base name prefix for column names.
    """

    def __init__(
        self,
        stats: tuple[str, ...] = _DEFAULT_STATS,
        name: str = "y",
    ):
        self.stats = stats
        self.name = name

    def fit(self, y: pd.Series) -> "ExpandingWindowFeatures":
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        shifted = y.shift(1)
        cols: dict[str, pd.Series] = {}
        expander = shifted.expanding(min_periods=1)
        for stat in self.stats:
            col = f"{self.name}-{stat}(0,t-1)"
            if stat == "mean":
                cols[col] = expander.mean()
            elif stat == "max":
                cols[col] = expander.max()
            elif stat == "min":
                cols[col] = expander.min()
            elif stat == "std":
                cols[col] = expander.std()
            elif stat == "median":
                cols[col] = expander.median()
            else:
                raise ValueError(f"Unsupported stat: {stat}")
        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        return [f"{self.name}-{s}(0,t-1)" for s in self.stats]
