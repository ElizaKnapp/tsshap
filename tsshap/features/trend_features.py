"""Polynomial trend feature extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsshap.features.base import FeatureExtractor


class TrendFeatures(FeatureExtractor):
    """
    Polynomial trend features based on integer time index.

    Produces features t^1, t^2, ..., t^degree where t is the integer position
    in the series (0-based).  These allow the surrogate model to capture linear
    and polynomial trends directly.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.  ``degree=1`` gives linear trend only.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, y: pd.Series) -> "TrendFeatures":
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        t = np.arange(len(y), dtype=float)
        cols = {f"t{d}": t**d for d in range(1, self.degree + 1)}
        return pd.DataFrame(cols, index=y.index)

    @property
    def feature_names(self) -> list[str]:
        return [f"t{d}" for d in range(1, self.degree + 1)]
