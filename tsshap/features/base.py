"""Abstract base class for all interpretable feature extractors."""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureExtractor(ABC):
    """
    Base class for interpretable time series feature extractors.

    Each extractor transforms a univariate time series into a DataFrame of
    interpretable features that can be used to train the surrogate model.
    All features at time t must only use values observed up to and including t
    (no look-ahead leakage).
    """

    @abstractmethod
    def fit(self, y: pd.Series) -> "FeatureExtractor":
        """Fit extractor parameters on the training series (e.g. period detection)."""

    @abstractmethod
    def transform(self, y: pd.Series) -> pd.DataFrame:
        """
        Transform the series into a feature DataFrame aligned to the same index.

        Parameters
        ----------
        y : pd.Series
            The full (or extended) time series to extract features from.

        Returns
        -------
        pd.DataFrame
            Feature matrix with the same index as y.  Rows with insufficient
            history are filled with NaN and should be dropped before training.
        """

    def fit_transform(self, y: pd.Series) -> pd.DataFrame:
        return self.fit(y).transform(y)

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of the columns produced by transform()."""
