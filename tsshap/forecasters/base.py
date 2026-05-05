"""Abstract base class for all forecaster wrappers."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Minimal interface required by TsSHAP.

    All forecasters must expose fit() and predict().  The predict() method
    returns a fixed-length forecast horizon starting from the end of the
    series passed to fit().
    """

    @abstractmethod
    def fit(self, y: pd.Series) -> "BaseForecaster":
        """Train the forecaster on the observed series y."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """
        Produce forecasts for the next `horizon` steps beyond the training data.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Point forecasts f_hat(T+1|T), ..., f_hat(T+horizon|T).
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
