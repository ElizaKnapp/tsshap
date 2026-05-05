"""
Pluggable black-box forecasters for TsSHAP (naive, smoothed, exponential smoothing,
lag-based XGBoost, and Prophet).  Notebooks ``02_synthetic_evaluation`` and
``03_tsice_comparison`` loop over this public set.
"""

from tsshap.forecasters.naive import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from tsshap.forecasters.statistical import ExponentialSmoothingForecaster
from tsshap.forecasters.ml import XGBoostForecaster, ProphetForecaster

__all__ = [
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "XGBoostForecaster",
    "ProphetForecaster",
]
