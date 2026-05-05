"""
TsSHAP: Feature-based explainability for univariate time series forecasting.

Based on "TsSHAP: Robust model agnostic feature-based explainability for
univariate time series forecasting" (Raykar et al., IBM Research, 2023).
"""

from tsshap.explainer import TsSHAPExplainer

__all__ = ["TsSHAPExplainer"]
