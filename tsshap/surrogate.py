"""
Surrogate model for TsSHAP.

The surrogate is a tree-ensemble regressor (XGBoost by default) that learns a
mapping from the interpretable feature space to the backtested forecast values.
After fitting, TreeSHAP is applied via the `shap` library to obtain SHAP values
for each feature at every time step.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import shap


SurrogateBackend = Literal["xgboost", "lightgbm", "catboost", "sklearn"]


class SurrogateModel:
    """
    Tree-ensemble surrogate for the black-box forecaster.

    Parameters
    ----------
    backend : {"xgboost", "lightgbm", "catboost", "sklearn"}
        Which tree library to use as the underlying regressor.
        ``"sklearn"`` uses ``GradientBoostingRegressor`` and requires no
        additional system dependencies (recommended if XGBoost/LightGBM are
        unavailable).
    model_params : dict or None
        Keyword arguments forwarded to the underlying model constructor.
    """

    def __init__(
        self,
        backend: SurrogateBackend | None = None,
        model_params: dict | None = None,
    ):
        self.backend = backend or self._default_backend()
        self.model_params = model_params or {}
        self._model = None
        self._explainer: shap.TreeExplainer | None = None
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SurrogateModel":
        """
        Train the surrogate on feature matrix X and backtested target y.

        Rows with NaN in either X or y are dropped before training.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples × n_features).
        y : pd.Series
            Backtested forecast targets aligned to X.
        """
        valid = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[valid]
        y_clean = y.loc[valid]

        if len(X_clean) < 5:
            raise ValueError(
                f"Only {len(X_clean)} valid training rows after dropping NaNs. "
                "Increase series length or reduce feature lag requirements."
            )

        self.feature_names_ = list(X_clean.columns)
        self._model = self._build_model()
        self._model.fit(X_clean.values, y_clean.values)
        self._explainer = shap.TreeExplainer(self._model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the surrogate model."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(X.values)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute TreeSHAP values for each row in X.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() first.")
        return self._explainer.shap_values(X.values)

    def expected_value(self) -> float:
        """Base value (mean training prediction) for SHAP waterfall plots."""
        if self._explainer is None:
            raise RuntimeError("Call fit() first.")
        ev = self._explainer.expected_value
        # TreeExplainer may return a scalar, 0-d array, or 1-element array
        import numpy as np
        return float(np.atleast_1d(ev).ravel()[0])

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """R² of surrogate predictions vs. backtested forecasts."""
        from sklearn.metrics import r2_score

        valid = ~(X.isna().any(axis=1) | y.isna())
        preds = self.predict(X.loc[valid])
        return float(r2_score(y.loc[valid].values, preds))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_backend() -> str:
        """Try XGBoost first, fall back to sklearn GradientBoosting."""
        try:
            from xgboost import XGBRegressor  # noqa: F401
            return "xgboost"
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor  # noqa: F401
            return "lightgbm"
        except Exception:
            pass
        return "sklearn"

    def _build_model(self):
        defaults: dict[str, dict] = {
            "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
                        "subsample": 0.8, "random_state": 42},
            "lightgbm": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
                         "subsample": 0.8, "random_state": 42, "verbose": -1},
            "catboost": {"iterations": 200, "depth": 4, "learning_rate": 0.1,
                         "random_seed": 42, "verbose": 0},
            "sklearn": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
                        "subsample": 0.8, "random_state": 42},
        }
        params = {**defaults.get(self.backend, {}), **self.model_params}

        if self.backend == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        elif self.backend == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**params)
        elif self.backend == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**params)
        elif self.backend == "sklearn":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
