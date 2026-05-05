"""
TsSHAPExplainer — main entry point for the TsSHAP algorithm.

Usage example
-------------
>>> from tsshap import TsSHAPExplainer
>>> from tsshap.features import STLFeatures, LagFeatures
>>> from tsshap.forecasters import MovingAverageForecaster
>>>
>>> forecaster = MovingAverageForecaster(k=6)
>>> explainer = TsSHAPExplainer(
...     forecaster=forecaster,
...     feature_extractors=[STLFeatures(period=12), LagFeatures(lags=3)],
... )
>>> explainer.fit(y_train)
>>> result = explainer.explain(scope="global")
>>> result.plot_importance()
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tsshap.backtesting import expanding_window_backtest, BacktestResult
from tsshap.features.base import FeatureExtractor
from tsshap.forecasters.base import BaseForecaster
from tsshap.surrogate import SurrogateModel, SurrogateBackend


Scope = Literal["local", "semi-local", "global"]


class TsSHAPExplainer:
    """
    Model-agnostic feature-based explainability for univariate time series.

    Implements the three-stage pipeline from the TsSHAP paper:
        1. Generate backtested forecasts from the black-box forecaster.
        2. Extract interpretable features from the original series.
        3. Train an XGBoost surrogate and compute TreeSHAP values.

    Parameters
    ----------
    forecaster : BaseForecaster
        The black-box forecaster to be explained.
    feature_extractors : list[FeatureExtractor]
        One or more feature extractor instances.  Their outputs are
        concatenated column-wise to form the surrogate feature matrix.
    horizon : int
        Forecast horizon H.  Defaults to 10% of the training series length.
    surrogate_backend : SurrogateBackend
        Tree library used for the surrogate ("xgboost", "lightgbm", "catboost").
    surrogate_params : dict or None
        Additional parameters forwarded to the surrogate model constructor.
    min_train_size : int or None
        Minimum number of training steps for the first backtesting split.
    backtest_step : int
        Step size between consecutive backtesting windows.
    refit_forecaster : bool
        Whether to refit the forecaster at each backtesting window.
        Set to False for ML/DL forecasters trained once on the full series.
    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        feature_extractors: list[FeatureExtractor] | None = None,
        horizon: int | None = None,
        surrogate_backend: SurrogateBackend | None = None,
        surrogate_params: dict | None = None,
        min_train_size: int | None = None,
        backtest_step: int = 1,
        refit_forecaster: bool = True,
    ):
        self.forecaster = forecaster
        self.feature_extractors = feature_extractors or []
        self._horizon = horizon
        self.surrogate_backend = surrogate_backend
        self.surrogate_params = surrogate_params
        self.min_train_size = min_train_size
        self.backtest_step = backtest_step
        self.refit_forecaster = refit_forecaster

        # Set after fit()
        self._surrogate: SurrogateModel | None = None
        self._feature_matrix: pd.DataFrame | None = None
        self._backtest_result: BacktestResult | None = None
        self._y_train: pd.Series | None = None
        self.horizon_: int | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series) -> "TsSHAPExplainer":
        """
        Fit the TsSHAP explainer on the observed time series y.

        Steps:
            1. Run expanding-window backtesting to generate surrogate targets.
            2. Extract interpretable features from y.
            3. Train the surrogate regressor.

        Parameters
        ----------
        y : pd.Series
            Training time series.
        """
        self._y_train = y.copy()
        self.horizon_ = self._horizon or max(1, int(0.1 * len(y)))

        # Stage 1: generate backtested forecasts
        self._backtest_result = expanding_window_backtest(
            y=y,
            forecaster=self.forecaster,
            horizon=self.horizon_,
            min_train_size=self.min_train_size,
            step=self.backtest_step,
            refit=self.refit_forecaster,
        )

        # Stage 2: extract features
        self._feature_matrix = self._extract_features(y)

        # Stage 3: train surrogate
        self._surrogate = SurrogateModel(
            backend=self.surrogate_backend,
            model_params=self.surrogate_params,
        )
        self._surrogate.fit(
            X=self._feature_matrix,
            y=self._backtest_result.forecasts,
        )
        return self

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def explain(
        self,
        scope: Scope = "global",
        t: int | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> "ExplanationResult":
        """
        Compute SHAP-based explanations at the requested scope.

        Parameters
        ----------
        scope : {"global", "semi-local", "local"}
            Explanation scope.
        t : int or None
            For ``scope="local"``: integer index (position in the series)
            of the time step to explain.  Negative indexing is supported.
        t_start, t_end : int or None
            For ``scope="semi-local"``: start and end integer indices of
            the time window to explain (inclusive, exclusive).

        Returns
        -------
        ExplanationResult
        """
        self._check_fitted()

        X = self._feature_matrix
        shap_vals = self._surrogate.shap_values(X.fillna(0))
        base_value = self._surrogate.expected_value()
        feature_names = list(X.columns)
        surrogate_preds = self._surrogate.predict(X.fillna(0))

        valid_mask = ~(X.isna().any(axis=1) | self._backtest_result.forecasts.isna())

        if scope == "global":
            # Mean absolute SHAP over all valid time steps
            shap_valid = shap_vals[valid_mask]
            mean_abs_shap = np.abs(shap_valid).mean(axis=0)
            shap_matrix = pd.DataFrame(shap_vals, index=X.index, columns=feature_names)
            return ExplanationResult(
                scope=scope,
                shap_matrix=shap_matrix,
                feature_importance=pd.Series(mean_abs_shap, index=feature_names),
                base_value=base_value,
                feature_names=feature_names,
                index=X.index,
                surrogate_preds=pd.Series(surrogate_preds, index=X.index),
                actuals=self._backtest_result.actuals,
                backtested_forecasts=self._backtest_result.forecasts,
            )

        elif scope == "local":
            if t is None:
                raise ValueError("Provide `t` for local explanation.")
            idx = t if t >= 0 else len(X) + t
            row = X.iloc[[idx]].fillna(0)
            shap_local = self._surrogate.shap_values(row)[0]
            shap_matrix = pd.DataFrame(
                shap_vals[[idx]], index=[X.index[idx]], columns=feature_names
            )
            return ExplanationResult(
                scope=scope,
                shap_matrix=shap_matrix,
                feature_importance=pd.Series(shap_local, index=feature_names),
                base_value=base_value,
                feature_names=feature_names,
                index=X.index[[idx]],
                surrogate_preds=pd.Series(surrogate_preds[[idx]], index=[X.index[idx]]),
                actuals=self._backtest_result.actuals,
                backtested_forecasts=self._backtest_result.forecasts,
            )

        elif scope == "semi-local":
            if t_start is None or t_end is None:
                raise ValueError("Provide `t_start` and `t_end` for semi-local explanation.")
            sub_shap = shap_vals[t_start:t_end]
            sub_index = X.index[t_start:t_end]
            mean_abs_shap = np.abs(sub_shap).mean(axis=0)
            shap_matrix = pd.DataFrame(sub_shap, index=sub_index, columns=feature_names)
            return ExplanationResult(
                scope=scope,
                shap_matrix=shap_matrix,
                feature_importance=pd.Series(mean_abs_shap, index=feature_names),
                base_value=base_value,
                feature_names=feature_names,
                index=sub_index,
                surrogate_preds=pd.Series(
                    surrogate_preds[t_start:t_end], index=sub_index
                ),
                actuals=self._backtest_result.actuals,
                backtested_forecasts=self._backtest_result.forecasts,
            )

        else:
            raise ValueError(f"Unknown scope: {scope!r}. Choose 'global', 'local', or 'semi-local'.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def surrogate(self) -> SurrogateModel:
        self._check_fitted()
        return self._surrogate

    @property
    def feature_matrix(self) -> pd.DataFrame:
        self._check_fitted()
        return self._feature_matrix

    @property
    def backtest_result(self) -> BacktestResult:
        self._check_fitted()
        return self._backtest_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_features(self, y: pd.Series) -> pd.DataFrame:
        frames = []
        for extractor in self.feature_extractors:
            extractor.fit(y)
            frames.append(extractor.transform(y))
        if not frames:
            raise ValueError("No feature extractors provided.")
        return pd.concat(frames, axis=1)

    def _check_fitted(self):
        if self._surrogate is None:
            raise RuntimeError("Call fit() before accessing explainer results.")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExplanationResult:
    """
    Container for TsSHAP explanation outputs.

    Attributes
    ----------
    scope : str
        Explanation scope used ("global", "local", "semi-local").
    shap_matrix : pd.DataFrame
        Full SHAP value matrix (n_time_steps × n_features).
    feature_importance : pd.Series
        Aggregated importance per feature (mean |SHAP| for global/semi-local,
        raw SHAP values for local).
    base_value : float
        TreeSHAP base value (mean surrogate prediction).
    feature_names : list[str]
    index : pd.Index
        Time index covered by this explanation.
    surrogate_preds : pd.Series
        Surrogate model predictions over the covered time range.
    actuals : pd.Series
        Original observed series values.
    backtested_forecasts : pd.Series
        Black-box backtested forecasts (surrogate training targets).
    """

    def __init__(
        self,
        scope: str,
        shap_matrix: pd.DataFrame,
        feature_importance: pd.Series,
        base_value: float,
        feature_names: list[str],
        index: pd.Index,
        surrogate_preds: pd.Series,
        actuals: pd.Series,
        backtested_forecasts: pd.Series,
    ):
        self.scope = scope
        self.shap_matrix = shap_matrix
        self.feature_importance = feature_importance
        self.base_value = base_value
        self.feature_names = feature_names
        self.index = index
        self.surrogate_preds = surrogate_preds
        self.actuals = actuals
        self.backtested_forecasts = backtested_forecasts

    def top_features(self, n: int = 10) -> pd.Series:
        """Return the top-n most important features sorted by importance."""
        return self.feature_importance.abs().sort_values(ascending=False).head(n)

    def __repr__(self) -> str:
        return (
            f"ExplanationResult(scope={self.scope!r}, "
            f"n_features={len(self.feature_names)}, "
            f"n_timesteps={len(self.index)})"
        )
