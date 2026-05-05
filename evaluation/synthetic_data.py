"""
Synthetic time series generation for controlled evaluation of TsSHAP.

Each generator produces a series from known ground-truth components so
that TsSHAP's recovered feature importances can be validated against the
true signal structure.

Generators
----------
make_trend_series        : Linear or polynomial trend + Gaussian noise.
make_seasonal_series     : Sine-wave seasonality + Gaussian noise.
make_trend_seasonal      : Additive trend + seasonality + noise.
make_component_series    : Full control — specify weights for trend,
                           seasonality, and an optional external regressor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticSeries:
    """Container for a synthetic time series with labelled components."""

    series: pd.Series
    """Full observed time series y(t) = trend + seasonal + noise + (regressor)."""

    trend: pd.Series
    """Pure trend component."""

    seasonal: pd.Series
    """Pure seasonal component."""

    noise: pd.Series
    """Gaussian noise component."""

    regressor: pd.Series | None
    """Optional external regressor series (None if not used)."""

    component_weights: dict[str, float]
    """True signal-to-noise weights for each component (used as ground truth)."""

    def component_fractions(self) -> dict[str, float]:
        """
        Fraction of total variance explained by each component.
        Useful as a ground-truth baseline for comparing with TsSHAP importances.
        """
        total = self.series.var()
        fracs: dict[str, float] = {}
        for name, comp in [
            ("trend", self.trend),
            ("seasonal", self.seasonal),
            ("noise", self.noise),
        ]:
            fracs[name] = float(comp.var() / total)
        if self.regressor is not None:
            fracs["regressor"] = float(
                (self.component_weights.get("regressor", 0.0) * self.regressor).var()
                / total
            )
        return fracs


def make_trend_series(
    n: int = 200,
    slope: float = 1.0,
    degree: int = 1,
    noise_std: float = 0.5,
    start: str = "2018-01-01",
    freq: str = "ME",
    seed: int = 0,
) -> SyntheticSeries:
    """
    Polynomial trend + Gaussian noise.

    y(t) = slope * t^degree + epsilon(t)

    Parameters
    ----------
    n : int
        Number of time steps.
    slope : float
        Coefficient of the trend polynomial.
    degree : int
        Polynomial degree (1 = linear).
    noise_std : float
        Standard deviation of Gaussian noise.
    start : str
        Start date for the DatetimeIndex.
    freq : str
        Pandas frequency string (e.g. "ME", "W", "D").
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    trend_vals = slope * (t / n) ** degree
    noise_vals = rng.normal(0, noise_std, size=n)
    y = trend_vals + noise_vals
    return SyntheticSeries(
        series=pd.Series(y, index=idx, name="y"),
        trend=pd.Series(trend_vals, index=idx, name="trend"),
        seasonal=pd.Series(np.zeros(n), index=idx, name="seasonal"),
        noise=pd.Series(noise_vals, index=idx, name="noise"),
        regressor=None,
        component_weights={"trend": slope, "noise": noise_std},
    )


def make_seasonal_series(
    n: int = 240,
    period: int = 12,
    amplitude: float = 2.0,
    noise_std: float = 0.5,
    start: str = "2003-01-01",
    freq: str = "ME",
    seed: int = 0,
) -> SyntheticSeries:
    """
    Sine-wave seasonality + Gaussian noise (no trend).

    y(t) = amplitude * sin(2π * t / period) + epsilon(t)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    seasonal_vals = amplitude * np.sin(2 * np.pi * t / period)
    noise_vals = rng.normal(0, noise_std, size=n)
    y = seasonal_vals + noise_vals
    return SyntheticSeries(
        series=pd.Series(y, index=idx, name="y"),
        trend=pd.Series(np.zeros(n), index=idx, name="trend"),
        seasonal=pd.Series(seasonal_vals, index=idx, name="seasonal"),
        noise=pd.Series(noise_vals, index=idx, name="noise"),
        regressor=None,
        component_weights={"seasonal": amplitude, "noise": noise_std},
    )


def make_trend_seasonal(
    n: int = 240,
    period: int = 12,
    trend_slope: float = 0.5,
    seasonal_amplitude: float = 2.0,
    noise_std: float = 0.3,
    start: str = "2003-01-01",
    freq: str = "ME",
    seed: int = 0,
) -> SyntheticSeries:
    """
    Additive trend + seasonality + noise.

    y(t) = trend_slope * t/n + seasonal_amplitude * sin(2π * t / period) + epsilon(t)

    This is the primary test case for TsSHAP because we know exactly which
    components should dominate.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    trend_vals = trend_slope * t / n
    seasonal_vals = seasonal_amplitude * np.sin(2 * np.pi * t / period)
    noise_vals = rng.normal(0, noise_std, size=n)
    y = trend_vals + seasonal_vals + noise_vals
    return SyntheticSeries(
        series=pd.Series(y, index=idx, name="y"),
        trend=pd.Series(trend_vals, index=idx, name="trend"),
        seasonal=pd.Series(seasonal_vals, index=idx, name="seasonal"),
        noise=pd.Series(noise_vals, index=idx, name="noise"),
        regressor=None,
        component_weights={
            "trend": trend_slope,
            "seasonal": seasonal_amplitude,
            "noise": noise_std,
        },
    )


def make_component_series(
    n: int = 240,
    period: int = 12,
    trend_weight: float = 1.0,
    seasonal_weight: float = 2.0,
    regressor_weight: float = 0.0,
    noise_std: float = 0.3,
    start: str = "2003-01-01",
    freq: str = "ME",
    seed: int = 0,
) -> SyntheticSeries:
    """
    Fully configurable additive decomposition with optional external regressor.

    y(t) = trend_weight * trend(t)
           + seasonal_weight * seasonal(t)
           + regressor_weight * z(t)
           + epsilon(t)

    The external regressor z(t) is a random walk clamped to [−1, 1],
    mimicking a promotion or discount signal.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)

    trend_vals = t / n
    seasonal_vals = np.sin(2 * np.pi * t / period)
    noise_vals = rng.normal(0, noise_std, size=n)

    # Simple random walk regressor
    z = np.clip(np.cumsum(rng.normal(0, 0.1, size=n)), -1, 1)

    y = (
        trend_weight * trend_vals
        + seasonal_weight * seasonal_vals
        + regressor_weight * z
        + noise_vals
    )

    return SyntheticSeries(
        series=pd.Series(y, index=idx, name="y"),
        trend=pd.Series(trend_weight * trend_vals, index=idx, name="trend"),
        seasonal=pd.Series(seasonal_weight * seasonal_vals, index=idx, name="seasonal"),
        noise=pd.Series(noise_vals, index=idx, name="noise"),
        regressor=pd.Series(z, index=idx, name="regressor"),
        component_weights={
            "trend": trend_weight,
            "seasonal": seasonal_weight,
            "regressor": regressor_weight,
            "noise": noise_std,
        },
    )


def block_bootstrap(
    y: pd.Series,
    block_length: int,
    seed: int | None = None,
) -> pd.Series:
    """
    Block bootstrap perturbation (§6.3 of TsSHAP paper).

    Decomposes y into trend-cycle (moving average) and residual, then
    block-bootstraps the residual and adds it back to the trend-cycle.

    Parameters
    ----------
    y : pd.Series
        Original time series.
    block_length : int
        Length of contiguous blocks used in the bootstrap.
    seed : int or None

    Returns
    -------
    pd.Series
        Perturbed time series with the same index and length as y.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    k = block_length // 2
    m = 2 * k + 1

    values = y.values.astype(float)
    # Trend-cycle via centred moving average
    trend_cycle = np.convolve(values, np.ones(m) / m, mode="same")
    # Handle edge effects with edge padding
    for i in range(k):
        trend_cycle[i] = values[:i + k + 1].mean()
        trend_cycle[n - 1 - i] = values[n - i - k - 1:].mean()

    residual = values - trend_cycle

    # Block bootstrap the residual
    blocks = [residual[i: i + block_length] for i in range(n - block_length + 1)]
    n_blocks_needed = int(np.ceil(n / block_length))
    chosen = [blocks[rng.integers(0, len(blocks))] for _ in range(n_blocks_needed)]
    bootstrapped_residual = np.concatenate(chosen)[:n]

    perturbed = trend_cycle + bootstrapped_residual
    return pd.Series(perturbed, index=y.index, name=y.name)
