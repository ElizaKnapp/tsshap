"""
Synthetic time series generation for controlled evaluation of TsSHAP.

Each generator produces a series from known ground-truth components so
that TsSHAP's recovered feature importances can be validated against the
true signal structure.

Generators
----------
make_trend_seasonal      : Additive trend + seasonality + noise.
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