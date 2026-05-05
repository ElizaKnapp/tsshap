"""Date and time encoding feature extractor."""

from __future__ import annotations

import pandas as pd

from tsshap.features.base import FeatureExtractor

_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


class DateFeatures(FeatureExtractor):
    """
    Calendar-based features extracted from a DatetimeIndex.

    Produces numeric and boolean features encoding temporal position:
    month, day-of-year, day-of-month, week-of-year, day-of-week, quarter,
    season (as integer 1-4), and several start/end-of-period indicators.

    The series index must be a pd.DatetimeIndex.

    Parameters
    ----------
    features : list[str] or None
        Subset of feature names to include.  If None, all features are included.
        Available names: "month", "day_of_year", "day_of_month", "week_of_year",
        "day_of_week", "is_weekend", "quarter", "season",
        "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
        "is_year_start", "is_year_end", "is_leap_year".
    """

    _ALL = [
        "month", "day_of_year", "day_of_month", "week_of_year",
        "day_of_week", "is_weekend", "quarter", "season",
        "is_month_start", "is_month_end", "is_quarter_start",
        "is_quarter_end", "is_year_start", "is_year_end", "is_leap_year",
    ]

    def __init__(self, features: list[str] | None = None):
        self.features = features if features is not None else self._ALL

    def fit(self, y: pd.Series) -> "DateFeatures":
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("DateFeatures requires a pd.DatetimeIndex.")
        return self

    def transform(self, y: pd.Series) -> pd.DataFrame:
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("DateFeatures requires a pd.DatetimeIndex.")

        idx = y.index
        all_cols: dict[str, pd.Series] = {
            "month":           pd.Series(idx.month, index=idx, dtype=float),
            "day_of_year":     pd.Series(idx.day_of_year, index=idx, dtype=float),
            "day_of_month":    pd.Series(idx.day, index=idx, dtype=float),
            "week_of_year":    pd.Series(idx.isocalendar().week.values, index=idx, dtype=float),
            "day_of_week":     pd.Series(idx.day_of_week, index=idx, dtype=float),
            "is_weekend":      pd.Series(idx.day_of_week >= 5, index=idx, dtype=float),
            "quarter":         pd.Series(idx.quarter, index=idx, dtype=float),
            "season":          pd.Series(
                [{"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}[_SEASON_MAP[m]]
                 for m in idx.month], index=idx, dtype=float,
            ),
            "is_month_start":  pd.Series(idx.is_month_start, index=idx, dtype=float),
            "is_month_end":    pd.Series(idx.is_month_end, index=idx, dtype=float),
            "is_quarter_start": pd.Series(idx.is_quarter_start, index=idx, dtype=float),
            "is_quarter_end":  pd.Series(idx.is_quarter_end, index=idx, dtype=float),
            "is_year_start":   pd.Series(idx.is_year_start, index=idx, dtype=float),
            "is_year_end":     pd.Series(idx.is_year_end, index=idx, dtype=float),
            "is_leap_year":    pd.Series(idx.is_leap_year, index=idx, dtype=float),
        }
        return pd.DataFrame(
            {k: v for k, v in all_cols.items() if k in self.features},
            index=idx,
        )

    @property
    def feature_names(self) -> list[str]:
        return [f for f in self._ALL if f in self.features]
