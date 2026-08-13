from __future__ import annotations

import hashlib
import json
from typing import Final

import numpy as np
import pandas as pd


FEATURE_SET_VERSION: Final = "bullionvault-hlc-causal-features-v1"
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "log_return_1",
    "log_return_2",
    "log_return_3",
    "log_return_6",
    "range_pct",
    "close_location",
    "rolling_return_mean_3",
    "rolling_return_std_3",
    "rolling_return_mean_6",
    "rolling_return_std_6",
    "rolling_range_mean_3",
    "rolling_range_mean_6",
)

_FEATURE_FORMULAS: Final[dict[str, str]] = {
    "log_return_1": "log(close_t / close_t-1)",
    "log_return_2": "log(close_t / close_t-2)",
    "log_return_3": "log(close_t / close_t-3)",
    "log_return_6": "log(close_t / close_t-6)",
    "range_pct": "(high_t - low_t) / close_t",
    "close_location": "(close_t - low_t) / (high_t - low_t); 0.5 when flat",
    "rolling_return_mean_3": "mean(log_return_1[t-2:t])",
    "rolling_return_std_3": "sample_std(log_return_1[t-2:t])",
    "rolling_return_mean_6": "mean(log_return_1[t-5:t])",
    "rolling_return_std_6": "sample_std(log_return_1[t-5:t])",
    "rolling_range_mean_3": "mean(range_pct[t-2:t])",
    "rolling_range_mean_6": "mean(range_pct[t-5:t])",
}


def feature_fingerprint_sha256() -> str:
    payload = {
        "version": FEATURE_SET_VERSION,
        "columns": list(FEATURE_COLUMNS),
        "formulas": _FEATURE_FORMULAS,
        "timezone_features": False,
        "future_data_used": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class CausalHlcFeatureBuilder:
    """Pure, horizon-agnostic HLC feature transform.

    Every feature for row t is a function only of HLC observations at timestamps
    <= t. Source timestamps remain timezone-naive and are never converted or used
    to create calendar features.
    """

    max_lookback_bars: Final = 6

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = ("timestamp_source", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg")
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"Feature source missing columns: {missing}.")
        if frame.empty:
            raise ValueError("Feature source must not be empty.")

        source = frame.loc[:, list(required)].copy()
        timestamps = pd.to_datetime(source["timestamp_source"], errors="raise")
        if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("Feature source timestamps must be unique and increasing.")

        high = pd.to_numeric(source["high_usd_per_kg"], errors="raise").astype(float)
        low = pd.to_numeric(source["low_usd_per_kg"], errors="raise").astype(float)
        close = pd.to_numeric(source["close_usd_per_kg"], errors="raise").astype(float)
        values = np.column_stack((high.to_numpy(), low.to_numpy(), close.to_numpy()))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Feature source HLC values must be finite and positive.")
        if (high < low).any() or (close > high).any() or (close < low).any():
            raise ValueError("Feature source contains invalid HLC relationships.")

        result = pd.DataFrame({"timestamp_source": timestamps})
        log_close = np.log(close)
        r1 = log_close - log_close.shift(1)
        range_pct = (high - low) / close
        width = high - low
        close_location = pd.Series(0.5, index=source.index, dtype=float)
        nonflat = width > 0.0
        close_location.loc[nonflat] = (
            (close.loc[nonflat] - low.loc[nonflat]) / width.loc[nonflat]
        )

        result["log_return_1"] = r1
        result["log_return_2"] = log_close - log_close.shift(2)
        result["log_return_3"] = log_close - log_close.shift(3)
        result["log_return_6"] = log_close - log_close.shift(6)
        result["range_pct"] = range_pct
        result["close_location"] = close_location
        result["rolling_return_mean_3"] = r1.rolling(3, min_periods=3).mean()
        result["rolling_return_std_3"] = r1.rolling(3, min_periods=3).std(ddof=1)
        result["rolling_return_mean_6"] = r1.rolling(6, min_periods=6).mean()
        result["rolling_return_std_6"] = r1.rolling(6, min_periods=6).std(ddof=1)
        result["rolling_range_mean_3"] = range_pct.rolling(3, min_periods=3).mean()
        result["rolling_range_mean_6"] = range_pct.rolling(6, min_periods=6).mean()

        finite_rows = result.loc[:, list(FEATURE_COLUMNS)].dropna()
        if not finite_rows.empty and not np.isfinite(finite_rows.to_numpy(dtype=float)).all():
            raise ValueError("Feature transform produced non-finite values.")
        return result
