from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


TARGET_VERSION: Final = "next-source-bar-log-return-v1"
TARGET_COLUMN: Final = "target_log_return"
TARGET_CLOSE_COLUMN: Final = "target_close_usd_per_kg"
TARGET_TIMESTAMP_COLUMN: Final = "target_timestamp_source"


class NextBarTargetBuilder:
    """Build an exact next-source-bar target without calendar-time inference."""

    def build(self, frame: pd.DataFrame, *, interval_seconds: int) -> pd.DataFrame:
        required = ("timestamp_source", "close_usd_per_kg")
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"Target source missing columns: {missing}.")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if frame.empty:
            raise ValueError("Target source must not be empty.")

        timestamps = pd.to_datetime(frame["timestamp_source"], errors="raise")
        if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("Target source timestamps must be unique and increasing.")
        deltas = timestamps.diff().dropna().dt.total_seconds().astype(int)
        if not deltas.empty and tuple(sorted(set(deltas.tolist()))) != (int(interval_seconds),):
            raise ValueError("Target source must preserve the exact registered source interval.")

        close = pd.to_numeric(frame["close_usd_per_kg"], errors="raise").astype(float)
        if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
            raise ValueError("Target close values must be finite and positive.")

        target_close = close.shift(-1)
        target_timestamp = timestamps.shift(-1)
        target_return = np.log(target_close / close)

        return pd.DataFrame(
            {
                "timestamp_source": timestamps,
                "current_close_usd_per_kg": close,
                TARGET_TIMESTAMP_COLUMN: target_timestamp,
                TARGET_CLOSE_COLUMN: target_close,
                TARGET_COLUMN: target_return,
            }
        )
