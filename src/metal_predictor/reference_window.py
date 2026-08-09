from __future__ import annotations

from pathlib import Path

import pandas as pd

from metal_predictor.market_source import DownloadWindow


class ParquetTimestampWindowProvider:
    """Derives an auxiliary-market download window from a canonical reference dataset."""

    def __init__(self, timestamp_name: str = "timestamp_utc") -> None:
        self._timestamp = timestamp_name

    def get(self, path: Path) -> DownloadWindow:
        timestamps = pd.read_parquet(path, columns=[self._timestamp])[self._timestamp]
        timestamps = pd.to_datetime(timestamps, utc=True, errors="raise")
        if timestamps.empty:
            raise ValueError(f"Reference dataset is empty: {path}")
        return DownloadWindow(start_utc=timestamps.min(), end_utc=timestamps.max())
