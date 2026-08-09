from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig

class ParquetDataLoader:
    """Loads the canonical silver dataset without mutating it."""
    def load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")
        return pd.read_parquet(path).copy(deep=True)

class SilverDatasetValidator:
    """Validates raw XAG H1 data invariants before feature generation."""
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self._c.required if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if frame.empty:
            raise ValueError("Input dataset is empty.")
        ts = pd.to_datetime(frame[self._c.timestamp], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError("timestamp_utc contains invalid or missing timestamps.")
        if ts.duplicated().any():
            raise ValueError("Duplicate timestamps detected.")
        if not ts.is_monotonic_increasing:
            raise ValueError("Timestamps must be strictly chronological.")
        price_cols = [self._c.open, self._c.high, self._c.low, self._c.close]
        prices = frame[price_cols].apply(pd.to_numeric, errors="coerce")
        values = prices.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("OHLC contains NaN or infinite values.")
        if (values <= 0).any():
            raise ValueError("OHLC must be strictly positive.")
        invalid = (
            (prices[self._c.high] < prices[self._c.low])
            | (prices[self._c.high] < prices[[self._c.open, self._c.close]].max(axis=1))
            | (prices[self._c.low] > prices[[self._c.open, self._c.close]].min(axis=1))
        )
        if invalid.any():
            raise ValueError(f"Invalid OHLC invariants in {int(invalid.sum())} rows.")
