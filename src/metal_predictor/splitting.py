from __future__ import annotations
import pandas as pd
from metal_predictor.core import ColumnConfig, SplitConfig

class ChronologicalPurgedSplitter:
    """Chronological split with time-based purge at future-label boundaries."""
    def __init__(self, columns: ColumnConfig, config: SplitConfig) -> None:
        self._c, self._cfg = columns, config

    def split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        ordered = frame.sort_values(self._c.timestamp).reset_index(drop=True)
        n = len(ordered)
        if n < 100:
            raise ValueError("Too few usable rows for a reliable chronological split.")
        train_end = int(n * self._cfg.train_ratio)
        val_end = train_end + int(n * self._cfg.validation_ratio)
        train_boundary = pd.Timestamp(ordered.iloc[train_end][self._c.timestamp])
        val_boundary = pd.Timestamp(ordered.iloc[val_end][self._c.timestamp])
        purge = pd.Timedelta(hours=self._cfg.purge_hours)
        ts = pd.to_datetime(ordered[self._c.timestamp], utc=True)
        target_ts = pd.to_datetime(ordered["target_timestamp_utc"], utc=True)
        train_mask = (ts < train_boundary - purge) & (target_ts < train_boundary)
        val_mask = (
            (ts >= train_boundary)
            & (ts < val_boundary - purge)
            & (target_ts < val_boundary)
        )
        test_mask = ts >= val_boundary
        splits = {
            "train": ordered.loc[train_mask].copy().reset_index(drop=True),
            "validation": ordered.loc[val_mask].copy().reset_index(drop=True),
            "test": ordered.loc[test_mask].copy().reset_index(drop=True),
        }
        if min(map(len, splits.values())) == 0:
            raise ValueError("At least one split is empty after purging.")
        return splits
