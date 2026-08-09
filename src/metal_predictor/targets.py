from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig

class NextHourTargetBuilder:
    """Creates a target only when the next source bar is exactly one hour later."""
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns
        self._names = (
            "target_log_return_1h",
            "target_close_usd_per_kg",
            "target_timestamp_utc",
        )

    @property
    def target_names(self) -> tuple[str, ...]:
        return self._names

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        ts = pd.to_datetime(out[self._c.timestamp], utc=True)
        close = out[self._c.close].astype(float)
        next_ts = ts.shift(-1)
        next_close = close.shift(-1)
        exact_next_hour = (next_ts - ts).eq(pd.Timedelta(hours=1))
        out["target_timestamp_utc"] = next_ts.where(exact_next_hour)
        out["target_close_usd_per_kg"] = next_close.where(exact_next_hour)
        out["target_log_return_1h"] = np.log(next_close / close).where(exact_next_hour)
        return out
