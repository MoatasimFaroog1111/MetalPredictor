from __future__ import annotations

import pandas as pd


class ExactTimestampAligner:
    """Left-aligns auxiliary data by identical UTC timestamps only. No as-of, tolerance or fill is allowed."""

    def align(
        self,
        base_timestamps: pd.Series,
        auxiliary: pd.DataFrame,
        columns: tuple[str, ...],
        prefix: str = "",
        timestamp_name: str = "timestamp_utc",
    ) -> pd.DataFrame:
        missing = set((timestamp_name, *columns)).difference(auxiliary.columns)
        if missing:
            raise ValueError(f"Auxiliary data missing alignment columns: {sorted(missing)}")

        base_ts = pd.to_datetime(base_timestamps, utc=True, errors="raise")
        aux = auxiliary.loc[:, [timestamp_name, *columns]].copy()
        aux[timestamp_name] = pd.to_datetime(aux[timestamp_name], utc=True, errors="raise")
        if aux[timestamp_name].duplicated().any():
            raise ValueError("Auxiliary timestamps must be unique before exact alignment.")
        aux = aux.set_index(timestamp_name).sort_index()

        requested = pd.DatetimeIndex(base_ts)
        aligned = aux.reindex(requested)
        aligned.index = base_timestamps.index
        aligned.columns = [f"{prefix}{name}" for name in columns]
        return aligned
