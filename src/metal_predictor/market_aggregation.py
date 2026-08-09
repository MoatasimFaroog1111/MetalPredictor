from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from metal_predictor.market_source import DownloadWindow, InstrumentSpec
from metal_predictor.price_normalization import HourlyPriceNormalizer, PreciousMetalUsdKgNormalizer


@dataclass(frozen=True)
class H1AggregationReport:
    source_rows: int
    duplicate_minute_rows: int
    duplicate_minute_timestamps: int
    conflicting_duplicate_timestamps: int
    invalid_minute_rows: int
    source_time_reversals: int
    raw_hours_over_60_rows: int
    excluded_suspicious_hours: int
    output_hours: int
    full_60_minute_hours: int
    partial_source_hours: int
    first_timestamp_utc: str
    last_timestamp_utc: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ConservativeH1Aggregator:
    """Quality-focused M1->H1 aggregation; source-unit conversion is injected as a strategy."""

    _PRICE = ("open", "high", "low", "close")

    def __init__(self, normalizer: HourlyPriceNormalizer | None = None) -> None:
        # Backward-compatible metal default; non-metal callers must inject their own strategy.
        self._normalizer = normalizer or PreciousMetalUsdKgNormalizer()

    def aggregate(
        self,
        minutes: pd.DataFrame,
        instrument: InstrumentSpec,
        window: DownloadWindow,
    ) -> tuple[pd.DataFrame, H1AggregationReport]:
        required = {
            "timestamp_utc", "archive_sequence", "source_row_number", "minute_valid_ohlc",
            *self._PRICE,
        }
        missing = required.difference(minutes.columns)
        if missing:
            raise ValueError(f"Minute data missing columns: {sorted(missing)}")
        if minutes.empty:
            raise ValueError("Minute data is empty.")

        raw = minutes.copy(deep=True)
        raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
        raw["hour_utc"] = raw["timestamp_utc"].dt.floor("h")

        duplicate_mask = raw.duplicated("timestamp_utc", keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_timestamps = int(raw.loc[duplicate_mask, "timestamp_utc"].nunique())
        conflict_timestamps = self._conflicting_duplicate_timestamps(raw.loc[duplicate_mask])
        conflict_hours = set(pd.DatetimeIndex(conflict_timestamps).floor("h"))

        invalid_hours = set(raw.loc[~raw["minute_valid_ohlc"].astype(bool), "hour_utc"])
        invalid_rows = int((~raw["minute_valid_ohlc"].astype(bool)).sum())

        raw_hour_counts = raw.groupby("hour_utc", sort=False).size()
        overfull_hours = set(raw_hour_counts.index[raw_hour_counts > 60])

        reversal_hours, reversal_count = self._source_time_reversal_hours(raw)
        suspicious_hours = conflict_hours | invalid_hours | overfull_hours | reversal_hours

        usable = raw.loc[~raw["hour_utc"].isin(suspicious_hours)].copy()
        usable = usable.sort_values(["timestamp_utc", "archive_sequence", "source_row_number"])
        usable = usable.drop_duplicates(subset=["timestamp_utc"], keep="first")

        hourly = usable.groupby("hour_utc", sort=True).agg(
            open_source=("open", "first"),
            high_source=("high", "max"),
            low_source=("low", "min"),
            close_source=("close", "last"),
            minute_count=("timestamp_utc", "size"),
        ).reset_index(names="timestamp_utc")

        start = pd.Timestamp(window.start_utc).tz_convert("UTC").floor("h")
        end = pd.Timestamp(window.end_utc).tz_convert("UTC").floor("h")
        hourly = hourly.loc[hourly["timestamp_utc"].between(start, end, inclusive="both")].copy()
        if hourly.empty:
            raise ValueError("No H1 bars remain inside the requested window.")
        if (hourly["minute_count"] > 60).any():
            raise AssertionError("H1 aggregation produced an impossible >60 unique-minute hour.")

        hourly = self._normalizer.normalize(hourly)
        hourly["asset"] = instrument.asset
        hourly["source_symbol"] = instrument.source_symbol
        hourly["source_provider"] = instrument.provider
        hourly["market_type"] = instrument.market_type
        hourly["currency"] = "USD"
        hourly["price_unit"] = self._normalizer.value_unit
        hourly["quality_flag"] = np.where(
            hourly["minute_count"].eq(60), "OK", "PARTIAL_SOURCE_HOUR"
        )
        hourly = hourly.sort_values("timestamp_utc").reset_index(drop=True)
        self.validate(hourly)

        report = H1AggregationReport(
            source_rows=int(len(raw)),
            duplicate_minute_rows=duplicate_rows,
            duplicate_minute_timestamps=duplicate_timestamps,
            conflicting_duplicate_timestamps=len(conflict_timestamps),
            invalid_minute_rows=invalid_rows,
            source_time_reversals=reversal_count,
            raw_hours_over_60_rows=len(overfull_hours),
            excluded_suspicious_hours=len(suspicious_hours),
            output_hours=int(len(hourly)),
            full_60_minute_hours=int(hourly["minute_count"].eq(60).sum()),
            partial_source_hours=int(hourly["minute_count"].lt(60).sum()),
            first_timestamp_utc=pd.Timestamp(hourly["timestamp_utc"].iloc[0]).isoformat(),
            last_timestamp_utc=pd.Timestamp(hourly["timestamp_utc"].iloc[-1]).isoformat(),
        )
        return hourly, report

    def validate(self, hourly: pd.DataFrame) -> None:
        ts = pd.to_datetime(hourly["timestamp_utc"], utc=True, errors="raise")
        if ts.duplicated().any() or not ts.is_monotonic_increasing:
            raise ValueError("Hourly market data timestamps must be unique and chronological.")
        value_columns = ["open_value", "high_value", "low_value", "close_value"]
        missing = set(value_columns).difference(hourly.columns)
        if missing:
            raise ValueError(f"Normalizer did not produce canonical value columns: {sorted(missing)}")
        values = hourly[value_columns].astype(float)
        if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
            raise ValueError("Hourly market data contains invalid normalized values.")
        invalid = (
            (values["high_value"] < values["low_value"])
            | (values["high_value"] < values[["open_value", "close_value"]].max(axis=1))
            | (values["low_value"] > values[["open_value", "close_value"]].min(axis=1))
        )
        if invalid.any():
            raise ValueError(f"Hourly market data has {int(invalid.sum())} OHLC invariant failures.")
        if not hourly["minute_count"].between(1, 60).all():
            raise ValueError("Hourly minute_count must be between 1 and 60.")

    def _conflicting_duplicate_timestamps(self, duplicates: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
        if duplicates.empty:
            return ()
        conflicts: list[pd.Timestamp] = []
        for timestamp, group in duplicates.groupby("timestamp_utc", sort=False):
            distinct = group.loc[:, self._PRICE].drop_duplicates()
            if len(distinct) > 1:
                conflicts.append(pd.Timestamp(timestamp))
        return tuple(conflicts)

    @staticmethod
    def _source_time_reversal_hours(raw: pd.DataFrame) -> tuple[set[pd.Timestamp], int]:
        bad_hours: set[pd.Timestamp] = set()
        count = 0
        ordered = raw.sort_values(["archive_sequence", "source_row_number"])
        for _, group in ordered.groupby("archive_sequence", sort=True):
            ts = pd.to_datetime(group["timestamp_utc"], utc=True)
            delta = ts.diff()
            positions = np.flatnonzero(delta.lt(pd.Timedelta(0)).to_numpy())
            count += len(positions)
            for pos in positions:
                if pos > 0:
                    bad_hours.add(pd.Timestamp(ts.iloc[pos - 1]).floor("h"))
                bad_hours.add(pd.Timestamp(ts.iloc[pos]).floor("h"))
        return bad_hours, count
