from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from metal_predictor.market_aggregation import ConservativeH1Aggregator, H1AggregationReport
from metal_predictor.market_source import DownloadWindow, InstrumentSpec, MinuteArchiveParser


class ArchiveSetProvider(Protocol):
    def archives(self) -> tuple[Path, ...]: ...


@dataclass(frozen=True)
class LongHistoryBuildReport:
    input_archives: int
    source_rows: int
    duplicate_minute_rows: int
    duplicate_minute_timestamps: int
    conflicting_duplicate_timestamps: int
    invalid_minute_rows: int
    source_time_reversals: int
    raw_hours_over_60_rows: int
    excluded_suspicious_hours: int
    overlapping_h1_timestamps: int
    conflicting_overlapping_h1_timestamps: int
    output_hours: int
    full_60_minute_hours: int
    partial_source_hours: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    archive_reports: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["archive_reports"] = list(self.archive_reports)
        return payload


class LongHistoryH1Stitcher:
    """Combine already-clean H1 archive fragments without guessing on conflicts."""

    _NUMERIC = (
        "open_usd_per_oz",
        "high_usd_per_oz",
        "low_usd_per_oz",
        "close_usd_per_oz",
        "open_usd_per_kg",
        "high_usd_per_kg",
        "low_usd_per_kg",
        "close_usd_per_kg",
        "minute_count",
    )
    _TEXT = ("quality_flag", "asset", "source_symbol", "source_provider", "market_type")

    def stitch(
        self,
        fragments: tuple[pd.DataFrame, ...],
        window: DownloadWindow,
    ) -> tuple[pd.DataFrame, int, int]:
        if not fragments:
            raise ValueError("No H1 fragments supplied for stitching.")
        combined = pd.concat(fragments, ignore_index=True, sort=False)
        combined["timestamp_utc"] = pd.to_datetime(
            combined["timestamp_utc"], utc=True, errors="raise"
        )
        duplicate_mask = combined.duplicated("timestamp_utc", keep=False)
        overlap_count = int(combined.loc[duplicate_mask, "timestamp_utc"].nunique())
        conflicting: set[pd.Timestamp] = set()

        if overlap_count:
            for timestamp, group in combined.loc[duplicate_mask].groupby(
                "timestamp_utc", sort=False
            ):
                comparable = group.loc[:, (*self._NUMERIC, *self._TEXT)].copy()
                if len(comparable.drop_duplicates()) > 1:
                    conflicting.add(pd.Timestamp(timestamp))

        usable = combined.loc[~combined["timestamp_utc"].isin(conflicting)].copy()
        usable = usable.sort_values(["timestamp_utc", "source_archive"])
        usable = usable.drop_duplicates("timestamp_utc", keep="first")

        start = pd.Timestamp(window.start_utc).tz_convert("UTC")
        end = pd.Timestamp(window.end_utc).tz_convert("UTC")
        usable = usable.loc[
            usable["timestamp_utc"].between(start, end, inclusive="both")
        ].copy()
        usable = usable.sort_values("timestamp_utc").reset_index(drop=True)
        if usable.empty:
            raise ValueError("No H1 rows remain inside the long-history window.")
        return usable, overlap_count, len(conflicting)


class LongHistoryH1Builder:
    """Orchestrates parser -> conservative aggregation -> conservative stitching.

    Dependencies are injected so local MetaStock archives and the existing Generic
    ASCII downloader can share the same quality and stitching policy.
    """

    def __init__(
        self,
        parser: MinuteArchiveParser,
        aggregator: ConservativeH1Aggregator,
        stitcher: LongHistoryH1Stitcher | None = None,
    ) -> None:
        self._parser = parser
        self._aggregator = aggregator
        self._stitcher = stitcher or LongHistoryH1Stitcher()

    def build(
        self,
        archives: tuple[Path, ...],
        instrument: InstrumentSpec,
        window: DownloadWindow,
    ) -> tuple[pd.DataFrame, LongHistoryBuildReport]:
        if not archives:
            raise ValueError("Long-history build requires at least one archive.")
        fragments: list[pd.DataFrame] = []
        reports: list[tuple[str, H1AggregationReport]] = []

        for archive in archives:
            minutes = self._parser.parse((archive,))
            timestamps = pd.to_datetime(minutes["timestamp_utc"], utc=True, errors="raise")
            local_start = timestamps.min()
            local_end = timestamps.max()
            if local_start >= local_end:
                raise ValueError(f"Archive {archive.name} has an invalid time span.")
            local_window = DownloadWindow(start_utc=local_start, end_utc=local_end)
            hourly, report = self._aggregator.aggregate(minutes, instrument, local_window)
            hourly = hourly.copy()
            hourly["source_archive"] = archive.name
            fragments.append(hourly)
            reports.append((archive.name, report))

        stitched, overlap_count, conflicting_overlap_count = self._stitcher.stitch(
            tuple(fragments), window
        )
        self._aggregator.validate(stitched)
        if stitched["timestamp_utc"].duplicated().any():
            raise AssertionError("Long-history stitching left duplicate H1 timestamps.")

        report = LongHistoryBuildReport(
            input_archives=len(archives),
            source_rows=sum(item.source_rows for _, item in reports),
            duplicate_minute_rows=sum(item.duplicate_minute_rows for _, item in reports),
            duplicate_minute_timestamps=sum(
                item.duplicate_minute_timestamps for _, item in reports
            ),
            conflicting_duplicate_timestamps=sum(
                item.conflicting_duplicate_timestamps for _, item in reports
            ),
            invalid_minute_rows=sum(item.invalid_minute_rows for _, item in reports),
            source_time_reversals=sum(item.source_time_reversals for _, item in reports),
            raw_hours_over_60_rows=sum(item.raw_hours_over_60_rows for _, item in reports),
            excluded_suspicious_hours=sum(
                item.excluded_suspicious_hours for _, item in reports
            ),
            overlapping_h1_timestamps=overlap_count,
            conflicting_overlapping_h1_timestamps=conflicting_overlap_count,
            output_hours=len(stitched),
            full_60_minute_hours=int(stitched["minute_count"].eq(60).sum()),
            partial_source_hours=int(stitched["minute_count"].lt(60).sum()),
            first_timestamp_utc=pd.Timestamp(stitched["timestamp_utc"].iloc[0]).isoformat(),
            last_timestamp_utc=pd.Timestamp(stitched["timestamp_utc"].iloc[-1]).isoformat(),
            archive_reports=tuple(
                {"archive": name, **item.as_dict()} for name, item in reports
            ),
        )
        return stitched, report
