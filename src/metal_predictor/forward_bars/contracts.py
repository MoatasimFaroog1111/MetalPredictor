from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping, Protocol, Sequence


FORWARD_BAR_VERSION = "bullionvault-forward-multi-horizon-v1"
FORWARD_SOURCE_STREAM = "BULLIONVAULT_READ_ONLY_MICROSTRUCTURE_SNAPSHOTS"
FORWARD_HORIZON_SECONDS: Mapping[str, int] = {
    "4h": 14_400,
    "12h": 43_200,
    "1d": 86_400,
    "2d": 172_800,
    "30d": 2_592_000,
}


@dataclass(frozen=True)
class QuoteSample:
    source_provider: str
    security_id: str
    currency: str
    captured_at_utc: datetime
    access_mode: str
    freshness_status: str
    best_bid_usd_per_kg: float
    best_ask_usd_per_kg: float

    def __post_init__(self) -> None:
        if self.captured_at_utc.tzinfo is None:
            raise ValueError("captured_at_utc must be timezone-aware.")
        if not self.source_provider or not self.security_id or not self.currency:
            raise ValueError("Quote sample source metadata is required.")
        for name, value in (
            ("best_bid_usd_per_kg", self.best_bid_usd_per_kg),
            ("best_ask_usd_per_kg", self.best_ask_usd_per_kg),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.best_ask_usd_per_kg <= self.best_bid_usd_per_kg:
            raise ValueError("Quote sample ask must be above bid.")

    @property
    def mid_usd_per_kg(self) -> float:
        return (self.best_bid_usd_per_kg + self.best_ask_usd_per_kg) / 2.0

    @property
    def spread_usd_per_kg(self) -> float:
        return self.best_ask_usd_per_kg - self.best_bid_usd_per_kg


@dataclass(frozen=True)
class ForwardBar:
    horizon_key: str
    interval_seconds: int
    bucket_start_utc: datetime
    bucket_end_utc: datetime
    source_provider: str
    security_id: str
    currency: str
    open_mid_usd_per_kg: float
    high_mid_usd_per_kg: float
    low_mid_usd_per_kg: float
    close_mid_usd_per_kg: float
    open_bid_usd_per_kg: float
    close_bid_usd_per_kg: float
    open_ask_usd_per_kg: float
    close_ask_usd_per_kg: float
    mean_spread_usd_per_kg: float
    max_spread_usd_per_kg: float
    close_spread_usd_per_kg: float
    snapshot_count: int
    expected_snapshot_count: int
    coverage_ratio: float
    first_sample_at_utc: datetime
    last_sample_at_utc: datetime
    access_mode_counts: Mapping[str, int]
    freshness_status_counts: Mapping[str, int]
    quality_status: str
    source_stream: str = FORWARD_SOURCE_STREAM
    bar_version: str = FORWARD_BAR_VERSION

    def __post_init__(self) -> None:
        if self.horizon_key not in FORWARD_HORIZON_SECONDS:
            raise ValueError(f"Unsupported horizon: {self.horizon_key}")
        if self.interval_seconds != FORWARD_HORIZON_SECONDS[self.horizon_key]:
            raise ValueError("Forward bar interval does not match horizon registry.")
        for dt_name, value in (
            ("bucket_start_utc", self.bucket_start_utc),
            ("bucket_end_utc", self.bucket_end_utc),
            ("first_sample_at_utc", self.first_sample_at_utc),
            ("last_sample_at_utc", self.last_sample_at_utc),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{dt_name} must be timezone-aware.")
        if self.bucket_end_utc <= self.bucket_start_utc:
            raise ValueError("Forward bar bucket end must be after start.")
        if self.snapshot_count < 2:
            raise ValueError("A materialized forward bar requires at least two observed snapshots.")
        if self.expected_snapshot_count <= 0:
            raise ValueError("expected_snapshot_count must be positive.")
        if not 0 < self.coverage_ratio <= 1.0:
            raise ValueError("coverage_ratio must be in (0, 1].")
        if not (
            self.bucket_start_utc <= self.first_sample_at_utc
            <= self.last_sample_at_utc < self.bucket_end_utc
        ):
            raise ValueError("Observed samples must lie inside the source bucket.")
        for value in (
            self.open_mid_usd_per_kg,
            self.high_mid_usd_per_kg,
            self.low_mid_usd_per_kg,
            self.close_mid_usd_per_kg,
            self.open_bid_usd_per_kg,
            self.close_bid_usd_per_kg,
            self.open_ask_usd_per_kg,
            self.close_ask_usd_per_kg,
            self.mean_spread_usd_per_kg,
            self.max_spread_usd_per_kg,
            self.close_spread_usd_per_kg,
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("Forward bar price/spread fields must be finite and positive.")
        if self.high_mid_usd_per_kg < max(
            self.open_mid_usd_per_kg, self.close_mid_usd_per_kg
        ):
            raise ValueError("Forward bar high cannot be below open/close.")
        if self.low_mid_usd_per_kg > min(
            self.open_mid_usd_per_kg, self.close_mid_usd_per_kg
        ):
            raise ValueError("Forward bar low cannot be above open/close.")

    def as_dict(self) -> dict[str, object]:
        return {
            "bar_version": self.bar_version,
            "horizon_key": self.horizon_key,
            "interval_seconds": self.interval_seconds,
            "bucket_start_utc": self.bucket_start_utc.isoformat(),
            "bucket_end_utc": self.bucket_end_utc.isoformat(),
            "source_stream": self.source_stream,
            "source_provider": self.source_provider,
            "security_id": self.security_id,
            "currency": self.currency,
            "open_mid_usd_per_kg": self.open_mid_usd_per_kg,
            "high_mid_usd_per_kg": self.high_mid_usd_per_kg,
            "low_mid_usd_per_kg": self.low_mid_usd_per_kg,
            "close_mid_usd_per_kg": self.close_mid_usd_per_kg,
            "open_bid_usd_per_kg": self.open_bid_usd_per_kg,
            "close_bid_usd_per_kg": self.close_bid_usd_per_kg,
            "open_ask_usd_per_kg": self.open_ask_usd_per_kg,
            "close_ask_usd_per_kg": self.close_ask_usd_per_kg,
            "mean_spread_usd_per_kg": self.mean_spread_usd_per_kg,
            "max_spread_usd_per_kg": self.max_spread_usd_per_kg,
            "close_spread_usd_per_kg": self.close_spread_usd_per_kg,
            "snapshot_count": self.snapshot_count,
            "expected_snapshot_count": self.expected_snapshot_count,
            "coverage_ratio": self.coverage_ratio,
            "first_sample_at_utc": self.first_sample_at_utc.isoformat(),
            "last_sample_at_utc": self.last_sample_at_utc.isoformat(),
            "access_mode_counts": dict(self.access_mode_counts),
            "freshness_status_counts": dict(self.freshness_status_counts),
            "quality_status": self.quality_status,
            "research_only": True,
            "edge_status": "NOT_PROVEN",
            "execution_enabled": False,
            "buy_sell_enabled": False,
        }


@dataclass(frozen=True)
class BucketAssessment:
    horizon_key: str
    interval_seconds: int
    bucket_start_utc: datetime
    bucket_end_utc: datetime
    status: str
    reason: str | None
    snapshot_count: int
    created_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon_key": self.horizon_key,
            "interval_seconds": self.interval_seconds,
            "bucket_start_utc": self.bucket_start_utc.isoformat(),
            "bucket_end_utc": self.bucket_end_utc.isoformat(),
            "status": self.status,
            "reason": self.reason,
            "snapshot_count": self.snapshot_count,
            "created_at_utc": self.created_at_utc.isoformat(),
        }


class QuoteSampleSource(Protocol):
    def first_sample_at(
        self,
        *,
        security_id: str,
        currency: str,
    ) -> datetime | None: ...

    def samples_between(
        self,
        start_utc: datetime,
        end_utc: datetime,
        *,
        security_id: str,
        currency: str,
    ) -> Sequence[QuoteSample]: ...


class ForwardBarRepository(Protocol):
    def has_assessment(self, horizon_key: str, bucket_start_utc: datetime) -> bool: ...

    def latest_assessed_end(self, horizon_key: str) -> datetime | None: ...

    def append_bar(self, bar: ForwardBar) -> bool: ...

    def append_gap(
        self,
        *,
        horizon_key: str,
        interval_seconds: int,
        bucket_start_utc: datetime,
        bucket_end_utc: datetime,
        reason: str,
        snapshot_count: int,
    ) -> bool: ...

    def latest_bar(self, horizon_key: str) -> ForwardBar | None: ...

    def history(self, horizon_key: str, limit: int = 100) -> list[ForwardBar]: ...

    def status_snapshot(self) -> dict[str, object]: ...
