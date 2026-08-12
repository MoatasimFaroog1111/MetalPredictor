from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo


SAUDI_TIMEZONE_NAME = "Asia/Riyadh"
_SAUDI_TIMEZONE = ZoneInfo(SAUDI_TIMEZONE_NAME)


def _saudi_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware.")
    return value.astimezone(_SAUDI_TIMEZONE).isoformat()


@dataclass(frozen=True)
class HourlySilverBar:
    timestamp_utc: datetime
    open_usd_per_kg: float
    high_usd_per_kg: float
    low_usd_per_kg: float
    close_usd_per_kg: float
    minute_count: int
    quality_flag: str
    source_provider: str
    source_symbol: str
    market_type: str

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["timestamp_utc"] = self.timestamp_utc.isoformat()
        data["timestamp_saudi"] = _saudi_iso(self.timestamp_utc)
        data["display_timezone"] = SAUDI_TIMEZONE_NAME
        return data


@dataclass(frozen=True)
class ForecastSnapshot:
    feature_timestamp_utc: datetime
    decision_time_utc: datetime
    current_price_usd_per_kg: float
    baseline_model: str
    baseline_log_return_1h: float
    baseline_predicted_price_usd_per_kg: float
    baseline_direction: str
    challenger_model: str
    challenger_log_return_1h: float
    challenger_predicted_price_usd_per_kg: float
    challenger_direction: str
    data_quality: str
    source_provider: str
    source_compatible_with_training: bool
    materialized_at_utc: datetime | None = None
    edge_status: str = "NOT_PROVEN"
    research_only: bool = True

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        feature = self.feature_timestamp_utc
        model_clock = self.decision_time_utc
        published = self.materialized_at_utc

        # Canonical H1 timestamps label bar STARTS. The feature bar at t becomes fully
        # observable at t+1h; its close is therefore the reference/current model price
        # at decision_time_utc. The frozen 1h target is the close of the NEXT H1 bar,
        # which is observed at t+2h. Expose both user-facing instants explicitly.
        current_price_time = model_clock
        forecast_target_time = model_clock + timedelta(hours=1)

        data["feature_timestamp_utc"] = feature.isoformat()
        data["decision_time_utc"] = model_clock.isoformat()
        data["model_clock_decision_time_utc"] = model_clock.isoformat()
        data["materialized_at_utc"] = published.isoformat() if published is not None else None

        data["feature_timestamp_saudi"] = _saudi_iso(feature)
        data["decision_time_saudi"] = _saudi_iso(model_clock)
        data["model_clock_decision_time_saudi"] = _saudi_iso(model_clock)
        data["materialized_at_saudi"] = _saudi_iso(published) if published is not None else None

        data["current_price_time_utc"] = current_price_time.isoformat()
        data["current_price_time_saudi"] = _saudi_iso(current_price_time)
        data["forecast_target_time_utc"] = forecast_target_time.isoformat()
        data["forecast_target_time_saudi"] = _saudi_iso(forecast_target_time)
        data["forecast_horizon_hours"] = 1
        data["display_timezone"] = SAUDI_TIMEZONE_NAME

        data["current_price_semantics"] = "LAST_COMPLETED_H1_CLOSE"
        data["forecast_target_semantics"] = "NEXT_H1_CLOSE"
        data["decision_time_semantics"] = "MODEL_CLOCK_HOUR_BOUNDARY"
        data["materialized_at_semantics"] = "ACTUAL_FORECAST_PUBLICATION_TIME"
        data["publication_delay_seconds"] = (
            (published - model_clock).total_seconds() if published is not None else None
        )
        return data


class MarketBarSource(Protocol):
    def fetch_completed_hour(self, hour_start_utc: datetime) -> HourlySilverBar: ...


class MarketBarBackfillSource(MarketBarSource, Protocol):
    def fetch_completed_range(
        self,
        start_hour_utc: datetime,
        end_hour_utc: datetime,
    ) -> list[HourlySilverBar]: ...


class ForecastRepository(Protocol):
    def put_bar(self, bar: HourlySilverBar) -> bool: ...
    def recent_bars(self, limit: int = 500) -> list[HourlySilverBar]: ...
    def put_forecast(self, snapshot: ForecastSnapshot) -> bool: ...
    def latest_forecast(self) -> ForecastSnapshot | None: ...
    def forecast_history(self, limit: int = 100) -> list[ForecastSnapshot]: ...


class NotificationPublisher(Protocol):
    def publish_forecast(self, snapshot: ForecastSnapshot) -> None: ...
