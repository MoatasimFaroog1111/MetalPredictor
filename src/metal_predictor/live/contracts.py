from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol


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
    edge_status: str = "NOT_PROVEN"
    research_only: bool = True

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["feature_timestamp_utc"] = self.feature_timestamp_utc.isoformat()
        data["decision_time_utc"] = self.decision_time_utc.isoformat()
        return data


class MarketBarSource(Protocol):
    def fetch_completed_hour(self, hour_start_utc: datetime) -> HourlySilverBar: ...


class ForecastRepository(Protocol):
    def put_bar(self, bar: HourlySilverBar) -> bool: ...
    def recent_bars(self, limit: int = 500) -> list[HourlySilverBar]: ...
    def put_forecast(self, snapshot: ForecastSnapshot) -> bool: ...
    def latest_forecast(self) -> ForecastSnapshot | None: ...
    def forecast_history(self, limit: int = 100) -> list[ForecastSnapshot]: ...


class NotificationPublisher(Protocol):
    def publish_forecast(self, snapshot: ForecastSnapshot) -> None: ...
