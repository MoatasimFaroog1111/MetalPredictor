from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DatasetState(StrEnum):
    READY = "READY"
    DATA_PENDING = "DATA_PENDING"


@dataclass(frozen=True)
class ForecastHorizonSpec:
    key: str
    label: str
    interval_seconds: int
    route: str
    dataset_state: DatasetState

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Horizon key must not be empty.")
        if self.interval_seconds <= 0:
            raise ValueError("Horizon interval_seconds must be positive.")
        if not self.route.startswith("/forecast/"):
            raise ValueError("Horizon route must live under /forecast/.")

    @property
    def interval_hours(self) -> float:
        return self.interval_seconds / 3600.0


@dataclass(frozen=True)
class ResearchGuardrails:
    edge_status: str = "NOT_PROVEN"
    research_only: bool = True
    buy_sell_enabled: bool = False
    execution_enabled: bool = False
    live_model_mutated: bool = False
    frozen_52_feature_graph_mutated: bool = False


@dataclass(frozen=True)
class HistoricalBarRecord:
    source_timestamp_text: str
    high_usd_per_kg: float
    low_usd_per_kg: float
    close_usd_per_kg: float


class HistoricalBarSource(Protocol):
    def load(self) -> tuple[HistoricalBarRecord, ...]: ...
