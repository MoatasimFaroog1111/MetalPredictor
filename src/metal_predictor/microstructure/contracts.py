from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Mapping, Protocol

from metal_predictor.live.quote_contracts import BidAskMarketQuote, MarketDepthLevel


MICROSTRUCTURE_FEATURE_VERSION = "bullionvault-microstructure-v1"


@dataclass(frozen=True)
class MicrostructureSnapshot:
    source_provider: str
    security_id: str
    currency: str
    captured_at_utc: datetime
    access_mode: str
    freshness_status: str
    bid_depth: tuple[MarketDepthLevel, ...]
    ask_depth: tuple[MarketDepthLevel, ...]

    def __post_init__(self) -> None:
        if self.captured_at_utc.tzinfo is None:
            raise ValueError("captured_at_utc must be timezone-aware.")
        if not self.bid_depth or not self.ask_depth:
            raise ValueError("Microstructure snapshot requires non-empty bid and ask depth.")
        if self.best_ask_usd_per_kg <= self.best_bid_usd_per_kg:
            raise ValueError("Microstructure snapshot must have ask above bid.")

    @classmethod
    def from_quote(cls, quote: BidAskMarketQuote) -> "MicrostructureSnapshot":
        return cls(
            source_provider=quote.source_provider,
            security_id=quote.security_id,
            currency=quote.currency,
            captured_at_utc=quote.fetched_at_utc,
            access_mode=quote.access_mode,
            freshness_status=quote.freshness_status,
            bid_depth=tuple(quote.bid_depth),
            ask_depth=tuple(quote.ask_depth),
        )

    @property
    def best_bid_usd_per_kg(self) -> float:
        return self.bid_depth[0].price_usd_per_kg

    @property
    def best_ask_usd_per_kg(self) -> float:
        return self.ask_depth[0].price_usd_per_kg

    @property
    def best_bid_quantity_kg(self) -> float:
        return self.bid_depth[0].quantity_kg

    @property
    def best_ask_quantity_kg(self) -> float:
        return self.ask_depth[0].quantity_kg

    @property
    def mid_usd_per_kg(self) -> float:
        return (self.best_bid_usd_per_kg + self.best_ask_usd_per_kg) / 2.0

    @property
    def spread_usd_per_kg(self) -> float:
        return self.best_ask_usd_per_kg - self.best_bid_usd_per_kg

    def as_dict(self) -> dict[str, object]:
        return {
            "source_provider": self.source_provider,
            "security_id": self.security_id,
            "currency": self.currency,
            "captured_at_utc": self.captured_at_utc.isoformat(),
            "access_mode": self.access_mode,
            "freshness_status": self.freshness_status,
            "best_bid_usd_per_kg": self.best_bid_usd_per_kg,
            "best_ask_usd_per_kg": self.best_ask_usd_per_kg,
            "mid_usd_per_kg": self.mid_usd_per_kg,
            "spread_usd_per_kg": self.spread_usd_per_kg,
            "best_bid_quantity_kg": self.best_bid_quantity_kg,
            "best_ask_quantity_kg": self.best_ask_quantity_kg,
            "bid_depth": [level.as_dict() for level in self.bid_depth],
            "ask_depth": [level.as_dict() for level in self.ask_depth],
            "read_only": True,
        }


@dataclass(frozen=True)
class MicrostructureFeatureVector:
    captured_at_utc: datetime
    feature_version: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.captured_at_utc.tzinfo is None:
            raise ValueError("captured_at_utc must be timezone-aware.")
        if not self.feature_version:
            raise ValueError("feature_version is required.")
        for name, value in self.values.items():
            if not name:
                raise ValueError("Microstructure feature names must be non-empty.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Microstructure feature {name} is not finite.")

    def as_dict(self) -> dict[str, object]:
        return {
            "captured_at_utc": self.captured_at_utc.isoformat(),
            "feature_version": self.feature_version,
            "values": {name: float(value) for name, value in self.values.items()},
        }


@dataclass(frozen=True)
class MicrostructureResearchRecord:
    snapshot_id: int
    snapshot: MicrostructureSnapshot
    features: MicrostructureFeatureVector

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot": self.snapshot.as_dict(),
            "features": self.features.as_dict(),
            "safety": {
                "research_only": True,
                "model_mutated": False,
                "frozen_feature_graph_mutated": False,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "order_submission_available": False,
            },
        }


class MicrostructureRepository(Protocol):
    def append(
        self,
        snapshot: MicrostructureSnapshot,
        features: MicrostructureFeatureVector,
    ) -> MicrostructureResearchRecord: ...

    def latest_snapshot(self) -> MicrostructureSnapshot | None: ...

    def latest_record(self) -> MicrostructureResearchRecord | None: ...

    def recent_records(self, limit: int = 100) -> list[MicrostructureResearchRecord]: ...

    def count(self) -> int: ...
