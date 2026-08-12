from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Protocol


@dataclass(frozen=True)
class MarketDepthLevel:
    price_usd_per_kg: float
    quantity_kg: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.price_usd_per_kg)) or self.price_usd_per_kg <= 0:
            raise ValueError("price_usd_per_kg must be finite and positive.")
        if not math.isfinite(float(self.quantity_kg)) or self.quantity_kg <= 0:
            raise ValueError("quantity_kg must be finite and positive.")

    def as_dict(self) -> dict[str, float]:
        return {
            "price_usd_per_kg": float(self.price_usd_per_kg),
            "quantity_kg": float(self.quantity_kg),
        }


@dataclass(frozen=True)
class BidAskMarketQuote:
    source_provider: str
    security_id: str
    currency: str
    best_bid_usd_per_kg: float
    best_ask_usd_per_kg: float
    best_bid_quantity_kg: float
    best_ask_quantity_kg: float
    bid_depth: tuple[MarketDepthLevel, ...]
    ask_depth: tuple[MarketDepthLevel, ...]
    fetched_at_utc: datetime
    access_mode: str
    freshness_status: str

    def __post_init__(self) -> None:
        if self.fetched_at_utc.tzinfo is None:
            raise ValueError("fetched_at_utc must be timezone-aware.")
        values = {
            "best_bid_usd_per_kg": self.best_bid_usd_per_kg,
            "best_ask_usd_per_kg": self.best_ask_usd_per_kg,
            "best_bid_quantity_kg": self.best_bid_quantity_kg,
            "best_ask_quantity_kg": self.best_ask_quantity_kg,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.best_ask_usd_per_kg <= self.best_bid_usd_per_kg:
            raise ValueError("Market quote must have ask above bid.")

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
            "best_bid_usd_per_kg": self.best_bid_usd_per_kg,
            "best_ask_usd_per_kg": self.best_ask_usd_per_kg,
            "mid_usd_per_kg": self.mid_usd_per_kg,
            "spread_usd_per_kg": self.spread_usd_per_kg,
            "best_bid_quantity_kg": self.best_bid_quantity_kg,
            "best_ask_quantity_kg": self.best_ask_quantity_kg,
            "bid_depth": [level.as_dict() for level in self.bid_depth],
            "ask_depth": [level.as_dict() for level in self.ask_depth],
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
            "access_mode": self.access_mode,
            "freshness_status": self.freshness_status,
            "execution_guaranteed": False,
            "read_only": True,
        }


class MarketQuoteProvider(Protocol):
    def fetch_quote(self) -> BidAskMarketQuote: ...
