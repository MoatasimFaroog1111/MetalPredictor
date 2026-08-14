from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from metal_predictor.live.quote_contracts import BidAskMarketQuote, MarketQuoteProvider
from metal_predictor.microstructure.contracts import MicrostructureRepository, MicrostructureSnapshot


class PersistedMicrostructureQuoteProvider(MarketQuoteProvider):
    """Serve the newest persisted BullionVault snapshot without network access.

    The microstructure collector remains the sole component allowed to call BullionVault.
    Public research APIs read this append-only snapshot view and fail closed when the
    newest observation is unavailable, too old, or has an implausible future timestamp.
    """

    delivery_mode = "PERSISTED_MICROSTRUCTURE_SNAPSHOT"

    def __init__(
        self,
        repository: MicrostructureRepository,
        *,
        max_age_seconds: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        max_age = int(max_age_seconds)
        if not 30 <= max_age <= 86_400:
            raise ValueError("max_age_seconds must be between 30 and 86400.")
        self._repository = repository
        self._max_age_seconds = max_age
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def max_age_seconds(self) -> int:
        return self._max_age_seconds

    def fetch_quote(self) -> BidAskMarketQuote:
        snapshot = self._repository.latest_snapshot()
        if snapshot is None:
            raise RuntimeError(
                "BullionVault persisted quote is not available yet; waiting for the read-only collector."
            )
        self._validate_freshness(snapshot)
        return self._to_quote(snapshot)

    def status(self) -> dict[str, object]:
        snapshot = self._repository.latest_snapshot()
        if snapshot is None:
            return {
                "delivery_mode": self.delivery_mode,
                "available": False,
                "max_age_seconds": self._max_age_seconds,
                "captured_at_utc": None,
                "age_seconds": None,
                "stale": True,
                "clock_anomaly": False,
                "direct_network_request": False,
            }

        now = self._now_utc()
        captured = snapshot.captured_at_utc.astimezone(timezone.utc)
        raw_age = (now - captured).total_seconds()
        clock_anomaly = raw_age < -5.0
        effective_age = max(0.0, raw_age)
        return {
            "delivery_mode": self.delivery_mode,
            "available": True,
            "max_age_seconds": self._max_age_seconds,
            "captured_at_utc": captured.isoformat(),
            "age_seconds": effective_age,
            "stale": bool(clock_anomaly or effective_age > self._max_age_seconds),
            "clock_anomaly": clock_anomaly,
            "direct_network_request": False,
            "access_mode": snapshot.access_mode,
            "freshness_status": snapshot.freshness_status,
        }

    def _validate_freshness(self, snapshot: MicrostructureSnapshot) -> None:
        now = self._now_utc()
        captured = snapshot.captured_at_utc.astimezone(timezone.utc)
        age_seconds = (now - captured).total_seconds()
        if age_seconds < -5.0:
            raise RuntimeError("BullionVault persisted quote timestamp is unexpectedly in the future.")
        if age_seconds > self._max_age_seconds:
            raise RuntimeError(
                "BullionVault persisted quote is stale; waiting for a fresh collector snapshot."
            )

    def _now_utc(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Persisted quote clock must return a timezone-aware datetime.")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _to_quote(snapshot: MicrostructureSnapshot) -> BidAskMarketQuote:
        return BidAskMarketQuote(
            source_provider=snapshot.source_provider,
            security_id=snapshot.security_id,
            currency=snapshot.currency,
            best_bid_usd_per_kg=snapshot.best_bid_usd_per_kg,
            best_ask_usd_per_kg=snapshot.best_ask_usd_per_kg,
            best_bid_quantity_kg=snapshot.best_bid_quantity_kg,
            best_ask_quantity_kg=snapshot.best_ask_quantity_kg,
            bid_depth=tuple(snapshot.bid_depth),
            ask_depth=tuple(snapshot.ask_depth),
            fetched_at_utc=snapshot.captured_at_utc,
            access_mode=snapshot.access_mode,
            freshness_status=snapshot.freshness_status,
        )
