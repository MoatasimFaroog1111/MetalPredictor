from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from metal_predictor.live.quote_contracts import BidAskMarketQuote, MarketDepthLevel
from metal_predictor.microstructure.collector import MicrostructureResearchCollector
from metal_predictor.microstructure.contracts import MicrostructureSnapshot
from metal_predictor.microstructure.features import MicrostructureFeatureBuilder
from metal_predictor.microstructure.repository import SQLiteMicrostructureRepository


def _quote(
    captured_at: datetime,
    *,
    bid: float = 2131.0,
    ask: float = 2139.0,
    bid_qty: float = 0.253,
    ask_qty: float = 15.416,
) -> BidAskMarketQuote:
    return BidAskMarketQuote(
        source_provider="BullionVault",
        security_id="AGXLN",
        currency="USD",
        best_bid_usd_per_kg=bid,
        best_ask_usd_per_kg=ask,
        best_bid_quantity_kg=bid_qty,
        best_ask_quantity_kg=ask_qty,
        bid_depth=(
            MarketDepthLevel(bid, bid_qty),
            MarketDepthLevel(bid - 1.0, 0.081),
            MarketDepthLevel(bid - 11.0, 35.0),
            MarketDepthLevel(bid - 13.0, 30.0),
            MarketDepthLevel(bid - 15.0, 30.0),
        ),
        ask_depth=(
            MarketDepthLevel(ask, ask_qty),
            MarketDepthLevel(ask + 1.0, 11.690),
            MarketDepthLevel(ask + 2.0, 16.843),
            MarketDepthLevel(ask + 9.0, 3.644),
            MarketDepthLevel(ask + 10.0, 35.0),
        ),
        fetched_at_utc=captured_at,
        access_mode="AUTHENTICATED_READ_ONLY",
        freshness_status="CURRENT_GUI_SOURCE",
    )


class SequenceProvider:
    def __init__(self, quotes: list[BidAskMarketQuote]) -> None:
        self._quotes = iter(quotes)

    def fetch_quote(self) -> BidAskMarketQuote:
        return next(self._quotes)


def test_feature_builder_uses_order_book_only_and_produces_finite_features() -> None:
    snapshot = MicrostructureSnapshot.from_quote(
        _quote(datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc))
    )

    vector = MicrostructureFeatureBuilder().build(snapshot)

    assert vector.feature_version == "bullionvault-microstructure-v1"
    assert vector.values["spread_usd_per_kg"] == 8.0
    assert vector.values["spread_bps"] == pytest.approx(8.0 / 2135.0 * 10_000.0)
    assert vector.values["top_quantity_imbalance"] == pytest.approx(
        (0.253 - 15.416) / (0.253 + 15.416)
    )
    assert vector.values["bid_depth_5_kg"] == pytest.approx(95.334)
    assert vector.values["ask_depth_5_kg"] == pytest.approx(82.593)
    assert vector.values["authenticated_quote_flag"] == 1.0
    assert vector.values["public_cached_quote_flag"] == 0.0
    assert vector.values["has_previous_snapshot"] == 0.0
    assert all(math.isfinite(float(value)) for value in vector.values.values())


def test_temporal_features_compare_only_to_earlier_snapshot() -> None:
    first = MicrostructureSnapshot.from_quote(
        _quote(datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc))
    )
    second = MicrostructureSnapshot.from_quote(
        _quote(
            datetime(2026, 8, 12, 10, 11, tzinfo=timezone.utc),
            bid=2132.0,
            ask=2138.0,
            bid_qty=2.0,
            ask_qty=4.0,
        )
    )

    vector = MicrostructureFeatureBuilder().build(second, first)

    assert vector.values["has_previous_snapshot"] == 1.0
    assert vector.values["seconds_since_previous"] == 60.0
    assert vector.values["mid_log_return_since_previous"] == pytest.approx(
        math.log(second.mid_usd_per_kg / first.mid_usd_per_kg)
    )
    assert vector.values["spread_bps_change"] != 0.0


def test_repository_preserves_raw_depth_and_versioned_features(tmp_path) -> None:
    repository = SQLiteMicrostructureRepository(tmp_path / "microstructure.sqlite3")
    snapshot = MicrostructureSnapshot.from_quote(
        _quote(datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc))
    )
    features = MicrostructureFeatureBuilder().build(snapshot)

    created = repository.append(snapshot, features)
    restored = repository.latest_record()

    assert created.snapshot_id == 1
    assert repository.count() == 1
    assert restored is not None
    assert restored.snapshot.security_id == "AGXLN"
    assert restored.snapshot.bid_depth == snapshot.bid_depth
    assert restored.snapshot.ask_depth == snapshot.ask_depth
    assert restored.features.feature_version == "bullionvault-microstructure-v1"
    assert restored.features.values["depth_imbalance_5"] == pytest.approx(
        features.values["depth_imbalance_5"]
    )
    assert restored.as_dict()["safety"]["frozen_feature_graph_mutated"] is False
    assert restored.as_dict()["safety"]["execution_enabled"] is False


def test_collector_appends_successive_causal_snapshots(tmp_path) -> None:
    first_time = datetime(2026, 8, 12, 10, 10, tzinfo=timezone.utc)
    provider = SequenceProvider(
        [
            _quote(first_time),
            _quote(
                first_time + timedelta(minutes=1),
                bid=2132.0,
                ask=2138.0,
                bid_qty=2.0,
                ask_qty=4.0,
            ),
        ]
    )
    repository = SQLiteMicrostructureRepository(tmp_path / "microstructure.sqlite3")
    collector = MicrostructureResearchCollector(provider, repository)

    first = collector.collect_once()
    second = collector.collect_once()

    assert first.features.values["has_previous_snapshot"] == 0.0
    assert second.features.values["has_previous_snapshot"] == 1.0
    assert second.features.values["seconds_since_previous"] == 60.0
    assert repository.count() == 2
    assert [record.snapshot_id for record in repository.recent_records(2)] == [1, 2]
