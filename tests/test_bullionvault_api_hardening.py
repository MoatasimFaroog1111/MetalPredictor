from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from metal_predictor.live.app import create_app
from metal_predictor.live.quote_contracts import BidAskMarketQuote, MarketDepthLevel
from metal_predictor.live.settings import LiveSettings
from metal_predictor.microstructure.contracts import MicrostructureSnapshot
from metal_predictor.microstructure.features import MicrostructureFeatureBuilder
from metal_predictor.microstructure.persisted_quote import PersistedMicrostructureQuoteProvider
from metal_predictor.microstructure.repository import SQLiteMicrostructureRepository


ROOT = Path(__file__).resolve().parents[1]


def _quote(captured_at_utc: datetime) -> BidAskMarketQuote:
    return BidAskMarketQuote(
        source_provider="BullionVault",
        security_id="AGXLN",
        currency="USD",
        best_bid_usd_per_kg=2131.0,
        best_ask_usd_per_kg=2139.0,
        best_bid_quantity_kg=0.253,
        best_ask_quantity_kg=15.416,
        bid_depth=(
            MarketDepthLevel(2131.0, 0.253),
            MarketDepthLevel(2130.0, 0.081),
        ),
        ask_depth=(
            MarketDepthLevel(2139.0, 15.416),
            MarketDepthLevel(2140.0, 11.690),
        ),
        fetched_at_utc=captured_at_utc,
        access_mode="AUTHENTICATED_READ_ONLY",
        freshness_status="CURRENT_GUI_SOURCE",
    )


def _append_quote(
    repository: SQLiteMicrostructureRepository,
    captured_at_utc: datetime,
) -> None:
    snapshot = MicrostructureSnapshot.from_quote(_quote(captured_at_utc))
    features = MicrostructureFeatureBuilder().build(snapshot, repository.latest_snapshot())
    repository.append(snapshot, features)


def test_persisted_quote_provider_replays_fresh_snapshot_without_network(tmp_path: Path) -> None:
    captured = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
    repository = SQLiteMicrostructureRepository(tmp_path / "microstructure.sqlite3")
    _append_quote(repository, captured)

    provider = PersistedMicrostructureQuoteProvider(
        repository,
        max_age_seconds=180,
        clock=lambda: captured + timedelta(seconds=60),
    )
    quote = provider.fetch_quote()
    status = provider.status()

    assert quote.source_provider == "BullionVault"
    assert quote.security_id == "AGXLN"
    assert quote.best_bid_usd_per_kg == 2131.0
    assert quote.best_ask_usd_per_kg == 2139.0
    assert quote.fetched_at_utc == captured
    assert quote.access_mode == "AUTHENTICATED_READ_ONLY"
    assert quote.freshness_status == "CURRENT_GUI_SOURCE"
    assert status["delivery_mode"] == "PERSISTED_MICROSTRUCTURE_SNAPSHOT"
    assert status["direct_network_request"] is False
    assert status["stale"] is False
    assert status["age_seconds"] == 60.0


def test_persisted_quote_provider_fails_closed_when_missing_or_stale(tmp_path: Path) -> None:
    captured = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
    repository = SQLiteMicrostructureRepository(tmp_path / "microstructure.sqlite3")
    missing = PersistedMicrostructureQuoteProvider(
        repository,
        max_age_seconds=180,
        clock=lambda: captured,
    )
    with pytest.raises(RuntimeError, match="not available yet"):
        missing.fetch_quote()

    _append_quote(repository, captured)
    stale = PersistedMicrostructureQuoteProvider(
        repository,
        max_age_seconds=180,
        clock=lambda: captured + timedelta(seconds=181),
    )
    with pytest.raises(RuntimeError, match="stale"):
        stale.fetch_quote()
    assert stale.status()["stale"] is True


class FailIfPublicApiCallsNetwork:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_quote(self) -> BidAskMarketQuote:
        self.calls += 1
        raise AssertionError("Public API must never call the BullionVault network provider.")


def test_public_bullionvault_quote_endpoint_reads_persisted_snapshot_only(tmp_path: Path) -> None:
    network_provider = FailIfPublicApiCallsNetwork()
    settings = LiveSettings(
        repository_root=ROOT,
        database_path=tmp_path / "live.sqlite3",
        bullionvault_microstructure_database_path=tmp_path / "microstructure.sqlite3",
        bullionvault_microstructure_enabled=False,
    )
    app = create_app(
        settings,
        bullionvault_network_quote_provider=network_provider,
    )
    captured = datetime.now(timezone.utc)
    _append_quote(app.state.microstructure_repository, captured)

    with TestClient(app) as client:
        response = client.get("/api/v1/research/bullionvault/quote")
        status_response = client.get("/api/v1/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["quote"]["source_provider"] == "BullionVault"
    assert body["quote"]["access_mode"] == "AUTHENTICATED_READ_ONLY"
    assert body["quote"]["fetched_at_utc"] == captured.isoformat()
    assert network_provider.calls == 0

    assert status_response.status_code == 200
    quote_status = status_response.json()["bullionvault_quote"]
    assert quote_status["network_access_owner"] == "MICROSTRUCTURE_COLLECTOR_ONLY"
    assert quote_status["public_api_direct_network_requests"] is False
    assert quote_status["persisted_snapshot"]["direct_network_request"] is False
    assert quote_status["persisted_snapshot"]["stale"] is False
