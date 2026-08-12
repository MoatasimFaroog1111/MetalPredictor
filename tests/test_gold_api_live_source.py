from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from metal_predictor.live.app import create_app
from metal_predictor.live.gold_api_source import GoldApiSilverOhlcSource
from metal_predictor.live.settings import LiveSettings
from metal_predictor.price_normalization import TROY_OZ_PER_KG


ROOT = Path(__file__).resolve().parents[1]


def test_gold_api_adapter_fetches_exact_completed_hour_and_converts_to_usd_per_kg() -> None:
    start = datetime(2026, 8, 12, 5, tzinfo=timezone.utc)
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["params"] = dict(request.url.params)
        observed["key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            request=request,
            json={
                "open": 38.0,
                "high": 38.4,
                "low": 37.8,
                "close": 38.2,
                "startTimestamp": int(start.timestamp()),
                "endTimestamp": int((start + timedelta(hours=1) - timedelta(seconds=1)).timestamp()),
            },
        )

    source = GoldApiSilverOhlcSource(
        "gold-test-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bar = source.fetch_completed_hour(start)

    assert observed["path"] == "/ohlc/XAG"
    assert observed["key"] == "gold-test-secret"
    params = observed["params"]
    assert isinstance(params, dict)
    assert int(params["startTimestamp"]) == int(start.timestamp())
    assert int(params["endTimestamp"]) == int(
        (start + timedelta(hours=1) - timedelta(seconds=1)).timestamp()
    )
    assert bar.timestamp_utc == start
    assert bar.source_provider == "GoldAPI"
    assert bar.source_symbol == "XAG"
    assert bar.market_type == "spot_quote"
    assert bar.quality_flag == "PROVIDER_AGGREGATED_H1"
    assert bar.minute_count == 1
    assert bar.open_usd_per_kg == pytest.approx(38.0 * TROY_OZ_PER_KG)
    assert bar.high_usd_per_kg == pytest.approx(38.4 * TROY_OZ_PER_KG)
    assert bar.low_usd_per_kg == pytest.approx(37.8 * TROY_OZ_PER_KG)
    assert bar.close_usd_per_kg == pytest.approx(38.2 * TROY_OZ_PER_KG)


def test_gold_api_free_tier_range_policy_fetches_only_latest_requested_hour() -> None:
    requested: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(dict(request.url.params))
        return httpx.Response(
            200,
            request=request,
            json={"open": 40.0, "high": 40.2, "low": 39.8, "close": 40.1},
        )

    source = GoldApiSilverOhlcSource(
        "gold-test-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    start = datetime(2026, 8, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=30)
    bars = source.fetch_completed_range(start, end)

    assert len(requested) == 1
    assert len(bars) == 1
    assert bars[0].timestamp_utc == end
    assert int(requested[0]["startTimestamp"]) == int(end.timestamp())


def test_gold_api_transport_error_redacts_key_and_reports_http_status() -> None:
    secret = "gold-super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request, json={"error": "rate limit"})

    source = GoldApiSilverOhlcSource(
        secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError) as captured:
        source.fetch_completed_hour(datetime(2026, 8, 12, 5, tzinfo=timezone.utc))

    message = str(captured.value)
    assert secret not in message
    assert message == "Gold API request failed with HTTP 429."


def test_auto_provider_prefers_gold_api_when_both_keys_are_present() -> None:
    settings = LiveSettings(
        market_provider="auto",
        gold_api_key="gold-key",
        twelvedata_api_key="twelve-key",
    )
    assert settings.resolved_market_provider == "GoldAPI"
    assert settings.market_source_symbol == "XAG"
    assert settings.market_source_mode == "PROVIDER_H1_OHLC_LATEST_ONLY"
    assert settings.market_source_enabled is True


def test_explicit_twelvedata_provider_remains_available() -> None:
    settings = LiveSettings(
        market_provider="twelvedata",
        gold_api_key="gold-key",
        twelvedata_api_key="twelve-key",
    )
    assert settings.resolved_market_provider == "TwelveData"
    assert settings.market_source_symbol == "XAG/USD"
    assert settings.market_source_mode == "M1_TO_H1_CONSERVATIVE_AGGREGATION"


def test_status_exposes_resolved_gold_api_provider(tmp_path: Path) -> None:
    settings = LiveSettings(
        repository_root=ROOT,
        database_path=tmp_path / "gold-live.sqlite3",
        market_provider="auto",
        gold_api_key="gold-key",
        twelvedata_api_key="twelve-key",
        auto_collect=False,
    )
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/v1/status")
        assert status.status_code == 200
        source = status.json()["market_source"]
        assert source == {
            "configured": True,
            "provider": "GoldAPI",
            "symbol": "XAG",
            "mode": "PROVIDER_H1_OHLC_LATEST_ONLY",
        }
