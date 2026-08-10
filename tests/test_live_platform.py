from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from metal_predictor.live.app import create_app
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.inference import LivePredictionEngine
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.repository import SQLiteForecastRepository
from metal_predictor.live.settings import LiveSettings


ROOT = Path(__file__).resolve().parents[1]


def _bar(timestamp: datetime, close: float = 2000.0) -> HourlySilverBar:
    return HourlySilverBar(
        timestamp_utc=timestamp,
        open_usd_per_kg=close,
        high_usd_per_kg=close * 1.001,
        low_usd_per_kg=close * 0.999,
        close_usd_per_kg=close,
        minute_count=60,
        quality_flag="OK",
        source_provider="Manual",
        source_symbol="XAGUSD",
        market_type="manual_input",
    )


def _next_historical_hour() -> tuple[datetime, float]:
    frame = pd.read_parquet(ROOT / "XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
    last = frame.sort_values("timestamp_utc").iloc[-1]
    timestamp = pd.Timestamp(last["timestamp_utc"]).tz_convert("UTC") + pd.Timedelta(hours=1)
    return timestamp.to_pydatetime(), float(last["close_usd_per_kg"])


def test_sqlite_repository_is_idempotent_and_rejects_revision(tmp_path: Path) -> None:
    repo = SQLiteForecastRepository(tmp_path / "live.sqlite3")
    timestamp, close = _next_historical_hour()
    original = _bar(timestamp, close)
    assert repo.put_bar(original) is True
    assert repo.put_bar(original) is False
    revised = _bar(timestamp, close * 1.01)
    with pytest.raises(ValueError, match="LIVE_BAR_REVISION_CONFLICT"):
        repo.put_bar(revised)


def test_frozen_live_engine_produces_finite_research_only_snapshot() -> None:
    timestamp, close = _next_historical_hour()
    engine = LivePredictionEngine(ROOT)
    snapshot = engine.predict([_bar(timestamp, close)])
    assert snapshot.baseline_model == "ridge_alpha_100"
    assert snapshot.challenger_model == "ridge_alpha_10"
    assert snapshot.edge_status == "NOT_PROVEN"
    assert snapshot.research_only is True
    assert snapshot.source_compatible_with_training is False
    assert snapshot.baseline_direction in {"UP", "DOWN", "FLAT"}
    assert snapshot.challenger_direction in {"UP", "DOWN", "FLAT"}
    assert np.isfinite(snapshot.baseline_log_return_1h)
    assert np.isfinite(snapshot.challenger_log_return_1h)
    assert snapshot.baseline_predicted_price_usd_per_kg > 0
    assert snapshot.challenger_predicted_price_usd_per_kg > 0


def test_fastapi_health_pwa_and_protected_manual_ingest(tmp_path: Path) -> None:
    settings = LiveSettings(
        repository_root=ROOT,
        database_path=tmp_path / "api.sqlite3",
        admin_token="test-admin-token",
    )
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/health").json()["buy_sell_enabled"] is False
    assert client.get("/").status_code == 200
    sw = client.get("/sw.js")
    assert sw.status_code == 200
    assert sw.headers["service-worker-allowed"] == "/"
    assert client.get("/api/v1/forecast/latest").status_code == 404

    timestamp, close = _next_historical_hour()
    payload = {
        "timestamp_utc": timestamp.isoformat(),
        "open_usd_per_kg": close,
        "high_usd_per_kg": close * 1.001,
        "low_usd_per_kg": close * 0.999,
        "close_usd_per_kg": close,
        "minute_count": 60,
        "quality_flag": "OK",
    }
    assert client.post("/api/v1/market/silver/hourly", json=payload).status_code == 401
    response = client.post(
        "/api/v1/market/silver/hourly",
        json=payload,
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bar_created"] is True
    assert body["forecast_created"] is True
    assert body["forecast"]["baseline_model"] == "ridge_alpha_100"
    assert body["forecast"]["research_only"] is True

    second = client.post(
        "/api/v1/market/silver/hourly",
        json=payload,
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert second.status_code == 200
    assert second.json()["bar_created"] is False
    assert second.json()["forecast_created"] is False
    assert len(client.get("/api/v1/forecast/history").json()) == 1


def test_twelvedata_adapter_aggregates_mocked_m1_to_full_h1() -> None:
    start = datetime(2026, 8, 10, 10, tzinfo=timezone.utc)
    values = []
    for minute in range(60):
        ts = start + timedelta(minutes=minute)
        price = 76.0 + minute * 0.001
        values.append(
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": f"{price:.6f}",
                "high": f"{price + 0.01:.6f}",
                "low": f"{price - 0.01:.6f}",
                "close": f"{price + 0.002:.6f}",
            }
        )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "XAG/USD"
        assert request.url.params["interval"] == "1min"
        assert request.url.params["timezone"] == "UTC"
        return httpx.Response(200, json={"meta": {"symbol": "XAG/USD"}, "values": values, "status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = TwelveDataSilverMinuteSource("secret-test-key", client=client)
    bar = source.fetch_completed_hour(start)
    assert bar.timestamp_utc == start
    assert bar.minute_count == 60
    assert bar.quality_flag == "OK"
    assert bar.source_provider == "TwelveData"
    assert bar.market_type == "spot_quote"
    assert bar.close_usd_per_kg > 2000
