from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from metal_predictor.live.app import create_app
from metal_predictor.live.catchup import LiveMarketCatchUpService
from metal_predictor.live.contracts import ForecastSnapshot, HourlySilverBar
from metal_predictor.live.inference import LivePredictionEngine
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.notifications import TelegramForecastPublisher
from metal_predictor.live.repository import SQLiteForecastRepository
from metal_predictor.live.scheduler import HourlyCollectionScheduler
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
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/v1/health")
        assert health.json()["buy_sell_enabled"] is False
        assert health.headers["cache-control"] == "no-store"

        page = client.get("/")
        assert page.status_code == 200
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert page.headers["x-frame-options"] == "DENY"

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
        unauthorized = client.post("/api/v1/market/silver/hourly", json=payload)
        assert unauthorized.status_code == 401, unauthorized.text
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
        return httpx.Response(
            200,
            json={"meta": {"symbol": "XAG/USD"}, "values": values, "status": "ok"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = TwelveDataSilverMinuteSource("secret-test-key", client=client)
    bar = source.fetch_completed_hour(start)
    assert bar.timestamp_utc == start
    assert bar.minute_count == 60
    assert bar.quality_flag == "OK"
    assert bar.source_provider == "TwelveData"
    assert bar.market_type == "spot_quote"
    assert bar.close_usd_per_kg > 2000


def test_twelvedata_catch_up_batches_stay_below_documented_point_limit() -> None:
    requests: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.url.params))
        return httpx.Response(200, json={"status": "ok", "values": []})

    source = TwelveDataSilverMinuteSource(
        "secret-test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=144)
    assert source.fetch_completed_range(start, end) == []
    assert len(requests) == 3
    assert all("outputsize" not in item for item in requests)
    for item in requests:
        request_start = datetime.fromisoformat(item["start_date"]).replace(tzinfo=timezone.utc)
        request_end = datetime.fromisoformat(item["end_date"]).replace(tzinfo=timezone.utc)
        covered_minutes = int((request_end - request_start).total_seconds() // 60) + 1
        assert covered_minutes <= 4320


def test_catch_up_stores_context_but_materializes_only_latest_forecast(tmp_path: Path) -> None:
    repository = SQLiteForecastRepository(tmp_path / "catchup.sqlite3")
    historical_last = datetime(2026, 8, 7, 21, tzinfo=timezone.utc)
    through = historical_last + timedelta(hours=3)
    bars = [
        _bar(historical_last + timedelta(hours=offset), 2000.0 + offset)
        for offset in (1, 2, 3)
    ]

    class FakeEngine:
        historical_last_datetime_utc = historical_last

    class FakeSource:
        def fetch_completed_range(self, start_hour_utc, end_hour_utc):
            assert start_hour_utc == historical_last + timedelta(hours=1)
            assert end_hour_utc == through
            return bars

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.materialize_calls = 0

        def ingest_bar(self, bar):
            return repository.put_bar(bar)

        def materialize_latest_forecast(self):
            self.materialize_calls += 1
            latest = repository.recent_bars(limit=1)[-1]
            snapshot = ForecastSnapshot(
                feature_timestamp_utc=latest.timestamp_utc,
                decision_time_utc=latest.timestamp_utc + timedelta(hours=1),
                current_price_usd_per_kg=latest.close_usd_per_kg,
                baseline_model="ridge_alpha_100",
                baseline_log_return_1h=0.001,
                baseline_predicted_price_usd_per_kg=latest.close_usd_per_kg * 1.001,
                baseline_direction="UP",
                challenger_model="ridge_alpha_10",
                challenger_log_return_1h=0.0011,
                challenger_predicted_price_usd_per_kg=latest.close_usd_per_kg * 1.0011,
                challenger_direction="UP",
                data_quality=latest.quality_flag,
                source_provider=latest.source_provider,
                source_compatible_with_training=False,
            )
            return snapshot, repository.put_forecast(snapshot)

    orchestrator = FakeOrchestrator()
    service = LiveMarketCatchUpService(
        FakeSource(), repository, FakeEngine(), orchestrator
    )
    result = service.catch_up(through)
    assert result.fetched_bars == 3
    assert result.created_bars == 3
    assert result.forecast_created is True
    assert result.forecast_timestamp_utc == through
    assert result.status == "FORECAST_MATERIALIZED"
    assert orchestrator.materialize_calls == 1
    assert len(repository.recent_bars(limit=10)) == 3
    forecasts = repository.forecast_history(limit=10)
    assert len(forecasts) == 1
    assert forecasts[0].feature_timestamp_utc == through


def test_catch_up_does_not_forecast_when_latest_requested_hour_is_unavailable(tmp_path: Path) -> None:
    repository = SQLiteForecastRepository(tmp_path / "catchup-gap.sqlite3")
    historical_last = datetime(2026, 8, 7, 21, tzinfo=timezone.utc)
    through = historical_last + timedelta(hours=3)
    bars = [_bar(historical_last + timedelta(hours=1), 2001.0)]

    class FakeEngine:
        historical_last_datetime_utc = historical_last

    class FakeSource:
        def fetch_completed_range(self, start_hour_utc, end_hour_utc):
            return bars

    class FakeOrchestrator:
        materialize_calls = 0

        def ingest_bar(self, bar):
            return repository.put_bar(bar)

        def materialize_latest_forecast(self):
            self.materialize_calls += 1
            raise AssertionError("Must not materialize a stale catch-up hour.")

    orchestrator = FakeOrchestrator()
    service = LiveMarketCatchUpService(
        FakeSource(), repository, FakeEngine(), orchestrator
    )
    result = service.catch_up(through)
    assert result.status == "LATEST_HOUR_NOT_AVAILABLE"
    assert result.forecast_created is False
    assert orchestrator.materialize_calls == 0
    assert len(repository.forecast_history(limit=10)) == 0


def test_scheduler_uses_fixed_utc_delay_after_each_hour() -> None:
    now = datetime(2026, 8, 10, 13, 4, 59, tzinfo=timezone.utc)
    assert HourlyCollectionScheduler.next_due_utc(now, 5) == datetime(
        2026, 8, 10, 13, 5, tzinfo=timezone.utc
    )
    now_after = datetime(2026, 8, 10, 13, 5, tzinfo=timezone.utc)
    assert HourlyCollectionScheduler.next_due_utc(now_after, 5) == datetime(
        2026, 8, 10, 14, 5, tzinfo=timezone.utc
    )
    assert HourlyCollectionScheduler.previous_completed_hour(now_after) == datetime(
        2026, 8, 10, 12, 0, tzinfo=timezone.utc
    )


def test_telegram_webhook_configuration_requires_https_and_sends_secret() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ok": True, "description": "Webhook was set"})

    publisher = TelegramForecastPublisher(
        "fake-bot-token",
        ["12345"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        publisher.configure_webhook("http://silver.example", "secret-token")

    result = publisher.configure_webhook("https://silver.example/", "secret-token")
    assert result["configured"] is True
    assert result["url"] == "https://silver.example/api/v1/telegram/webhook"
    assert str(observed["path"]).endswith("/setWebhook")
    sent = observed["json"]
    assert isinstance(sent, dict)
    assert sent["secret_token"] == "secret-token"
    assert sent["allowed_updates"] == ["message"]


def test_twelvedata_transport_error_redacts_api_key() -> None:
    secret = "td-super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, json={"status": "error"})

    source = TwelveDataSilverMinuteSource(
        secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError) as captured:
        source.fetch_completed_hour(datetime(2026, 8, 10, 10, tzinfo=timezone.utc))
    assert secret not in str(captured.value)
    assert "Twelve Data transport request failed" in str(captured.value)


def test_telegram_transport_error_redacts_bot_token() -> None:
    token = "123456:telegram-super-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, json={"ok": False})

    publisher = TelegramForecastPublisher(
        token,
        ["12345"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError) as captured:
        publisher.send_text("12345", "test")
    assert token not in str(captured.value)
    assert "Telegram sendMessage transport request failed" in str(captured.value)
