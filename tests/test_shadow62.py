from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.settings import LiveSettings
from metal_predictor.live.shadow62_runtime import install_shadow62_runtime
from metal_predictor.precious_metals.confirmation import CANDIDATE_FEATURES, CANDIDATE_ID
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures
from metal_predictor.shadow62.contracts import (
    SHADOW_EARLIEST_FINAL_SCORE_UTC,
    SHADOW_FIRST_FEATURE_BAR_START_UTC,
    SHADOW_FIXED_WINDOW_DAYS,
    SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC,
    SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES,
    ShadowForecastSnapshot,
    ShadowOutcome,
)
from metal_predictor.shadow62.features import ConfirmedPreciousMetalsShadowFeatures
from metal_predictor.shadow62.repository import SQLiteShadowRepository
from metal_predictor.shadow62.scheduler import Shadow62Scheduler
from metal_predictor.shadow62.service import Shadow62Service


def _market_frame(start: str, rows: int, base: float, slope: float) -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    open_price = base + np.arange(rows, dtype=float) * slope
    close = open_price * (1.0 + 0.001 * np.sin(np.arange(rows, dtype=float) / 3.0))
    high = np.maximum(open_price, close) * 1.002
    low = np.minimum(open_price, close) * 0.998
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "open_usd_per_kg": open_price,
            "high_usd_per_kg": high,
            "low_usd_per_kg": low,
            "close_usd_per_kg": close,
            "quality_flag": "TEST_H1",
        }
    )


def _silver_frame(start: str, rows: int) -> pd.DataFrame:
    frame = _market_frame(start, rows, 2200.0, 1.5)
    frame["minute_count"] = 60
    return frame


def _snapshot(ts: datetime) -> ShadowForecastSnapshot:
    return ShadowForecastSnapshot(
        feature_timestamp_utc=ts,
        materialized_at_utc=ts + timedelta(hours=1, minutes=8),
        reference_close_usd_per_kg=2200.0,
        baseline_model="ridge_alpha_100",
        baseline_model_sha256="a" * 64,
        baseline_log_return_1h=0.001,
        baseline_predicted_price_usd_per_kg=2202.0,
        candidate_id=CANDIDATE_ID,
        candidate_model="ridge_alpha_100",
        candidate_model_sha256="b" * 64,
        candidate_log_return_1h=0.002,
        candidate_predicted_price_usd_per_kg=2204.0,
        xpt_exact_current=True,
        xpd_exact_current=True,
        auxiliary_provider="Dukascopy Public Historical Feed / H1 Bid",
    )


def test_shadow_protocol_window_is_fixed_and_separate() -> None:
    assert SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC - SHADOW_FIRST_FEATURE_BAR_START_UTC == timedelta(
        days=SHADOW_FIXED_WINDOW_DAYS
    )
    assert SHADOW_FIXED_WINDOW_DAYS == 180
    assert SHADOW_MINIMUM_EXACT_HOUR_OUTCOMES == 2500
    assert SHADOW_EARLIEST_FINAL_SCORE_UTC == SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC + timedelta(
        hours=2
    )


def test_shadow_features_equal_confirmed_subset_of_original_family() -> None:
    silver = _silver_frame("2026-08-12T00:00:00Z", 36)
    xpt = _market_frame("2026-08-12T00:00:00Z", 36, 47000.0, 7.0)
    xpd = _market_frame("2026-08-12T00:00:00Z", 36, 36000.0, 5.0)

    original = PlatinumPalladiumCrossAssetFeatures(
        xpt,
        xpd,
        ExactTimestampAligner(),
        ColumnConfig(),
    ).transform(silver)
    newest = silver.iloc[[-1]].copy(deep=True)
    shadow = ConfirmedPreciousMetalsShadowFeatures().transform(newest, xpt, xpd)

    assert ConfirmedPreciousMetalsShadowFeatures().feature_names == CANDIDATE_FEATURES
    for feature in CANDIDATE_FEATURES:
        expected = float(original.loc[original.index[-1], feature])
        actual = float(shadow.iloc[0][feature])
        assert np.isclose(actual, expected, rtol=0.0, atol=1e-14, equal_nan=True), feature


def test_shadow_features_preserve_missing_exact_hour_without_fill() -> None:
    silver = _silver_frame("2026-08-12T00:00:00Z", 30)
    newest = silver.iloc[[-1]].copy(deep=True)
    target_ts = newest["timestamp_utc"].iloc[0]
    xpt = _market_frame("2026-08-12T00:00:00Z", 30, 47000.0, 7.0)
    xpd = _market_frame("2026-08-12T00:00:00Z", 30, 36000.0, 5.0)
    xpt = xpt.loc[xpt["timestamp_utc"].ne(target_ts)].reset_index(drop=True)

    result = ConfirmedPreciousMetalsShadowFeatures().transform(newest, xpt, xpd)

    assert pd.isna(result.iloc[0]["xpt_candle_range_pct"])
    assert pd.isna(result.iloc[0]["xpt_candle_body_pct"])
    assert pd.isna(result.iloc[0]["xpt_log_return_1h"])
    assert np.isfinite(float(result.iloc[0]["xpd_log_return_1h"]))


def test_shadow_repository_is_append_only_and_idempotent(tmp_path) -> None:
    repository = SQLiteShadowRepository(tmp_path / "shadow.sqlite3")
    ts = SHADOW_FIRST_FEATURE_BAR_START_UTC
    snapshot = _snapshot(ts)

    assert repository.put_prediction(snapshot) is True
    assert repository.put_prediction(snapshot) is False
    assert repository.prediction_count() == 1
    assert repository.has_prediction(ts) is True
    assert repository.has_prediction_for_target(ts + timedelta(hours=1)) is True

    outcome = ShadowOutcome(
        target_bar_start_utc=ts + timedelta(hours=1),
        observed_at_utc=ts + timedelta(hours=2, minutes=8),
        actual_close_usd_per_kg=2210.0,
        source_provider="GoldAPI",
        quality_flag="PROVIDER_H1",
    )
    assert repository.put_outcome(outcome) is True
    assert repository.put_outcome(outcome) is False
    assert repository.outcome_count() == 1


class _OneBarRepository:
    def __init__(self, bar: HourlySilverBar) -> None:
        self._bar = bar

    def recent_bars(self, limit: int = 500):
        return [self._bar]


class _NeverSource:
    def fetch_hourly(self, instrument, start_utc, end_utc):
        raise AssertionError("Auxiliary source must not be called after decision evidence is stale.")


class _NeverEngine:
    def predict(self, *args, **kwargs):
        raise AssertionError("Engine must not run after the target close can already be known.")

    def status(self):
        return {}


def test_shadow_service_refuses_retroactive_prediction_after_target_close(tmp_path) -> None:
    ts = SHADOW_FIRST_FEATURE_BAR_START_UTC
    bar = HourlySilverBar(
        timestamp_utc=ts,
        open_usd_per_kg=2200.0,
        high_usd_per_kg=2210.0,
        low_usd_per_kg=2190.0,
        close_usd_per_kg=2205.0,
        minute_count=60,
        quality_flag="TEST",
        source_provider="GoldAPI",
        source_symbol="XAG",
        market_type="provider_h1",
    )
    shadow_repository = SQLiteShadowRepository(tmp_path / "shadow.sqlite3")
    service = Shadow62Service(
        _OneBarRepository(bar),
        shadow_repository,
        _NeverEngine(),
        _NeverSource(),
    )

    result = service.run_once(now_utc=ts + timedelta(hours=2))

    assert result["status"] == "MISSED_REALTIME_DECISION_WINDOW"
    assert result["performance_metrics_computed"] is False
    assert shadow_repository.prediction_count() == 0


class _NoopService:
    def run_once(self):
        return None


def test_shadow_scheduler_runs_at_fixed_delay_after_hour() -> None:
    scheduler = Shadow62Scheduler(_NoopService(), delay_minutes=8)
    before = datetime(2026, 8, 14, 12, 7, 0, tzinfo=timezone.utc)
    after = datetime(2026, 8, 14, 12, 9, 0, tzinfo=timezone.utc)

    assert scheduler.seconds_until_next_run(before) == 60.0
    assert scheduler.seconds_until_next_run(after) == 59 * 60.0


def test_shadow_runtime_is_disabled_by_default_and_prediction_values_are_sealed(tmp_path) -> None:
    app = FastAPI()
    app.state.settings = LiveSettings(
        repository_root=tmp_path,
        shadow62_database_path=tmp_path / "shadow.sqlite3",
    )
    app.state.repository = object()

    installed = install_shadow62_runtime(app)
    paths = {route.path for route in installed.routes}

    assert installed is app
    assert installed.state.shadow62_engine is None
    assert installed.state.shadow62_service is None
    assert installed.state.shadow62_scheduler is None
    assert "/api/v1/research/shadow62/status" in paths
    assert "/api/v1/research/shadow62/latest" not in paths
