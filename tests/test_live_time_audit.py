from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from metal_predictor.live.contracts import ForecastSnapshot
from metal_predictor.live.repository import SQLiteForecastRepository


def _snapshot(materialized_at: datetime) -> ForecastSnapshot:
    feature = datetime(2026, 8, 12, 6, 0, tzinfo=timezone.utc)
    return ForecastSnapshot(
        feature_timestamp_utc=feature,
        decision_time_utc=datetime(2026, 8, 12, 7, 0, tzinfo=timezone.utc),
        current_price_usd_per_kg=2123.40,
        baseline_model="ridge_alpha_100",
        baseline_log_return_1h=-0.0015,
        baseline_predicted_price_usd_per_kg=2120.20,
        baseline_direction="DOWN",
        challenger_model="ridge_alpha_10",
        challenger_log_return_1h=-0.0018,
        challenger_predicted_price_usd_per_kg=2119.60,
        challenger_direction="DOWN",
        data_quality="PROVIDER_AGGREGATED_H1",
        source_provider="GoldAPI",
        source_compatible_with_training=False,
        materialized_at_utc=materialized_at,
    )


def test_forecast_serialization_distinguishes_model_clock_from_publication_time() -> None:
    published = datetime(2026, 8, 12, 7, 5, 12, tzinfo=timezone.utc)
    data = _snapshot(published).as_dict()

    assert data["decision_time_utc"] == "2026-08-12T07:00:00+00:00"
    assert data["model_clock_decision_time_utc"] == "2026-08-12T07:00:00+00:00"
    assert data["materialized_at_utc"] == "2026-08-12T07:05:12+00:00"
    assert data["feature_timestamp_saudi"] == "2026-08-12T09:00:00+03:00"
    assert data["decision_time_saudi"] == "2026-08-12T10:00:00+03:00"
    assert data["materialized_at_saudi"] == "2026-08-12T10:05:12+03:00"
    assert data["display_timezone"] == "Asia/Riyadh"
    assert data["decision_time_semantics"] == "MODEL_CLOCK_HOUR_BOUNDARY"
    assert data["materialized_at_semantics"] == "ACTUAL_FORECAST_PUBLICATION_TIME"
    assert data["publication_delay_seconds"] == 312.0


def test_repository_round_trip_preserves_materialization_time(tmp_path: Path) -> None:
    published = datetime(2026, 8, 12, 7, 5, 12, 345678, tzinfo=timezone.utc)
    repository = SQLiteForecastRepository(tmp_path / "audit.sqlite3")
    snapshot = _snapshot(published)

    assert repository.put_forecast(snapshot) is True
    restored = repository.latest_forecast()

    assert restored is not None
    assert restored.materialized_at_utc == published
    assert restored.as_dict()["materialized_at_saudi"] == "2026-08-12T10:05:12.345678+03:00"
