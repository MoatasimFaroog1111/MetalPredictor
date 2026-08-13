from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from metal_predictor.live.contracts import ForecastRepository
from metal_predictor.precious_metals.contracts import PALLADIUM, PLATINUM, HistoricalMetalSource
from metal_predictor.shadow62.contracts import (
    SHADOW_FIRST_FEATURE_BAR_START_UTC,
    SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC,
    ShadowOutcome,
    ShadowRepository,
)
from metal_predictor.shadow62.engine import Shadow62InferenceEngine


logger = logging.getLogger(__name__)


class Shadow62Service:
    """Coordinates one real-time research observation without interim scoring."""

    def __init__(
        self,
        live_repository: ForecastRepository,
        shadow_repository: ShadowRepository,
        engine: Shadow62InferenceEngine,
        auxiliary_source: HistoricalMetalSource,
    ) -> None:
        self._live_repository = live_repository
        self._shadow_repository = shadow_repository
        self._engine = engine
        self._source = auxiliary_source

    def run_once(self, now_utc: datetime | None = None) -> dict[str, object]:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        bars = self._live_repository.recent_bars(limit=5000)
        if not bars:
            return self._result("WAITING_FOR_SILVER_BAR")

        latest_bar = bars[-1]
        feature_timestamp = latest_bar.timestamp_utc.astimezone(timezone.utc)
        outcome_created = False

        if self._shadow_repository.has_prediction_for_target(feature_timestamp):
            outcome_created = self._shadow_repository.put_outcome(
                ShadowOutcome(
                    target_bar_start_utc=feature_timestamp,
                    observed_at_utc=now,
                    actual_close_usd_per_kg=latest_bar.close_usd_per_kg,
                    source_provider=latest_bar.source_provider,
                    quality_flag=latest_bar.quality_flag,
                )
            )

        if feature_timestamp < SHADOW_FIRST_FEATURE_BAR_START_UTC:
            return self._result("WAITING_FOR_HOLDOUT_START", outcome_created=outcome_created)
        if feature_timestamp >= SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC:
            return self._result("HOLDOUT_WINDOW_COMPLETE", outcome_created=outcome_created)
        if self._shadow_repository.has_prediction(feature_timestamp):
            return self._result("PREDICTION_ALREADY_RECORDED", outcome_created=outcome_created)

        target_close_available = feature_timestamp + timedelta(hours=2)
        if now >= target_close_available:
            logger.warning(
                "Shadow observation skipped because the target close could already be known: %s",
                feature_timestamp.isoformat(),
            )
            return self._result("MISSED_REALTIME_DECISION_WINDOW", outcome_created=outcome_created)

        auxiliary_start = feature_timestamp - timedelta(hours=24)
        platinum = self._source.fetch_hourly(PLATINUM, auxiliary_start, feature_timestamp)
        palladium = self._source.fetch_hourly(PALLADIUM, auxiliary_start, feature_timestamp)
        snapshot = self._engine.predict(
            bars,
            platinum,
            palladium,
            materialized_at_utc=now,
        )
        prediction_created = self._shadow_repository.put_prediction(snapshot)
        return {
            **self._result(
                "PREDICTION_RECORDED" if prediction_created else "PREDICTION_ALREADY_RECORDED",
                outcome_created=outcome_created,
            ),
            "feature_timestamp_utc": snapshot.feature_timestamp_utc.isoformat(),
            "materialized_at_utc": snapshot.materialized_at_utc.isoformat(),
            "xpt_exact_current": snapshot.xpt_exact_current,
            "xpd_exact_current": snapshot.xpd_exact_current,
        }

    def status(self) -> dict[str, object]:
        latest = self._shadow_repository.latest_prediction()
        return {
            "component": "xpt-xpd-shadow-62-research",
            "protocol": self._engine.status(),
            "holdout": {
                "first_feature_bar_start_utc": SHADOW_FIRST_FEATURE_BAR_START_UTC.isoformat(),
                "last_feature_bar_start_exclusive_utc": SHADOW_LAST_FEATURE_BAR_START_EXCLUSIVE_UTC.isoformat(),
                "performance_metrics_available": False,
                "interim_scoring_enabled": False,
            },
            "prediction_count": self._shadow_repository.prediction_count(),
            "outcome_count": self._shadow_repository.outcome_count(),
            "latest_feature_timestamp_utc": latest.feature_timestamp_utc.isoformat() if latest else None,
            "latest_materialized_at_utc": latest.materialized_at_utc.isoformat() if latest else None,
            "safety": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
            },
        }

    @staticmethod
    def _result(status: str, *, outcome_created: bool = False) -> dict[str, object]:
        return {
            "status": status,
            "outcome_created": outcome_created,
            "performance_metrics_computed": False,
            "edge_status": "NOT_PROVEN",
            "research_only": True,
        }
