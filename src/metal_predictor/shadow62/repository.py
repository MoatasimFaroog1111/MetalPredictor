from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from metal_predictor.shadow62.contracts import ShadowForecastSnapshot, ShadowOutcome


class SQLiteShadowRepository:
    """Append-only persistence for shadow predictions and realized closes.

    No scoring, aggregate error, directional-accuracy, or model-selection query exists
    here. The repository records immutable evidence only; the sealed final scorer is a
    separate future concern.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_predictions (
                    feature_timestamp_utc TEXT PRIMARY KEY,
                    decision_time_utc TEXT NOT NULL,
                    target_bar_start_utc TEXT NOT NULL UNIQUE,
                    target_close_available_utc TEXT NOT NULL,
                    materialized_at_utc TEXT NOT NULL,
                    reference_close_usd_per_kg REAL NOT NULL,
                    baseline_model TEXT NOT NULL,
                    baseline_model_sha256 TEXT NOT NULL,
                    baseline_log_return_1h REAL NOT NULL,
                    baseline_predicted_price_usd_per_kg REAL NOT NULL,
                    candidate_id TEXT NOT NULL,
                    candidate_model TEXT NOT NULL,
                    candidate_model_sha256 TEXT NOT NULL,
                    candidate_log_return_1h REAL NOT NULL,
                    candidate_predicted_price_usd_per_kg REAL NOT NULL,
                    xpt_exact_current INTEGER NOT NULL CHECK (xpt_exact_current IN (0,1)),
                    xpd_exact_current INTEGER NOT NULL CHECK (xpd_exact_current IN (0,1)),
                    auxiliary_provider TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shadow_outcomes (
                    target_bar_start_utc TEXT PRIMARY KEY,
                    observed_at_utc TEXT NOT NULL,
                    actual_close_usd_per_kg REAL NOT NULL,
                    source_provider TEXT NOT NULL,
                    quality_flag TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_shadow_prediction_materialized
                ON shadow_predictions(materialized_at_utc);
                """
            )

    def put_prediction(self, snapshot: ShadowForecastSnapshot) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO shadow_predictions (
                    feature_timestamp_utc,
                    decision_time_utc,
                    target_bar_start_utc,
                    target_close_available_utc,
                    materialized_at_utc,
                    reference_close_usd_per_kg,
                    baseline_model,
                    baseline_model_sha256,
                    baseline_log_return_1h,
                    baseline_predicted_price_usd_per_kg,
                    candidate_id,
                    candidate_model,
                    candidate_model_sha256,
                    candidate_log_return_1h,
                    candidate_predicted_price_usd_per_kg,
                    xpt_exact_current,
                    xpd_exact_current,
                    auxiliary_provider
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.feature_timestamp_utc.astimezone(timezone.utc).isoformat(),
                    snapshot.decision_time_utc.astimezone(timezone.utc).isoformat(),
                    snapshot.target_bar_start_utc.astimezone(timezone.utc).isoformat(),
                    snapshot.target_close_available_utc.astimezone(timezone.utc).isoformat(),
                    snapshot.materialized_at_utc.astimezone(timezone.utc).isoformat(),
                    float(snapshot.reference_close_usd_per_kg),
                    snapshot.baseline_model,
                    snapshot.baseline_model_sha256,
                    float(snapshot.baseline_log_return_1h),
                    float(snapshot.baseline_predicted_price_usd_per_kg),
                    snapshot.candidate_id,
                    snapshot.candidate_model,
                    snapshot.candidate_model_sha256,
                    float(snapshot.candidate_log_return_1h),
                    float(snapshot.candidate_predicted_price_usd_per_kg),
                    int(snapshot.xpt_exact_current),
                    int(snapshot.xpd_exact_current),
                    snapshot.auxiliary_provider,
                ),
            )
            return cursor.rowcount == 1

    def put_outcome(self, outcome: ShadowOutcome) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO shadow_outcomes (
                    target_bar_start_utc,
                    observed_at_utc,
                    actual_close_usd_per_kg,
                    source_provider,
                    quality_flag
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    outcome.target_bar_start_utc.astimezone(timezone.utc).isoformat(),
                    outcome.observed_at_utc.astimezone(timezone.utc).isoformat(),
                    float(outcome.actual_close_usd_per_kg),
                    outcome.source_provider,
                    outcome.quality_flag,
                ),
            )
            return cursor.rowcount == 1

    def latest_prediction(self) -> ShadowForecastSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM shadow_predictions
                ORDER BY feature_timestamp_utc DESC LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return ShadowForecastSnapshot(
            feature_timestamp_utc=datetime.fromisoformat(row["feature_timestamp_utc"]),
            materialized_at_utc=datetime.fromisoformat(row["materialized_at_utc"]),
            reference_close_usd_per_kg=float(row["reference_close_usd_per_kg"]),
            baseline_model=str(row["baseline_model"]),
            baseline_model_sha256=str(row["baseline_model_sha256"]),
            baseline_log_return_1h=float(row["baseline_log_return_1h"]),
            baseline_predicted_price_usd_per_kg=float(row["baseline_predicted_price_usd_per_kg"]),
            candidate_id=str(row["candidate_id"]),
            candidate_model=str(row["candidate_model"]),
            candidate_model_sha256=str(row["candidate_model_sha256"]),
            candidate_log_return_1h=float(row["candidate_log_return_1h"]),
            candidate_predicted_price_usd_per_kg=float(row["candidate_predicted_price_usd_per_kg"]),
            xpt_exact_current=bool(row["xpt_exact_current"]),
            xpd_exact_current=bool(row["xpd_exact_current"]),
            auxiliary_provider=str(row["auxiliary_provider"]),
        )

    def prediction_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0])

    def outcome_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM shadow_outcomes").fetchone()[0])

    def has_prediction(self, feature_timestamp_utc: datetime) -> bool:
        key = feature_timestamp_utc.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM shadow_predictions WHERE feature_timestamp_utc=? LIMIT 1",
                (key,),
            ).fetchone() is not None

    def has_prediction_for_target(self, target_bar_start_utc: datetime) -> bool:
        key = target_bar_start_utc.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM shadow_predictions WHERE target_bar_start_utc=? LIMIT 1",
                (key,),
            ).fetchone() is not None
