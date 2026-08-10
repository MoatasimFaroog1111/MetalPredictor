from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

import numpy as np

from metal_predictor.live.contracts import ForecastSnapshot, HourlySilverBar


class SQLiteForecastRepository:
    """Small operational store with idempotent writes and revision rejection.

    SQLite is the default zero-setup runtime. The repository boundary allows a
    PostgreSQL implementation to replace it later without changing inference/API code.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS hourly_bars (
                    timestamp_utc TEXT PRIMARY KEY,
                    open_usd_per_kg REAL NOT NULL,
                    high_usd_per_kg REAL NOT NULL,
                    low_usd_per_kg REAL NOT NULL,
                    close_usd_per_kg REAL NOT NULL,
                    minute_count INTEGER NOT NULL,
                    quality_flag TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    source_symbol TEXT NOT NULL,
                    market_type TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS forecasts (
                    feature_timestamp_utc TEXT PRIMARY KEY,
                    decision_time_utc TEXT NOT NULL,
                    current_price_usd_per_kg REAL NOT NULL,
                    baseline_model TEXT NOT NULL,
                    baseline_log_return_1h REAL NOT NULL,
                    baseline_predicted_price_usd_per_kg REAL NOT NULL,
                    baseline_direction TEXT NOT NULL,
                    challenger_model TEXT NOT NULL,
                    challenger_log_return_1h REAL NOT NULL,
                    challenger_predicted_price_usd_per_kg REAL NOT NULL,
                    challenger_direction TEXT NOT NULL,
                    data_quality TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    source_compatible_with_training INTEGER NOT NULL,
                    edge_status TEXT NOT NULL,
                    research_only INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )

    def put_bar(self, bar: HourlySilverBar) -> bool:
        self._validate_bar(bar)
        timestamp = self._iso(bar.timestamp_utc)
        values = (
            float(bar.open_usd_per_kg),
            float(bar.high_usd_per_kg),
            float(bar.low_usd_per_kg),
            float(bar.close_usd_per_kg),
            int(bar.minute_count),
            str(bar.quality_flag),
            str(bar.source_provider),
            str(bar.source_symbol),
            str(bar.market_type),
        )
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM hourly_bars WHERE timestamp_utc = ?", (timestamp,)
            ).fetchone()
            if existing is not None:
                self._assert_bar_same(existing, bar)
                return False
            conn.execute(
                """
                INSERT INTO hourly_bars (
                    timestamp_utc, open_usd_per_kg, high_usd_per_kg,
                    low_usd_per_kg, close_usd_per_kg, minute_count,
                    quality_flag, source_provider, source_symbol, market_type,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, *values, self._now_iso()),
            )
            return True

    def recent_bars(self, limit: int = 500) -> list[HourlySilverBar]:
        limit = self._validated_limit(limit, maximum=5000)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hourly_bars ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_bar(row) for row in reversed(rows)]

    def put_forecast(self, snapshot: ForecastSnapshot) -> bool:
        timestamp = self._iso(snapshot.feature_timestamp_utc)
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM forecasts WHERE feature_timestamp_utc = ?", (timestamp,)
            ).fetchone()
            if existing is not None:
                self._assert_forecast_same(existing, snapshot)
                return False
            conn.execute(
                """
                INSERT INTO forecasts (
                    feature_timestamp_utc, decision_time_utc,
                    current_price_usd_per_kg,
                    baseline_model, baseline_log_return_1h,
                    baseline_predicted_price_usd_per_kg, baseline_direction,
                    challenger_model, challenger_log_return_1h,
                    challenger_predicted_price_usd_per_kg, challenger_direction,
                    data_quality, source_provider, source_compatible_with_training,
                    edge_status, research_only, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    self._iso(snapshot.decision_time_utc),
                    snapshot.current_price_usd_per_kg,
                    snapshot.baseline_model,
                    snapshot.baseline_log_return_1h,
                    snapshot.baseline_predicted_price_usd_per_kg,
                    snapshot.baseline_direction,
                    snapshot.challenger_model,
                    snapshot.challenger_log_return_1h,
                    snapshot.challenger_predicted_price_usd_per_kg,
                    snapshot.challenger_direction,
                    snapshot.data_quality,
                    snapshot.source_provider,
                    int(snapshot.source_compatible_with_training),
                    snapshot.edge_status,
                    int(snapshot.research_only),
                    self._now_iso(),
                ),
            )
            return True

    def latest_forecast(self) -> ForecastSnapshot | None:
        history = self.forecast_history(limit=1)
        return history[0] if history else None

    def forecast_history(self, limit: int = 100) -> list[ForecastSnapshot]:
        limit = self._validated_limit(limit, maximum=1000)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forecasts ORDER BY feature_timestamp_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_forecast(row) for row in rows]

    @staticmethod
    def _validated_limit(limit: int, maximum: int) -> int:
        value = int(limit)
        if not 1 <= value <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")
        return value

    @classmethod
    def _validate_bar(cls, bar: HourlySilverBar) -> None:
        ts = cls._utc(bar.timestamp_utc)
        if ts.minute or ts.second or ts.microsecond:
            raise ValueError("Hourly bar timestamp must be aligned to an exact UTC hour.")
        prices = np.asarray(
            [bar.open_usd_per_kg, bar.high_usd_per_kg, bar.low_usd_per_kg, bar.close_usd_per_kg],
            dtype=float,
        )
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Hourly bar prices must be finite and positive.")
        if bar.high_usd_per_kg < max(bar.open_usd_per_kg, bar.close_usd_per_kg):
            raise ValueError("Hourly high is below open/close.")
        if bar.low_usd_per_kg > min(bar.open_usd_per_kg, bar.close_usd_per_kg):
            raise ValueError("Hourly low is above open/close.")
        if bar.high_usd_per_kg < bar.low_usd_per_kg:
            raise ValueError("Hourly high is below low.")
        if not 1 <= int(bar.minute_count) <= 60:
            raise ValueError("minute_count must be between 1 and 60.")

    @classmethod
    def _assert_bar_same(cls, row: sqlite3.Row, bar: HourlySilverBar) -> None:
        numeric = {
            "open_usd_per_kg": bar.open_usd_per_kg,
            "high_usd_per_kg": bar.high_usd_per_kg,
            "low_usd_per_kg": bar.low_usd_per_kg,
            "close_usd_per_kg": bar.close_usd_per_kg,
            "minute_count": bar.minute_count,
        }
        for column, expected in numeric.items():
            if not np.isclose(float(row[column]), float(expected), rtol=1e-12, atol=1e-9):
                raise ValueError(f"LIVE_BAR_REVISION_CONFLICT {row['timestamp_utc']} {column}")
        for column, expected in {
            "quality_flag": bar.quality_flag,
            "source_provider": bar.source_provider,
            "source_symbol": bar.source_symbol,
            "market_type": bar.market_type,
        }.items():
            if str(row[column]) != str(expected):
                raise ValueError(f"LIVE_BAR_REVISION_CONFLICT {row['timestamp_utc']} {column}")

    @classmethod
    def _assert_forecast_same(cls, row: sqlite3.Row, snapshot: ForecastSnapshot) -> None:
        expected = snapshot.as_dict()
        numeric_columns = (
            "current_price_usd_per_kg",
            "baseline_log_return_1h",
            "baseline_predicted_price_usd_per_kg",
            "challenger_log_return_1h",
            "challenger_predicted_price_usd_per_kg",
        )
        for column in numeric_columns:
            if not np.isclose(float(row[column]), float(expected[column]), rtol=1e-12, atol=1e-9):
                raise ValueError(
                    f"LIVE_FORECAST_REVISION_CONFLICT {row['feature_timestamp_utc']} {column}"
                )
        text_columns = (
            "baseline_model", "baseline_direction", "challenger_model",
            "challenger_direction", "data_quality", "source_provider", "edge_status",
        )
        for column in text_columns:
            if str(row[column]) != str(expected[column]):
                raise ValueError(
                    f"LIVE_FORECAST_REVISION_CONFLICT {row['feature_timestamp_utc']} {column}"
                )

    @classmethod
    def _row_to_bar(cls, row: sqlite3.Row) -> HourlySilverBar:
        return HourlySilverBar(
            timestamp_utc=cls._utc(datetime.fromisoformat(row["timestamp_utc"])),
            open_usd_per_kg=float(row["open_usd_per_kg"]),
            high_usd_per_kg=float(row["high_usd_per_kg"]),
            low_usd_per_kg=float(row["low_usd_per_kg"]),
            close_usd_per_kg=float(row["close_usd_per_kg"]),
            minute_count=int(row["minute_count"]),
            quality_flag=str(row["quality_flag"]),
            source_provider=str(row["source_provider"]),
            source_symbol=str(row["source_symbol"]),
            market_type=str(row["market_type"]),
        )

    @classmethod
    def _row_to_forecast(cls, row: sqlite3.Row) -> ForecastSnapshot:
        return ForecastSnapshot(
            feature_timestamp_utc=cls._utc(datetime.fromisoformat(row["feature_timestamp_utc"])),
            decision_time_utc=cls._utc(datetime.fromisoformat(row["decision_time_utc"])),
            current_price_usd_per_kg=float(row["current_price_usd_per_kg"]),
            baseline_model=str(row["baseline_model"]),
            baseline_log_return_1h=float(row["baseline_log_return_1h"]),
            baseline_predicted_price_usd_per_kg=float(row["baseline_predicted_price_usd_per_kg"]),
            baseline_direction=str(row["baseline_direction"]),
            challenger_model=str(row["challenger_model"]),
            challenger_log_return_1h=float(row["challenger_log_return_1h"]),
            challenger_predicted_price_usd_per_kg=float(row["challenger_predicted_price_usd_per_kg"]),
            challenger_direction=str(row["challenger_direction"]),
            data_quality=str(row["data_quality"]),
            source_provider=str(row["source_provider"]),
            source_compatible_with_training=bool(row["source_compatible_with_training"]),
            edge_status=str(row["edge_status"]),
            research_only=bool(row["research_only"]),
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @classmethod
    def _iso(cls, value: datetime) -> str:
        return cls._utc(value).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
