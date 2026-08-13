from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from metal_predictor.forward_bars.contracts import ForwardBar
from metal_predictor.forward_bars.serialization import bar_from_payload


class SQLiteForwardBarRepository:
    """Append-only bucket assessment store with immutable serialized bars."""

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._path

    def has_assessment(self, horizon_key: str, bucket_start_utc: datetime) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM forward_bar_assessments WHERE horizon_key=? AND bucket_start_utc=?",
                (horizon_key, self._iso(bucket_start_utc)),
            ).fetchone()
        return row is not None

    def latest_assessed_end(self, horizon_key: str) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT bucket_end_utc FROM forward_bar_assessments
                WHERE horizon_key=? ORDER BY bucket_end_utc DESC LIMIT 1
                """,
                (horizon_key,),
            ).fetchone()
        return (
            datetime.fromisoformat(str(row["bucket_end_utc"])).astimezone(timezone.utc)
            if row is not None
            else None
        )

    def append_bar(self, bar: ForwardBar) -> bool:
        return self._append(
            horizon_key=bar.horizon_key,
            interval_seconds=bar.interval_seconds,
            bucket_start_utc=bar.bucket_start_utc,
            bucket_end_utc=bar.bucket_end_utc,
            status="BAR_MATERIALIZED",
            reason=None,
            snapshot_count=bar.snapshot_count,
            bar_json=json.dumps(bar.as_dict(), sort_keys=True, separators=(",", ":")),
        )

    def append_gap(
        self,
        *,
        horizon_key: str,
        interval_seconds: int,
        bucket_start_utc: datetime,
        bucket_end_utc: datetime,
        reason: str,
        snapshot_count: int,
    ) -> bool:
        return self._append(
            horizon_key=horizon_key,
            interval_seconds=interval_seconds,
            bucket_start_utc=bucket_start_utc,
            bucket_end_utc=bucket_end_utc,
            status="GAP_RECORDED",
            reason=reason,
            snapshot_count=snapshot_count,
            bar_json=None,
        )

    def latest_bar(self, horizon_key: str) -> ForwardBar | None:
        bars = self.history(horizon_key, limit=1)
        return bars[-1] if bars else None

    def history(self, horizon_key: str, limit: int = 100) -> list[ForwardBar]:
        if not 1 <= int(limit) <= 5000:
            raise ValueError("limit must be between 1 and 5000.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT bar_json FROM forward_bar_assessments
                WHERE horizon_key=? AND status='BAR_MATERIALIZED'
                ORDER BY bucket_start_utc DESC LIMIT ?
                """,
                (horizon_key, int(limit)),
            ).fetchall()
        return [
            bar_from_payload(json.loads(str(row["bar_json"])))
            for row in reversed(rows)
        ]

    def status_snapshot(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT horizon_key,
                       SUM(status='BAR_MATERIALIZED') AS bar_count,
                       SUM(status='GAP_RECORDED') AS gap_count,
                       MAX(bucket_end_utc) AS latest_assessed_end_utc
                FROM forward_bar_assessments GROUP BY horizon_key
                """
            ).fetchall()
        result: dict[str, object] = {}
        for row in rows:
            key = str(row["horizon_key"])
            latest = self.latest_bar(key)
            result[key] = {
                "bar_count": int(row["bar_count"] or 0),
                "gap_count": int(row["gap_count"] or 0),
                "latest_assessed_end_utc": row["latest_assessed_end_utc"],
                "latest_bar": (
                    {
                        "bucket_start_utc": latest.bucket_start_utc.isoformat(),
                        "bucket_end_utc": latest.bucket_end_utc.isoformat(),
                        "snapshot_count": latest.snapshot_count,
                        "expected_snapshot_count": latest.expected_snapshot_count,
                        "coverage_ratio": latest.coverage_ratio,
                        "quality_status": latest.quality_status,
                        "access_mode_counts": dict(latest.access_mode_counts),
                        "freshness_status_counts": dict(latest.freshness_status_counts),
                    }
                    if latest
                    else None
                ),
            }
        return result

    def _append(
        self,
        *,
        horizon_key: str,
        interval_seconds: int,
        bucket_start_utc: datetime,
        bucket_end_utc: datetime,
        status: str,
        reason: str | None,
        snapshot_count: int,
        bar_json: str | None,
    ) -> bool:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO forward_bar_assessments
                    (horizon_key, interval_seconds, bucket_start_utc, bucket_end_utc,
                     status, reason, snapshot_count, bar_json, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        horizon_key,
                        int(interval_seconds),
                        self._iso(bucket_start_utc),
                        self._iso(bucket_end_utc),
                        status,
                        reason,
                        int(snapshot_count),
                        bar_json,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            connection.commit()
        return True

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS forward_bar_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    horizon_key TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL CHECK(interval_seconds > 0),
                    bucket_start_utc TEXT NOT NULL,
                    bucket_end_utc TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('BAR_MATERIALIZED','GAP_RECORDED')),
                    reason TEXT,
                    snapshot_count INTEGER NOT NULL CHECK(snapshot_count >= 0),
                    bar_json TEXT,
                    created_at_utc TEXT NOT NULL,
                    UNIQUE(horizon_key, bucket_start_utc),
                    CHECK((status='BAR_MATERIALIZED' AND bar_json IS NOT NULL)
                       OR (status='GAP_RECORDED' AND bar_json IS NULL))
                );
                CREATE INDEX IF NOT EXISTS idx_forward_bar_end
                    ON forward_bar_assessments(horizon_key, bucket_end_utc);
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=20.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _iso(value: datetime) -> str:
        if value.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return value.astimezone(timezone.utc).isoformat()
