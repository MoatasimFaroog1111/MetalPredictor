from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from metal_predictor.forward_bars.contracts import QuoteSample


class SQLiteMicrostructureQuoteSampleSource:
    """Read-only adapter over the append-only BullionVault microstructure snapshot store."""

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)

    @property
    def database_path(self) -> Path:
        return self._path

    def first_sample_at(
        self,
        *,
        security_id: str,
        currency: str,
    ) -> datetime | None:
        if not self._path.exists():
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT captured_at_utc
                FROM microstructure_snapshots
                WHERE security_id = ? AND currency = ?
                ORDER BY captured_at_utc ASC
                LIMIT 1
                """,
                (security_id, currency),
            ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(str(row["captured_at_utc"])).astimezone(timezone.utc)

    def samples_between(
        self,
        start_utc: datetime,
        end_utc: datetime,
        *,
        security_id: str,
        currency: str,
    ) -> list[QuoteSample]:
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("Sample-window timestamps must be timezone-aware.")
        start = start_utc.astimezone(timezone.utc)
        end = end_utc.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("Sample-window end must be after start.")
        if not self._path.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    source_provider,
                    security_id,
                    currency,
                    captured_at_utc,
                    access_mode,
                    freshness_status,
                    best_bid_usd_per_kg,
                    best_ask_usd_per_kg
                FROM microstructure_snapshots
                WHERE security_id = ?
                  AND currency = ?
                  AND captured_at_utc >= ?
                  AND captured_at_utc < ?
                ORDER BY captured_at_utc ASC, id ASC
                """,
                (
                    security_id,
                    currency,
                    start.isoformat(),
                    end.isoformat(),
                ),
            ).fetchall()
        return [
            QuoteSample(
                source_provider=str(row["source_provider"]),
                security_id=str(row["security_id"]),
                currency=str(row["currency"]),
                captured_at_utc=datetime.fromisoformat(
                    str(row["captured_at_utc"])
                ).astimezone(timezone.utc),
                access_mode=str(row["access_mode"]),
                freshness_status=str(row["freshness_status"]),
                best_bid_usd_per_kg=float(row["best_bid_usd_per_kg"]),
                best_ask_usd_per_kg=float(row["best_ask_usd_per_kg"]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=20.0)
        connection.row_factory = sqlite3.Row
        return connection
