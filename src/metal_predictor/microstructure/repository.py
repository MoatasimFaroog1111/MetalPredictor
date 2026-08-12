from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from metal_predictor.live.quote_contracts import MarketDepthLevel
from metal_predictor.microstructure.contracts import (
    MicrostructureFeatureVector,
    MicrostructureResearchRecord,
    MicrostructureSnapshot,
)


class SQLiteMicrostructureRepository:
    """Append-only persistence for raw order books and derived research features."""

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._path

    def append(
        self,
        snapshot: MicrostructureSnapshot,
        features: MicrostructureFeatureVector,
    ) -> MicrostructureResearchRecord:
        if features.captured_at_utc != snapshot.captured_at_utc:
            raise ValueError("Snapshot and feature timestamps must match.")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO microstructure_snapshots (
                    source_provider, security_id, currency, captured_at_utc,
                    access_mode, freshness_status, best_bid_usd_per_kg,
                    best_ask_usd_per_kg, best_bid_quantity_kg, best_ask_quantity_kg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.source_provider,
                    snapshot.security_id,
                    snapshot.currency,
                    snapshot.captured_at_utc.astimezone(timezone.utc).isoformat(),
                    snapshot.access_mode,
                    snapshot.freshness_status,
                    snapshot.best_bid_usd_per_kg,
                    snapshot.best_ask_usd_per_kg,
                    snapshot.best_bid_quantity_kg,
                    snapshot.best_ask_quantity_kg,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            self._insert_levels(connection, snapshot_id, "BID", snapshot.bid_depth)
            self._insert_levels(connection, snapshot_id, "ASK", snapshot.ask_depth)
            connection.execute(
                """
                INSERT INTO microstructure_features (
                    snapshot_id, feature_version, feature_json, created_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    features.feature_version,
                    json.dumps(
                        {name: float(value) for name, value in features.values.items()},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        return MicrostructureResearchRecord(snapshot_id, snapshot, features)

    def latest_snapshot(self) -> MicrostructureSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM microstructure_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self._snapshot_from_row(connection, row)

    def latest_record(self) -> MicrostructureResearchRecord | None:
        records = self.recent_records(limit=1)
        return records[-1] if records else None

    def recent_records(self, limit: int = 100) -> list[MicrostructureResearchRecord]:
        if not 1 <= int(limit) <= 10_000:
            raise ValueError("limit must be between 1 and 10000.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.*, f.feature_version, f.feature_json
                FROM microstructure_snapshots AS s
                JOIN microstructure_features AS f ON f.snapshot_id = s.id
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            records: list[MicrostructureResearchRecord] = []
            for row in reversed(rows):
                snapshot = self._snapshot_from_row(connection, row)
                values = json.loads(str(row["feature_json"]))
                feature_vector = MicrostructureFeatureVector(
                    captured_at_utc=snapshot.captured_at_utc,
                    feature_version=str(row["feature_version"]),
                    values={name: float(value) for name, value in values.items()},
                )
                records.append(
                    MicrostructureResearchRecord(
                        snapshot_id=int(row["id"]),
                        snapshot=snapshot,
                        features=feature_vector,
                    )
                )
            return records

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM microstructure_snapshots"
            ).fetchone()
            return int(row["n"])

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS microstructure_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_provider TEXT NOT NULL,
                    security_id TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    access_mode TEXT NOT NULL,
                    freshness_status TEXT NOT NULL,
                    best_bid_usd_per_kg REAL NOT NULL,
                    best_ask_usd_per_kg REAL NOT NULL,
                    best_bid_quantity_kg REAL NOT NULL,
                    best_ask_quantity_kg REAL NOT NULL,
                    UNIQUE(source_provider, security_id, currency, captured_at_utc)
                );

                CREATE TABLE IF NOT EXISTS microstructure_levels (
                    snapshot_id INTEGER NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BID', 'ASK')),
                    level_index INTEGER NOT NULL CHECK(level_index >= 1),
                    price_usd_per_kg REAL NOT NULL,
                    quantity_kg REAL NOT NULL,
                    PRIMARY KEY(snapshot_id, side, level_index),
                    FOREIGN KEY(snapshot_id) REFERENCES microstructure_snapshots(id)
                        ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS microstructure_features (
                    snapshot_id INTEGER PRIMARY KEY,
                    feature_version TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(snapshot_id) REFERENCES microstructure_snapshots(id)
                        ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_microstructure_time
                    ON microstructure_snapshots(captured_at_utc);
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=20.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _insert_levels(
        connection: sqlite3.Connection,
        snapshot_id: int,
        side: str,
        levels: tuple[MarketDepthLevel, ...],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO microstructure_levels (
                snapshot_id, side, level_index, price_usd_per_kg, quantity_kg
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    side,
                    index,
                    level.price_usd_per_kg,
                    level.quantity_kg,
                )
                for index, level in enumerate(levels, start=1)
            ],
        )

    def _snapshot_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> MicrostructureSnapshot:
        bid_depth = self._levels_for(connection, int(row["id"]), "BID")
        ask_depth = self._levels_for(connection, int(row["id"]), "ASK")
        return MicrostructureSnapshot(
            source_provider=str(row["source_provider"]),
            security_id=str(row["security_id"]),
            currency=str(row["currency"]),
            captured_at_utc=datetime.fromisoformat(str(row["captured_at_utc"])),
            access_mode=str(row["access_mode"]),
            freshness_status=str(row["freshness_status"]),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

    @staticmethod
    def _levels_for(
        connection: sqlite3.Connection,
        snapshot_id: int,
        side: str,
    ) -> tuple[MarketDepthLevel, ...]:
        rows = connection.execute(
            """
            SELECT price_usd_per_kg, quantity_kg
            FROM microstructure_levels
            WHERE snapshot_id = ? AND side = ?
            ORDER BY level_index ASC
            """,
            (snapshot_id, side),
        ).fetchall()
        return tuple(
            MarketDepthLevel(
                price_usd_per_kg=float(row["price_usd_per_kg"]),
                quantity_kg=float(row["quantity_kg"]),
            )
            for row in rows
        )
