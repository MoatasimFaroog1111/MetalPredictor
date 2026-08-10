from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class LedgerAppendResult:
    existing_rows: int
    appended_rows: int
    final_rows: int
    final_hash: str


class CsvHashChainLedger:
    """Append-only CSV ledger with deterministic SHA-256 row chaining.

    Existing rows are never rewritten semantically. Every collection run verifies the
    complete stored chain before appending. A source revision to an already-recorded
    timestamp is rejected by the caller rather than silently mutating the holdout.
    """

    def __init__(
        self,
        path: Path,
        data_columns: tuple[str, ...],
        timestamp_column: str,
    ) -> None:
        if timestamp_column not in data_columns:
            raise ValueError("timestamp_column must be one of data_columns.")
        self.path = path
        self.data_columns = data_columns
        self.timestamp_column = timestamp_column
        self.columns = (*data_columns, "previous_row_hash", "row_hash")

    def read_verified(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=self.columns)
        frame = pd.read_csv(self.path, dtype="string", keep_default_na=False)
        if tuple(frame.columns) != self.columns:
            raise ValueError(
                f"Ledger schema mismatch at {self.path}: {tuple(frame.columns)} != {self.columns}"
            )
        previous = GENESIS_HASH
        last_timestamp: pd.Timestamp | None = None
        for position, row in frame.iterrows():
            if row["previous_row_hash"] != previous:
                raise ValueError(f"Ledger chain predecessor mismatch at row {position}.")
            expected = self._row_hash(previous, row)
            if row["row_hash"] != expected:
                raise ValueError(f"Ledger row hash mismatch at row {position}.")
            timestamp = pd.Timestamp(row[self.timestamp_column])
            if timestamp.tzinfo is None:
                raise ValueError("Ledger timestamps must be timezone-aware.")
            if last_timestamp is not None and timestamp <= last_timestamp:
                raise ValueError("Ledger timestamps must be strictly increasing.")
            last_timestamp = timestamp
            previous = row["row_hash"]
        return frame

    def append(self, rows: pd.DataFrame) -> LedgerAppendResult:
        existing = self.read_verified()
        if rows.empty:
            return LedgerAppendResult(
                existing_rows=len(existing),
                appended_rows=0,
                final_rows=len(existing),
                final_hash=(existing["row_hash"].iloc[-1] if len(existing) else GENESIS_HASH),
            )
        missing = set(self.data_columns).difference(rows.columns)
        if missing:
            raise ValueError(f"Rows missing ledger columns: {sorted(missing)}")
        clean = rows.loc[:, self.data_columns].copy()
        timestamps = pd.to_datetime(clean[self.timestamp_column], utc=True, errors="raise")
        clean[self.timestamp_column] = timestamps.map(lambda value: value.isoformat())
        clean = clean.sort_values(self.timestamp_column).reset_index(drop=True)
        if clean[self.timestamp_column].duplicated().any():
            raise ValueError("Cannot append duplicate ledger timestamps.")
        if len(existing):
            last_existing = pd.Timestamp(existing[self.timestamp_column].iloc[-1])
            if pd.Timestamp(clean[self.timestamp_column].iloc[0]) <= last_existing:
                raise ValueError("Append rows must be strictly later than the ledger tail.")

        previous = existing["row_hash"].iloc[-1] if len(existing) else GENESIS_HASH
        encoded_rows: list[dict[str, str]] = []
        for _, row in clean.iterrows():
            data = {column: self._canonical_scalar(row[column]) for column in self.data_columns}
            material = {**data, "previous_row_hash": previous}
            row_hash = self._hash_material(material)
            encoded = {**data, "previous_row_hash": previous, "row_hash": row_hash}
            encoded_rows.append(encoded)
            previous = row_hash
        appended = pd.DataFrame(encoded_rows, columns=self.columns)
        combined = pd.concat([existing, appended], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self.path, index=False, lineterminator="\n")
        verified = self.read_verified()
        if len(verified) != len(combined):
            raise AssertionError("Ledger verification row count changed after write.")
        return LedgerAppendResult(
            existing_rows=len(existing),
            appended_rows=len(appended),
            final_rows=len(combined),
            final_hash=previous,
        )

    def _row_hash(self, previous: str, row: pd.Series) -> str:
        material = {
            column: self._canonical_scalar(row[column])
            for column in self.data_columns
        }
        material["previous_row_hash"] = previous
        return self._hash_material(material)

    @staticmethod
    def _canonical_scalar(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return ""
            return format(float(value), ".17g")
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if pd.isna(value):
            return ""
        return str(value)

    @staticmethod
    def _hash_material(material: dict[str, str]) -> str:
        canonical = json.dumps(
            material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
