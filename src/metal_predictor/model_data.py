from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_names: tuple[str, ...]
    target_name: str = "target_log_return_1h"
    timestamp_name: str = "timestamp_utc"
    target_timestamp_name: str = "target_timestamp_utc"
    current_close_name: str = "close_usd_per_kg"
    target_close_name: str = "target_close_usd_per_kg"

    @property
    def development(self) -> pd.DataFrame:
        return pd.concat([self.train, self.validation], ignore_index=True).sort_values(
            self.timestamp_name
        ).reset_index(drop=True)


class PreparedDatasetLoader:
    """Loads immutable artifacts emitted by the training-data pipeline."""

    def load(self, processed_dir: Path) -> PreparedDataset:
        manifest_path = processed_dir / "feature_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = tuple(manifest.get("features", ()))
        if not features:
            raise ValueError("Feature manifest contains no features.")
        dataset = PreparedDataset(
            train=self._normalize(pd.read_parquet(processed_dir / "train.parquet")),
            validation=self._normalize(pd.read_parquet(processed_dir / "validation.parquet")),
            test=self._normalize(pd.read_parquet(processed_dir / "test.parquet")),
            feature_names=features,
        )
        self._validate(dataset)
        return dataset

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        for name in ("timestamp_utc", "target_timestamp_utc"):
            out[name] = pd.to_datetime(out[name], utc=True, errors="raise")
        return out.sort_values("timestamp_utc").reset_index(drop=True)

    @staticmethod
    def _validate(dataset: PreparedDataset) -> None:
        required = set(dataset.feature_names) | {
            dataset.target_name, dataset.timestamp_name, dataset.target_timestamp_name,
            dataset.current_close_name, dataset.target_close_name,
        }
        previous_last_target = None
        for split_name in ("train", "validation", "test"):
            frame = getattr(dataset, split_name)
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{split_name} missing columns: {sorted(missing)}")
            if frame.empty:
                raise ValueError(f"{split_name} is empty.")
            ts = frame[dataset.timestamp_name]
            target_ts = frame[dataset.target_timestamp_name]
            if ts.duplicated().any() or not ts.is_monotonic_increasing:
                raise ValueError(f"{split_name} timestamps are not unique chronological values.")
            if not target_ts.sub(ts).eq(pd.Timedelta(hours=1)).all():
                raise ValueError(f"{split_name} contains non-1h targets.")
            target = pd.to_numeric(frame[dataset.target_name], errors="coerce").to_numpy(float)
            if not np.isfinite(target).all():
                raise ValueError(f"{split_name} target contains NaN/Inf.")
            if previous_last_target is not None and not previous_last_target < ts.iloc[0]:
                raise ValueError(f"{split_name} overlaps the previous split label horizon.")
            previous_last_target = target_ts.iloc[-1]
