from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from metal_predictor.multi_horizon.contracts import DatasetState
from metal_predictor.multi_horizon.feature_set import (
    FEATURE_COLUMNS,
    FEATURE_SET_VERSION,
    CausalHlcFeatureBuilder,
    feature_fingerprint_sha256,
)
from metal_predictor.multi_horizon.provenance import BullionVaultChartCsvLoader
from metal_predictor.multi_horizon.registry import get_horizon
from metal_predictor.multi_horizon.targets import (
    TARGET_CLOSE_COLUMN,
    TARGET_COLUMN,
    TARGET_TIMESTAMP_COLUMN,
    TARGET_VERSION,
    NextBarTargetBuilder,
)


STAGE1_MANIFEST_VERSION: Final = "bullionvault-multi-horizon-provenance-v1"
DATASET_VERSION: Final = "bullionvault-multi-horizon-causal-dataset-v1"


class DataPendingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CausalHorizonDataset:
    horizon_key: str
    interval_seconds: int
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    target_close_column: str
    target_timestamp_column: str
    source_sha256: str
    source_timestamp_semantics: str
    source_provider: str

    def __post_init__(self) -> None:
        if self.frame.empty:
            raise ValueError("Causal horizon dataset must contain model rows.")
        if tuple(self.feature_columns) != FEATURE_COLUMNS:
            raise ValueError("Unexpected multi-horizon feature registry.")
        if self.target_column != TARGET_COLUMN:
            raise ValueError("Unexpected multi-horizon target column.")

    @property
    def model_row_count(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class DatasetBuildReport:
    horizon_key: str
    state: str
    interval_seconds: int
    source_sha256: str | None
    raw_row_count: int
    source_rows_after_incomplete_exclusion: int
    model_row_count: int
    warmup_rows_dropped: int
    unlabeled_tail_rows_dropped: int
    first_feature_timestamp_source: str | None
    last_feature_timestamp_source: str | None
    last_target_timestamp_source: str | None
    feature_set_version: str
    feature_count: int
    feature_fingerprint_sha256: str
    target_version: str
    target_semantics: str
    timestamp_timezone_semantics: str
    newest_incomplete_source_row_excluded: bool
    performance_metrics_computed: bool = False
    live_model_mutated: bool = False
    frozen_52_feature_graph_mutated: bool = False
    future_holdout_read: bool = False
    shadow62_mutated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon_key": self.horizon_key,
            "state": self.state,
            "interval_seconds": self.interval_seconds,
            "source_sha256": self.source_sha256,
            "raw_row_count": self.raw_row_count,
            "source_rows_after_incomplete_exclusion": self.source_rows_after_incomplete_exclusion,
            "model_row_count": self.model_row_count,
            "warmup_rows_dropped": self.warmup_rows_dropped,
            "unlabeled_tail_rows_dropped": self.unlabeled_tail_rows_dropped,
            "first_feature_timestamp_source": self.first_feature_timestamp_source,
            "last_feature_timestamp_source": self.last_feature_timestamp_source,
            "last_target_timestamp_source": self.last_target_timestamp_source,
            "feature_set_version": self.feature_set_version,
            "feature_count": self.feature_count,
            "feature_fingerprint_sha256": self.feature_fingerprint_sha256,
            "target_version": self.target_version,
            "target_semantics": self.target_semantics,
            "timestamp_timezone_semantics": self.timestamp_timezone_semantics,
            "newest_incomplete_source_row_excluded": self.newest_incomplete_source_row_excluded,
            "performance_metrics_computed": self.performance_metrics_computed,
            "live_model_mutated": self.live_model_mutated,
            "frozen_52_feature_graph_mutated": self.frozen_52_feature_graph_mutated,
            "future_holdout_read": self.future_holdout_read,
            "shadow62_mutated": self.shadow62_mutated,
        }


class Stage1ManifestRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, object]:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if payload.get("manifest_version") != STAGE1_MANIFEST_VERSION:
            raise ValueError("Unexpected Stage-1 BullionVault provenance manifest version.")
        return payload

    def dataset_record(self, horizon_key: str) -> dict[str, object]:
        payload = self.load()
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict) or horizon_key not in datasets:
            raise KeyError(f"Stage-1 manifest has no dataset record for {horizon_key!r}.")
        record = datasets[horizon_key]
        if not isinstance(record, dict):
            raise ValueError("Malformed Stage-1 dataset record.")
        return record


class MultiHorizonDatasetBuilder:
    """Compose immutable Stage-1 data, causal features and next-bar targets."""

    def __init__(
        self,
        *,
        repo_root: str | Path = ".",
        loader: BullionVaultChartCsvLoader | None = None,
        feature_builder: CausalHlcFeatureBuilder | None = None,
        target_builder: NextBarTargetBuilder | None = None,
    ) -> None:
        self._repo_root = Path(repo_root)
        self._manifest = Stage1ManifestRepository(
            self._repo_root / "research_data/bullionvault_horizons/manifest.json"
        )
        self._loader = loader or BullionVaultChartCsvLoader()
        self._features = feature_builder or CausalHlcFeatureBuilder()
        self._targets = target_builder or NextBarTargetBuilder()

    def build(self, horizon_key: str) -> tuple[CausalHorizonDataset, DatasetBuildReport]:
        spec = get_horizon(horizon_key)
        record = self._manifest.dataset_record(spec.key)
        state = str(record.get("state", ""))
        if spec.dataset_state == DatasetState.DATA_PENDING or state == DatasetState.DATA_PENDING.value:
            raise DataPendingError(f"{spec.key} dataset is DATA_PENDING.")
        if state != DatasetState.READY.value:
            raise ValueError(f"{spec.key} dataset state is not READY.")

        raw_path = record.get("raw_path")
        source_sha = record.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{spec.key} READY dataset has no raw_path.")
        if not isinstance(source_sha, str) or len(source_sha) != 64:
            raise ValueError(f"{spec.key} READY dataset has no valid SHA256.")

        loaded = self._loader.load(
            self._repo_root / raw_path,
            expected_interval_seconds=spec.interval_seconds,
            include_potentially_incomplete_newest=False,
        )
        if loaded.audit.sha256 != source_sha:
            raise ValueError(f"{spec.key} source SHA256 no longer matches Stage-1 provenance.")

        source = loaded.frame
        features = self._features.build(source)
        targets = self._targets.build(source, interval_seconds=spec.interval_seconds)
        merged = features.merge(
            targets,
            on="timestamp_source",
            how="inner",
            validate="one_to_one",
            sort=True,
        )

        required_model_columns = [
            *FEATURE_COLUMNS,
            "current_close_usd_per_kg",
            TARGET_TIMESTAMP_COLUMN,
            TARGET_CLOSE_COLUMN,
            TARGET_COLUMN,
        ]
        model = merged.dropna(subset=required_model_columns).reset_index(drop=True)
        if model.empty:
            raise ValueError(f"{spec.key} has no usable causal model rows after warmup.")

        if not np.isfinite(model.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{spec.key} causal feature matrix contains non-finite values.")
        if not np.isfinite(model[TARGET_COLUMN].to_numpy(dtype=float)).all():
            raise ValueError(f"{spec.key} target contains non-finite values.")

        feature_ts = pd.to_datetime(model["timestamp_source"], errors="raise")
        target_ts = pd.to_datetime(model[TARGET_TIMESTAMP_COLUMN], errors="raise")
        target_delta = (target_ts - feature_ts).dt.total_seconds().astype(int)
        if tuple(sorted(set(target_delta.tolist()))) != (spec.interval_seconds,):
            raise ValueError(f"{spec.key} target is not the exact next registered source bar.")

        dataset = CausalHorizonDataset(
            horizon_key=spec.key,
            interval_seconds=spec.interval_seconds,
            frame=model,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            target_close_column=TARGET_CLOSE_COLUMN,
            target_timestamp_column=TARGET_TIMESTAMP_COLUMN,
            source_sha256=source_sha,
            source_timestamp_semantics=str(record.get("timestamp_timezone_semantics")),
            source_provider=str(record.get("source_provider")),
        )

        source_rows = int(len(source))
        unlabeled_tail = 1
        warmup = source_rows - unlabeled_tail - dataset.model_row_count
        report = DatasetBuildReport(
            horizon_key=spec.key,
            state=DatasetState.READY.value,
            interval_seconds=spec.interval_seconds,
            source_sha256=source_sha,
            raw_row_count=int(loaded.audit.row_count),
            source_rows_after_incomplete_exclusion=source_rows,
            model_row_count=dataset.model_row_count,
            warmup_rows_dropped=int(warmup),
            unlabeled_tail_rows_dropped=unlabeled_tail,
            first_feature_timestamp_source=feature_ts.iloc[0].isoformat(),
            last_feature_timestamp_source=feature_ts.iloc[-1].isoformat(),
            last_target_timestamp_source=target_ts.iloc[-1].isoformat(),
            feature_set_version=FEATURE_SET_VERSION,
            feature_count=len(FEATURE_COLUMNS),
            feature_fingerprint_sha256=feature_fingerprint_sha256(),
            target_version=TARGET_VERSION,
            target_semantics="EXACT_NEXT_SOURCE_BAR_CLOSE_LOG_RETURN",
            timestamp_timezone_semantics=str(record.get("timestamp_timezone_semantics")),
            newest_incomplete_source_row_excluded=loaded.potentially_incomplete_newest_excluded,
        )
        return dataset, report

    def build_report_for_all(self) -> dict[str, object]:
        report: dict[str, object] = {
            "dataset_version": DATASET_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_fingerprint_sha256": feature_fingerprint_sha256(),
            "feature_count": len(FEATURE_COLUMNS),
            "target_version": TARGET_VERSION,
            "performance_metrics_computed": False,
            "guardrails": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
                "future_holdout_read": False,
                "shadow62_mutated": False,
            },
            "horizons": {},
        }
        horizons = report["horizons"]
        assert isinstance(horizons, dict)
        for key in ("4h", "12h", "1d", "2d", "30d"):
            try:
                _, built = self.build(key)
            except DataPendingError:
                spec = get_horizon(key)
                horizons[key] = {
                    "horizon_key": key,
                    "state": DatasetState.DATA_PENDING.value,
                    "interval_seconds": spec.interval_seconds,
                    "model_row_count": 0,
                    "performance_metrics_computed": False,
                }
            else:
                horizons[key] = built.as_dict()
        return report
