from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.modeling import ModelSpec


@dataclass(frozen=True)
class FrozenRidgePayload:
    schema_version: int
    model_name: str
    alpha: float
    feature_names: tuple[str, ...]
    imputer_median: tuple[float, ...]
    missing_indicator_feature_indices: tuple[int, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    ridge_coefficients: tuple[float, ...]
    ridge_intercept: float
    training_rows: int
    training_first_timestamp_utc: str
    training_last_timestamp_utc: str
    training_last_target_timestamp_utc: str
    source_dataset_git_blob_sha: str
    research_code_cutoff_commit: str
    target: str
    prediction_semantics: str
    model_payload_sha256: str


class FrozenRidgeExporter:
    """Exports a fitted sklearn Ridge pipeline into a version-independent numeric payload."""

    def export(
        self,
        spec: ModelSpec,
        development: pd.DataFrame,
        feature_names: tuple[str, ...],
        source_dataset_git_blob_sha: str,
        research_code_cutoff_commit: str,
    ) -> dict[str, object]:
        if spec.family != "ridge":
            raise ValueError("FrozenRidgeExporter accepts Ridge ModelSpec objects only.")
        required = {
            "timestamp_utc", "target_timestamp_utc", "target_log_return_1h",
            *feature_names,
        }
        missing = required.difference(development.columns)
        if missing:
            raise ValueError(f"Frozen Ridge development frame missing columns: {sorted(missing)}")
        ordered = development.sort_values("timestamp_utc").reset_index(drop=True)
        model = spec.factory()
        model.fit(ordered.loc[:, feature_names], ordered["target_log_return_1h"])
        if not hasattr(model, "named_steps"):
            raise TypeError("Expected sklearn Pipeline for Ridge export.")
        imputer = model.named_steps["imputer"]
        scaler = model.named_steps["scaler"]
        ridge = model.named_steps["regressor"]
        indicator = tuple(int(value) for value in imputer.indicator_.features_.tolist())
        payload: dict[str, object] = {
            "schema_version": 1,
            "model_name": spec.name,
            "family": "frozen_standardized_ridge",
            "alpha": float(ridge.alpha),
            "feature_names": list(feature_names),
            "imputer_median": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "missing_indicator_feature_indices": list(indicator),
            "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
            "ridge_coefficients": np.asarray(ridge.coef_, dtype=float).tolist(),
            "ridge_intercept": float(ridge.intercept_),
            "training_rows": int(len(ordered)),
            "training_first_timestamp_utc": pd.Timestamp(
                ordered["timestamp_utc"].iloc[0]
            ).isoformat(),
            "training_last_timestamp_utc": pd.Timestamp(
                ordered["timestamp_utc"].iloc[-1]
            ).isoformat(),
            "training_last_target_timestamp_utc": pd.Timestamp(
                ordered["target_timestamp_utc"].iloc[-1]
            ).isoformat(),
            "training_scope": "historical Train+Validation only; historical Test excluded",
            "source_dataset_git_blob_sha": source_dataset_git_blob_sha,
            "research_code_cutoff_commit": research_code_cutoff_commit,
            "target": "target_log_return_1h",
            "prediction_semantics": "next exact-hour XAG/USD log return",
        }
        payload["model_payload_sha256"] = self.compute_hash(payload)
        return payload

    @staticmethod
    def compute_hash(payload: dict[str, object]) -> str:
        clean = dict(payload)
        clean.pop("model_payload_sha256", None)
        canonical = json.dumps(
            clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def write(self, payload: dict[str, object], path: Path) -> None:
        if payload.get("model_payload_sha256") != self.compute_hash(payload):
            raise ValueError("Frozen model payload hash is invalid before write.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FrozenRidgeRegressor:
    """Pure NumPy inference for an exported median-impute/indicator/scale/Ridge pipeline."""

    def __init__(self, payload: dict[str, object]) -> None:
        if payload.get("model_payload_sha256") != FrozenRidgeExporter.compute_hash(payload):
            raise ValueError("Frozen Ridge payload hash mismatch.")
        self._payload = payload
        self.feature_names = tuple(str(value) for value in payload["feature_names"])
        self.model_name = str(payload["model_name"])
        self.model_payload_sha256 = str(payload["model_payload_sha256"])
        self._median = np.asarray(payload["imputer_median"], dtype=float)
        self._indicator = np.asarray(
            payload["missing_indicator_feature_indices"], dtype=int
        )
        self._mean = np.asarray(payload["scaler_mean"], dtype=float)
        self._scale = np.asarray(payload["scaler_scale"], dtype=float)
        self._coef = np.asarray(payload["ridge_coefficients"], dtype=float)
        self._intercept = float(payload["ridge_intercept"])
        if len(self._median) != len(self.feature_names):
            raise ValueError("Frozen Ridge imputer width does not match feature list.")
        transformed_width = len(self.feature_names) + len(self._indicator)
        if not (
            len(self._mean) == len(self._scale) == len(self._coef) == transformed_width
        ):
            raise ValueError("Frozen Ridge transformed widths are inconsistent.")
        if (self._scale <= 0).any() or not np.isfinite(self._scale).all():
            raise ValueError("Frozen Ridge scaler contains invalid scale values.")

    @classmethod
    def from_path(cls, path: Path) -> "FrozenRidgeRegressor":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing_columns = set(self.feature_names).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Frozen Ridge prediction missing features: {sorted(missing_columns)}")
        values = frame.loc[:, self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        missing = np.isnan(values)
        finite_or_nan = np.isfinite(values) | missing
        if not finite_or_nan.all():
            raise ValueError("Frozen Ridge features contain infinite values.")
        filled = np.where(missing, self._median.reshape(1, -1), values)
        if len(self._indicator):
            transformed = np.column_stack([
                filled,
                missing[:, self._indicator].astype(float),
            ])
        else:
            transformed = filled
        standardized = (transformed - self._mean) / self._scale
        prediction = self._intercept + standardized @ self._coef
        if not np.isfinite(prediction).all():
            raise ValueError("Frozen Ridge produced non-finite predictions.")
        return prediction.astype(float)
