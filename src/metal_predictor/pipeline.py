from __future__ import annotations
import json
import numpy as np
import pandas as pd


class TrainingDataPipeline:
    """Coordinates independent components; contains no feature or model logic."""

    def __init__(self, config, loader, validator, feature_components,
                 target_builder, splitter, leakage_guard, writer) -> None:
        self._config = config
        self._loader = loader
        self._validator = validator
        self._features = feature_components
        self._target_builder = target_builder
        self._splitter = splitter
        self._leakage_guard = leakage_guard
        self._writer = writer

    def run(self):
        frame = self._loader.load(self._config.input_path)
        frame = self._normalize(frame)
        self._validator.validate(frame)
        raw_rows = len(frame)
        if self._config.strict_quality_only and self._config.columns.quality in frame.columns:
            frame = frame.loc[frame[self._config.columns.quality].eq("OK")].copy()
        post_quality_rows = len(frame)

        feature_names = []
        for component in self._features:
            frame = component.transform(frame)
            feature_names.extend(component.feature_names)

        frame = self._target_builder.build(frame)
        target_names = self._target_builder.target_names
        usable = self._finalize(frame, tuple(feature_names), target_names)
        splits = self._splitter.split(usable)
        self._leakage_guard.validate(usable, splits, tuple(feature_names), target_names)
        self._writer.write(splits, tuple(feature_names), target_names, self._config.output_dir)

        report = self._quality_report(raw_rows, post_quality_rows, usable, splits, tuple(feature_names))
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        (self._config.output_dir / "data_quality_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    def _normalize(self, frame):
        c = self._config.columns
        out = frame.copy()
        out[c.timestamp] = pd.to_datetime(out[c.timestamp], utc=True, errors="raise")
        out = out.sort_values(c.timestamp).reset_index(drop=True)
        for name in (c.open, c.high, c.low, c.close):
            out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
        return out

    def _finalize(self, frame, feature_names, target_names):
        # Targets are mandatory. Long-horizon exact-time features are allowed to be
        # missing around genuine market gaps; their availability is represented by
        # has_exact_* features and imputation, if required by a model, must be fit
        # on Train only in the model-preprocessing stage.
        usable = frame.dropna(subset=list(target_names)).copy()
        matrix = usable.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        inf_mask = np.isinf(matrix.to_numpy(float)).any(axis=1)
        usable = usable.loc[~inf_mask].copy()
        return usable.sort_values(self._config.columns.timestamp).reset_index(drop=True)

    def _quality_report(self, raw_rows, post_quality_rows, usable, splits, feature_names):
        c = self._config.columns
        ts = pd.to_datetime(usable[c.timestamp], utc=True)
        feature_frame = usable.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        missing_cells = int(feature_frame.isna().sum().sum())
        rows_with_optional_missing = int(feature_frame.isna().any(axis=1).sum())
        return {
            "status": "PASS",
            "input_rows": int(raw_rows),
            "rows_after_quality_policy": int(post_quality_rows),
            "usable_rows": int(len(usable)),
            "dropped_rows_total": int(raw_rows - len(usable)),
            "feature_count": len(feature_names),
            "rows_with_optional_feature_missingness": rows_with_optional_missing,
            "optional_feature_missing_cells": missing_cells,
            "first_usable_timestamp_utc": ts.min().isoformat(),
            "last_usable_timestamp_utc": ts.max().isoformat(),
            "split_rows": {k: int(len(v)) for k, v in splits.items()},
            "target_horizon_hours": self._config.target_horizon_hours,
            "strict_quality_only": self._config.strict_quality_only,
            "missing_feature_policy": "Preserve exact-time NaN; impute on Train only if model requires it.",
            "leakage_checks": "PASS",
        }
