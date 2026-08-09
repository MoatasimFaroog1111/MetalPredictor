from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.metrics import PairedBlockBootstrapComparison, RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import PurgedWalkForwardSplitter


@dataclass(frozen=True)
class DevelopmentFeatureSet:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    label: str


class DevelopmentFeatureSetLoader:
    """Research loader that deliberately reads Train + Validation only. Test is never opened."""

    def load(self, processed_dir: Path, label: str) -> DevelopmentFeatureSet:
        manifest_path = processed_dir / "feature_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = tuple(manifest.get("features", ()))
        if not features:
            raise ValueError(f"{label} feature manifest is empty.")
        parts = []
        for split in ("train", "validation"):
            path = processed_dir / f"{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            parts.append(pd.read_parquet(path))
        frame = pd.concat(parts, ignore_index=True)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
        self._validate(frame, features, label)
        return DevelopmentFeatureSet(frame=frame, feature_names=features, label=label)

    @staticmethod
    def _validate(frame: pd.DataFrame, features: tuple[str, ...], label: str) -> None:
        required = set(features) | {
            "timestamp_utc", "target_timestamp_utc", "target_log_return_1h",
            "close_usd_per_kg", "target_close_usd_per_kg",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} development data missing columns: {sorted(missing)}")
        if frame.empty or frame["timestamp_utc"].duplicated().any():
            raise ValueError(f"{label} development timestamps are empty or duplicated.")
        if not frame["timestamp_utc"].is_monotonic_increasing:
            raise ValueError(f"{label} development timestamps are not chronological.")
        if not frame["target_timestamp_utc"].sub(frame["timestamp_utc"]).eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"{label} contains non-exact-hour targets.")
        target = pd.to_numeric(frame["target_log_return_1h"], errors="coerce").to_numpy(float)
        if not np.isfinite(target).all():
            raise ValueError(f"{label} contains invalid targets.")


@dataclass(frozen=True)
class FeatureSetComparisonConfig:
    output_dir: Path = Path("artifacts/cross_asset_gold")
    base_set_id: str = "A"
    enhanced_set_id: str = "B"
    artifact_prefix: str = "gold_ab"
    bootstrap_block_rows: int = 24
    bootstrap_resamples: int = 5000
    strong_min_better_folds: int = 4
    promising_min_better_folds: int = 3

    def __post_init__(self) -> None:
        if not self.base_set_id or not self.enhanced_set_id or self.base_set_id == self.enhanced_set_id:
            raise ValueError("Feature-set IDs must be non-empty and distinct.")
        if not self.artifact_prefix or "/" in self.artifact_prefix or "\\" in self.artifact_prefix:
            raise ValueError("artifact_prefix must be a simple file prefix.")


@dataclass(frozen=True)
class FeatureSetFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    validation_first_timestamp_utc: str
    validation_last_timestamp_utc: str
    base_mae_return: float
    enhanced_mae_return: float
    mae_improvement: float
    base_directional_accuracy: float
    enhanced_directional_accuracy: float
    base_pearson_ic: float | None
    enhanced_pearson_ic: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FeatureSetComparator:
    """Paired Walk-Forward comparison on development data only using one frozen estimator recipe."""

    def __init__(
        self,
        config: FeatureSetComparisonConfig,
        splitter: PurgedWalkForwardSplitter,
        metrics: RegressionForecastMetrics,
    ) -> None:
        self._config = config
        self._splitter = splitter
        self._metrics = metrics

    def compare(
        self,
        base: DevelopmentFeatureSet,
        enhanced: DevelopmentFeatureSet,
        model_spec: ModelSpec,
    ) -> dict[str, object]:
        self._validate_pair(base, enhanced)
        enhanced_indexed = enhanced.frame.set_index("timestamp_utc", drop=False)
        fold_rows: list[FeatureSetFoldResult] = []
        oof_parts: list[pd.DataFrame] = []

        for fold in self._splitter.split(base.frame):
            train_ts = pd.DatetimeIndex(fold.train["timestamp_utc"])
            validation_ts = pd.DatetimeIndex(fold.validation["timestamp_utc"])
            enhanced_train = enhanced_indexed.loc[train_ts].reset_index(drop=True)
            enhanced_validation = enhanced_indexed.loc[validation_ts].reset_index(drop=True)
            self._assert_targets_equal(fold.train, enhanced_train, f"fold {fold.number} train")
            self._assert_targets_equal(fold.validation, enhanced_validation, f"fold {fold.number} validation")

            base_model = model_spec.factory()
            enhanced_model = model_spec.factory()
            base_model.fit(fold.train.loc[:, base.feature_names], fold.train["target_log_return_1h"])
            enhanced_model.fit(
                enhanced_train.loc[:, enhanced.feature_names], enhanced_train["target_log_return_1h"]
            )
            base_prediction = np.asarray(
                base_model.predict(fold.validation.loc[:, base.feature_names]), dtype=float
            )
            enhanced_prediction = np.asarray(
                enhanced_model.predict(enhanced_validation.loc[:, enhanced.feature_names]), dtype=float
            )
            if not np.isfinite(base_prediction).all() or not np.isfinite(enhanced_prediction).all():
                raise ValueError("Feature-set comparison produced non-finite predictions.")

            base_metrics = self._metrics.calculate(
                fold.validation["target_log_return_1h"], base_prediction,
                fold.validation["close_usd_per_kg"], fold.validation["target_close_usd_per_kg"],
            )
            enhanced_metrics = self._metrics.calculate(
                enhanced_validation["target_log_return_1h"], enhanced_prediction,
                enhanced_validation["close_usd_per_kg"], enhanced_validation["target_close_usd_per_kg"],
            )
            fold_rows.append(FeatureSetFoldResult(
                fold=fold.number,
                train_rows=len(fold.train),
                validation_rows=len(fold.validation),
                validation_first_timestamp_utc=pd.Timestamp(fold.validation["timestamp_utc"].iloc[0]).isoformat(),
                validation_last_timestamp_utc=pd.Timestamp(fold.validation["timestamp_utc"].iloc[-1]).isoformat(),
                base_mae_return=base_metrics.mae_return,
                enhanced_mae_return=enhanced_metrics.mae_return,
                mae_improvement=base_metrics.mae_return - enhanced_metrics.mae_return,
                base_directional_accuracy=base_metrics.directional_accuracy,
                enhanced_directional_accuracy=enhanced_metrics.directional_accuracy,
                base_pearson_ic=base_metrics.pearson_ic,
                enhanced_pearson_ic=enhanced_metrics.pearson_ic,
            ))
            oof_parts.append(pd.DataFrame({
                "timestamp_utc": fold.validation["timestamp_utc"].to_numpy(),
                "target_log_return_1h": fold.validation["target_log_return_1h"].to_numpy(float),
                "base_prediction": base_prediction,
                "enhanced_prediction": enhanced_prediction,
            }))

        oof = pd.concat(oof_parts, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
        bootstrap = PairedBlockBootstrapComparison(
            block_size_rows=self._config.bootstrap_block_rows,
            resamples=self._config.bootstrap_resamples,
            random_state=42,
        ).compare(oof["target_log_return_1h"], oof["enhanced_prediction"], oof["base_prediction"])

        better_folds = sum(row.enhanced_mae_return < row.base_mae_return for row in fold_rows)
        promotion = self._promotion_decision(bootstrap, better_folds, len(fold_rows))
        new_features = sorted(set(enhanced.feature_names).difference(base.feature_names))
        report = {
            "status": "PASS",
            "research_policy": {
                "test_data_read": False,
                "data_used": "original Train + Validation only",
                "old_test_status": "historical benchmark; forbidden for feature selection",
                "estimator": model_spec.name,
                "estimator_hyperparameters_frozen_before_feature_set_comparison": True,
                "paired_folds": True,
            },
            "feature_sets": {
                self._config.base_set_id: {"label": base.label, "feature_count": len(base.feature_names)},
                self._config.enhanced_set_id: {"label": enhanced.label, "feature_count": len(enhanced.feature_names)},
                "new_feature_count": len(new_features),
                "new_features": new_features,
            },
            "walk_forward": {
                "folds": len(fold_rows),
                "enhanced_better_folds": better_folds,
                "fold_results": [row.as_dict() for row in fold_rows],
            },
            "paired_oof_mae": {
                "base": bootstrap.baseline_mae_return,
                "enhanced": bootstrap.selected_mae_return,
                "improvement": bootstrap.mae_improvement_vs_baseline,
                "improvement_percent": bootstrap.mae_improvement_percent,
            },
            "paired_block_bootstrap": bootstrap.as_dict(),
            "decision": promotion,
        }
        self._write(report, fold_rows, oof)
        return report

    def _promotion_decision(self, bootstrap, better_folds: int, total_folds: int) -> dict[str, object]:
        positive = bootstrap.mae_improvement_vs_baseline > 0
        if (
            positive
            and better_folds >= self._config.strong_min_better_folds
            and bootstrap.improvement_ci95_low > 0
            and bootstrap.probability_selected_better >= 0.95
        ):
            level = "STRONG_EVIDENCE"
            promote = True
        elif (
            positive
            and better_folds >= self._config.promising_min_better_folds
            and bootstrap.probability_selected_better >= 0.80
        ):
            level = "PROMISING_NOT_CONCLUSIVE"
            promote = True
        else:
            level = "NO_STABLE_EVIDENCE"
            promote = False
        return {
            "promote_enhanced_feature_set": promote,
            "evidence_level": level,
            "better_folds": better_folds,
            "total_folds": total_folds,
            "rule_fixed_before_result": True,
        }

    def _write(self, report, fold_rows, oof) -> None:
        out = self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        prefix = self._config.artifact_prefix
        pd.DataFrame([row.as_dict() for row in fold_rows]).to_csv(out / f"{prefix}_folds.csv", index=False)
        oof.to_parquet(out / f"{prefix}_oof_predictions.parquet", index=False)
        oof.to_csv(out / f"{prefix}_oof_predictions.csv", index=False)
        (out / f"{prefix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    @staticmethod
    def _validate_pair(base: DevelopmentFeatureSet, enhanced: DevelopmentFeatureSet) -> None:
        if len(base.frame) != len(enhanced.frame):
            raise ValueError("Feature sets must contain the same development rows.")
        if not base.frame["timestamp_utc"].equals(enhanced.frame["timestamp_utc"]):
            raise ValueError("Feature sets are not aligned on identical development timestamps.")
        FeatureSetComparator._assert_targets_equal(base.frame, enhanced.frame, "full development")

    @staticmethod
    def _assert_targets_equal(base: pd.DataFrame, enhanced: pd.DataFrame, context: str) -> None:
        if not pd.Series(base["timestamp_utc"].to_numpy()).equals(pd.Series(enhanced["timestamp_utc"].to_numpy())):
            raise ValueError(f"Timestamp mismatch in {context}.")
        if not np.allclose(
            base["target_log_return_1h"].to_numpy(float),
            enhanced["target_log_return_1h"].to_numpy(float),
            rtol=0.0, atol=1e-15,
        ):
            raise ValueError(f"Target mismatch in {context}.")
