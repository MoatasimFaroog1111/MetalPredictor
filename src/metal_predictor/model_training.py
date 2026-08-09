from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from metal_predictor.metrics import PairedBlockBootstrapComparison, RegressionForecastMetrics
from metal_predictor.model_data import PreparedDataset, PreparedDatasetLoader
from metal_predictor.modeling import DefaultModelRegistry, ModelSpec
from metal_predictor.selection import CandidateSummary, FinalModelSelectionPolicy, ValidationResult, WalkForwardSelectionPolicy
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig, WalkForwardEvaluator


@dataclass(frozen=True)
class ModelTrainingConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("artifacts/model_evaluation")
    n_walk_forward_splits: int = 5
    initial_train_fraction: float = 0.50
    min_train_rows: int = 5000


class ModelTrainingPipeline:
    """Coordinates evaluation while keeping data, models, metrics and selection replaceable."""

    def __init__(self, config: ModelTrainingConfig, loader: PreparedDatasetLoader,
                 registry: DefaultModelRegistry, wf_policy: WalkForwardSelectionPolicy,
                 final_policy: FinalModelSelectionPolicy,
                 metric_calculator: RegressionForecastMetrics) -> None:
        self._config = config
        self._loader = loader
        self._registry = registry
        self._wf_policy = wf_policy
        self._final_policy = final_policy
        self._metrics = metric_calculator

    def run(self) -> dict[str, object]:
        dataset = self._loader.load(self._config.processed_dir)
        evaluator = self._make_evaluator(dataset)
        specs = self._registry.candidates()

        all_fold_results = []
        summaries = []
        for spec in specs:
            results = evaluator.evaluate(dataset.train, spec)
            all_fold_results.extend(results)
            summaries.append(self._wf_policy.summarize(results))

        family_winners = self._wf_policy.family_winners(specs, tuple(summaries))
        summary_by_name = {s.model_name: s for s in summaries}
        validation_results = []
        for spec in family_winners:
            _, _, metrics = evaluator.fit_and_score(dataset.train, dataset.validation, spec)
            summary = summary_by_name[spec.name]
            validation_results.append(ValidationResult(
                model_name=spec.name,
                family=spec.family,
                history_days=spec.history_days,
                wf_mae_mean=summary.wf_mae_mean,
                validation_mae_return=metrics.mae_return,
                validation_rmse_return=metrics.rmse_return,
                validation_directional_accuracy=metrics.directional_accuracy,
                validation_pearson_ic=metrics.pearson_ic,
                validation_price_mae_usd_per_kg=metrics.price_mae_usd_per_kg,
            ))

        selected_row = self._final_policy.choose(tuple(validation_results))
        selected_spec = self._find_spec(specs, selected_row.model_name)

        final_model, final_train = evaluator.fit_final(dataset.development, dataset.test, selected_spec)
        test_prediction = np.asarray(final_model.predict(dataset.test.loc[:, dataset.feature_names]), dtype=float)
        test_metrics = self._metrics.calculate(
            dataset.test[dataset.target_name], test_prediction,
            dataset.test[dataset.current_close_name], dataset.test[dataset.target_close_name])

        zero_prediction = np.zeros(len(dataset.test), dtype=float)
        zero_test_metrics = self._metrics.calculate(
            dataset.test[dataset.target_name], zero_prediction,
            dataset.test[dataset.current_close_name], dataset.test[dataset.target_close_name])
        bootstrap = PairedBlockBootstrapComparison(
            block_size_rows=24, resamples=2000, random_state=42).compare(
                dataset.test[dataset.target_name], test_prediction, zero_prediction)

        report = self._build_report(
            dataset=dataset, specs=specs, summaries=tuple(summaries),
            family_winners=family_winners, validation_results=tuple(validation_results),
            selected=selected_row, selected_spec=selected_spec, final_train=final_train,
            test_metrics=test_metrics.as_dict(), zero_test_metrics=zero_test_metrics.as_dict(),
            bootstrap_comparison=bootstrap.as_dict())
        self._write_artifacts(
            dataset=dataset, final_model=final_model, selected_spec=selected_spec,
            all_fold_results=all_fold_results, summaries=tuple(summaries),
            validation_results=tuple(validation_results), test_prediction=test_prediction,
            report=report)
        return report

    def _make_evaluator(self, dataset: PreparedDataset) -> WalkForwardEvaluator:
        return WalkForwardEvaluator(
            splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
                n_splits=self._config.n_walk_forward_splits,
                initial_train_fraction=self._config.initial_train_fraction,
                min_train_rows=self._config.min_train_rows,
                timestamp_name=dataset.timestamp_name,
                target_timestamp_name=dataset.target_timestamp_name)),
            metrics=self._metrics,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            timestamp_name=dataset.timestamp_name,
            target_timestamp_name=dataset.target_timestamp_name,
            current_close_name=dataset.current_close_name,
            target_close_name=dataset.target_close_name)

    @staticmethod
    def _find_spec(specs: tuple[ModelSpec, ...], name: str) -> ModelSpec:
        for spec in specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def _build_report(self, *, dataset: PreparedDataset, specs: tuple[ModelSpec, ...],
                      summaries: tuple[CandidateSummary, ...], family_winners: tuple[ModelSpec, ...],
                      validation_results: tuple[ValidationResult, ...], selected: ValidationResult,
                      selected_spec: ModelSpec, final_train: pd.DataFrame,
                      test_metrics: dict[str, object], zero_test_metrics: dict[str, object],
                      bootstrap_comparison: dict[str, object]) -> dict[str, object]:
        zero_validation = next((row for row in validation_results if row.family == "baseline_zero"), None)
        zero_validation_mae = float(zero_validation.validation_mae_return) if zero_validation else None
        return {
            "status": "PASS",
            "methodology": {
                "selection_metric": "validation_mae_return",
                "hyperparameters_selected_with": "5-fold purged expanding walk-forward on Train only",
                "family_comparison_with": "original Validation split only",
                "test_policy": "one-time score after model and configuration are frozen",
                "test_used_for_selection": False,
                "scaling_imputation_policy": "fit inside each training fold only; final fit on development only",
            },
            "dataset": {
                "train_rows": len(dataset.train), "validation_rows": len(dataset.validation),
                "test_rows": len(dataset.test), "feature_count": len(dataset.feature_names),
                "target": dataset.target_name,
            },
            "candidate_count": len(specs),
            "families": sorted({spec.family for spec in specs}),
            "walk_forward_summaries": [s.as_dict() for s in summaries],
            "family_winners": [spec.name for spec in family_winners],
            "validation_results": [row.as_dict() for row in validation_results],
            "selected_model": {
                "name": selected_spec.name, "family": selected_spec.family,
                "history_days": selected_spec.history_days,
                "validation_mae_return": selected.validation_mae_return,
                "validation_rmse_return": selected.validation_rmse_return,
                "validation_directional_accuracy": selected.validation_directional_accuracy,
                "validation_pearson_ic": selected.validation_pearson_ic,
            },
            "final_training": {
                "rows": len(final_train),
                "first_feature_timestamp_utc": pd.Timestamp(final_train[dataset.timestamp_name].iloc[0]).isoformat(),
                "last_target_timestamp_utc": pd.Timestamp(final_train[dataset.target_timestamp_name].iloc[-1]).isoformat(),
            },
            "final_test_metrics": test_metrics,
            "zero_return_test_metrics": zero_test_metrics,
            "test_mae_bootstrap_vs_zero": bootstrap_comparison,
            "baseline_reference": {
                "zero_return_validation_mae": zero_validation_mae,
                "selected_validation_mae_improvement_vs_zero_percent": None if zero_validation_mae is None else
                    (zero_validation_mae - selected.validation_mae_return) / zero_validation_mae * 100.0,
            },
            "software": self._software_versions(),
        }

    def _write_artifacts(self, *, dataset: PreparedDataset, final_model, selected_spec: ModelSpec,
                         all_fold_results, summaries: tuple[CandidateSummary, ...],
                         validation_results: tuple[ValidationResult, ...],
                         test_prediction: np.ndarray, report: dict[str, object]) -> None:
        out = self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row.flat_dict() for row in all_fold_results]).to_csv(out / "walk_forward_folds.csv", index=False)
        pd.DataFrame([row.as_dict() for row in summaries]).sort_values(
            ["family", "wf_mae_mean"]).to_csv(out / "walk_forward_candidate_summary.csv", index=False)
        pd.DataFrame([row.as_dict() for row in validation_results]).sort_values(
            "validation_mae_return").to_csv(out / "family_winners_validation.csv", index=False)

        predictions = dataset.test[[dataset.timestamp_name, dataset.target_timestamp_name,
                                    dataset.current_close_name, dataset.target_close_name,
                                    dataset.target_name]].copy()
        predictions["predicted_log_return_1h"] = test_prediction
        predictions["predicted_close_usd_per_kg"] = predictions[dataset.current_close_name].astype(float) * np.exp(test_prediction)
        predictions["return_error"] = predictions["predicted_log_return_1h"] - predictions[dataset.target_name]
        predictions.to_parquet(out / "test_predictions.parquet", index=False)
        predictions.to_csv(out / "test_predictions.csv", index=False)

        joblib.dump(final_model, out / "best_model.joblib")
        (out / "best_model_manifest.json").write_text(json.dumps({
            "model_name": selected_spec.name, "family": selected_spec.family,
            "history_days": selected_spec.history_days, "feature_names": list(dataset.feature_names),
            "target": dataset.target_name, "prediction_semantics": "next exact-hour XAG/USD log return",
            "test_used_for_selection": False,
        }, indent=2), encoding="utf-8")
        (out / "model_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    @staticmethod
    def _software_versions() -> dict[str, str]:
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__}
        try:
            import xgboost
            versions["xgboost"] = xgboost.__version__
        except Exception:
            versions["xgboost"] = "unavailable"
        try:
            import lightgbm
            versions["lightgbm"] = lightgbm.__version__
        except Exception:
            versions["lightgbm"] = "unavailable"
        return versions
