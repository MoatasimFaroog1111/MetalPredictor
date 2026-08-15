from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.direct_horizon.dataset import Stage7DatasetBuilder, Stage7HorizonDataset
from metal_predictor.direct_horizon.models import Stage7ModelFactory
from metal_predictor.direct_horizon.preregistration import (
    Stage7CandidateSpec,
    stage7_candidates,
    stage7_horizons,
    stage7_preregistration_fingerprint_sha256,
    stage7_preregistration_payload,
)
from metal_predictor.direct_horizon.split import Stage7PurgedExpandingPlanner, Stage7SplitPlan
from metal_predictor.multi_horizon.statistics import (
    paired_block_bootstrap_mae_improvement,
    regression_metrics,
)


@dataclass(frozen=True)
class Stage7DevelopmentGate:
    """Apply only the development thresholds frozen before Stage-7 results exist."""

    def evaluate(
        self,
        *,
        candidate_mae: float,
        baseline_mae: float,
        better_folds: int,
        directional_accuracy_delta: float,
        bootstrap_ci_low: float,
        bootstrap_probability_better: float,
    ) -> dict[str, object]:
        gate = stage7_preregistration_payload()["development_selection_gate"]
        if not isinstance(gate, dict):
            raise ValueError("Malformed Stage-7 development selection gate.")
        checks = {
            "beats_random_walk_oof_mae": candidate_mae < baseline_mae,
            "minimum_better_folds": better_folds >= int(gate["minimum_better_folds"]),
            "bootstrap_ci_lower_positive": bootstrap_ci_low > 0.0,
            "bootstrap_probability_threshold": bootstrap_probability_better
            >= float(gate["probability_candidate_mae_better_minimum"]),
            "directional_accuracy_floor": directional_accuracy_delta
            >= float(gate["directional_accuracy_delta_minimum"]),
        }
        return {"passed": all(checks.values()), "checks": checks}


class Stage7DevelopmentEvaluator:
    """Compare preregistered candidates on development folds only."""

    baseline_id = "random_walk_zero_return"

    def __init__(self, *, model_factory: Stage7ModelFactory | None = None) -> None:
        self._models = model_factory or Stage7ModelFactory()
        self._gate = Stage7DevelopmentGate()
        self._gate_payload = stage7_preregistration_payload()["development_selection_gate"]
        if not isinstance(self._gate_payload, dict):
            raise ValueError("Malformed Stage-7 gate payload.")

    def evaluate(
        self,
        dataset: Stage7HorizonDataset,
        plan: Stage7SplitPlan,
    ) -> dict[str, object]:
        if plan.total_rows != len(dataset.frame):
            raise ValueError("Stage-7 split plan and dataset row counts differ.")
        if plan.historical_test_rows <= 0:
            raise ValueError("Stage-7 locked historical test must be non-empty.")

        # Scientific firewall: only this chronological development prefix is ever
        # materialized into model X/y arrays. Historical-test target values are not
        # passed to metrics, estimators, bootstraps, or selection.
        development = dataset.frame.iloc[: plan.development_end_exclusive].copy()
        x_dev = development.loc[:, list(dataset.feature_names)]
        y_dev = pd.to_numeric(development[dataset.target_name], errors="raise").to_numpy(float)
        if not np.isfinite(y_dev).all():
            raise ValueError("Stage-7 development targets must be finite.")

        baseline_actual_parts: list[np.ndarray] = []
        baseline_prediction_parts: list[np.ndarray] = []
        baseline_fold_metrics: dict[int, object] = {}
        fold_audit: list[dict[str, object]] = []

        for fold in plan.folds:
            if fold.validation_end_exclusive > plan.development_end_exclusive:
                raise ValueError("Stage-7 validation fold crosses into historical test.")
            y_val = y_dev[fold.validation_start : fold.validation_end_exclusive]
            zero = np.zeros(y_val.size, dtype=float)
            metrics = regression_metrics(y_val, zero)
            baseline_actual_parts.append(y_val.copy())
            baseline_prediction_parts.append(zero)
            baseline_fold_metrics[fold.fold_number] = metrics

            train_target_ts = pd.to_datetime(
                development.iloc[: fold.train_end_exclusive][dataset.target_timestamp_name],
                utc=True,
            )
            if train_target_ts.max() >= fold.validation_start_timestamp_utc:
                raise ValueError("Stage-7 purge firewall failed: Train label reaches Validation.")
            fold_audit.append(
                {
                    "fold_number": fold.fold_number,
                    "train_rows": fold.train_row_count,
                    "validation_rows": fold.validation_row_count,
                    "purged_rows_before_validation": fold.purged_rows_before_validation,
                    "purge_hours": fold.purge_hours,
                    "embargo_hours": fold.embargo_hours,
                    "validation_start_timestamp_utc": fold.validation_start_timestamp_utc.isoformat(),
                    "last_train_target_timestamp_utc": train_target_ts.max().isoformat(),
                    "train_labels_end_before_validation": True,
                    "future_rows_used_for_training": False,
                }
            )

        baseline_actual_oof = np.concatenate(baseline_actual_parts)
        baseline_prediction_oof = np.concatenate(baseline_prediction_parts)
        baseline_oof = regression_metrics(baseline_actual_oof, baseline_prediction_oof)

        candidate_rows: list[dict[str, object]] = []
        passing_rows: list[dict[str, object]] = []
        for spec in stage7_candidates():
            row = self._evaluate_candidate(
                dataset=dataset,
                development=development,
                x_dev=x_dev,
                y_dev=y_dev,
                plan=plan,
                spec=spec,
                baseline_actual_oof=baseline_actual_oof,
                baseline_prediction_oof=baseline_prediction_oof,
                baseline_fold_metrics=baseline_fold_metrics,
                baseline_oof_mae=baseline_oof.mae,
            )
            candidate_rows.append(row)
            gate = row["development_gate"]
            if isinstance(gate, dict) and gate.get("passed") is True:
                passing_rows.append(row)

        if passing_rows:
            passing_rows.sort(key=lambda row: float(row["mae_improvement"]), reverse=True)
            selected_id = str(passing_rows[0]["candidate_id"])
            selected_kind = "CANDIDATE"
            selection_reason = str(self._gate_payload["winner_rule"])
        else:
            selected_id = self.baseline_id
            selected_kind = "BASELINE"
            selection_reason = str(self._gate_payload["if_no_candidate_passes"])

        return {
            "horizon_key": dataset.horizon.key,
            "horizon_hours": dataset.horizon.hours,
            "feature_count": len(dataset.feature_names),
            "feature_graph_version": dataset.feature_graph_version,
            "target": dataset.target_name,
            "target_semantics": "EXACT_UTC_T_PLUS_H_LOG_RETURN",
            "total_model_rows": len(dataset.frame),
            "development_rows": plan.development_rows,
            "development_oof_rows": baseline_oof.row_count,
            "split_firewall": {
                "historical_test_boundary_utc": plan.historical_test_boundary_utc.isoformat(),
                "historical_test_rows": plan.historical_test_rows,
                "historical_test_metrics_read": False,
                "historical_test_predictions_computed": False,
                "historical_test_used_for_fit": False,
                "historical_test_used_for_selection": False,
                "historical_test_target_values_consumed_by_model": False,
                "purge_hours": plan.purge_hours,
                "embargo_hours": plan.embargo_hours,
                "folds": fold_audit,
            },
            "random_walk_development_oof": baseline_oof.as_dict(),
            "candidates": candidate_rows,
            "selection": {
                "selected_id": selected_id,
                "selected_kind": selected_kind,
                "reason": selection_reason,
            },
            "performance_scope": "DEVELOPMENT_ONLY",
        }

    def _evaluate_candidate(
        self,
        *,
        dataset: Stage7HorizonDataset,
        development: pd.DataFrame,
        x_dev: pd.DataFrame,
        y_dev: np.ndarray,
        plan: Stage7SplitPlan,
        spec: Stage7CandidateSpec,
        baseline_actual_oof: np.ndarray,
        baseline_prediction_oof: np.ndarray,
        baseline_fold_metrics: dict[int, object],
        baseline_oof_mae: float,
    ) -> dict[str, object]:
        actual_parts: list[np.ndarray] = []
        predicted_parts: list[np.ndarray] = []
        fold_rows: list[dict[str, object]] = []

        for fold in plan.folds:
            x_train = x_dev.iloc[: fold.train_end_exclusive]
            y_train = y_dev[: fold.train_end_exclusive]
            x_val = x_dev.iloc[fold.validation_start : fold.validation_end_exclusive]
            y_val = y_dev[fold.validation_start : fold.validation_end_exclusive]

            estimator = self._models.create(spec)
            estimator.fit(x_train, y_train)
            predicted = np.asarray(estimator.predict(x_val), dtype=float).reshape(-1)
            if predicted.size != y_val.size or not np.isfinite(predicted).all():
                raise ValueError(f"{spec.candidate_id} emitted invalid validation predictions.")

            candidate_metrics = regression_metrics(y_val, predicted)
            baseline_metrics = baseline_fold_metrics[fold.fold_number]
            baseline_mae = float(getattr(baseline_metrics, "mae"))
            baseline_direction = float(getattr(baseline_metrics, "directional_accuracy"))
            actual_parts.append(y_val.copy())
            predicted_parts.append(predicted)
            fold_rows.append(
                {
                    "fold_number": fold.fold_number,
                    "train_rows": len(x_train),
                    "validation_rows": len(x_val),
                    "baseline_mae": baseline_mae,
                    "candidate_mae": candidate_metrics.mae,
                    "candidate_mae_better": candidate_metrics.mae < baseline_mae,
                    "mae_improvement": baseline_mae - candidate_metrics.mae,
                    "baseline_directional_accuracy": baseline_direction,
                    "candidate_directional_accuracy": candidate_metrics.directional_accuracy,
                    "directional_accuracy_delta": candidate_metrics.directional_accuracy
                    - baseline_direction,
                }
            )

        actual_oof = np.concatenate(actual_parts)
        predicted_oof = np.concatenate(predicted_parts)
        if not np.array_equal(actual_oof, baseline_actual_oof):
            raise ValueError("Stage-7 candidate and baseline OOF rows are not identical.")
        candidate_oof = regression_metrics(actual_oof, predicted_oof)
        block_length = max(24, int(dataset.horizon.hours))
        bootstrap = paired_block_bootstrap_mae_improvement(
            actual_oof,
            baseline_prediction_oof,
            predicted_oof,
            iterations=int(self._gate_payload["paired_block_bootstrap_iterations"]),
            block_length_rows=block_length,
            seed=int(self._gate_payload["paired_block_bootstrap_seed"]),
            ci_level=float(self._gate_payload["mae_improvement_ci"]),
        )
        better_folds = sum(bool(row["candidate_mae_better"]) for row in fold_rows)
        directional_delta = (
            candidate_oof.directional_accuracy
            - regression_metrics(baseline_actual_oof, baseline_prediction_oof).directional_accuracy
        )
        gate = self._gate.evaluate(
            candidate_mae=candidate_oof.mae,
            baseline_mae=baseline_oof_mae,
            better_folds=better_folds,
            directional_accuracy_delta=directional_delta,
            bootstrap_ci_low=bootstrap.ci_low,
            bootstrap_probability_better=bootstrap.probability_candidate_mae_better,
        )
        improvement = baseline_oof_mae - candidate_oof.mae
        return {
            "candidate_id": spec.candidate_id,
            "development_oof": candidate_oof.as_dict(),
            "better_folds": better_folds,
            "mae_improvement": improvement,
            "mae_improvement_pct": (
                improvement / baseline_oof_mae * 100.0 if baseline_oof_mae > 0.0 else 0.0
            ),
            "directional_accuracy_delta": directional_delta,
            "paired_block_bootstrap": bootstrap.as_dict(),
            "development_gate": gate,
            "folds": fold_rows,
        }


class Stage7DevelopmentRunner:
    """Orchestrate all five independent direct-horizon development studies."""

    def __init__(
        self,
        *,
        repo_root: Path = Path("."),
        dataset_builder: Stage7DatasetBuilder | None = None,
        planner: Stage7PurgedExpandingPlanner | None = None,
        evaluator: Stage7DevelopmentEvaluator | None = None,
    ) -> None:
        self._root = Path(repo_root)
        self._builder = dataset_builder or Stage7DatasetBuilder(repo_root=self._root)
        self._planner = planner or Stage7PurgedExpandingPlanner()
        self._evaluator = evaluator or Stage7DevelopmentEvaluator()

    def run_all(self) -> dict[str, object]:
        horizons: dict[str, object] = {}
        for horizon in stage7_horizons():
            dataset = self._builder.build(horizon)
            plan = self._planner.plan(dataset)
            horizons[horizon.key] = self._evaluator.evaluate(dataset, plan)

        return {
            "stage7_version": "xag-h1-direct-multi-horizon-stage7-development-v1",
            "preregistration_fingerprint_sha256": stage7_preregistration_fingerprint_sha256(),
            "source": {
                "path": "XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet",
                "bar_interval": "H1",
                "direct_targets": True,
                "bullionvault_forward_history_merged": False,
                "microstructure_merged": False,
            },
            "performance_scope": "DEVELOPMENT_ONLY",
            "historical_test_metrics_read": False,
            "historical_test_predictions_computed": False,
            "historical_test_target_values_consumed_by_model": False,
            "formal_future_holdout_read": False,
            "forward_bars_used_for_fit": False,
            "microstructure_used_for_fit": False,
            "candidate_artifacts_written": False,
            "production_forecast_routes_mutated": False,
            "guardrails": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "automatic_live_promotion": False,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
                "shadow62_mutated": False,
            },
            "horizons": horizons,
        }
