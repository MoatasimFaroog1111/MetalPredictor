from __future__ import annotations

from typing import Final

import numpy as np

from metal_predictor.multi_horizon.dataset import (
    CausalHorizonDataset,
    DataPendingError,
    MultiHorizonDatasetBuilder,
)
from metal_predictor.multi_horizon.development import (
    CandidateDevelopmentResult,
    FoldComparison,
    HorizonDevelopmentResult,
)
from metal_predictor.multi_horizon.models import random_walk_zero_return
from metal_predictor.multi_horizon.registry import get_horizon
from metal_predictor.multi_horizon.selection import DevelopmentCandidateEvidence
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner, WalkForwardPlan
from metal_predictor.multi_horizon.stage6_models import Stage6DevelopmentModelFactory
from metal_predictor.multi_horizon.stage6_preregistration import (
    stage6_candidate_registry,
    stage6_preregistration_fingerprint_sha256,
    stage6_preregistration_payload,
)
from metal_predictor.multi_horizon.stage6_selection import Stage6DevelopmentSelectionGate
from metal_predictor.multi_horizon.statistics import (
    paired_block_bootstrap_mae_improvement,
    regression_metrics,
)


STAGE6_DEVELOPMENT_VERSION: Final = (
    "bullionvault-multi-horizon-stage6-development-v1"
)


class Stage6DevelopmentOnlyEvaluator:
    """Evaluate new candidate families on development folds without touching test rows."""

    def __init__(
        self,
        *,
        model_factory: Stage6DevelopmentModelFactory | None = None,
        selection_gate: Stage6DevelopmentSelectionGate | None = None,
    ) -> None:
        self._models = model_factory or Stage6DevelopmentModelFactory()
        self._gate = selection_gate or Stage6DevelopmentSelectionGate()
        self._prereg = stage6_preregistration_payload()

    def evaluate(
        self,
        dataset: CausalHorizonDataset,
        plan: WalkForwardPlan,
    ) -> HorizonDevelopmentResult:
        if plan.total_rows != dataset.model_row_count:
            raise ValueError("Walk-forward plan row count does not match the dataset.")
        if plan.historical_test.row_count <= 0:
            raise ValueError("Locked historical-test partition must remain non-empty.")

        development = dataset.frame.iloc[: plan.development_end_exclusive].copy()
        x_development = development.loc[:, list(dataset.feature_columns)].to_numpy(dtype=float)
        y_development = development[dataset.target_column].to_numpy(dtype=float)
        if not np.isfinite(x_development).all() or not np.isfinite(y_development).all():
            raise ValueError("Stage-6 development matrices must be finite.")

        baseline_actual_parts: list[np.ndarray] = []
        baseline_prediction_parts: list[np.ndarray] = []
        baseline_fold_metrics = {}
        for fold in plan.folds:
            if fold.validation_end_exclusive > plan.development_end_exclusive:
                raise ValueError("Validation fold crosses into the locked historical test.")
            actual = y_development[fold.validation_start : fold.validation_end_exclusive]
            predicted = random_walk_zero_return(actual.size)
            baseline_actual_parts.append(actual.copy())
            baseline_prediction_parts.append(predicted)
            baseline_fold_metrics[fold.fold_number] = regression_metrics(actual, predicted)

        baseline_actual_oof = np.concatenate(baseline_actual_parts)
        baseline_prediction_oof = np.concatenate(baseline_prediction_parts)
        baseline_oof_metrics = regression_metrics(
            baseline_actual_oof,
            baseline_prediction_oof,
        )

        gate_payload = self._prereg["development_selection_gate"]
        if not isinstance(gate_payload, dict):
            raise ValueError("Malformed Stage-6 preregistered development gate.")

        results: list[CandidateDevelopmentResult] = []
        evidences: list[DevelopmentCandidateEvidence] = []
        decisions = []

        for candidate_spec in stage6_candidate_registry():
            candidate_actual_parts: list[np.ndarray] = []
            candidate_prediction_parts: list[np.ndarray] = []
            fold_comparisons: list[FoldComparison] = []

            for fold in plan.folds:
                x_train = x_development[fold.train_start : fold.train_end_exclusive]
                y_train = y_development[fold.train_start : fold.train_end_exclusive]
                x_validation = x_development[
                    fold.validation_start : fold.validation_end_exclusive
                ]
                y_validation = y_development[
                    fold.validation_start : fold.validation_end_exclusive
                ]

                estimator = self._models.create(candidate_spec)
                estimator.fit(x_train, y_train)
                predicted = np.asarray(estimator.predict(x_validation), dtype=float).reshape(-1)
                if predicted.size != y_validation.size or not np.isfinite(predicted).all():
                    raise ValueError(
                        f"{candidate_spec.candidate_id} produced invalid validation predictions."
                    )

                candidate_metrics = regression_metrics(y_validation, predicted)
                baseline_metrics = baseline_fold_metrics[fold.fold_number]
                candidate_actual_parts.append(y_validation.copy())
                candidate_prediction_parts.append(predicted)
                fold_comparisons.append(
                    FoldComparison(
                        fold_number=fold.fold_number,
                        train_row_count=fold.train_row_count,
                        validation_row_count=fold.validation_row_count,
                        baseline_mae=baseline_metrics.mae,
                        candidate_mae=candidate_metrics.mae,
                        baseline_directional_accuracy=baseline_metrics.directional_accuracy,
                        candidate_directional_accuracy=candidate_metrics.directional_accuracy,
                    )
                )

            candidate_actual_oof = np.concatenate(candidate_actual_parts)
            candidate_prediction_oof = np.concatenate(candidate_prediction_parts)
            if not np.array_equal(candidate_actual_oof, baseline_actual_oof):
                raise ValueError("Stage-6 candidate OOF rows diverged from baseline rows.")

            candidate_oof_metrics = regression_metrics(
                candidate_actual_oof,
                candidate_prediction_oof,
            )
            bootstrap = paired_block_bootstrap_mae_improvement(
                candidate_actual_oof,
                baseline_prediction_oof,
                candidate_prediction_oof,
                iterations=int(gate_payload["paired_block_bootstrap_iterations"]),
                block_length_rows=int(
                    gate_payload["paired_block_bootstrap_block_length_rows"]
                ),
                seed=int(gate_payload["paired_block_bootstrap_seed"]),
                ci_level=float(gate_payload["mae_improvement_ci"]),
            )
            evidence = DevelopmentCandidateEvidence(
                candidate_id=candidate_spec.candidate_id,
                candidate_oof_mae=candidate_oof_metrics.mae,
                baseline_oof_mae=baseline_oof_metrics.mae,
                better_folds=sum(
                    comparison.candidate_mae_better for comparison in fold_comparisons
                ),
                directional_accuracy_delta=(
                    candidate_oof_metrics.directional_accuracy
                    - baseline_oof_metrics.directional_accuracy
                ),
                bootstrap_ci_low=bootstrap.ci_low,
                bootstrap_probability_better=bootstrap.probability_candidate_mae_better,
            )
            decision = self._gate.evaluate(evidence)
            evidences.append(evidence)
            decisions.append(decision)
            results.append(
                CandidateDevelopmentResult(
                    candidate_id=candidate_spec.candidate_id,
                    oof_metrics=candidate_oof_metrics,
                    baseline_oof_metrics=baseline_oof_metrics,
                    fold_comparisons=tuple(fold_comparisons),
                    bootstrap=bootstrap,
                    gate_passed=decision.passed,
                    gate_checks=decision.checks,
                )
            )

        winner = self._gate.choose_winner(tuple(evidences), tuple(decisions))
        return HorizonDevelopmentResult(
            horizon_key=dataset.horizon_key,
            interval_seconds=dataset.interval_seconds,
            total_model_rows=dataset.model_row_count,
            development_rows=plan.development_end_exclusive,
            development_oof_rows=baseline_oof_metrics.row_count,
            historical_test_start=plan.historical_test.start,
            historical_test_end_exclusive=plan.historical_test.end_exclusive,
            historical_test_rows=plan.historical_test.row_count,
            baseline_oof_metrics=baseline_oof_metrics,
            candidates=tuple(results),
            selected_id=winner.selected_id,
            selected_kind=winner.selected_kind,
            selection_reason=winner.reason,
        )


class Stage6DevelopmentRunner:
    def __init__(
        self,
        *,
        dataset_builder: MultiHorizonDatasetBuilder | None = None,
        planner: ExpandingWalkForwardPlanner | None = None,
        evaluator: Stage6DevelopmentOnlyEvaluator | None = None,
    ) -> None:
        self._builder = dataset_builder or MultiHorizonDatasetBuilder()
        self._planner = planner or ExpandingWalkForwardPlanner()
        self._evaluator = evaluator or Stage6DevelopmentOnlyEvaluator()

    def run_all(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage6_development_version": STAGE6_DEVELOPMENT_VERSION,
            "preregistration_fingerprint_sha256": (
                stage6_preregistration_fingerprint_sha256()
            ),
            "performance_metrics_computed": True,
            "performance_scope": "DEVELOPMENT_ONLY",
            "historical_test_metrics_read": False,
            "historical_test_predictions_computed": False,
            "future_holdout_read": False,
            "forward_bars_used_for_fit": False,
            "microstructure_used_for_fit": False,
            "selection_gate": Stage6DevelopmentSelectionGate().thresholds,
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
            "horizons": {},
        }
        horizons = payload["horizons"]
        assert isinstance(horizons, dict)
        for key in ("4h", "12h", "1d", "2d", "30d"):
            try:
                dataset, _ = self._builder.build(key)
            except DataPendingError:
                spec = get_horizon(key)
                horizons[key] = {
                    "horizon_key": key,
                    "interval_seconds": spec.interval_seconds,
                    "state": "DATA_PENDING",
                    "performance_metrics_computed": False,
                    "selection": None,
                }
                continue
            plan = self._planner.plan(dataset.model_row_count)
            horizons[key] = self._evaluator.evaluate(dataset, plan).as_dict()
        return payload
