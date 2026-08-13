from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from metal_predictor.multi_horizon.dataset import (
    CausalHorizonDataset,
    DataPendingError,
    MultiHorizonDatasetBuilder,
)
from metal_predictor.multi_horizon.models import (
    DevelopmentModelFactory,
    random_walk_zero_return,
)
from metal_predictor.multi_horizon.preregistration import (
    candidate_registry,
    preregistration_fingerprint_sha256,
    preregistration_payload,
)
from metal_predictor.multi_horizon.registry import get_horizon
from metal_predictor.multi_horizon.selection import (
    DevelopmentCandidateEvidence,
    DevelopmentSelectionGate,
)
from metal_predictor.multi_horizon.split import (
    ExpandingWalkForwardPlanner,
    WalkForwardPlan,
)
from metal_predictor.multi_horizon.statistics import (
    PairedBlockBootstrapResult,
    RegressionMetrics,
    paired_block_bootstrap_mae_improvement,
    regression_metrics,
)


STAGE3_VERSION: Final = "bullionvault-multi-horizon-development-selection-v1"


@dataclass(frozen=True)
class FoldComparison:
    fold_number: int
    train_row_count: int
    validation_row_count: int
    baseline_mae: float
    candidate_mae: float
    baseline_directional_accuracy: float
    candidate_directional_accuracy: float

    @property
    def candidate_mae_better(self) -> bool:
        return self.candidate_mae < self.baseline_mae

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_number": self.fold_number,
            "train_row_count": self.train_row_count,
            "validation_row_count": self.validation_row_count,
            "baseline_mae": self.baseline_mae,
            "candidate_mae": self.candidate_mae,
            "mae_improvement": self.baseline_mae - self.candidate_mae,
            "candidate_mae_better": self.candidate_mae_better,
            "baseline_directional_accuracy": self.baseline_directional_accuracy,
            "candidate_directional_accuracy": self.candidate_directional_accuracy,
            "directional_accuracy_delta": (
                self.candidate_directional_accuracy - self.baseline_directional_accuracy
            ),
        }


@dataclass(frozen=True)
class CandidateDevelopmentResult:
    candidate_id: str
    oof_metrics: RegressionMetrics
    baseline_oof_metrics: RegressionMetrics
    fold_comparisons: tuple[FoldComparison, ...]
    bootstrap: PairedBlockBootstrapResult
    gate_passed: bool
    gate_checks: dict[str, bool]

    @property
    def better_folds(self) -> int:
        return sum(comparison.candidate_mae_better for comparison in self.fold_comparisons)

    @property
    def mae_improvement(self) -> float:
        return self.baseline_oof_metrics.mae - self.oof_metrics.mae

    @property
    def mae_improvement_pct(self) -> float:
        if self.baseline_oof_metrics.mae == 0.0:
            return 0.0
        return 100.0 * self.mae_improvement / self.baseline_oof_metrics.mae

    @property
    def directional_accuracy_delta(self) -> float:
        return (
            self.oof_metrics.directional_accuracy
            - self.baseline_oof_metrics.directional_accuracy
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "development_oof": self.oof_metrics.as_dict(),
            "random_walk_development_oof": self.baseline_oof_metrics.as_dict(),
            "mae_improvement": self.mae_improvement,
            "mae_improvement_pct": self.mae_improvement_pct,
            "directional_accuracy_delta": self.directional_accuracy_delta,
            "better_folds": self.better_folds,
            "folds": [comparison.as_dict() for comparison in self.fold_comparisons],
            "paired_block_bootstrap": self.bootstrap.as_dict(),
            "development_gate": {
                "passed": self.gate_passed,
                "checks": dict(self.gate_checks),
            },
        }


@dataclass(frozen=True)
class HorizonDevelopmentResult:
    horizon_key: str
    interval_seconds: int
    total_model_rows: int
    development_rows: int
    development_oof_rows: int
    historical_test_start: int
    historical_test_end_exclusive: int
    historical_test_rows: int
    baseline_oof_metrics: RegressionMetrics
    candidates: tuple[CandidateDevelopmentResult, ...]
    selected_id: str
    selected_kind: str
    selection_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon_key": self.horizon_key,
            "interval_seconds": self.interval_seconds,
            "total_model_rows": self.total_model_rows,
            "development_rows": self.development_rows,
            "development_oof_rows": self.development_oof_rows,
            "performance_scope": "DEVELOPMENT_ONLY",
            "random_walk_development_oof": self.baseline_oof_metrics.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "selection": {
                "selected_id": self.selected_id,
                "selected_kind": self.selected_kind,
                "reason": self.selection_reason,
            },
            "historical_test": {
                "start": self.historical_test_start,
                "end_exclusive": self.historical_test_end_exclusive,
                "row_count": self.historical_test_rows,
                "metrics_read": False,
                "predictions_computed": False,
                "used_for_fit": False,
                "used_for_selection": False,
                "outcomes_reported": False,
            },
        }


class DevelopmentOnlyEvaluator:
    """Fit and compare candidates only on preregistered development folds."""

    def __init__(
        self,
        *,
        model_factory: DevelopmentModelFactory | None = None,
        selection_gate: DevelopmentSelectionGate | None = None,
    ) -> None:
        self._models = model_factory or DevelopmentModelFactory()
        self._gate = selection_gate or DevelopmentSelectionGate()
        self._prereg = preregistration_payload()

    def evaluate(
        self,
        dataset: CausalHorizonDataset,
        plan: WalkForwardPlan,
    ) -> HorizonDevelopmentResult:
        if plan.total_rows != dataset.model_row_count:
            raise ValueError("Walk-forward plan row count does not match the dataset.")
        if plan.development_end_exclusive <= 0:
            raise ValueError("Development partition must contain rows.")
        if plan.historical_test.row_count <= 0:
            raise ValueError("Locked historical-test partition must remain non-empty.")

        # Scientific firewall: all matrices used below are materialized from the
        # development prefix only. Historical-test rows are never sliced into X/y,
        # fitted, predicted, scored, or reported in Stage 3.
        development = dataset.frame.iloc[: plan.development_end_exclusive].copy()
        x_development = development.loc[:, list(dataset.feature_columns)].to_numpy(dtype=float)
        y_development = development[dataset.target_column].to_numpy(dtype=float)
        if not np.isfinite(x_development).all() or not np.isfinite(y_development).all():
            raise ValueError("Development data must be finite.")

        baseline_actual_parts: list[np.ndarray] = []
        baseline_prediction_parts: list[np.ndarray] = []
        baseline_fold_metrics: dict[int, RegressionMetrics] = {}
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
            raise ValueError("Malformed preregistered development gate.")

        results: list[CandidateDevelopmentResult] = []
        evidences: list[DevelopmentCandidateEvidence] = []
        decisions = []

        for candidate_spec in candidate_registry():
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
                if not np.isfinite(predicted).all():
                    raise ValueError(
                        f"{candidate_spec.candidate_id} produced non-finite validation predictions."
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
                raise ValueError("Candidate OOF rows diverged from baseline OOF rows.")
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


class Stage3DevelopmentRunner:
    def __init__(
        self,
        *,
        dataset_builder: MultiHorizonDatasetBuilder | None = None,
        planner: ExpandingWalkForwardPlanner | None = None,
        evaluator: DevelopmentOnlyEvaluator | None = None,
    ) -> None:
        self._builder = dataset_builder or MultiHorizonDatasetBuilder()
        self._planner = planner or ExpandingWalkForwardPlanner()
        self._evaluator = evaluator or DevelopmentOnlyEvaluator()

    def run_all(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage3_version": STAGE3_VERSION,
            "preregistration_fingerprint_sha256": preregistration_fingerprint_sha256(),
            "performance_metrics_computed": True,
            "performance_scope": "DEVELOPMENT_ONLY",
            "historical_test_metrics_read": False,
            "historical_test_predictions_computed": False,
            "future_holdout_read": False,
            "selection_gate": DevelopmentSelectionGate().thresholds,
            "guardrails": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
                "future_holdout_read": False,
                "shadow62_mutated": False,
                "automatic_live_promotion": False,
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
