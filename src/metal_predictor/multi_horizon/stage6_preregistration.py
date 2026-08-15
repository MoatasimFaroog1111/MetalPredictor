from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final

from metal_predictor.multi_horizon.feature_set import (
    FEATURE_COLUMNS,
    FEATURE_SET_VERSION,
    feature_fingerprint_sha256,
)
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner
from metal_predictor.multi_horizon.targets import TARGET_VERSION


STAGE6_PREREGISTRATION_VERSION: Final = (
    "bullionvault-multi-horizon-candidate-expansion-v2"
)
STAGE3_REPORT_SHA256: Final = (
    "279b824c0775710b9b60a03a39564519e5ed728a54caca4febcefd71a24586f9"
)


@dataclass(frozen=True)
class Stage6CandidateSpec:
    candidate_id: str
    estimator: str
    parameters: dict[str, object]
    preprocessing: tuple[str, ...]
    rationale: str


def stage6_candidate_registry() -> tuple[Stage6CandidateSpec, ...]:
    """Return the complete candidate set locked before Stage-6 development scoring."""

    return (
        Stage6CandidateSpec(
            candidate_id="train_median_return_v2",
            estimator="metal_predictor.multi_horizon.stage6_models.TrainMedianReturnRegressor",
            parameters={},
            preprocessing=(),
            rationale=(
                "Absolute-error-optimal constant estimated from train rows only; tests whether "
                "a stable non-zero conditional drift beats the zero-return random-walk baseline."
            ),
        ),
        Stage6CandidateSpec(
            candidate_id="ridge_alpha_1000_v2",
            estimator="sklearn.linear_model.Ridge",
            parameters={"alpha": 1000.0},
            preprocessing=("StandardScaler(train_only)",),
            rationale=(
                "Very strong linear shrinkage toward the zero-return baseline to reduce the "
                "variance that hurt the Stage-3 linear candidates."
            ),
        ),
        Stage6CandidateSpec(
            candidate_id="random_forest_shallow_v2",
            estimator="sklearn.ensemble.RandomForestRegressor",
            parameters={
                "n_estimators": 500,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "max_features": 0.7,
                "bootstrap": True,
                "random_state": 20260815,
                "n_jobs": 1,
            },
            preprocessing=(),
            rationale=(
                "Low-depth bagged nonlinear model with deliberately large leaves for the small "
                "historical samples; no hyperparameter search is permitted."
            ),
        ),
        Stage6CandidateSpec(
            candidate_id="hist_gradient_boosting_shallow_v2",
            estimator="sklearn.ensemble.HistGradientBoostingRegressor",
            parameters={
                "loss": "absolute_error",
                "learning_rate": 0.03,
                "max_iter": 120,
                "max_leaf_nodes": 3,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 20260815,
            },
            preprocessing=(),
            rationale=(
                "Regularized shallow boosting under absolute loss, fixed in advance to test "
                "small nonlinear interactions without a tuning search."
            ),
        ),
    )


def stage6_preregistration_payload() -> dict[str, object]:
    planner = ExpandingWalkForwardPlanner()
    candidates = stage6_candidate_registry()
    return {
        "preregistration_version": STAGE6_PREREGISTRATION_VERSION,
        "created_for": "MULTI_HORIZON_STAGE6_DEVELOPMENT_ONLY",
        "predecessor_stage3_report_sha256": STAGE3_REPORT_SHA256,
        "scientific_reason_for_new_version": (
            "Stage 3 was a valid negative result. Stage 6 introduces genuinely new, "
            "predeclared candidate families while preserving the same immutable source data, "
            "causal feature graph, target, split planner, and sealed historical-test boundary."
        ),
        "scope": {
            "horizons": ["4h", "12h", "2d", "30d"],
            "blocked_horizons": {
                "1d": (
                    "DATA_PENDING: no direct historical 1d BullionVault chart dataset exists; "
                    "live forward bars are not resampled into training history."
                )
            },
            "cross_horizon_pooling": False,
            "one_independent_selection_per_horizon": True,
            "historical_source": "BullionVault Chart Export HLC",
            "forward_bars_used_for_fit": False,
            "bullionvault_microstructure_used_for_fit": False,
            "source_domain_merging": False,
        },
        "features": {
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_fingerprint_sha256": feature_fingerprint_sha256(),
            "feature_set_changed_from_stage3": False,
            "causal_only": True,
            "calendar_features_allowed": False,
            "forward_fill": False,
            "backward_fill": False,
            "interpolation": False,
            "resampling": False,
        },
        "target": {
            "target_version": TARGET_VERSION,
            "semantics": "log(next_source_bar_close / current_source_bar_close)",
            "changed_from_stage3": False,
        },
        "baseline": {
            "baseline_id": "random_walk_zero_return",
            "predicted_log_return": 0.0,
        },
        "candidates": [
            {
                "candidate_id": spec.candidate_id,
                "estimator": spec.estimator,
                "parameters": spec.parameters,
                "preprocessing": list(spec.preprocessing),
                "rationale": spec.rationale,
            }
            for spec in candidates
        ],
        "walk_forward": {
            "planner": "ExpandingWalkForwardPlanner",
            "fold_count": planner.fold_count,
            "purge_bars": planner.purge_bars,
            "minimum_train_rows": planner.minimum_train_rows,
            "minimum_validation_rows": planner.minimum_validation_rows,
            "final_historical_test_fraction": planner.final_test_fraction,
            "minimum_final_test_rows": planner.minimum_test_rows,
            "split_changed_from_stage3": False,
            "historical_test_used_for_stage6_development": False,
        },
        "development_selection_gate": {
            "primary_metric": "MAE(target_log_return)",
            "candidate_must_beat_random_walk_mae": True,
            "minimum_better_folds": 3,
            "fold_count": planner.fold_count,
            "paired_block_bootstrap_iterations": 5000,
            "paired_block_bootstrap_block_length_rows": 4,
            "paired_block_bootstrap_seed": 20260815,
            "mae_improvement_ci": 0.95,
            "mae_improvement_ci_lower_must_be_positive": True,
            "probability_candidate_mae_better_minimum": 0.99,
            "directional_accuracy_delta_minimum": -0.02,
            "if_no_candidate_passes": "RETAIN_RANDOM_WALK_BASELINE",
            "winner_rule": (
                "largest development OOF MAE improvement among candidates passing every gate"
            ),
            "tie_break_order": [spec.candidate_id for spec in candidates],
            "hyperparameter_search_allowed": False,
            "post_result_candidate_editing_allowed": False,
        },
        "historical_confirmation_policy": {
            "authorized_during_preregistration": False,
            "authorization_condition": (
                "Exactly the selected Stage-6 development winner for a horizon may read the "
                "still-sealed historical-test block in a separate, subsequent confirmation run."
            ),
            "test_read_before_development_pass": False,
            "post_test_tuning_allowed": False,
            "failure_action": (
                "DO_NOT_DEPLOY_CANDIDATE; keep baseline and require a newly versioned future study"
            ),
        },
        "production_policy": {
            "candidate_model_artifacts_created_now": False,
            "candidate_forecast_routes_changed_now": False,
            "automatic_promotion": False,
            "live_52_feature_model_mutated": False,
            "shadow62_mutated": False,
        },
        "guardrails": {
            "edge_status": "NOT_PROVEN",
            "research_only": True,
            "buy_sell_enabled": False,
            "execution_enabled": False,
            "future_holdout_read": False,
            "historical_test_metrics_read": False,
        },
    }


def stage6_preregistration_fingerprint_sha256() -> str:
    canonical = json.dumps(
        stage6_preregistration_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
