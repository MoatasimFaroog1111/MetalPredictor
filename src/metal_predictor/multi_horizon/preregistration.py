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


PREREGISTRATION_VERSION: Final = "bullionvault-multi-horizon-model-selection-v1"


@dataclass(frozen=True)
class CandidateModelSpec:
    candidate_id: str
    estimator: str
    parameters: dict[str, object]
    preprocessing: tuple[str, ...]


def candidate_registry() -> tuple[CandidateModelSpec, ...]:
    return (
        CandidateModelSpec(
            candidate_id="ridge_alpha_10",
            estimator="sklearn.linear_model.Ridge",
            parameters={"alpha": 10.0},
            preprocessing=("StandardScaler(train_only)",),
        ),
        CandidateModelSpec(
            candidate_id="huber_v1",
            estimator="sklearn.linear_model.HuberRegressor",
            parameters={"epsilon": 1.35, "alpha": 0.0001, "max_iter": 1000},
            preprocessing=("StandardScaler(train_only)",),
        ),
        CandidateModelSpec(
            candidate_id="elastic_net_v1",
            estimator="sklearn.linear_model.ElasticNet",
            parameters={
                "alpha": 0.0005,
                "l1_ratio": 0.20,
                "max_iter": 10000,
                "selection": "cyclic",
            },
            preprocessing=("StandardScaler(train_only)",),
        ),
    )


def preregistration_payload() -> dict[str, object]:
    planner = ExpandingWalkForwardPlanner()
    candidates = candidate_registry()
    return {
        "preregistration_version": PREREGISTRATION_VERSION,
        "scope": {
            "horizons": ["4h", "12h", "2d", "30d"],
            "blocked_horizons": {"1d": "DATA_PENDING"},
            "one_model_artifact_per_horizon": True,
            "cross_horizon_pooling": False,
            "historical_source": "BullionVault Chart Export HLC",
            "timestamp_semantics": "UNVERIFIED_EXPORT_CLIENT_TIMEZONE",
            "calendar_or_hour_of_day_features_allowed": False,
        },
        "features": {
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_fingerprint_sha256": feature_fingerprint_sha256(),
            "causal_only": True,
            "max_lookback_bars": 6,
            "no_forward_fill": True,
            "no_backward_fill": True,
            "no_interpolation": True,
            "no_resampling": True,
            "no_open_fabrication": True,
        },
        "target": {
            "target_version": TARGET_VERSION,
            "column": "target_log_return",
            "semantics": "log(next_source_bar_close / current_source_bar_close)",
            "target_timestamp": "exactly one registered source interval after feature timestamp",
            "baseline_reconstruction": "predicted_close = current_close * exp(predicted_log_return)",
        },
        "baseline": {
            "baseline_id": "random_walk_zero_return",
            "predicted_log_return": 0.0,
            "predicted_close": "current_close_usd_per_kg",
        },
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "estimator": candidate.estimator,
                "parameters": candidate.parameters,
                "preprocessing": list(candidate.preprocessing),
            }
            for candidate in candidates
        ],
        "walk_forward": {
            "fold_count": planner.fold_count,
            "purge_bars": planner.purge_bars,
            "expanding_train": True,
            "final_historical_test_fraction": planner.final_test_fraction,
            "minimum_final_test_rows": planner.minimum_test_rows,
            "minimum_train_rows": planner.minimum_train_rows,
            "minimum_validation_rows": planner.minimum_validation_rows,
            "model_selection_uses_historical_test": False,
            "preprocessing_fit_train_only": True,
        },
        "development_selection_gate": {
            "primary_metric": "MAE(target_log_return)",
            "secondary_metric": "directional_accuracy",
            "candidate_must_beat_random_walk_mae": True,
            "minimum_better_folds": 3,
            "fold_count": planner.fold_count,
            "paired_block_bootstrap_iterations": 5000,
            "paired_block_bootstrap_block_length_rows": 4,
            "paired_block_bootstrap_seed": 20260813,
            "mae_improvement_ci": 0.95,
            "mae_improvement_ci_lower_must_be_positive": True,
            "probability_candidate_mae_better_minimum": 0.9833333333333333,
            "probability_note": (
                "Bootstrap probability is descriptive evidence, not a classical p-value; "
                "the high threshold is preregistered to account conservatively for three candidates."
            ),
            "directional_accuracy_delta_minimum": -0.02,
            "if_no_candidate_passes": "RETAIN_RANDOM_WALK_BASELINE",
            "winner_rule": "largest development OOF MAE improvement among candidates that pass all gates",
            "tie_break_order": ["ridge_alpha_10", "huber_v1", "elastic_net_v1"],
        },
        "locked_historical_confirmation_gate": {
            "test_read_policy": "READ_ONCE_AFTER_CANDIDATE_AND_FEATURE_SET_ARE_LOCKED",
            "primary_metric": "MAE(target_log_return)",
            "candidate_must_beat_random_walk_mae": True,
            "paired_block_bootstrap_iterations": 5000,
            "paired_block_bootstrap_block_length_rows": 4,
            "paired_block_bootstrap_seed": 20260814,
            "mae_improvement_ci": 0.95,
            "mae_improvement_ci_lower_must_be_positive": True,
            "probability_candidate_mae_better_minimum": 0.95,
            "directional_accuracy_delta_minimum": -0.02,
            "post_test_feature_or_hyperparameter_tuning_allowed": False,
            "failure_action": "DO_NOT_PROMOTE; NEW_VERSION_AND_NEW_FORWARD_SHADOW_REQUIRED",
        },
        "stage2_execution": {
            "fit_candidate_models": False,
            "read_locked_historical_test_metrics": False,
            "publish_forecasts": False,
            "create_ui_pages": False,
            "performance_metrics_computed": False,
        },
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
    }


def preregistration_fingerprint_sha256() -> str:
    canonical = json.dumps(
        preregistration_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
