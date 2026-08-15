from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final


STAGE7_PREREGISTRATION_VERSION: Final = "xag-h1-direct-multi-horizon-stage7-prereg-v1"
STAGE7_SOURCE_PATH: Final = "XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet"
STAGE7_SOURCE_GIT_BLOB_SHA1: Final = "9b95fcc5aa2679208c6b5c44c830ce6b1eaa5829"
STAGE7_FEATURE_GRAPH_VERSION: Final = "canonical-h1-52-causal-v1"


@dataclass(frozen=True)
class Stage7HorizonSpec:
    key: str
    hours: int


@dataclass(frozen=True)
class Stage7CandidateSpec:
    candidate_id: str
    estimator: str
    parameters: tuple[tuple[str, object], ...]
    preprocessing: tuple[str, ...] = ()


_HORIZONS: Final = (
    Stage7HorizonSpec("4h", 4),
    Stage7HorizonSpec("12h", 12),
    Stage7HorizonSpec("1d", 24),
    Stage7HorizonSpec("2d", 48),
    Stage7HorizonSpec("30d", 720),
)

_CANDIDATES: Final = (
    Stage7CandidateSpec(
        candidate_id="ridge_alpha_100_stage7",
        estimator="sklearn.linear_model.Ridge",
        parameters=(("alpha", 100.0),),
        preprocessing=(
            "SimpleImputer(strategy=median,add_indicator=True,keep_empty_features=True,train_only)",
            "StandardScaler(train_only)",
        ),
    ),
    Stage7CandidateSpec(
        candidate_id="histgb_absolute_stage7",
        estimator="sklearn.ensemble.HistGradientBoostingRegressor",
        parameters=(
            ("loss", "absolute_error"),
            ("learning_rate", 0.04),
            ("max_iter", 350),
            ("max_leaf_nodes", 15),
            ("min_samples_leaf", 60),
            ("l2_regularization", 0.1),
            ("early_stopping", False),
            ("random_state", 42),
        ),
        preprocessing=(
            "SimpleImputer(strategy=median,add_indicator=True,keep_empty_features=True,train_only)",
        ),
    ),
    Stage7CandidateSpec(
        candidate_id="extra_trees_shallow_stage7",
        estimator="sklearn.ensemble.ExtraTreesRegressor",
        parameters=(
            ("n_estimators", 250),
            ("min_samples_leaf", 24),
            ("max_features", 0.65),
            ("max_depth", 16),
            ("random_state", 42),
            ("n_jobs", 2),
        ),
        preprocessing=(
            "SimpleImputer(strategy=median,add_indicator=True,keep_empty_features=True,train_only)",
        ),
    ),
)


def stage7_horizons() -> tuple[Stage7HorizonSpec, ...]:
    return _HORIZONS


def stage7_candidates() -> tuple[Stage7CandidateSpec, ...]:
    return _CANDIDATES


def stage7_preregistration_payload() -> dict[str, object]:
    return {
        "version": STAGE7_PREREGISTRATION_VERSION,
        "source": {
            "path": STAGE7_SOURCE_PATH,
            "git_blob_sha1": STAGE7_SOURCE_GIT_BLOB_SHA1,
            "bar_interval": "H1",
            "instrument": "XAG/USD",
            "unit": "USD_PER_KG",
            "forward_bars_merged": False,
            "microstructure_merged": False,
        },
        "horizons": [
            {"key": spec.key, "hours": spec.hours} for spec in stage7_horizons()
        ],
        "feature_graph": {
            "version": STAGE7_FEATURE_GRAPH_VERSION,
            "feature_count": 52,
            "components": [
                "PriceActionFeatures",
                "MomentumFeatures",
                "VolatilityFeatures",
                "TrendFeatures",
                "TemporalFeatures",
                "QualityFeatures",
            ],
            "causal_only": True,
            "frozen_live_52_graph_mutated": False,
        },
        "target": {
            "definition": "log(close_at_exact_t_plus_h / close_at_t)",
            "future_timestamp_rule": "EXACT_UTC_CLOCK_ONLY",
            "nearest_or_asof_allowed": False,
            "fill_or_interpolation_allowed": False,
            "one_independent_target_per_horizon": True,
        },
        "development_protocol": {
            "historical_test_fraction": 0.20,
            "historical_test_locked_during_development": True,
            "development_initial_train_fraction": 0.50,
            "expanding_walk_forward_folds": 4,
            "purge_hours": "EQUAL_TO_TARGET_HORIZON_HOURS",
            "embargo_hours": "EQUAL_TO_TARGET_HORIZON_HOURS",
            "future_training_after_validation_allowed": False,
            "test_metrics_authorized_during_stage7": False,
            "hyperparameter_search_allowed": False,
            "post_result_candidate_editing_allowed": False,
        },
        "baseline": {
            "id": "random_walk_zero_return",
            "prediction": 0.0,
        },
        "candidates": [
            {
                "candidate_id": spec.candidate_id,
                "estimator": spec.estimator,
                "parameters": dict(spec.parameters),
                "preprocessing": list(spec.preprocessing),
            }
            for spec in stage7_candidates()
        ],
        "development_selection_gate": {
            "primary_metric": "MAE(target_log_return_h)",
            "candidate_must_beat_random_walk_mae": True,
            "minimum_better_folds": 3,
            "fold_count": 4,
            "paired_block_bootstrap_iterations": 5000,
            "paired_block_bootstrap_seed": 20260815,
            "paired_block_bootstrap_block_length_rows": "MAX_24_OR_HORIZON_HOURS",
            "mae_improvement_ci": 0.95,
            "mae_improvement_ci_lower_must_be_positive": True,
            "probability_candidate_mae_better_minimum": 0.99,
            "directional_accuracy_delta_minimum": -0.02,
            "winner_rule": "largest development OOF MAE improvement among candidates passing every gate",
            "if_no_candidate_passes": "RETAIN_RANDOM_WALK_BASELINE",
        },
        "guardrails": {
            "edge_status": "NOT_PROVEN",
            "research_only": True,
            "buy_sell_enabled": False,
            "execution_enabled": False,
            "automatic_live_promotion": False,
            "live_model_mutated": False,
            "formal_future_holdout_read": False,
            "shadow62_mutated": False,
        },
    }


def stage7_preregistration_fingerprint_sha256() -> str:
    canonical = json.dumps(
        stage7_preregistration_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
