from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.frozen_ridge import FrozenRidgeExporter, FrozenRidgeRegressor
from metal_predictor.modeling import DefaultModelRegistry


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("forward_holdout")
SOURCE_DATASET_BLOB_SHA = "9b95fcc5aa2679208c6b5c44c830ce6b1eaa5829"
RESEARCH_CUTOFF_COMMIT = "8331dea6dd8c63e432a5df963f085721410e49b2"


def _development() -> tuple[pd.DataFrame, tuple[str, ...]]:
    manifest = json.loads(
        (PROCESSED_DIR / "feature_manifest.json").read_text(encoding="utf-8")
    )
    feature_names = tuple(manifest["features"])
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    development = pd.concat([train, validation], ignore_index=True)
    development["timestamp_utc"] = pd.to_datetime(
        development["timestamp_utc"], utc=True, errors="raise"
    )
    development["target_timestamp_utc"] = pd.to_datetime(
        development["target_timestamp_utc"], utc=True, errors="raise"
    )
    development = development.sort_values("timestamp_utc").reset_index(drop=True)
    return development, feature_names


def _equivalence_error(spec, payload, development, feature_names) -> float:
    sklearn_model = spec.factory()
    sklearn_model.fit(
        development.loc[:, feature_names],
        development["target_log_return_1h"],
    )
    # Use a deterministic spread across the full development history, not the historical Test.
    positions = np.linspace(
        0, len(development) - 1, num=min(512, len(development)), dtype=int
    )
    sample = development.iloc[positions]
    expected = np.asarray(
        sklearn_model.predict(sample.loc[:, feature_names]), dtype=float
    )
    frozen = FrozenRidgeRegressor(payload).predict(sample.loc[:, feature_names])
    return float(np.max(np.abs(expected - frozen)))


def build() -> dict[str, object]:
    development, feature_names = _development()
    registry = {
        spec.name: spec
        for spec in DefaultModelRegistry(random_state=42).candidates()
    }
    exporter = FrozenRidgeExporter()
    model_plan = {
        "primary": registry["ridge_alpha_10"],
        "benchmark": registry["ridge_alpha_100"],
    }
    models: dict[str, dict[str, object]] = {}
    equivalence: dict[str, float] = {}
    for role, spec in model_plan.items():
        payload = exporter.export(
            spec,
            development,
            feature_names,
            SOURCE_DATASET_BLOB_SHA,
            RESEARCH_CUTOFF_COMMIT,
        )
        path = OUTPUT_DIR / "models" / f"{spec.name}.json"
        exporter.write(payload, path)
        error = _equivalence_error(spec, payload, development, feature_names)
        if error > 1e-12:
            raise AssertionError(
                f"Frozen model {spec.name} does not reproduce sklearn; max error={error}"
            )
        equivalence[spec.name] = error
        models[role] = {
            "name": spec.name,
            "path": str(path),
            "payload_sha256": payload["model_payload_sha256"],
            "alpha": payload["alpha"],
        }

    manifest = {
        "schema_version": 1,
        "freeze_id": "future-holdout-v1-20260810",
        "protocol_predeclared_utc": "2026-08-10T07:46:14Z",
        "research_code_cutoff_commit": RESEARCH_CUTOFF_COMMIT,
        "historical_source_dataset_path": "XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet",
        "historical_source_dataset_git_blob_sha": SOURCE_DATASET_BLOB_SHA,
        "historical_test_forbidden_for_freeze_fit_or_tuning": True,
        "context_start_utc": "2026-08-07T22:00:00Z",
        "holdout_first_bar_start_utc": "2026-08-11T00:00:00Z",
        "holdout_last_feature_bar_start_exclusive_utc": "2027-02-07T00:00:00Z",
        "earliest_final_score_utc": "2027-02-07T02:00:00Z",
        "minimum_exact_hour_outcomes": 2500,
        "fixed_window_days": 180,
        "source": {
            "provider": "HistData",
            "symbol": "XAGUSD",
            "market_type": "spot_bid",
            "source_timezone_policy": "fixed EST UTC-05:00 converted to UTC",
            "aggregation": "M1 to H1 conservative aggregation; suspicious source hours excluded",
            "batch_future_holdout_not_live_execution": True,
        },
        "models": models,
        "primary_hypothesis": {
            "model": "ridge_alpha_10",
            "selection_locked_before_future_window": True,
            "selection_reason": (
                "Stage 6 found ridge_alpha_10 had the best observed pre-cost strategy Sharpe "
                "among the common development OOF strategy matrix, but its Deflated Sharpe "
                "probability was only 26.58%; therefore it is a single predeclared future "
                "challenger rather than a proven model."
            ),
        },
        "benchmark_hypothesis": {
            "model": "ridge_alpha_100",
            "role": "frozen baseline-v1 benchmark",
        },
        "final_score_rules": {
            "one_shot_fixed_window": True,
            "no_interim_performance_metrics": True,
            "bootstrap_block_rows": 24,
            "bootstrap_resamples": 5000,
            "random_state": 42,
            "six_sequential_blocks_days_each": 30,
            "required_positive_blocks": 4,
            "all_required_gates": [
                "primary directional-accuracy 95% block-bootstrap lower bound > 50%",
                "primary pre-cost directional strategy mean 95% lower bound > 0",
                "primary MAE improvement vs zero-return 95% lower bound > 0",
                "primary MAE improvement vs ridge_alpha_100 95% lower bound > 0",
                "positive pre-cost strategy mean in at least 4 of 6 predeclared 30-day blocks",
            ],
        },
        "interpretation_limit": (
            "A pass can establish frozen future pre-cost predictive evidence only. "
            "Spot-bid H1 history does not establish executable spread, slippage, fees, "
            "or live order-fill performance."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = {
        "status": "PASS",
        "development_rows": int(len(development)),
        "feature_count": len(feature_names),
        "training_first_timestamp_utc": development["timestamp_utc"].min().isoformat(),
        "training_last_timestamp_utc": development["timestamp_utc"].max().isoformat(),
        "historical_test_loaded": False,
        "models": models,
        "max_prediction_equivalence_error": equivalence,
        "performance_metrics_computed": False,
    }
    (OUTPUT_DIR / "model_freeze_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
