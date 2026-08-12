from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.cross_asset_experiment import DevelopmentFeatureSetLoader
from metal_predictor.metrics import PairedBlockBootstrapComparison, RegressionForecastMetrics
from metal_predictor.modeling import DefaultModelRegistry, ModelSpec
from metal_predictor.precious_metals.confirmation import (
    CANDIDATE_FEATURES,
    CANDIDATE_FAMILIES,
    CANDIDATE_ID,
    CONFIRMATION_VERSION,
    HistoricalConfirmationPolicy,
    candidate_fingerprint,
)


BASE_DIR = Path("data/processed")
ENHANCED_DIR = Path("data/processed_precious_metals")
OUTPUT_DIR = Path("artifacts/cross_asset_precious_metals/historical_confirmation_v1")


def _model(name: str) -> ModelSpec:
    return next(spec for spec in DefaultModelRegistry(random_state=42).candidates() if spec.name == name)


def _load_test(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path / "test.parquet")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="raise")
    if frame.empty or frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"Historical Test frame is empty or duplicated: {path}")
    if not frame["timestamp_utc"].is_monotonic_increasing:
        raise ValueError(f"Historical Test frame is not chronological: {path}")
    if not frame["target_timestamp_utc"].sub(frame["timestamp_utc"]).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Historical Test contains non-exact-hour targets.")
    return frame


def _assert_same_test(base: pd.DataFrame, enhanced: pd.DataFrame) -> None:
    if not base["timestamp_utc"].equals(enhanced["timestamp_utc"]):
        raise ValueError("Base and enhanced historical Test timestamps differ.")
    if not np.allclose(
        base["target_log_return_1h"].to_numpy(float),
        enhanced["target_log_return_1h"].to_numpy(float),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Base and enhanced historical Test targets differ.")


def _coverage(frame: pd.DataFrame) -> dict[str, float]:
    required = {"xpt_has_exact_current", "xpd_has_exact_current", "both_metals_have_exact_current"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical confirmation coverage fields missing: {sorted(missing)}")
    n = len(frame)
    return {
        "xpt": float(frame["xpt_has_exact_current"].eq(1).sum() / n),
        "xpd": float(frame["xpd_has_exact_current"].eq(1).sum() / n),
        "joint": float(frame["both_metals_have_exact_current"].eq(1).sum() / n),
    }


def _evaluate(
    model_spec: ModelSpec,
    base_train: pd.DataFrame,
    enhanced_train: pd.DataFrame,
    base_test: pd.DataFrame,
    enhanced_test: pd.DataFrame,
    base_features: tuple[str, ...],
    candidate_features: tuple[str, ...],
    policy: HistoricalConfirmationPolicy,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    base_model = model_spec.factory()
    candidate_model = model_spec.factory()
    base_model.fit(base_train.loc[:, base_features], base_train["target_log_return_1h"])
    candidate_model.fit(
        enhanced_train.loc[:, candidate_features],
        enhanced_train["target_log_return_1h"],
    )
    base_prediction = np.asarray(base_model.predict(base_test.loc[:, base_features]), dtype=float)
    candidate_prediction = np.asarray(
        candidate_model.predict(enhanced_test.loc[:, candidate_features]), dtype=float
    )
    metrics = RegressionForecastMetrics()
    base_metrics = metrics.calculate(
        base_test["target_log_return_1h"],
        base_prediction,
        base_test["close_usd_per_kg"],
        base_test["target_close_usd_per_kg"],
    ).as_dict()
    candidate_metrics = metrics.calculate(
        enhanced_test["target_log_return_1h"],
        candidate_prediction,
        enhanced_test["close_usd_per_kg"],
        enhanced_test["target_close_usd_per_kg"],
    ).as_dict()
    bootstrap = PairedBlockBootstrapComparison(
        block_size_rows=policy.bootstrap_block_rows,
        resamples=policy.bootstrap_resamples,
        random_state=42,
    ).compare(
        enhanced_test["target_log_return_1h"],
        candidate_prediction,
        base_prediction,
    ).as_dict()
    return {
        "model": model_spec.name,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_block_bootstrap": bootstrap,
    }, base_prediction, candidate_prediction


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base_dev = loader.load(BASE_DIR, label="A_frozen_silver_52")
    enhanced_dev = loader.load(ENHANCED_DIR, label="B_xpt_xpd_95")
    base_test = _load_test(BASE_DIR)
    enhanced_test = _load_test(ENHANCED_DIR)
    _assert_same_test(base_test, enhanced_test)

    candidate_features = tuple(base_dev.feature_names) + CANDIDATE_FEATURES
    missing = set(candidate_features).difference(enhanced_dev.frame.columns)
    if missing:
        raise ValueError(f"Locked candidate features missing from enhanced development data: {sorted(missing)}")

    policy = HistoricalConfirmationPolicy()
    coverage = _coverage(enhanced_test)

    primary, base_pred, candidate_pred = _evaluate(
        _model("ridge_alpha_100"),
        base_dev.frame,
        enhanced_dev.frame,
        base_test,
        enhanced_test,
        base_dev.feature_names,
        candidate_features,
        policy,
    )
    decision = policy.decide(
        bootstrap=primary["paired_block_bootstrap"],
        base_metrics=primary["base_metrics"],
        candidate_metrics=primary["candidate_metrics"],
        joint_coverage=coverage["joint"],
    )

    challenger, _, _ = _evaluate(
        _model("ridge_alpha_10"),
        base_dev.frame,
        enhanced_dev.frame,
        base_test,
        enhanced_test,
        base_dev.feature_names,
        candidate_features,
        policy,
    )

    report = {
        "execution_status": "PASS",
        "confirmation_version": CONFIRMATION_VERSION,
        "candidate_id": CANDIDATE_ID,
        "candidate_fingerprint_sha256": candidate_fingerprint(),
        "candidate_families": list(CANDIDATE_FAMILIES),
        "candidate_feature_count": len(CANDIDATE_FEATURES),
        "total_model_feature_count": len(candidate_features),
        "candidate_features": list(CANDIDATE_FEATURES),
        "policy": policy.as_dict(),
        "historical_test": {
            "rows": len(base_test),
            "first_timestamp_utc": base_test["timestamp_utc"].iloc[0].isoformat(),
            "last_timestamp_utc": base_test["timestamp_utc"].iloc[-1].isoformat(),
            "coverage": coverage,
        },
        "primary_confirmation": primary,
        "primary_decision": decision,
        "frozen_challenger_diagnostic": challenger,
        "research_policy": {
            "old_test_read": True,
            "old_test_used_for_feature_selection": False,
            "candidate_locked_before_old_test_read": True,
            "no_post_test_feature_tuning_allowed": True,
            "future_holdout_read": False,
            "formal_future_holdout_status": "SEALED_UNREAD",
            "live_promotion_eligible": False,
            "automatic_live_promotion": False,
        },
        "guardrails": {
            "research_only": True,
            "edge_status": "NOT_PROVEN",
            "live_model_mutated": False,
            "frozen_52_feature_graph_mutated": False,
            "buy_sell_enabled": False,
            "execution_enabled": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "historical_confirmation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "timestamp_utc": base_test["timestamp_utc"],
            "target_log_return_1h": base_test["target_log_return_1h"].to_numpy(float),
            "base_prediction": base_pred,
            "candidate_prediction": candidate_pred,
        }
    ).to_parquet(OUTPUT_DIR / "historical_confirmation_predictions.parquet", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
