from __future__ import annotations

import json
from pathlib import Path

from metal_predictor.cross_asset_experiment import (
    DevelopmentFeatureSetLoader,
    FeatureSetComparator,
    FeatureSetComparisonConfig,
)
from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


BASE_DIR = Path("data/processed")
RATES_DIR = Path("data/processed_rates")
OUTPUT_DIR = Path("artifacts/cross_asset_rates")
REPORT_PATH = OUTPUT_DIR / "rates_ad_report.json"


def frozen_baseline_model():
    return next(
        spec for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_silver_only")
    enhanced = loader.load(RATES_DIR, label="D_silver_plus_treasury_rates")
    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(
            output_dir=OUTPUT_DIR,
            base_set_id="A",
            enhanced_set_id="D",
            artifact_prefix="rates_ad",
        ),
        splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )),
        metrics=RegressionForecastMetrics(),
    )
    report = comparator.compare(base, enhanced, frozen_baseline_model())
    comparator_promotes = bool(report["decision"]["promote_enhanced_feature_set"])
    report["rates_data_integrity"] = {
        "release_timing_point_in_time": True,
        "current_historical_values": True,
        "historical_revision_risk": (
            "Treasury current official historical values may include later corrections. "
            "Documented H.15 publication omissions are delayed explicitly, but full ALFRED-style "
            "vintage reconstruction is not available without a vintage-capable source."
        ),
        "baseline_promotion_allowed_from_this_experiment": False,
        "reason": (
            "This experiment can reject weak rate features, but any positive result must be "
            "confirmed with a vintage-safe source before changing baseline-v1."
        ),
    }
    report["decision"]["provisional_feature_evidence"] = report["decision"]["evidence_level"]
    report["decision"]["baseline_promotion_allowed"] = False
    report["decision"]["requires_vintage_safe_confirmation"] = comparator_promotes
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
