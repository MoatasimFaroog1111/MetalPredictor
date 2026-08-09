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
VIX_DIR = Path("data/processed_vix")
OUTPUT_DIR = Path("artifacts/cross_asset_vix")
REPORT_PATH = OUTPUT_DIR / "vix_ae_report.json"


def frozen_baseline_model():
    return next(
        spec for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_silver_only")
    enhanced = loader.load(VIX_DIR, label="E_silver_plus_vix")
    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(
            output_dir=OUTPUT_DIR,
            base_set_id="A",
            enhanced_set_id="E",
            artifact_prefix="vix_ae",
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
    report["vix_data_integrity"] = {
        "release_timing_point_in_time": True,
        "current_historical_values": True,
        "historical_revision_risk": (
            "Cboe provides a current official daily history file rather than a vintage archive. "
            "The daily close is hidden until 16:15 America/New_York, but later corrections to "
            "historical rows, if any, cannot be reconstructed from this source."
        ),
        "baseline_promotion_allowed_from_this_experiment": False,
        "reason": (
            "This development-only experiment can reject weak VIX features. Any positive VIX "
            "evidence remains a research candidate until confirmed on genuinely future data "
            "that was not available during feature design."
        ),
    }
    report["decision"]["provisional_feature_evidence"] = report["decision"]["evidence_level"]
    report["decision"]["baseline_promotion_allowed"] = False
    report["decision"]["requires_future_holdout_confirmation"] = comparator_promotes
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
