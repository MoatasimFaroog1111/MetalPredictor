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
CFTC_DIR = Path("data/processed_cftc")
OUTPUT_DIR = Path("artifacts/cross_asset_cftc")
REPORT_PATH = OUTPUT_DIR / "cftc_af_report.json"


def frozen_baseline_model():
    return next(
        spec for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_silver_only")
    enhanced = loader.load(CFTC_DIR, label="F_silver_plus_cftc_cot")
    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(
            output_dir=OUTPUT_DIR,
            base_set_id="A",
            enhanced_set_id="F",
            artifact_prefix="cftc_af",
        ),
        splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )),
        metrics=RegressionForecastMetrics(),
    )
    report = comparator.compare(base, enhanced, frozen_baseline_model())
    report["cftc_data_integrity"] = {
        "publication_timing_point_in_time": True,
        "report_state_backdated_to_tuesday": False,
        "historical_release_date_limit": (
            "CFTC states that a complete historical release-date list is unavailable. "
            "Documented disruptions are exact overrides and remaining holiday weeks use "
            "a conservative delayed availability policy."
        ),
        "historical_test_read": False,
        "baseline_promotion_allowed": bool(
            report["decision"]["promote_enhanced_feature_set"]
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
