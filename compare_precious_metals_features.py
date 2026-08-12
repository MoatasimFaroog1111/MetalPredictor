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
from metal_predictor.precious_metals.coverage import PreciousMetalsCoverageValidator
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


BASE_DIR = Path("data/processed")
ENHANCED_DIR = Path("data/processed_precious_metals")
OUTPUT_DIR = Path("artifacts/cross_asset_precious_metals")


def frozen_baseline_model():
    candidates = DefaultModelRegistry(random_state=42).candidates()
    return next(spec for spec in candidates if spec.name == "ridge_alpha_100")


def research_splitter() -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=5,
        initial_train_fraction=0.50,
        min_train_rows=8000,
    ))


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_frozen_silver_52")
    enhanced = loader.load(ENHANCED_DIR, label="B_silver_plus_pre_registered_XPT_XPD_v1")

    # Availability is checked before fitting either model. This prevents a sparse or
    # truncated provider download from being silently converted into mostly-imputed
    # candidate features and then presented as a valid paired experiment.
    coverage_gate = PreciousMetalsCoverageValidator().validate(
        enhanced.frame,
        research_splitter(),
    )

    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(
            output_dir=OUTPUT_DIR,
            base_set_id="A",
            enhanced_set_id="B",
            artifact_prefix="precious_metals_ab",
        ),
        splitter=research_splitter(),
        metrics=RegressionForecastMetrics(),
    )
    result = comparator.compare(base, enhanced, frozen_baseline_model())
    result["coverage_gate"] = coverage_gate
    result["guardrails"] = {
        "development_only": True,
        "future_holdout_read": False,
        "old_test_read": False,
        "live_model_mutated": False,
        "frozen_52_feature_graph_mutated": False,
        "promotion_requires_predeclared_rule": True,
        "provider_coverage_validated_before_model_fit": True,
    }
    (OUTPUT_DIR / "precious_metals_ab_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
