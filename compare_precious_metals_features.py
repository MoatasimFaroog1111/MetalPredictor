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
PRECIOUS_METALS_DIR = Path("data/processed_precious_metals")
ARTIFACT_DIR = Path("artifacts/cross_asset_precious_metals")


def frozen_baseline_model():
    candidates = DefaultModelRegistry(random_state=42).candidates()
    return next(spec for spec in candidates if spec.name == "ridge_alpha_100")


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_silver_only")
    enhanced = loader.load(
        PRECIOUS_METALS_DIR,
        label="B_silver_plus_platinum_palladium",
    )
    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(output_dir=ARTIFACT_DIR),
        splitter=PurgedWalkForwardSplitter(
            WalkForwardConfig(
                n_splits=5,
                initial_train_fraction=0.50,
                min_train_rows=8000,
            )
        ),
        metrics=RegressionForecastMetrics(),
    )
    report = comparator.compare(base, enhanced, frozen_baseline_model())
    report["research_policy"] = {
        "development_only": True,
        "future_holdout_touched": False,
        "frozen_live_model_mutated": False,
        "frozen_52_feature_graph_mutated": False,
        "promotion_automatic": False,
        "edge_status": "NOT_PROVEN",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "precious_metals_ab_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
