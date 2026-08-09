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
GOLD_DIR = Path("data/processed_gold")


def frozen_baseline_model():
    candidates = DefaultModelRegistry(random_state=42).candidates()
    return next(spec for spec in candidates if spec.name == "ridge_alpha_100")


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_silver_only")
    enhanced = loader.load(GOLD_DIR, label="B_silver_plus_gold")
    comparator = FeatureSetComparator(
        config=FeatureSetComparisonConfig(),
        splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )),
        metrics=RegressionForecastMetrics(),
    )
    return comparator.compare(base, enhanced, frozen_baseline_model())


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
