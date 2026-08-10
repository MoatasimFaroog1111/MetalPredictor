from __future__ import annotations

import json
from pathlib import Path

from metal_predictor.cross_asset_experiment import DevelopmentFeatureSetLoader
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.selective_prediction import (
    NestedSelectivePredictionEvaluator,
    SelectivePredictionConfig,
)
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("artifacts/selective_prediction")


def frozen_baseline_model():
    return next(
        spec for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def run() -> dict[str, object]:
    development = DevelopmentFeatureSetLoader().load(
        PROCESSED_DIR, label="silver_only_development"
    )
    evaluator = NestedSelectivePredictionEvaluator(
        config=SelectivePredictionConfig(),
        outer_splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )),
        feature_names=development.feature_names,
    )
    report = evaluator.evaluate(development.frame, frozen_baseline_model())
    oof = report.pop("oof_predictions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUTPUT_DIR / "selective_oof_predictions.parquet", index=False)
    oof.to_csv(OUTPUT_DIR / "selective_oof_predictions.csv", index=False)
    (OUTPUT_DIR / "selective_prediction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
