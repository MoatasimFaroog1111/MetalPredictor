from __future__ import annotations

import json
from pathlib import Path

from metal_predictor.cross_asset_experiment import DevelopmentFeatureSetLoader
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.oof_stacking import NestedOOFStackingEvaluator, StackingConfig
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("artifacts/oof_stacking")


def model_specs():
    registry = {spec.name: spec for spec in DefaultModelRegistry(random_state=42).candidates()}
    baseline = registry["ridge_alpha_100"]
    bases = (
        registry["ridge_alpha_100"],
        registry["histgb_absolute_regularized"],
        registry["lgbm_l1_small"],
    )
    return baseline, bases


def run() -> dict[str, object]:
    development = DevelopmentFeatureSetLoader().load(
        PROCESSED_DIR, label="silver_only_development"
    )
    baseline, bases = model_specs()
    evaluator = NestedOOFStackingEvaluator(
        config=StackingConfig(),
        outer_splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )),
        feature_names=development.feature_names,
        baseline_spec=baseline,
        base_specs=bases,
    )
    report = evaluator.evaluate(development.frame)
    oof = report.pop("oof_predictions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUTPUT_DIR / "stacking_oof_predictions.parquet", index=False)
    oof.to_csv(OUTPUT_DIR / "stacking_oof_predictions.csv", index=False)
    (OUTPUT_DIR / "stacking_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
