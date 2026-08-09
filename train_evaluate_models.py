from __future__ import annotations

import json

from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.model_data import PreparedDatasetLoader
from metal_predictor.model_training import ModelTrainingConfig, ModelTrainingPipeline
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.selection import FinalModelSelectionPolicy, WalkForwardSelectionPolicy


def build_pipeline() -> ModelTrainingPipeline:
    return ModelTrainingPipeline(
        config=ModelTrainingConfig(),
        loader=PreparedDatasetLoader(),
        registry=DefaultModelRegistry(random_state=42),
        wf_policy=WalkForwardSelectionPolicy(),
        final_policy=FinalModelSelectionPolicy(),
        metric_calculator=RegressionForecastMetrics(),
    )


if __name__ == "__main__":
    print(json.dumps(build_pipeline().run(), indent=2))
