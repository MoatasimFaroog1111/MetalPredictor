from metal_predictor.artifacts import ParquetArtifactWriter
from metal_predictor.data import ParquetDataLoader, SilverDatasetValidator
from metal_predictor.core import PipelineConfig
from metal_predictor.features import (
    MomentumFeatures, PriceActionFeatures, QualityFeatures,
    TemporalFeatures, TrendFeatures, VolatilityFeatures,
)
from metal_predictor.leakage import StrictLeakageGuard
from metal_predictor.pipeline import TrainingDataPipeline
from metal_predictor.splitting import ChronologicalPurgedSplitter
from metal_predictor.targets import NextHourTargetBuilder


def build_pipeline(config: PipelineConfig) -> TrainingDataPipeline:
    c, f = config.columns, config.features
    return TrainingDataPipeline(
        config,
        ParquetDataLoader(),
        SilverDatasetValidator(c),
        (
            PriceActionFeatures(c),
            MomentumFeatures(c, f),
            VolatilityFeatures(c, f),
            TrendFeatures(c, f),
            TemporalFeatures(c),
            QualityFeatures(c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


if __name__ == "__main__":
    report = build_pipeline(PipelineConfig()).run()
    for key, value in report.items():
        print(f"{key}: {value}")
