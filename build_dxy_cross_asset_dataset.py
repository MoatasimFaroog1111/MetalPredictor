from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.artifacts import ParquetArtifactWriter
from metal_predictor.core import PipelineConfig
from metal_predictor.data import ParquetDataLoader, SilverDatasetValidator
from metal_predictor.dxy_features import DollarIndexCrossAssetFeatures
from metal_predictor.features import (
    MomentumFeatures,
    PriceActionFeatures,
    QualityFeatures,
    TemporalFeatures,
    TrendFeatures,
    VolatilityFeatures,
)
from metal_predictor.leakage import StrictLeakageGuard
from metal_predictor.pipeline import TrainingDataPipeline
from metal_predictor.splitting import ChronologicalPurgedSplitter
from metal_predictor.targets import NextHourTargetBuilder


DXY_PATH = Path("data/market/UDXUSD_H1_INDEX.parquet")
OUTPUT_DIR = Path("data/processed_dxy")


def build_pipeline(config: PipelineConfig, dxy: pd.DataFrame) -> TrainingDataPipeline:
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
            DollarIndexCrossAssetFeatures(dxy, ExactTimestampAligner(), c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    if not DXY_PATH.exists():
        raise FileNotFoundError(f"Build DXY first: {DXY_PATH}")
    dxy = pd.read_parquet(DXY_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    report = build_pipeline(config, dxy).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "cross_asset": "UDX/USD Dollar Index",
        "base_plus_dxy_feature_count": int(manifest["feature_count"]),
        "dxy_source_path": str(DXY_PATH),
        "exact_timestamp_alignment": True,
        "forward_fill": False,
    }
    (OUTPUT_DIR / "cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
