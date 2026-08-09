from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.artifacts import ParquetArtifactWriter
from metal_predictor.core import PipelineConfig
from metal_predictor.cross_asset_features import GoldSilverCrossAssetFeatures
from metal_predictor.data import ParquetDataLoader, SilverDatasetValidator
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


GOLD_PATH = Path("data/market/XAUUSD_H1_USD_PER_KG.parquet")
OUTPUT_DIR = Path("data/processed_gold")


def build_pipeline(config: PipelineConfig, gold: pd.DataFrame) -> TrainingDataPipeline:
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
            GoldSilverCrossAssetFeatures(gold, ExactTimestampAligner(), c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Build gold first: {GOLD_PATH}")
    gold = pd.read_parquet(GOLD_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    report = build_pipeline(config, gold).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "cross_asset": "XAU/USD",
        "base_plus_gold_feature_count": int(manifest["feature_count"]),
        "gold_source_path": str(GOLD_PATH),
        "exact_timestamp_alignment": True,
        "forward_fill": False,
    }
    (OUTPUT_DIR / "cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
