from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.artifacts import ParquetArtifactWriter
from metal_predictor.core import PipelineConfig
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
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures
from metal_predictor.splitting import ChronologicalPurgedSplitter
from metal_predictor.targets import NextHourTargetBuilder


PLATINUM_PATH = Path("data/market/XPTUSD_H1_USD_PER_KG.parquet")
PALLADIUM_PATH = Path("data/market/XPDUSD_H1_USD_PER_KG.parquet")
OUTPUT_DIR = Path("data/processed_precious_metals")


def build_pipeline(
    config: PipelineConfig,
    platinum: pd.DataFrame,
    palladium: pd.DataFrame,
) -> TrainingDataPipeline:
    c, f = config.columns, config.features
    precious = PlatinumPalladiumCrossAssetFeatures(
        platinum,
        palladium,
        ExactTimestampAligner(),
        c,
    )
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
            precious,
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    missing = [path for path in (PLATINUM_PATH, PALLADIUM_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Build precious-metal source data first: " + ", ".join(str(path) for path in missing)
        )
    platinum = pd.read_parquet(PLATINUM_PATH)
    palladium = pd.read_parquet(PALLADIUM_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    component = PlatinumPalladiumCrossAssetFeatures(
        platinum,
        palladium,
        ExactTimestampAligner(),
        config.columns,
    )
    report = build_pipeline(config, platinum, palladium).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "candidate_family": "XPT_XPD_cross_asset",
        "feature_version": component.feature_version,
        "candidate_feature_count": len(component.feature_names),
        "base_plus_candidate_feature_count": int(manifest["feature_count"]),
        "source_paths": [str(PLATINUM_PATH), str(PALLADIUM_PATH)],
        "exact_timestamp_alignment": True,
        "forward_fill": False,
        "future_holdout_read": False,
        "frozen_live_feature_graph_mutated": False,
    }
    (OUTPUT_DIR / "precious_metals_cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
