from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
from metal_predictor.published_state import PublishedStateAligner
from metal_predictor.splitting import ChronologicalPurgedSplitter
from metal_predictor.targets import NextHourTargetBuilder
from metal_predictor.treasury_rate_features import TreasuryRateFeatures


RATES_PATH = Path("data/market/UST_2Y_10Y_H15_PUBLICATION_AWARE.parquet")
OUTPUT_DIR = Path("data/processed_rates")


def build_pipeline(config: PipelineConfig, rates: pd.DataFrame) -> TrainingDataPipeline:
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
            TreasuryRateFeatures(rates, PublishedStateAligner(), c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    if not RATES_PATH.exists():
        raise FileNotFoundError(f"Build Treasury rates first: {RATES_PATH}")
    rates = pd.read_parquet(RATES_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    report = build_pipeline(config, rates).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "cross_asset": "US Treasury 2Y/10Y publication-aware state",
        "base_plus_rates_feature_count": int(manifest["feature_count"]),
        "rates_source_path": str(RATES_PATH),
        "alignment": "latest state with available_from_utc <= feature timestamp",
        "raw_observation_date_used_for_alignment": False,
        "market_price_forward_fill": False,
    }
    (OUTPUT_DIR / "cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
