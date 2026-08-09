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
from metal_predictor.vix_features import VixDailyFeatures


VIX_PATH = Path("data/market/VIX_DAILY_PUBLICATION_AWARE.parquet")
OUTPUT_DIR = Path("data/processed_vix")


def build_pipeline(config: PipelineConfig, vix: pd.DataFrame) -> TrainingDataPipeline:
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
            VixDailyFeatures(vix, PublishedStateAligner(), c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    if not VIX_PATH.exists():
        raise FileNotFoundError(f"Build VIX first: {VIX_PATH}")
    vix = pd.read_parquet(VIX_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    report = build_pipeline(config, vix).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "cross_asset": "Cboe VIX daily closing state",
        "base_plus_vix_feature_count": int(manifest["feature_count"]),
        "vix_source_path": str(VIX_PATH),
        "bar_label_semantics": "timestamp_utc is H1 bar start",
        "decision_time_semantics": "timestamp_utc + 1 hour after current H1 completes",
        "alignment": "latest VIX daily close with available_from_utc <= completed-bar decision time",
        "intraday_daily_close_backfill": False,
        "market_price_forward_fill": False,
    }
    (OUTPUT_DIR / "cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
