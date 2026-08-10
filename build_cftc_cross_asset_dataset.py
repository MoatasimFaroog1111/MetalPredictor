from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.artifacts import ParquetArtifactWriter
from metal_predictor.cftc_cot_features import CftcSilverCotFeatures
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


COT_PATH = Path("data/market/CFTC_SILVER_084691_COT_PUBLICATION_AWARE.parquet")
OUTPUT_DIR = Path("data/processed_cftc")


def build_pipeline(config: PipelineConfig, cot: pd.DataFrame) -> TrainingDataPipeline:
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
            CftcSilverCotFeatures(cot, PublishedStateAligner(), c),
        ),
        NextHourTargetBuilder(c),
        ChronologicalPurgedSplitter(c, config.split),
        StrictLeakageGuard(c),
        ParquetArtifactWriter(c),
    )


def build() -> dict[str, object]:
    if not COT_PATH.exists():
        raise FileNotFoundError(f"Build CFTC Silver COT first: {COT_PATH}")
    cot = pd.read_parquet(COT_PATH)
    config = PipelineConfig(output_dir=OUTPUT_DIR)
    report = build_pipeline(config, cot).run()
    manifest = json.loads((OUTPUT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    report = {
        **report,
        "cross_asset": "CFTC COMEX Silver COT/Open Interest",
        "base_plus_cftc_feature_count": int(manifest["feature_count"]),
        "cftc_source_path": str(COT_PATH),
        "decision_time_semantics": "timestamp_utc + 1 hour after current H1 completes",
        "alignment": "latest COT state with available_from_utc <= completed-bar decision time",
        "backdate_to_report_date": False,
    }
    (OUTPUT_DIR / "cross_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
