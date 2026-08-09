from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from metal_predictor.data import SilverDatasetValidator
from metal_predictor.core import ColumnConfig, FeatureConfig, SplitConfig
from metal_predictor.features import (
    MomentumFeatures, PriceActionFeatures, QualityFeatures,
    TemporalFeatures, TrendFeatures, VolatilityFeatures,
)
from metal_predictor.leakage import StrictLeakageGuard
from metal_predictor.pipeline import TrainingDataPipeline
from metal_predictor.splitting import ChronologicalPurgedSplitter
from metal_predictor.targets import NextHourTargetBuilder

C = ColumnConfig()


def sample_frame(rows=400):
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.1 + np.sin(np.arange(rows) / 12)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.2,
        C.high: close + 0.5,
        C.low: close - 0.5,
        C.close: close,
        C.quality: "OK",
    })


def test_validator_rejects_duplicate_timestamp():
    f = sample_frame()
    f.loc[1, C.timestamp] = f.loc[0, C.timestamp]
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        SilverDatasetValidator(C).validate(f)


def test_target_requires_exact_next_hour():
    f = sample_frame(10)
    f.loc[5:, C.timestamp] = f.loc[5:, C.timestamp] + pd.Timedelta(hours=2)
    out = NextHourTargetBuilder(C).build(f)
    assert pd.isna(out.loc[4, "target_log_return_1h"])
    assert out.loc[3, "target_timestamp_utc"] == out.loc[4, C.timestamp]


def test_momentum_is_causal_under_future_perturbation():
    f = sample_frame()
    comp = MomentumFeatures(C, FeatureConfig())
    baseline = comp.transform(f)
    perturbed = f.copy()
    cutoff = 250
    perturbed.loc[cutoff + 1:, [C.open, C.high, C.low, C.close]] *= 100.0
    changed = comp.transform(perturbed)
    for name in comp.feature_names:
        pd.testing.assert_series_equal(
            baseline.loc[:cutoff, name], changed.loc[:cutoff, name], check_names=False
        )


def test_purged_split_has_label_isolation():
    f = NextHourTargetBuilder(C).build(sample_frame()).dropna(subset=["target_timestamp_utc"]).reset_index(drop=True)
    s = ChronologicalPurgedSplitter(C, SplitConfig()).split(f)
    assert s["train"]["target_timestamp_utc"].max() < s["validation"][C.timestamp].min()
    assert s["validation"]["target_timestamp_utc"].max() < s["test"][C.timestamp].min()


def test_leakage_guard_rejects_future_feature_name():
    f = NextHourTargetBuilder(C).build(sample_frame()).dropna(subset=["target_timestamp_utc"]).reset_index(drop=True)
    f["future_price"] = 1.0
    s = ChronologicalPurgedSplitter(C, SplitConfig()).split(f)
    with pytest.raises(ValueError, match="future-looking"):
        StrictLeakageGuard(C).validate(
            f,
            s,
            ("future_price",),
            ("target_log_return_1h", "target_close_usd_per_kg", "target_timestamp_utc"),
        )


def test_full_pipeline_in_memory(tmp_path):
    from metal_predictor.core import PipelineConfig
    source = sample_frame(1000)

    class MemoryLoader:
        def load(self, path):
            return source.copy()

    class MemoryWriter:
        def __init__(self):
            self.splits = None
        def write(self, splits, feature_names, target_names, output_dir):
            self.splits = splits

    cfg = PipelineConfig(input_path=tmp_path / "unused.parquet", output_dir=tmp_path / "out")
    writer = MemoryWriter()
    pipeline = TrainingDataPipeline(
        cfg,
        MemoryLoader(),
        SilverDatasetValidator(C),
        (
            PriceActionFeatures(C),
            MomentumFeatures(C, cfg.features),
            VolatilityFeatures(C, cfg.features),
            TrendFeatures(C, cfg.features),
            TemporalFeatures(C),
            QualityFeatures(C),
        ),
        NextHourTargetBuilder(C),
        ChronologicalPurgedSplitter(C, cfg.split),
        StrictLeakageGuard(C),
        writer,
    )
    report = pipeline.run()
    assert report["status"] == "PASS"
    assert report["leakage_checks"] == "PASS"
    assert report["feature_count"] > 30
    assert writer.splits is not None
    assert all(len(writer.splits[name]) > 0 for name in ("train", "validation", "test"))


def test_hour_features_require_exact_timestamp_lag():
    frame = sample_frame(8)
    frame.loc[5:, C.timestamp] = frame.loc[5:, C.timestamp] + pd.Timedelta(hours=3)
    out = MomentumFeatures(C, FeatureConfig()).transform(frame)
    assert pd.isna(out.loc[5, "log_return_1h"])
    expected_ts = out.loc[5, C.timestamp] - pd.Timedelta(hours=3)
    exists = bool((out[C.timestamp] == expected_ts).any())
    assert pd.isna(out.loc[5, "log_return_3h"]) is (not exists)
