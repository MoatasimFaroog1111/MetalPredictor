from __future__ import annotations

import pandas as pd
import pytest

from metal_predictor.precious_metals.coverage import (
    PreciousMetalsCoveragePolicy,
    PreciousMetalsCoverageValidator,
)
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


def _development(rows: int = 600, joint_missing_from: int | None = None) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    xpt = pd.Series(1, index=range(rows), dtype="int8")
    xpd = pd.Series(1, index=range(rows), dtype="int8")
    if joint_missing_from is not None:
        xpt.iloc[joint_missing_from:] = 0
        xpd.iloc[joint_missing_from:] = 0
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "xpt_has_exact_current": xpt,
        "xpd_has_exact_current": xpd,
        "both_metals_have_exact_current": (xpt.eq(1) & xpd.eq(1)).astype("int8"),
    })


def _splitter() -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=3,
        initial_train_fraction=0.50,
        min_train_rows=200,
    ))


def _policy() -> PreciousMetalsCoveragePolicy:
    return PreciousMetalsCoveragePolicy(
        min_full_metal_coverage=0.50,
        min_train_metal_coverage=0.40,
        min_validation_metal_coverage=0.60,
        min_validation_joint_coverage=0.50,
        min_train_joint_rows=100,
        min_validation_joint_rows=50,
    )


def test_coverage_gate_passes_before_model_fit_when_every_fold_is_supported() -> None:
    report = PreciousMetalsCoverageValidator(_policy()).validate(_development(), _splitter())
    assert report["status"] == "PASS"
    assert report["policy_fixed_before_result"] is True
    assert all(row["passed"] for row in report["folds"])


def test_coverage_gate_fails_closed_on_truncated_recent_provider_data() -> None:
    sparse = _development(joint_missing_from=400)
    with pytest.raises(ValueError, match="coverage gate failed fold"):
        PreciousMetalsCoverageValidator(_policy()).validate(sparse, _splitter())
