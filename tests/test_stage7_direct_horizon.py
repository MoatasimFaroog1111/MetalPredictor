from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.direct_horizon.dataset import (
    Stage7DatasetBuilder,
    Stage7ExactClockTargetBuilder,
)
from metal_predictor.direct_horizon.models import Stage7ModelFactory
from metal_predictor.direct_horizon.preregistration import (
    Stage7HorizonSpec,
    stage7_candidates,
    stage7_horizons,
    stage7_preregistration_fingerprint_sha256,
    stage7_preregistration_payload,
)
from metal_predictor.direct_horizon.split import Stage7PurgedExpandingPlanner


def test_stage7_preregistration_lock_matches_code_payload() -> None:
    locked = json.loads(
        Path("research_data/direct_horizon_stage7/preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert locked == stage7_preregistration_payload()
    assert [spec.hours for spec in stage7_horizons()] == [4, 12, 24, 48, 720]
    fingerprint = stage7_preregistration_fingerprint_sha256()
    assert len(fingerprint) == 64
    assert fingerprint == stage7_preregistration_fingerprint_sha256()


def test_stage7_target_requires_exact_future_clock() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T03:00:00Z",
                    "2026-01-01T05:00:00Z",
                ],
                utc=True,
            ),
            "close_usd_per_kg": [100.0, 101.0, 103.0, 105.0],
        }
    )
    out, target_name, target_close_name = Stage7ExactClockTargetBuilder().build(
        frame, Stage7HorizonSpec("4h", 4)
    )

    # 00:00 wants 04:00. Nearby 03:00 and 05:00 must never be used.
    assert pd.isna(out.loc[0, target_name])
    assert pd.isna(out.loc[0, target_close_name])
    assert pd.isna(out.loc[0, "target_timestamp_utc"])

    # 01:00 -> 05:00 is exact and therefore valid.
    assert out.loc[1, target_close_name] == 105.0
    assert out.loc[1, "target_timestamp_utc"] == pd.Timestamp("2026-01-01T05:00:00Z")
    assert np.isclose(out.loc[1, target_name], np.log(105.0 / 101.0))


def test_stage7_real_4h_dataset_reuses_exactly_52_causal_features() -> None:
    dataset = Stage7DatasetBuilder().build(Stage7HorizonSpec("4h", 4))
    assert len(dataset.feature_names) == 52
    assert len(set(dataset.feature_names)) == 52
    assert dataset.feature_graph_version == "canonical-h1-52-causal-v1"
    delta = pd.to_datetime(dataset.frame[dataset.target_timestamp_name], utc=True) - pd.to_datetime(
        dataset.frame[dataset.timestamp_name], utc=True
    )
    assert delta.eq(pd.Timedelta(hours=4)).all()


def test_stage7_real_4h_split_keeps_labels_before_validation_and_test() -> None:
    dataset = Stage7DatasetBuilder().build(Stage7HorizonSpec("4h", 4))
    plan = Stage7PurgedExpandingPlanner().plan(dataset)
    assert plan.historical_test_rows > 0
    assert plan.purge_hours == 4
    assert plan.embargo_hours == 4

    dev = dataset.frame.iloc[: plan.development_end_exclusive]
    dev_target_ts = pd.to_datetime(dev[dataset.target_timestamp_name], utc=True)
    assert dev_target_ts.max() < plan.historical_test_boundary_utc

    for fold in plan.folds:
        train_target_ts = pd.to_datetime(
            dev.iloc[: fold.train_end_exclusive][dataset.target_timestamp_name], utc=True
        )
        assert train_target_ts.max() < fold.validation_start_timestamp_utc
        assert fold.train_end_exclusive <= fold.validation_start
        assert fold.purge_hours == 4
        assert fold.embargo_hours == 4
        assert fold.validation_row_count > 0


def test_stage7_model_factory_builds_only_preregistered_candidates() -> None:
    factory = Stage7ModelFactory()
    assert len(stage7_candidates()) == 3
    for spec in stage7_candidates():
        assert factory.create(spec) is not None
