from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metal_predictor.multi_horizon.dataset import (
    DataPendingError,
    MultiHorizonDatasetBuilder,
)
from metal_predictor.multi_horizon.feature_set import (
    FEATURE_COLUMNS,
    CausalHlcFeatureBuilder,
    feature_fingerprint_sha256,
)
from metal_predictor.multi_horizon.preregistration import (
    candidate_registry,
    preregistration_fingerprint_sha256,
    preregistration_payload,
)
from metal_predictor.multi_horizon.provenance import BullionVaultChartCsvLoader
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("key", "rows"),
    [("4h", 172), ("12h", 172), ("2d", 172), ("30d", 232)],
)
def test_stage2_builds_expected_causal_rows(key: str, rows: int) -> None:
    dataset, report = MultiHorizonDatasetBuilder(repo_root=REPO_ROOT).build(key)
    assert dataset.model_row_count == rows
    assert report.model_row_count == rows
    assert report.warmup_rows_dropped == 6
    assert report.unlabeled_tail_rows_dropped == 1
    assert report.newest_incomplete_source_row_excluded is True
    assert dataset.feature_columns == FEATURE_COLUMNS
    assert report.feature_count == 12
    assert report.performance_metrics_computed is False
    assert report.future_holdout_read is False
    assert report.shadow62_mutated is False


def test_daily_horizon_remains_blocked() -> None:
    with pytest.raises(DataPendingError):
        MultiHorizonDatasetBuilder(repo_root=REPO_ROOT).build("1d")


def test_target_is_exact_next_registered_source_bar() -> None:
    dataset, _ = MultiHorizonDatasetBuilder(repo_root=REPO_ROOT).build("4h")
    delta = (
        pd.to_datetime(dataset.frame["target_timestamp_source"])
        - pd.to_datetime(dataset.frame["timestamp_source"])
    ).dt.total_seconds()
    assert set(delta.astype(int).tolist()) == {14_400}

    expected = np.log(
        dataset.frame["target_close_usd_per_kg"].to_numpy()
        / dataset.frame["current_close_usd_per_kg"].to_numpy()
    )
    np.testing.assert_allclose(dataset.frame["target_log_return"].to_numpy(), expected)


def test_future_perturbation_cannot_change_past_features() -> None:
    manifest = json.loads(
        (REPO_ROOT / "research_data/bullionvault_horizons/manifest.json").read_text()
    )
    record = manifest["datasets"]["4h"]
    source = BullionVaultChartCsvLoader().load(
        REPO_ROOT / record["raw_path"],
        expected_interval_seconds=14_400,
        include_potentially_incomplete_newest=False,
    ).frame

    cutoff = 100
    original = CausalHlcFeatureBuilder().build(source)
    changed = source.copy()
    changed.loc[
        cutoff + 1 :,
        ["high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"],
    ] *= 1.25
    perturbed = CausalHlcFeatureBuilder().build(changed)

    pd.testing.assert_frame_equal(
        original.loc[:cutoff, ["timestamp_source", *FEATURE_COLUMNS]],
        perturbed.loc[:cutoff, ["timestamp_source", *FEATURE_COLUMNS]],
        check_exact=True,
    )


def test_feature_registry_is_locked_and_contains_no_calendar_features() -> None:
    assert (
        feature_fingerprint_sha256()
        == "5ad621a8b432f874566b115887200e0008a8fe5e4bba207a689894b5de242043"
    )
    forbidden = ("hour", "weekday", "day_of_week", "month", "calendar")
    assert not any(any(token in name for token in forbidden) for name in FEATURE_COLUMNS)


def test_walk_forward_plan_is_expanding_purged_and_locks_tail() -> None:
    planner = ExpandingWalkForwardPlanner()
    plan = planner.plan(172)
    assert plan.development_end_exclusive == 137
    assert plan.historical_test.row_count == 35
    assert len(plan.folds) == 4
    assert [fold.train_row_count for fold in plan.folds] == [60, 79, 98, 117]
    assert [fold.validation_row_count for fold in plan.folds] == [19, 19, 19, 19]
    for fold in plan.folds:
        assert fold.validation_start - fold.train_end_exclusive == 1
        assert fold.validation_end_exclusive <= plan.development_end_exclusive


def test_preregistration_is_exactly_checked_in_and_does_not_fit_models() -> None:
    expected = preregistration_payload()
    checked = json.loads(
        (
            REPO_ROOT
            / "research_data/bullionvault_horizons/stage2_preregistration.json"
        ).read_text()
    )
    assert checked == expected
    assert (
        preregistration_fingerprint_sha256()
        == "fcf19e14ef55932093cd5406034700469b1e04723ac3d11b6c543345cb33b1d6"
    )
    assert [candidate.candidate_id for candidate in candidate_registry()] == [
        "ridge_alpha_10",
        "huber_v1",
        "elastic_net_v1",
    ]
    assert expected["baseline"]["baseline_id"] == "random_walk_zero_return"
    assert expected["stage2_execution"]["fit_candidate_models"] is False
    assert expected["stage2_execution"]["read_locked_historical_test_metrics"] is False
    assert expected["stage2_execution"]["performance_metrics_computed"] is False
    assert expected["guardrails"]["edge_status"] == "NOT_PROVEN"
    assert expected["guardrails"]["execution_enabled"] is False


def test_all_stage2_reports_keep_production_research_guards() -> None:
    payload = MultiHorizonDatasetBuilder(repo_root=REPO_ROOT).build_report_for_all()
    assert payload["performance_metrics_computed"] is False
    assert payload["guardrails"] == {
        "edge_status": "NOT_PROVEN",
        "research_only": True,
        "buy_sell_enabled": False,
        "execution_enabled": False,
        "live_model_mutated": False,
        "frozen_52_feature_graph_mutated": False,
        "future_holdout_read": False,
        "shadow62_mutated": False,
    }
    assert payload["horizons"]["1d"]["state"] == "DATA_PENDING"
