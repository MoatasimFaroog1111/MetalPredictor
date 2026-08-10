from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metal_predictor.append_only_ledger import CsvHashChainLedger
from metal_predictor.frozen_ridge import FrozenRidgeExporter, FrozenRidgeRegressor
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.future_holdout_scorer import FutureHoldoutScorer


def _synthetic_development(rows: int = 2500) -> tuple[pd.DataFrame, tuple[str, ...]]:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    x1[::17] = np.nan
    target = np.nan_to_num(x1, nan=0.0) * 0.0003 - x2 * 0.0002 + rng.normal(
        scale=0.0005, size=rows
    )
    frame = pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "x1": x1,
        "x2": x2,
        "target_log_return_1h": target,
    })
    return frame, ("x1", "x2")


def test_frozen_ridge_reproduces_sklearn_pipeline_with_missing_indicators() -> None:
    development, features = _synthetic_development()
    registry = {
        spec.name: spec
        for spec in DefaultModelRegistry(random_state=42).candidates()
    }
    spec = registry["ridge_alpha_10"]
    payload = FrozenRidgeExporter().export(
        spec,
        development,
        features,
        source_dataset_git_blob_sha="abc123",
        research_code_cutoff_commit="def456",
    )
    frozen = FrozenRidgeRegressor(payload)
    sklearn_model = spec.factory()
    sklearn_model.fit(development.loc[:, features], development["target_log_return_1h"])
    expected = sklearn_model.predict(development.loc[:, features])
    actual = frozen.predict(development.loc[:, features])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_hash_chain_ledger_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    ledger = CsvHashChainLedger(
        path,
        ("timestamp_utc", "value"),
        "timestamp_utc",
    )
    rows = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2026-08-11 00:00:00+00:00",
            "2026-08-11 01:00:00+00:00",
        ]),
        "value": [1.0, 2.0],
    })
    result = ledger.append(rows)
    assert result.appended_rows == 2
    stored = pd.read_csv(path)
    stored.loc[0, "value"] = "999"
    stored.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row hash mismatch"):
        ledger.read_verified()


def test_hash_chain_rejects_backdated_append(tmp_path: Path) -> None:
    ledger = CsvHashChainLedger(
        tmp_path / "ledger.csv",
        ("timestamp_utc", "value"),
        "timestamp_utc",
    )
    ledger.append(pd.DataFrame({
        "timestamp_utc": [pd.Timestamp("2026-08-11 02:00:00", tz="UTC")],
        "value": [2.0],
    }))
    with pytest.raises(ValueError, match="strictly later"):
        ledger.append(pd.DataFrame({
            "timestamp_utc": [pd.Timestamp("2026-08-11 01:00:00", tz="UTC")],
            "value": [1.0],
        }))


def test_final_scorer_refuses_to_compute_metrics_before_time_lock(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    ledger = tmp_path / "ledger"
    (frozen / "forward_holdout").mkdir(parents=True)
    (ledger / "forward_holdout").mkdir(parents=True)
    manifest = {
        "freeze_id": "test",
        "holdout_first_bar_start_utc": "2026-08-11T00:00:00Z",
        "holdout_last_feature_bar_start_exclusive_utc": "2027-02-07T00:00:00Z",
        "earliest_final_score_utc": "2027-02-07T02:00:00Z",
        "minimum_exact_hour_outcomes": 2500,
        "final_score_rules": {
            "bootstrap_block_rows": 24,
            "bootstrap_resamples": 1000,
            "random_state": 42,
        },
    }
    (frozen / "forward_holdout/freeze_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = FutureHoldoutScorer(frozen, ledger).score(
        pd.Timestamp("2026-12-01T00:00:00Z")
    )
    assert report["status"] == "LOCKED_TIME"
    assert report["performance_metrics_computed"] is False
    assert "directional_accuracy" not in report
    assert not (ledger / "forward_holdout/final_score.json").exists()
