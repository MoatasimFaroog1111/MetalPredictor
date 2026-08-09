from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec, ZeroReturnRegressor
from metal_predictor.selection import CandidateSummary, FinalModelSelectionPolicy, ValidationResult, WalkForwardSelectionPolicy
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


def sample_frame(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.1
    ret = np.sin(np.arange(rows) / 19.0) * 0.001
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "close_usd_per_kg": close,
        "target_close_usd_per_kg": close * np.exp(ret),
        "target_log_return_1h": ret,
        "f1": np.cos(np.arange(rows) / 13.0),
    })


def test_walk_forward_purges_training_labels_at_each_boundary() -> None:
    folds = PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=4, initial_train_fraction=0.5, min_train_rows=400)).split(sample_frame())
    assert len(folds) == 4
    for fold in folds:
        assert fold.train["target_timestamp_utc"].max() < fold.validation["timestamp_utc"].min()


def test_metrics_zero_return_baseline_is_exact_for_zero_targets() -> None:
    y = np.zeros(10)
    close = np.repeat(1000.0, 10)
    result = RegressionForecastMetrics().calculate(y, y, close, close)
    assert result.mae_return == 0.0
    assert result.rmse_return == 0.0
    assert result.price_mae_usd_per_kg == 0.0


def test_walk_forward_policy_chooses_best_config_inside_each_family() -> None:
    specs = (
        ModelSpec("a1", "family_a", lambda: ZeroReturnRegressor()),
        ModelSpec("a2", "family_a", lambda: ZeroReturnRegressor()),
        ModelSpec("b1", "family_b", lambda: ZeroReturnRegressor()),
    )
    summaries = (
        CandidateSummary("a1", "family_a", None, 5, 0.20, 0.01, 0.3, 0.51, 0.01, 4.0),
        CandidateSummary("a2", "family_a", None, 5, 0.10, 0.01, 0.2, 0.50, 0.02, 3.0),
        CandidateSummary("b1", "family_b", None, 5, 0.15, 0.01, 0.2, 0.52, 0.03, 3.5),
    )
    assert [w.name for w in WalkForwardSelectionPolicy().family_winners(specs, summaries)] == ["a2", "b1"]


def test_final_selection_uses_validation_mae_not_direction() -> None:
    rows = (
        ValidationResult("a", "fa", None, 0.1, 0.20, 0.25, 0.70, 0.2, 5.0),
        ValidationResult("b", "fb", None, 0.2, 0.10, 0.30, 0.40, -0.1, 4.0),
    )
    assert FinalModelSelectionPolicy().choose(rows).model_name == "b"
