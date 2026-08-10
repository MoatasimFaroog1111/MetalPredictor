from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.selective_prediction import (
    NestedSelectivePredictionEvaluator,
    SelectivePredictionConfig,
)
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


def _frame(rows: int = 7000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    noise = rng.normal(scale=0.8, size=rows)
    target = (0.0007 * x1 - 0.0004 * x2 + 0.0007 * noise)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "f1": x1,
        "f2": x2,
        "target_log_return_1h": target,
    })


def _ridge():
    return next(
        spec for spec in DefaultModelRegistry(random_state=42).candidates()
        if spec.name == "ridge_alpha_100"
    )


def _evaluator() -> NestedSelectivePredictionEvaluator:
    return NestedSelectivePredictionEvaluator(
        config=SelectivePredictionConfig(
            coverage_targets=(0.50, 0.25, 0.10),
            inner_splits=3,
            inner_initial_train_fraction=0.55,
            inner_min_train_rows=1000,
            min_inner_signals=100,
            bootstrap_resamples=500,
        ),
        outer_splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=3,
            initial_train_fraction=0.55,
            min_train_rows=2000,
        )),
        feature_names=("f1", "f2"),
    )


def test_selective_thresholds_do_not_change_when_outer_validation_targets_change() -> None:
    frame = _frame()
    baseline = _evaluator().evaluate(frame, _ridge())

    changed = frame.copy()
    outer = PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=3, initial_train_fraction=0.55, min_train_rows=2000
    )).split(changed)
    first_validation_ts = outer[0].validation["timestamp_utc"].iloc[0]
    changed.loc[changed["timestamp_utc"] >= first_validation_ts, "target_log_return_1h"] *= -20.0
    perturbed = _evaluator().evaluate(changed, _ridge())

    base_threshold = baseline["outer_walk_forward"]["fold_results"][0]["threshold"]
    changed_threshold = perturbed["outer_walk_forward"]["fold_results"][0]["threshold"]
    assert np.isclose(base_threshold, changed_threshold, rtol=0.0, atol=1e-15)


def test_selective_output_has_explicit_no_trade_rows_and_valid_coverage() -> None:
    result = _evaluator().evaluate(_frame(), _ridge())
    oof = result["oof_predictions"]
    assert oof["selected"].any()
    assert (~oof["selected"]).any()
    coverage = result["pooled_selected_signals"]["coverage"]
    assert 0.0 < coverage < 1.0
    assert result["research_policy"]["outer_validation_used_for_threshold_selection"] is False


def test_selective_threshold_audit_uses_only_inner_oof_rows() -> None:
    result = _evaluator().evaluate(_frame(), _ridge())
    for audit, fold in zip(
        result["threshold_audit"],
        result["outer_walk_forward"]["fold_results"],
        strict=True,
    ):
        assert audit["inner_oof_rows"] < fold["train_rows"]
        assert audit["chosen"]["selected_rows"] >= 100
        assert audit["chosen"]["absolute_prediction_threshold"] >= 0.0
