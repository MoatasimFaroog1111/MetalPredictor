from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.oof_stacking import NestedOOFStackingEvaluator, StackingConfig
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


def _frame(rows: int = 7000) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ts = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    x3 = rng.normal(size=rows)
    target = 0.00045 * x1 - 0.00025 * x2 + 0.00010 * x1 * x3 + rng.normal(
        scale=0.0009, size=rows
    )
    close = 1000.0 * np.exp(np.cumsum(target * 0.03))
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "target_log_return_1h": target,
        "close_usd_per_kg": close,
        "target_close_usd_per_kg": close * np.exp(target),
    })


def _evaluator() -> NestedOOFStackingEvaluator:
    registry = {spec.name: spec for spec in DefaultModelRegistry(random_state=42).candidates()}
    return NestedOOFStackingEvaluator(
        config=StackingConfig(
            inner_splits=3,
            inner_initial_train_fraction=0.55,
            inner_min_train_rows=1000,
            meta_alpha=1.0,
            bootstrap_resamples=500,
        ),
        outer_splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=3,
            initial_train_fraction=0.55,
            min_train_rows=2000,
        )),
        feature_names=("x1", "x2", "x3"),
        baseline_spec=registry["ridge_alpha_100"],
        base_specs=(registry["ridge_alpha_10"], registry["ridge_alpha_100"]),
    )


def test_outer_validation_targets_do_not_change_first_fold_stacked_predictions() -> None:
    frame = _frame()
    first = _evaluator().evaluate(frame)

    changed = frame.copy()
    splitter = PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=3, initial_train_fraction=0.55, min_train_rows=2000
    ))
    first_validation = splitter.split(changed)[0].validation["timestamp_utc"].iloc[0]
    changed.loc[
        changed["timestamp_utc"] >= first_validation,
        "target_log_return_1h",
    ] *= -15.0
    second = _evaluator().evaluate(changed)

    a = first["oof_predictions"]
    b = second["oof_predictions"]
    first_fold_a = a.loc[a["fold"].eq(1), "stacked_prediction"].to_numpy(float)
    first_fold_b = b.loc[b["fold"].eq(1), "stacked_prediction"].to_numpy(float)
    np.testing.assert_allclose(first_fold_a, first_fold_b, rtol=0.0, atol=1e-15)


def test_meta_model_uses_inner_oof_predictions_only() -> None:
    report = _evaluator().evaluate(_frame())
    assert report["research_policy"]["outer_validation_used_for_meta_fit"] is False
    for fold in report["outer_walk_forward"]["fold_results"]:
        assert 500 <= fold["meta_oof_rows"] < fold["train_rows"]


def test_stacking_outputs_finite_predictions_for_every_outer_row() -> None:
    report = _evaluator().evaluate(_frame())
    oof = report["oof_predictions"]
    assert len(oof) > 0
    assert np.isfinite(oof["stacked_prediction"].to_numpy(float)).all()
    assert np.isfinite(oof["baseline_prediction"].to_numpy(float)).all()
