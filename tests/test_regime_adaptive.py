from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.regime_adaptive import (
    RegimeAdaptiveConfig,
    RegimeAdaptiveEvaluator,
    TrainOnlyRegimeDetector,
)
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


def _frame(rows: int = 7000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    trend = rng.normal(scale=0.012, size=rows)
    volatility = np.abs(rng.normal(loc=0.0025, scale=0.001, size=rows))
    x = rng.normal(size=rows)
    target = 0.0004 * x + np.where(
        volatility > np.quantile(volatility, 0.67),
        0.00025 * np.sign(trend),
        -0.00010 * np.sign(trend),
    ) + rng.normal(scale=0.0008, size=rows)
    close = 1000.0 * np.exp(np.cumsum(target * 0.05))
    target_close = close * np.exp(target)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "realized_vol_24h": volatility,
        "close_vs_sma_72h": trend,
        "x": x,
        "target_log_return_1h": target,
        "close_usd_per_kg": close,
        "target_close_usd_per_kg": target_close,
    })


def _registry():
    return {spec.name: spec for spec in DefaultModelRegistry(random_state=42).candidates()}


def _evaluator() -> RegimeAdaptiveEvaluator:
    registry = _registry()
    return RegimeAdaptiveEvaluator(
        config=RegimeAdaptiveConfig(
            inner_splits=3,
            inner_initial_train_fraction=0.55,
            inner_min_train_rows=1000,
            min_regime_train_rows=150,
            min_regime_inner_validation_rows=50,
            bootstrap_resamples=500,
        ),
        outer_splitter=PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=3,
            initial_train_fraction=0.55,
            min_train_rows=2000,
        )),
        feature_names=("realized_vol_24h", "close_vs_sma_72h", "x"),
        baseline_spec=registry["ridge_alpha_100"],
        specialist_specs=(registry["ridge_alpha_10"], registry["ridge_alpha_100"]),
    )


def test_regime_thresholds_are_fit_only_from_training_rows() -> None:
    frame = _frame(3000)
    train = frame.iloc[:2000].copy()
    detector = TrainOnlyRegimeDetector().fit(train)
    original = detector.thresholds

    future = frame.iloc[2000:].copy()
    future["realized_vol_24h"] *= 1000.0
    future["close_vs_sma_72h"] *= 1000.0
    transformed = detector.transform(future)

    assert detector.thresholds == original
    assert len(transformed) == len(future)


def test_outer_validation_targets_do_not_change_first_fold_model_selection() -> None:
    frame = _frame()
    baseline = _evaluator().evaluate(frame)

    changed = frame.copy()
    splitter = PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=3, initial_train_fraction=0.55, min_train_rows=2000
    ))
    first_validation_ts = splitter.split(changed)[0].validation["timestamp_utc"].iloc[0]
    changed.loc[
        changed["timestamp_utc"] >= first_validation_ts,
        "target_log_return_1h",
    ] *= -20.0
    perturbed = _evaluator().evaluate(changed)

    first_base = baseline["outer_walk_forward"]["fold_results"][0]
    first_changed = perturbed["outer_walk_forward"]["fold_results"][0]
    assert first_base["thresholds"] == first_changed["thresholds"]
    assert first_base["model_choices"] == first_changed["model_choices"]


def test_regime_adaptive_outputs_one_prediction_for_every_outer_oof_row() -> None:
    report = _evaluator().evaluate(_frame())
    oof = report["oof_predictions"]
    assert len(oof) > 0
    assert np.isfinite(oof["adaptive_prediction"].to_numpy(float)).all()
    assert np.isfinite(oof["baseline_prediction"].to_numpy(float)).all()
    assert set(oof["regime"].unique()).issubset(
        {"HIGH_VOL", "TREND_UP", "TREND_DOWN", "RANGE"}
    )
    assert report["research_policy"]["outer_validation_used_for_model_selection"] is False
