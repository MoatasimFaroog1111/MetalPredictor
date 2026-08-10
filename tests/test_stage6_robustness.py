from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.cpcv import CPCVConfig, CombinatorialPurgedSplitter
from metal_predictor.deflated_sharpe import DeflatedSharpeEvaluator
from metal_predictor.multiple_testing import BlockBootstrapHolmTester
from metal_predictor.pbo import CSCVPBOEstimator
from metal_predictor.trial_ledger import ResearchTrialLedger


def _time_frame(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp_utc": ts,
        "target_timestamp_utc": ts + pd.Timedelta(hours=1),
        "x": np.arange(rows, dtype=float),
    })


def test_cpcv_creates_all_combinations_and_purges_label_overlap() -> None:
    frame = _time_frame()
    splitter = CombinatorialPurgedSplitter(CPCVConfig(
        n_groups=6, test_groups=2, embargo_rows=1
    ))
    splits = splitter.split(frame)
    assert len(splits) == 15
    for split in splits:
        assert len(split.train) > 0 and len(split.test) > 0
        assert not set(split.train["timestamp_utc"]) & set(split.test["timestamp_utc"])
        test_ts = set(split.test["timestamp_utc"])
        for _, row in split.train.iterrows():
            assert row["timestamp_utc"] not in test_ts


def test_cscv_pbo_is_low_for_a_persistent_winner() -> None:
    rng = np.random.default_rng(42)
    rows = 2400
    persistent = rng.normal(loc=0.0020, scale=0.01, size=rows)
    returns = pd.DataFrame({
        "persistent": persistent,
        "noise_a": rng.normal(loc=0.0, scale=0.01, size=rows),
        "noise_b": rng.normal(loc=-0.0002, scale=0.01, size=rows),
        "noise_c": rng.normal(loc=0.0001, scale=0.01, size=rows),
    })
    report = CSCVPBOEstimator(n_blocks=8).estimate(
        returns, tuple(returns.columns)
    )
    assert 0.0 <= report["pbo"] <= 1.0
    assert report["pbo"] < 0.25
    assert report["selected_strategy_counts"]["persistent"] > 0


def test_deflated_sharpe_probability_decreases_as_trial_count_increases() -> None:
    rng = np.random.default_rng(5)
    rows = 3000
    returns = pd.DataFrame({
        "selected": rng.normal(loc=0.0007, scale=0.01, size=rows),
        "trial_b": rng.normal(loc=0.0001, scale=0.01, size=rows),
        "trial_c": rng.normal(loc=-0.0001, scale=0.01, size=rows),
        "trial_d": rng.normal(loc=0.0002, scale=0.01, size=rows),
    })
    evaluator = DeflatedSharpeEvaluator()
    names = tuple(returns.columns)
    few = evaluator.evaluate(returns, names, "selected", counted_trials=4)
    many = evaluator.evaluate(returns, names, "selected", counted_trials=40)
    assert many.selection_bias_benchmark_sharpe >= few.selection_bias_benchmark_sharpe
    assert many.deflated_sharpe_probability <= few.deflated_sharpe_probability


def test_holm_adjustment_is_never_smaller_than_raw_p_value() -> None:
    rng = np.random.default_rng(8)
    rows = 2400
    returns = pd.DataFrame({
        "strong": rng.normal(loc=0.002, scale=0.01, size=rows),
        "weak": rng.normal(loc=0.0001, scale=0.01, size=rows),
        "null": rng.normal(loc=0.0, scale=0.01, size=rows),
    })
    report = BlockBootstrapHolmTester(
        block_size_rows=24, resamples=1000, random_state=9
    ).test(returns, tuple(returns.columns))
    for row in report["results"]:
        assert row["holm_adjusted_p_value"] >= row["raw_one_sided_p_value"]
        assert 0.0 <= row["holm_adjusted_p_value"] <= 1.0


def test_trial_ledger_counts_all_completed_research_trials_conservatively() -> None:
    ledger = ResearchTrialLedger(initial_model_registry_trials=16)
    assert ledger.total_trials == 40
    assert ledger.as_dict()["total_counted_trials"] == 40
