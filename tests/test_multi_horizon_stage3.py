from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from metal_predictor.multi_horizon.dataset import DataPendingError, MultiHorizonDatasetBuilder
from metal_predictor.multi_horizon.development import DevelopmentOnlyEvaluator
from metal_predictor.multi_horizon.models import DevelopmentModelFactory
from metal_predictor.multi_horizon.preregistration import candidate_registry
from metal_predictor.multi_horizon.selection import DevelopmentCandidateEvidence, DevelopmentSelectionGate
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner
from metal_predictor.multi_horizon.statistics import paired_block_bootstrap_mae_improvement

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_model_factory_uses_locked_registry() -> None:
    factory = DevelopmentModelFactory()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 12))
    y = 0.01 * x[:, 0] - 0.003 * x[:, 1]
    for spec in candidate_registry():
        model = factory.create(spec)
        model.fit(x[:60], y[:60])
        prediction = model.predict(x[60:])
        assert prediction.shape == (20,)
        assert np.isfinite(prediction).all()


def test_bootstrap_is_deterministic() -> None:
    actual = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.04, 0.02, 0.01])
    baseline = np.zeros_like(actual)
    candidate = actual * 0.75
    kwargs = dict(iterations=500, block_length_rows=4, seed=20260813, ci_level=0.95)
    first = paired_block_bootstrap_mae_improvement(actual, baseline, candidate, **kwargs)
    second = paired_block_bootstrap_mae_improvement(actual, baseline, candidate, **kwargs)
    assert first == second
    assert first.ci_low > 0.0
    assert first.probability_candidate_mae_better == 1.0


def test_development_result_is_independent_of_reserved_test_rows() -> None:
    builder = MultiHorizonDatasetBuilder(repo_root=REPO_ROOT)
    dataset, _ = builder.build("4h")
    plan = ExpandingWalkForwardPlanner().plan(dataset.model_row_count)
    evaluator = DevelopmentOnlyEvaluator()
    before = evaluator.evaluate(dataset, plan).as_dict()

    frame = dataset.frame.copy(deep=True)
    indices = frame.index[plan.historical_test.start:plan.historical_test.end_exclusive]
    frame.loc[indices, list(dataset.feature_columns)] = 123456.0
    frame.loc[indices, dataset.target_column] = -123.0
    after = evaluator.evaluate(replace(dataset, frame=frame), plan).as_dict()

    assert before == after
    assert before["historical_test"]["metrics_read"] is False
    assert before["historical_test"]["predictions_computed"] is False
    assert before["historical_test"]["used_for_fit"] is False
    assert before["historical_test"]["used_for_selection"] is False


def test_weak_evidence_keeps_random_walk() -> None:
    gate = DevelopmentSelectionGate()
    evidence = DevelopmentCandidateEvidence(
        candidate_id="ridge_alpha_10",
        candidate_oof_mae=0.011,
        baseline_oof_mae=0.010,
        better_folds=1,
        directional_accuracy_delta=0.0,
        bootstrap_ci_low=-0.001,
        bootstrap_probability_better=0.25,
    )
    decision = gate.evaluate(evidence)
    winner = gate.choose_winner((evidence,), (decision,))
    assert decision.passed is False
    assert winner.selected_id == "random_walk_zero_return"


def test_daily_track_stays_pending() -> None:
    builder = MultiHorizonDatasetBuilder(repo_root=REPO_ROOT)
    try:
        builder.build("1d")
    except DataPendingError:
        return
    raise AssertionError("1d must remain DATA_PENDING without a direct daily source.")
