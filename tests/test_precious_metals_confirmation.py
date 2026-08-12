from __future__ import annotations

import pytest

from metal_predictor.precious_metals.confirmation import (
    CANDIDATE_FEATURES,
    CANDIDATE_FAMILIES,
    CANDIDATE_ID,
    CONFIRMATION_VERSION,
    HistoricalConfirmationPolicy,
    candidate_fingerprint,
)


def _metrics(directional: float) -> dict[str, object]:
    return {"directional_accuracy": directional}


def _bootstrap(
    improvement: float,
    probability: float,
    ci_low: float,
) -> dict[str, object]:
    return {
        "mae_improvement_vs_baseline": improvement,
        "probability_selected_better": probability,
        "improvement_ci95_low": ci_low,
    }


def test_locked_candidate_is_exactly_ten_features_from_two_families() -> None:
    assert CONFIRMATION_VERSION == "precious-metals-historical-confirmation-v1"
    assert CANDIDATE_ID == "xpt-xpd-candle-shape-own-returns-v1"
    assert CANDIDATE_FAMILIES == ("candle_shape", "own_returns")
    assert len(CANDIDATE_FEATURES) == 10
    assert len(set(CANDIDATE_FEATURES)) == 10
    assert len(candidate_fingerprint()) == 64


def test_confirmation_passes_only_when_all_one_shot_gates_pass() -> None:
    policy = HistoricalConfirmationPolicy()
    decision = policy.decide(
        bootstrap=_bootstrap(0.0001, 0.99, 0.00002),
        base_metrics=_metrics(0.51),
        candidate_metrics=_metrics(0.52),
        joint_coverage=0.99,
    )
    assert decision["status"] == "CONFIRMED"
    assert decision["confirmed"] is True
    assert all(decision["checks"].values())


def test_confirmation_rejects_candidate_without_positive_interval() -> None:
    policy = HistoricalConfirmationPolicy()
    decision = policy.decide(
        bootstrap=_bootstrap(0.0001, 0.99, -0.00001),
        base_metrics=_metrics(0.51),
        candidate_metrics=_metrics(0.52),
        joint_coverage=0.99,
    )
    assert decision["status"] == "REJECTED_ON_HISTORICAL_TEST"
    assert decision["confirmed"] is False
    assert decision["checks"]["ci95_low_positive"] is False


def test_confirmation_rejects_material_directional_degradation() -> None:
    policy = HistoricalConfirmationPolicy()
    decision = policy.decide(
        bootstrap=_bootstrap(0.0001, 0.99, 0.00002),
        base_metrics=_metrics(0.52),
        candidate_metrics=_metrics(0.50),
        joint_coverage=0.99,
    )
    assert decision["confirmed"] is False
    assert decision["checks"]["directional_accuracy_not_materially_worse"] is False


def test_confirmation_policy_is_one_shot_and_strictly_covered() -> None:
    policy = HistoricalConfirmationPolicy()
    assert policy.minimum_joint_coverage == pytest.approx(0.90)
    assert policy.minimum_bootstrap_probability_better == pytest.approx(0.95)
    assert policy.bootstrap_resamples == 5000
