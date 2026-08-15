from __future__ import annotations

import numpy as np

from metal_predictor.multi_horizon.dataset import MultiHorizonDatasetBuilder
from metal_predictor.multi_horizon.split import ExpandingWalkForwardPlanner
from metal_predictor.multi_horizon.stage6_development import Stage6DevelopmentOnlyEvaluator
from metal_predictor.multi_horizon.stage6_models import (
    Stage6DevelopmentModelFactory,
    TrainMedianReturnRegressor,
)
from metal_predictor.multi_horizon.stage6_preregistration import stage6_candidate_registry


def test_train_median_regressor_learns_train_only_constant() -> None:
    x = np.arange(15, dtype=float).reshape(5, 3)
    y = np.array([-0.4, -0.1, 0.2, 0.9, 1.1], dtype=float)
    model = TrainMedianReturnRegressor().fit(x, y)
    predicted = model.predict(np.ones((3, 3), dtype=float))
    assert np.array_equal(predicted, np.array([0.2, 0.2, 0.2]))


def test_stage6_factory_builds_every_preregistered_candidate() -> None:
    factory = Stage6DevelopmentModelFactory()
    for spec in stage6_candidate_registry():
        assert factory.create(spec) is not None


def test_stage6_4h_evaluation_never_reads_locked_test_for_model_matrices() -> None:
    dataset, _ = MultiHorizonDatasetBuilder().build("4h")
    plan = ExpandingWalkForwardPlanner().plan(dataset.model_row_count)
    result = Stage6DevelopmentOnlyEvaluator().evaluate(dataset, plan)

    assert result.horizon_key == "4h"
    assert result.historical_test_rows > 0
    assert result.development_rows == plan.development_end_exclusive
    assert len(result.candidates) == 4
    assert result.selected_kind in {"BASELINE", "CANDIDATE"}
    allowed_ids = {"random_walk_zero_return"}
    allowed_ids.update(spec.candidate_id for spec in stage6_candidate_registry())
    assert result.selected_id in allowed_ids
    payload = result.as_dict()
    assert payload["performance_scope"] == "DEVELOPMENT_ONLY"
    assert payload["historical_test"]["metrics_read"] is False
    assert payload["historical_test"]["predictions_computed"] is False
    assert payload["historical_test"]["used_for_fit"] is False
    assert payload["historical_test"]["used_for_selection"] is False
