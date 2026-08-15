from __future__ import annotations

import json
from pathlib import Path

from metal_predictor.multi_horizon.feature_set import feature_fingerprint_sha256
from metal_predictor.multi_horizon.stage6_preregistration import (
    stage6_candidate_registry,
    stage6_preregistration_fingerprint_sha256,
    stage6_preregistration_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage6_preregistration_file_matches_code_exactly() -> None:
    locked = json.loads(
        (
            ROOT
            / "research_data/bullionvault_horizons/stage6_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    assert locked == stage6_preregistration_payload()


def test_stage6_preregistration_preserves_scientific_firewalls() -> None:
    payload = stage6_preregistration_payload()
    assert payload["features"]["feature_fingerprint_sha256"] == feature_fingerprint_sha256()
    assert payload["scope"]["forward_bars_used_for_fit"] is False
    assert payload["scope"]["bullionvault_microstructure_used_for_fit"] is False
    assert payload["scope"]["source_domain_merging"] is False
    assert payload["walk_forward"]["historical_test_used_for_stage6_development"] is False
    assert payload["historical_confirmation_policy"]["authorized_during_preregistration"] is False
    assert payload["production_policy"]["candidate_model_artifacts_created_now"] is False
    assert payload["production_policy"]["candidate_forecast_routes_changed_now"] is False
    assert payload["guardrails"]["future_holdout_read"] is False
    assert payload["guardrails"]["historical_test_metrics_read"] is False


def test_stage6_candidate_set_is_fixed_and_fingerprint_is_stable_shape() -> None:
    candidates = stage6_candidate_registry()
    assert [item.candidate_id for item in candidates] == [
        "train_median_return_v2",
        "ridge_alpha_1000_v2",
        "random_forest_shallow_v2",
        "hist_gradient_boosting_shallow_v2",
    ]
    assert len(stage6_preregistration_fingerprint_sha256()) == 64
