from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.precious_metals.ablation import (
    ABLATION_VERSION,
    FamilyEvidencePolicy,
    PreciousMetalsFeatureFamilyRegistry,
)
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures


C = ColumnConfig()


def _metal_frame(asset: str, rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    offset = np.arange(rows, dtype=float)
    close = (31000.0 if asset == "XPT" else 33000.0) + offset
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open_usd_per_kg": close - 1.0,
            "high_usd_per_kg": close + 2.0,
            "low_usd_per_kg": close - 2.0,
            "close_usd_per_kg": close,
            "quality_flag": "PROVIDER_H1_BID",
        }
    )


def _component() -> PlatinumPalladiumCrossAssetFeatures:
    return PlatinumPalladiumCrossAssetFeatures(
        _metal_frame("XPT"),
        _metal_frame("XPD"),
        ExactTimestampAligner(),
        C,
    )


def _report(
    *,
    improvement: float,
    probability: float,
    ci_low: float,
    ci_high: float,
    better_folds: int,
) -> dict[str, object]:
    return {
        "paired_block_bootstrap": {
            "mae_improvement_vs_baseline": improvement,
            "mae_improvement_percent": improvement * 1000.0,
            "improvement_ci95_low": ci_low,
            "improvement_ci95_high": ci_high,
            "probability_selected_better": probability,
        },
        "walk_forward": {
            "enhanced_better_folds": better_folds,
            "folds": 5,
        },
    }


def test_ablation_registry_exactly_partitions_all_43_candidate_features() -> None:
    registry = PreciousMetalsFeatureFamilyRegistry()
    component = _component()

    registry.validate_candidate_features(component.feature_names)

    assert registry.version == ABLATION_VERSION
    assert len(registry.families) == 7
    assert len(registry.all_features) == 43
    assert len(set(registry.all_features)) == 43
    assert sum(len(family.features) for family in registry.families) == 43
    assert len(registry.fingerprint()) == 64


def test_registry_fails_closed_when_candidate_graph_changes() -> None:
    registry = PreciousMetalsFeatureFamilyRegistry()
    features = list(_component().feature_names)
    features.pop()
    features.append("unexpected_future_feature")

    with pytest.raises(ValueError, match="does not exactly partition"):
        registry.validate_candidate_features(tuple(features))


def test_strong_family_gate_is_conservative_across_seven_families() -> None:
    policy = FamilyEvidencePolicy()
    assert policy.strong_probability == pytest.approx(1.0 - 0.05 / 7.0)
    assert policy.strong_probability > 0.99


def test_core_family_requires_strong_addon_and_conditional_evidence() -> None:
    family = PreciousMetalsFeatureFamilyRegistry().families[2]
    policy = FamilyEvidencePolicy()
    strong = _report(
        improvement=0.0001,
        probability=1.0,
        ci_low=0.00002,
        ci_high=0.00015,
        better_folds=4,
    )

    assessment = policy.assess(family, strong, strong)

    assert assessment["classification"] == "CORE"
    assert assessment["retain_in_parsimonious_candidate"] is True


def test_conditional_strong_family_is_retained_even_if_not_standalone() -> None:
    family = PreciousMetalsFeatureFamilyRegistry().families[6]
    policy = FamilyEvidencePolicy()
    weak_addon = _report(
        improvement=-0.00001,
        probability=0.4,
        ci_low=-0.00005,
        ci_high=0.00003,
        better_folds=2,
    )
    strong_conditional = _report(
        improvement=0.00008,
        probability=1.0,
        ci_low=0.00002,
        ci_high=0.00012,
        better_folds=5,
    )

    assessment = policy.assess(family, weak_addon, strong_conditional)

    assert assessment["classification"] == "COMPLEMENTARY_CORE"
    assert assessment["retain_in_parsimonious_candidate"] is True


def test_strong_conditional_harm_forces_exclusion() -> None:
    family = PreciousMetalsFeatureFamilyRegistry().families[1]
    policy = FamilyEvidencePolicy()
    supportive_addon = _report(
        improvement=0.00005,
        probability=0.95,
        ci_low=-0.00001,
        ci_high=0.00009,
        better_folds=3,
    )
    harmful_conditional = _report(
        improvement=-0.00008,
        probability=0.0,
        ci_low=-0.00012,
        ci_high=-0.00002,
        better_folds=0,
    )

    assessment = policy.assess(family, supportive_addon, harmful_conditional)

    assert assessment["classification"] == "HARMFUL_CONDITIONAL"
    assert assessment["retain_in_parsimonious_candidate"] is False
