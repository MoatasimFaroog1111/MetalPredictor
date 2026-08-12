from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from metal_predictor.cross_asset_experiment import (
    DevelopmentFeatureSet,
    DevelopmentFeatureSetLoader,
    FeatureSetComparator,
    FeatureSetComparisonConfig,
)
from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import DefaultModelRegistry, ModelSpec
from metal_predictor.precious_metals.ablation import (
    FamilyEvidencePolicy,
    PreciousMetalsFeatureFamily,
    PreciousMetalsFeatureFamilyRegistry,
)
from metal_predictor.precious_metals.coverage import PreciousMetalsCoverageValidator
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


BASE_DIR = Path("data/processed")
ENHANCED_DIR = Path("data/processed_precious_metals")
OUTPUT_DIR = Path("artifacts/cross_asset_precious_metals/ablation_v1")


def frozen_baseline_model() -> ModelSpec:
    candidates = DefaultModelRegistry(random_state=42).candidates()
    return next(spec for spec in candidates if spec.name == "ridge_alpha_100")


def research_splitter() -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(
        WalkForwardConfig(
            n_splits=5,
            initial_train_fraction=0.50,
            min_train_rows=8000,
        )
    )


def _feature_set(
    source: DevelopmentFeatureSet,
    feature_names: tuple[str, ...],
    label: str,
) -> DevelopmentFeatureSet:
    missing = set(feature_names).difference(source.frame.columns)
    if missing:
        raise ValueError(f"{label} references missing features: {sorted(missing)}")
    return DevelopmentFeatureSet(
        frame=source.frame,
        feature_names=feature_names,
        label=label,
    )


def _comparator(key: str) -> FeatureSetComparator:
    return FeatureSetComparator(
        config=FeatureSetComparisonConfig(
            output_dir=OUTPUT_DIR / "comparisons" / key,
            base_set_id="A",
            enhanced_set_id="B",
            artifact_prefix=key,
            bootstrap_block_rows=24,
            bootstrap_resamples=5000,
            strong_min_better_folds=4,
            promising_min_better_folds=3,
        ),
        splitter=research_splitter(),
        metrics=RegressionForecastMetrics(),
    )


def _compare(
    key: str,
    base: DevelopmentFeatureSet,
    enhanced: DevelopmentFeatureSet,
    model: ModelSpec,
) -> dict[str, object]:
    return _comparator(key).compare(base, enhanced, model)


def _compact(report: Mapping[str, object]) -> dict[str, object]:
    bootstrap = report["paired_block_bootstrap"]
    walk_forward = report["walk_forward"]
    paired = report["paired_oof_mae"]
    if not isinstance(bootstrap, Mapping) or not isinstance(walk_forward, Mapping):
        raise ValueError("Comparison report structure is invalid.")
    if not isinstance(paired, Mapping):
        raise ValueError("Comparison report paired OOF structure is invalid.")
    fold_rows = walk_forward["fold_results"]
    if not isinstance(fold_rows, list):
        raise ValueError("Comparison report fold structure is invalid.")
    return {
        "base_mae": float(paired["base"]),
        "enhanced_mae": float(paired["enhanced"]),
        "mae_improvement": float(paired["improvement"]),
        "mae_improvement_percent": float(paired["improvement_percent"]),
        "ci95_low": float(bootstrap["improvement_ci95_low"]),
        "ci95_high": float(bootstrap["improvement_ci95_high"]),
        "probability_better": float(bootstrap["probability_selected_better"]),
        "better_folds": int(walk_forward["enhanced_better_folds"]),
        "total_folds": int(walk_forward["folds"]),
        "fold_mae_improvements": [float(row["mae_improvement"]) for row in fold_rows],
    }


def _candidate_features(
    base: DevelopmentFeatureSet,
    enhanced: DevelopmentFeatureSet,
) -> tuple[str, ...]:
    base_names = set(base.feature_names)
    return tuple(name for name in enhanced.feature_names if name not in base_names)


def _addon_set(
    base: DevelopmentFeatureSet,
    enhanced: DevelopmentFeatureSet,
    family: PreciousMetalsFeatureFamily,
) -> DevelopmentFeatureSet:
    features = tuple(base.feature_names) + tuple(family.features)
    return _feature_set(enhanced, features, f"base_plus_{family.name}")


def _without_family_set(
    enhanced: DevelopmentFeatureSet,
    family: PreciousMetalsFeatureFamily,
) -> DevelopmentFeatureSet:
    removed = set(family.features)
    features = tuple(name for name in enhanced.feature_names if name not in removed)
    return _feature_set(enhanced, features, f"full_without_{family.name}")


def _retained_feature_set(
    base: DevelopmentFeatureSet,
    enhanced: DevelopmentFeatureSet,
    retained_features: set[str],
) -> DevelopmentFeatureSet:
    base_names = set(base.feature_names)
    ordered = tuple(
        name
        for name in enhanced.feature_names
        if name in base_names or name in retained_features
    )
    return _feature_set(enhanced, ordered, "parsimonious_xpt_xpd_candidate")


def run() -> dict[str, object]:
    loader = DevelopmentFeatureSetLoader()
    base = loader.load(BASE_DIR, label="A_frozen_silver_52")
    enhanced = loader.load(
        ENHANCED_DIR,
        label="B_silver_plus_pre_registered_XPT_XPD_v1",
    )

    candidate_features = _candidate_features(base, enhanced)
    registry = PreciousMetalsFeatureFamilyRegistry()
    registry.validate_candidate_features(candidate_features)
    policy = FamilyEvidencePolicy(family_count=len(registry.families))

    # This gate runs before the first estimator fit in this script.
    coverage = PreciousMetalsCoverageValidator().validate(
        enhanced.frame,
        research_splitter(),
    )

    model = frozen_baseline_model()
    full_reference = _compare("full_43_vs_base_52", base, enhanced, model)

    family_rows: list[dict[str, object]] = []
    retained_family_names: list[str] = []
    retained_features: set[str] = set()

    for family in registry.families:
        addon = _addon_set(base, enhanced, family)
        without_family = _without_family_set(enhanced, family)

        addon_report = _compare(
            f"addon_{family.name}",
            base,
            addon,
            model,
        )
        conditional_report = _compare(
            f"conditional_{family.name}",
            without_family,
            enhanced,
            model,
        )
        assessment = policy.assess(family, addon_report, conditional_report)
        assessment["addon_comparison"] = _compact(addon_report)
        assessment["conditional_comparison"] = _compact(conditional_report)
        family_rows.append(assessment)

        if bool(assessment["retain_in_parsimonious_candidate"]):
            retained_family_names.append(family.name)
            retained_features.update(family.features)

    retained = _retained_feature_set(base, enhanced, retained_features)
    if retained_features:
        retained_vs_base = _compare("retained_subset_vs_base_52", base, retained, model)
    else:
        retained_vs_base = None

    all_candidate_set = set(candidate_features)
    if retained_features and retained_features != all_candidate_set:
        full_vs_retained = _compare("full_43_vs_retained_subset", retained, enhanced, model)
    else:
        full_vs_retained = None

    ranking = pd.DataFrame(
        [
            {
                "family": row["family"],
                "classification": row["classification"],
                "retain": row["retain_in_parsimonious_candidate"],
                "feature_count": row["feature_count"],
                "addon_mae_improvement_percent": row["addon_evidence"]["mae_improvement_percent"],
                "addon_probability_better": row["addon_evidence"]["probability_better"],
                "addon_better_folds": row["addon_evidence"]["better_folds"],
                "conditional_mae_improvement_percent": row["conditional_evidence"]["mae_improvement_percent"],
                "conditional_probability_better": row["conditional_evidence"]["probability_better"],
                "conditional_better_folds": row["conditional_evidence"]["better_folds"],
            }
            for row in family_rows
        ]
    ).sort_values(
        ["retain", "conditional_mae_improvement_percent", "addon_mae_improvement_percent"],
        ascending=[False, False, False],
    )

    report = {
        "status": "PASS",
        "ablation_version": registry.version,
        "registry_fingerprint_sha256": registry.fingerprint(),
        "pre_registered_before_family_results": True,
        "family_count": len(registry.families),
        "candidate_feature_count": len(candidate_features),
        "evidence_policy": policy.as_dict(),
        "coverage_gate": coverage,
        "full_43_reference": _compact(full_reference),
        "families": family_rows,
        "parsimonious_candidate": {
            "retained_families": retained_family_names,
            "retained_candidate_feature_count": len(retained_features),
            "retained_candidate_features": [
                feature for feature in registry.all_features if feature in retained_features
            ],
            "total_model_feature_count": len(base.feature_names) + len(retained_features),
            "selection_rule_fixed_before_family_results": True,
            "selection_derived_on_same_development_data": True,
            "live_promotion_eligible": False,
            "requires_new_or_nested_validation_before_any_promotion": True,
            "diagnostic_vs_base": (
                _compact(retained_vs_base) if retained_vs_base is not None else None
            ),
            "diagnostic_full_43_vs_retained": (
                _compact(full_vs_retained) if full_vs_retained is not None else None
            ),
        },
        "research_policy": {
            "data_used": "original Train + Validation only",
            "old_test_read": False,
            "future_holdout_read": False,
            "family_results_used_for_live_model": False,
            "automatic_live_promotion": False,
            "estimator": model.name,
            "estimator_hyperparameters_frozen_before_ablation": True,
            "paired_walk_forward_folds": True,
            "bootstrap_block_rows": 24,
            "bootstrap_resamples_per_comparison": 5000,
        },
        "guardrails": {
            "research_only": True,
            "edge_status": "NOT_PROVEN",
            "live_model_mutated": False,
            "frozen_52_feature_graph_mutated": False,
            "future_holdout_read": False,
            "old_test_read": False,
            "buy_sell_enabled": False,
            "execution_enabled": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "precious_metals_family_ablation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    ranking.to_csv(OUTPUT_DIR / "precious_metals_family_ablation_ranking.csv", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
