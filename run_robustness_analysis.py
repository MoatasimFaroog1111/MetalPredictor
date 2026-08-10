from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.cpcv import CPCVConfig, CombinatorialPurgedSplitter
from metal_predictor.cpcv_stacking import CPCVStackingConfig, CPCVStackingEvaluator
from metal_predictor.cross_asset_experiment import DevelopmentFeatureSetLoader
from metal_predictor.deflated_sharpe import DeflatedSharpeEvaluator
from metal_predictor.modeling import DefaultModelRegistry
from metal_predictor.multiple_testing import BlockBootstrapHolmTester
from metal_predictor.pbo import CSCVPBOEstimator
from metal_predictor.strategy_matrix import WalkForwardStrategyMatrixBuilder
from metal_predictor.trial_ledger import ResearchTrialLedger
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("artifacts/robustness")


def _walk_forward_splitter() -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(WalkForwardConfig(
        n_splits=5,
        initial_train_fraction=0.50,
        min_train_rows=8000,
    ))


def run() -> dict[str, object]:
    development = DevelopmentFeatureSetLoader().load(
        PROCESSED_DIR, label="silver_only_development"
    )
    registry = DefaultModelRegistry(random_state=42).candidates()
    by_name = {spec.name: spec for spec in registry}
    baseline = by_name["ridge_alpha_100"]
    stack_bases = (
        by_name["ridge_alpha_100"],
        by_name["histgb_absolute_regularized"],
        by_name["lgbm_l1_small"],
    )

    strategy_matrix = WalkForwardStrategyMatrixBuilder(
        splitter=_walk_forward_splitter(),
        feature_names=development.feature_names,
        model_specs=registry,
        stacking_base_specs=stack_bases,
        baseline_spec=baseline,
    ).build(development.frame)

    pbo = CSCVPBOEstimator(n_blocks=8).estimate(
        strategy_matrix.returns,
        strategy_matrix.strategy_names,
    )
    trial_ledger = ResearchTrialLedger(initial_model_registry_trials=len(registry))
    dsr_evaluator = DeflatedSharpeEvaluator()
    stack_dsr = dsr_evaluator.evaluate(
        strategy_matrix.returns,
        strategy_matrix.strategy_names,
        selected_strategy="oof_stack",
        counted_trials=trial_ledger.total_trials,
    )
    sharpes = {
        name: float(
            strategy_matrix.returns[name].mean()
            / strategy_matrix.returns[name].std(ddof=1)
        )
        for name in strategy_matrix.strategy_names
    }
    best_strategy = max(sharpes, key=sharpes.get)
    best_dsr = dsr_evaluator.evaluate(
        strategy_matrix.returns,
        strategy_matrix.strategy_names,
        selected_strategy=best_strategy,
        counted_trials=trial_ledger.total_trials,
    )

    multiple_testing = BlockBootstrapHolmTester(
        block_size_rows=24,
        resamples=5000,
        random_state=42,
    ).test(strategy_matrix.returns, strategy_matrix.strategy_names)
    holm_by_name = {
        row["strategy"]: row
        for row in multiple_testing["results"]
    }

    cpcv = CPCVStackingEvaluator(
        config=CPCVStackingConfig(),
        splitter=CombinatorialPurgedSplitter(CPCVConfig(
            n_groups=6,
            test_groups=2,
            embargo_rows=1,
        )),
        feature_names=development.feature_names,
        baseline_spec=baseline,
        base_specs=stack_bases,
    ).evaluate(development.frame)

    stack_holm = holm_by_name["oof_stack"]
    gates = {
        "cpcv_stack_better_at_least_two_thirds": cpcv["stack_better_fraction"] >= (2.0 / 3.0),
        "pbo_below_20pct": pbo["pbo"] < 0.20,
        "stack_dsr_probability_at_least_95pct": (
            stack_dsr.deflated_sharpe_probability >= 0.95
        ),
        "stack_holm_adjusted_positive_mean_p_below_5pct": (
            stack_holm["holm_adjusted_p_value"] < 0.05
            and stack_holm["mean_return_per_period"] > 0
        ),
    }
    robustness_pass = all(gates.values())
    report = {
        "status": "PASS",
        "research_policy": {
            "historical_test_read": False,
            "data_used": "original Train + Validation development data only",
            "trading_return_definition": (
                "sign(predicted 1h log return) * realized 1h log return; before transaction costs"
            ),
            "cpcv_role": "robustness diagnostic, not live chronological performance estimate",
            "pbo_method": "CSCV from Bailey, Borwein, Lopez de Prado and Zhu",
            "dsr_method": "Bailey and Lopez de Prado Deflated Sharpe Ratio",
            "multiple_testing": "24-hour block bootstrap + Holm FWER correction",
        },
        "strategy_matrix": {
            "rows": len(strategy_matrix.returns),
            "active_strategies": list(strategy_matrix.strategy_names),
            "excluded_constant_strategies": list(
                strategy_matrix.excluded_constant_strategies
            ),
            "per_period_sharpes": sharpes,
            "best_pre_cost_strategy_by_sharpe": best_strategy,
        },
        "trial_ledger": trial_ledger.as_dict(),
        "cpcv": cpcv,
        "pbo": pbo,
        "deflated_sharpe": {
            "oof_stack": stack_dsr.as_dict(),
            "best_observed_strategy": best_dsr.as_dict(),
        },
        "multiple_testing": multiple_testing,
        "stack_robustness_gates": gates,
        "decision": {
            "stack_passes_stage6_robustness": robustness_pass,
            "evidence_level": (
                "ROBUST_AFTER_MULTIPLE_TESTING" if robustness_pass
                else "NOT_PROVEN_AFTER_MULTIPLE_TESTING"
            ),
            "baseline_v1_replacement_allowed": False,
            "reason": (
                "Even a Stage-6 pass remains a research candidate until a genuinely future "
                "holdout is accumulated and evaluated under the frozen Stage-7 protocol."
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_matrix.predictions.to_parquet(
        OUTPUT_DIR / "comparable_oof_predictions.parquet", index=False
    )
    strategy_matrix.returns.to_parquet(
        OUTPUT_DIR / "comparable_strategy_returns.parquet", index=False
    )
    strategy_matrix.returns.to_csv(
        OUTPUT_DIR / "comparable_strategy_returns.csv", index=False
    )
    pd.DataFrame(cpcv["split_results"]).to_csv(
        OUTPUT_DIR / "cpcv_stack_split_results.csv", index=False
    )
    pd.DataFrame(pbo["split_results"]).to_csv(
        OUTPUT_DIR / "cscv_pbo_split_results.csv", index=False
    )
    (OUTPUT_DIR / "robustness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
