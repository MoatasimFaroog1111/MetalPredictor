from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.stress_contracts import HistoricalStressConfig
from metal_predictor.stress_statistics import (
    YearStratifiedCircularBlockBootstrap,
    balanced_direction_accuracy,
    position_turnover_units,
)


class HistoricalStressReportBuilder:
    """Build descriptive and bootstrap summaries from already-produced OOF predictions."""

    def __init__(
        self,
        config: HistoricalStressConfig,
        metrics: RegressionForecastMetrics | None = None,
    ) -> None:
        self._config = config
        self._metrics = metrics or RegressionForecastMetrics()

    def build(
        self,
        protocol: str,
        folds: pd.DataFrame,
        oof: pd.DataFrame,
    ) -> dict[str, object]:
        report: dict[str, object] = {
            "protocol": protocol,
            "evaluation_years": sorted(int(value) for value in oof["year"].unique()),
            "rows": len(oof),
            "models_are_preexisting_fixed_specifications": True,
            "model_selection_performed": False,
            "models": {
                name: self._model_summary(name, folds, oof)
                for name in self._config.model_names
            },
            "baselines": {
                name: self._model_summary(name, folds, oof, uncertainty=False)
                for name in ("zero_return", "train_majority_sign")
            },
            "periods": {
                "legacy_2012_2020": self._period_summary(
                    oof.loc[oof["year"].le(2020)]
                ),
                "modern_2021_2026": self._period_summary(
                    oof.loc[oof["year"].ge(2021)]
                ),
            },
        }
        return report

    def _model_summary(
        self,
        model_name: str,
        folds: pd.DataFrame,
        oof: pd.DataFrame,
        uncertainty: bool = True,
    ) -> dict[str, object]:
        prediction = oof[model_name].to_numpy(float)
        actual = oof["actual"].to_numpy(float)
        metric = self._metrics.calculate(
            actual,
            prediction,
            oof["current_close_usd_per_kg"],
            oof["target_close_usd_per_kg"],
        )
        model_folds = folds.loc[folds["model"].eq(model_name)].set_index("year")
        majority_folds = folds.loc[folds["model"].eq("train_majority_sign")].set_index(
            "year"
        )
        zero_folds = folds.loc[folds["model"].eq("zero_return")].set_index("year")
        turnover_units, flip_rate = position_turnover_units(
            prediction, oof["year"].to_numpy()
        )
        strategy = np.sign(prediction) * actual
        break_even_bps = (
            float(np.mean(strategy) / turnover_units * 10_000.0)
            if turnover_units > 0
            else None
        )
        summary: dict[str, object] = {
            "forecast_metrics": metric.as_dict(),
            "balanced_directional_accuracy": balanced_direction_accuracy(
                actual, prediction
            ),
            "strategy_mean_log_return_pre_cost": float(np.mean(strategy)),
            "mean_turnover_position_units_per_row": turnover_units,
            "position_flip_rate": flip_rate,
            "break_even_cost_bps_per_position_unit": break_even_bps,
            "years": len(model_folds),
            "years_direction_above_50pct": int(
                model_folds["directional_accuracy"].gt(0.5).sum()
            ),
            "years_direction_above_training_majority": int(
                model_folds["directional_accuracy"]
                .gt(majority_folds["directional_accuracy"])
                .sum()
            ),
            "years_positive_pre_cost_strategy_mean": int(
                model_folds["strategy_mean_log_return"].gt(0.0).sum()
            ),
            "years_strategy_above_training_majority": int(
                model_folds["strategy_mean_log_return"]
                .gt(majority_folds["strategy_mean_log_return"])
                .sum()
            ),
            "years_mae_better_than_zero_return": int(
                model_folds["mae_return"].lt(zero_folds["mae_return"]).sum()
            ),
        }
        if uncertainty and model_name in self._config.model_names:
            summary["uncertainty"] = self._uncertainty(model_name, oof)
        return summary

    def _uncertainty(
        self,
        model_name: str,
        oof: pd.DataFrame,
    ) -> dict[str, object]:
        actual = oof["actual"].to_numpy(float)
        prediction = oof[model_name].to_numpy(float)
        majority = oof["train_majority_sign"].to_numpy(float)
        zero = oof["zero_return"].to_numpy(float)
        stats = pd.DataFrame(
            {
                "year": oof["year"].to_numpy(),
                "strategy": np.sign(prediction) * actual,
                "strategy_uplift_vs_majority": (
                    np.sign(prediction) - np.sign(majority)
                )
                * actual,
                "mae_improvement_vs_zero": np.abs(actual - zero)
                - np.abs(actual - prediction),
            }
        )
        if model_name == "ridge_alpha_10":
            stats["mae_improvement_vs_ridge_alpha_100"] = np.abs(
                actual - oof["ridge_alpha_100"].to_numpy(float)
            ) - np.abs(actual - prediction)

        nonzero = actual != 0.0
        direction_stats = pd.DataFrame(
            {
                "year": oof.loc[nonzero, "year"].to_numpy(),
                "correct": (
                    np.sign(prediction[nonzero]) == np.sign(actual[nonzero])
                ).astype(float),
                "direction_uplift_vs_majority": (
                    np.sign(prediction[nonzero]) == np.sign(actual[nonzero])
                ).astype(float)
                - (
                    np.sign(majority[nonzero]) == np.sign(actual[nonzero])
                ).astype(float),
            }
        )

        result: dict[str, object] = {}
        for block in self._config.block_sizes_rows:
            bootstrap = YearStratifiedCircularBlockBootstrap(
                block_size_rows=block,
                resamples=self._config.bootstrap_resamples,
                random_state=self._config.random_state,
            )
            sampled_level = bootstrap.sample_means(
                stats, tuple(column for column in stats.columns if column != "year")
            )
            sampled_direction = bootstrap.sample_means(
                direction_stats,
                ("correct", "direction_uplift_vs_majority"),
            )
            block_result: dict[str, object] = {
                "directional_accuracy": bootstrap.summarize(
                    direction_stats["correct"], sampled_direction["correct"], 0.5
                ).as_dict(),
                "direction_uplift_vs_training_majority": bootstrap.summarize(
                    direction_stats["direction_uplift_vs_majority"],
                    sampled_direction["direction_uplift_vs_majority"],
                    0.0,
                ).as_dict(),
                "strategy_mean_log_return_pre_cost": bootstrap.summarize(
                    stats["strategy"], sampled_level["strategy"], 0.0
                ).as_dict(),
                "strategy_uplift_vs_training_majority": bootstrap.summarize(
                    stats["strategy_uplift_vs_majority"],
                    sampled_level["strategy_uplift_vs_majority"],
                    0.0,
                ).as_dict(),
                "mae_improvement_vs_zero_return": bootstrap.summarize(
                    stats["mae_improvement_vs_zero"],
                    sampled_level["mae_improvement_vs_zero"],
                    0.0,
                ).as_dict(),
            }
            if "mae_improvement_vs_ridge_alpha_100" in stats:
                block_result["mae_improvement_vs_ridge_alpha_100"] = bootstrap.summarize(
                    stats["mae_improvement_vs_ridge_alpha_100"],
                    sampled_level["mae_improvement_vs_ridge_alpha_100"],
                    0.0,
                ).as_dict()
            result[f"block_{block}_rows"] = block_result
        return result

    def _period_summary(self, subset: pd.DataFrame) -> dict[str, object]:
        if subset.empty:
            return {"rows": 0}
        payload: dict[str, object] = {"rows": len(subset), "models": {}}
        for name in (*self._config.model_names, "train_majority_sign", "zero_return"):
            prediction = subset[name].to_numpy(float)
            metric = self._metrics.calculate(
                subset["actual"],
                prediction,
                subset["current_close_usd_per_kg"],
                subset["target_close_usd_per_kg"],
            )
            payload["models"][name] = {
                "mae_return": metric.mae_return,
                "directional_accuracy": metric.directional_accuracy,
                "balanced_directional_accuracy": balanced_direction_accuracy(
                    subset["actual"].to_numpy(float), prediction
                ),
                "strategy_mean_log_return_pre_cost": float(
                    np.mean(np.sign(prediction) * subset["actual"].to_numpy(float))
                ),
            }
        return payload


class LongHistoryStressDecisionPolicy:
    """Pure policy object: classify evidence without mutating any model or dataset."""

    def decide(
        self,
        primary: dict[str, object],
        strict: dict[str, object],
    ) -> dict[str, object]:
        details = {}
        for model in ("ridge_alpha_10", "ridge_alpha_100"):
            details[model] = {
                "primary_direction_gate": self._direction_gate(primary, model),
                "strict_quality_direction_gate": self._direction_gate(strict, model),
                "primary_mae_gate_vs_zero": self._mae_gate(primary, model),
                "strict_quality_mae_gate_vs_zero": self._mae_gate(strict, model),
            }
        challenger = details["ridge_alpha_10"]
        if (
            challenger["primary_direction_gate"]
            and challenger["strict_quality_direction_gate"]
            and challenger["primary_mae_gate_vs_zero"]
            and challenger["strict_quality_mae_gate_vs_zero"]
        ):
            verdict = "ROBUST_LONG_HISTORY_PREDICTIVE_EVIDENCE"
        elif challenger["primary_direction_gate"] and not challenger[
            "strict_quality_direction_gate"
        ]:
            verdict = "QUALITY_SENSITIVE_DIRECTIONAL_SIGNAL_NOT_PROVEN"
        elif challenger["primary_direction_gate"]:
            verdict = "DIRECTIONAL_ONLY_SIGNAL_WITHOUT_MAE_EDGE"
        else:
            verdict = "NO_ROBUST_LONG_HISTORY_EDGE"
        return {
            "verdict": verdict,
            "model_gates": details,
            "baseline_or_future_holdout_changed": False,
            "interpretation": (
                "Long-history stress is supporting evidence only. It cannot replace the "
                "already-frozen one-shot future holdout, and no result here is allowed to "
                "retune that future protocol."
            ),
        }

    @staticmethod
    def _direction_gate(report: dict[str, object], model: str) -> bool:
        uncertainty = report["models"][model]["uncertainty"]["block_120_rows"]
        return (
            uncertainty["directional_accuracy"]["ci95_low"] > 0.5
            and uncertainty["direction_uplift_vs_training_majority"]["ci95_low"] > 0.0
        )

    @staticmethod
    def _mae_gate(report: dict[str, object], model: str) -> bool:
        uncertainty = report["models"][model]["uncertainty"]["block_120_rows"]
        return uncertainty["mae_improvement_vs_zero_return"]["ci95_low"] > 0.0
