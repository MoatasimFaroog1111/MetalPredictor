from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import DefaultModelRegistry, ModelSpec
from metal_predictor.stress_split import PurgedCalendarYearSplitter
from metal_predictor.stress_statistics import (
    YearStratifiedCircularBlockBootstrap,
    balanced_direction_accuracy,
    position_turnover_units,
)


@dataclass(frozen=True)
class HistoricalStressConfig:
    target_name: str = "target_log_return_1h"
    timestamp_name: str = "timestamp_utc"
    target_timestamp_name: str = "target_timestamp_utc"
    current_close_name: str = "close_usd_per_kg"
    target_close_name: str = "target_close_usd_per_kg"
    current_quality_name: str = "quality_flag"
    target_quality_name: str = "target_quality_flag"
    strict_quality_value: str = "OK"
    model_names: tuple[str, ...] = ("ridge_alpha_10", "ridge_alpha_100")
    block_sizes_rows: tuple[int, ...] = (24, 120)
    bootstrap_resamples: int = 5000
    random_state: int = 42


@dataclass(frozen=True)
class StressFoldMetric:
    protocol: str
    year: int
    model: str
    train_rows: int
    validation_rows: int
    train_first_timestamp_utc: str
    train_last_target_timestamp_utc: str
    validation_first_timestamp_utc: str
    validation_last_timestamp_utc: str
    mae_return: float
    rmse_return: float
    r2_return: float
    directional_accuracy: float
    balanced_directional_accuracy: float
    strategy_mean_log_return: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class HistoricalStressEvaluator:
    """Evaluate pre-existing model specifications; never tune or select a new model."""

    def __init__(
        self,
        splitter: PurgedCalendarYearSplitter,
        config: HistoricalStressConfig | None = None,
    ) -> None:
        self._splitter = splitter
        self._config = config or HistoricalStressConfig()
        registry = {item.name: item for item in DefaultModelRegistry().candidates()}
        missing = set(self._config.model_names).difference(registry)
        if missing:
            raise ValueError(f"Stress model registry missing: {sorted(missing)}")
        self._specs: tuple[ModelSpec, ...] = tuple(
            registry[name] for name in self._config.model_names
        )
        self._metrics = RegressionForecastMetrics()

    def evaluate(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        protocol: str,
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        c = self._config
        required = {
            c.target_name,
            c.timestamp_name,
            c.target_timestamp_name,
            c.current_close_name,
            c.target_close_name,
            *feature_names,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Stress input missing columns: {sorted(missing)}")
        clean = frame.loc[
            frame[c.target_name].notna() & frame[c.target_timestamp_name].notna()
        ].copy()
        if clean.empty:
            raise ValueError("Stress input has no labeled rows.")
        clean[c.timestamp_name] = pd.to_datetime(clean[c.timestamp_name], utc=True)
        clean[c.target_timestamp_name] = pd.to_datetime(
            clean[c.target_timestamp_name], utc=True
        )

        fold_metrics: list[StressFoldMetric] = []
        prediction_parts: list[pd.DataFrame] = []
        for fold in self._splitter.split(clean):
            majority_sign = self._training_majority_sign(fold.train[c.target_name])
            predictions: dict[str, np.ndarray] = {}
            for spec in self._specs:
                model = spec.factory()
                model.fit(fold.train.loc[:, feature_names], fold.train[c.target_name])
                prediction = np.asarray(
                    model.predict(fold.validation.loc[:, feature_names]), dtype=float
                )
                if not np.isfinite(prediction).all():
                    raise ValueError(f"{spec.name} produced non-finite stress predictions.")
                predictions[spec.name] = prediction

            predictions["zero_return"] = np.zeros(len(fold.validation), dtype=float)
            predictions["train_majority_sign"] = np.full(
                len(fold.validation), majority_sign * 1e-12, dtype=float
            )
            for model_name, prediction in predictions.items():
                metric = self._metrics.calculate(
                    fold.validation[c.target_name],
                    prediction,
                    fold.validation[c.current_close_name],
                    fold.validation[c.target_close_name],
                )
                fold_metrics.append(
                    StressFoldMetric(
                        protocol=protocol,
                        year=fold.year,
                        model=model_name,
                        train_rows=len(fold.train),
                        validation_rows=len(fold.validation),
                        train_first_timestamp_utc=pd.Timestamp(
                            fold.train[c.timestamp_name].iloc[0]
                        ).isoformat(),
                        train_last_target_timestamp_utc=pd.Timestamp(
                            fold.train[c.target_timestamp_name].iloc[-1]
                        ).isoformat(),
                        validation_first_timestamp_utc=pd.Timestamp(
                            fold.validation[c.timestamp_name].iloc[0]
                        ).isoformat(),
                        validation_last_timestamp_utc=pd.Timestamp(
                            fold.validation[c.timestamp_name].iloc[-1]
                        ).isoformat(),
                        mae_return=metric.mae_return,
                        rmse_return=metric.rmse_return,
                        r2_return=metric.r2_return,
                        directional_accuracy=metric.directional_accuracy,
                        balanced_directional_accuracy=balanced_direction_accuracy(
                            fold.validation[c.target_name].to_numpy(float), prediction
                        ),
                        strategy_mean_log_return=float(
                            np.mean(
                                np.sign(prediction)
                                * fold.validation[c.target_name].to_numpy(float)
                            )
                        ),
                    )
                )

            part = pd.DataFrame(
                {
                    "year": fold.year,
                    c.timestamp_name: fold.validation[c.timestamp_name].to_numpy(),
                    "actual": fold.validation[c.target_name].to_numpy(float),
                    "current_close_usd_per_kg": fold.validation[
                        c.current_close_name
                    ].to_numpy(float),
                    "target_close_usd_per_kg": fold.validation[
                        c.target_close_name
                    ].to_numpy(float),
                    **predictions,
                }
            )
            prediction_parts.append(part)

        folds = pd.DataFrame([item.as_dict() for item in fold_metrics])
        oof = pd.concat(prediction_parts, ignore_index=True).sort_values(
            ["year", c.timestamp_name]
        ).reset_index(drop=True)
        report = self._build_report(protocol, folds, oof)
        return report, folds, oof

    def _build_report(
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
            "models": {},
            "baselines": {},
            "periods": {},
        }
        model_summaries = {
            name: self._model_summary(name, folds, oof)
            for name in self._config.model_names
        }
        report["models"] = model_summaries
        report["baselines"] = {
            name: self._model_summary(name, folds, oof, uncertainty=False)
            for name in ("zero_return", "train_majority_sign")
        }
        report["periods"] = {
            "legacy_2012_2020": self._period_summary(oof.loc[oof["year"].le(2020)]),
            "modern_2021_2026": self._period_summary(oof.loc[oof["year"].ge(2021)]),
        }
        return report

    def _model_summary(
        self,
        model_name: str,
        folds: pd.DataFrame,
        oof: pd.DataFrame,
        uncertainty: bool = True,
    ) -> dict[str, object]:
        c = self._config
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
        if uncertainty and model_name in c.model_names:
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

    @staticmethod
    def _training_majority_sign(target: pd.Series) -> float:
        sign = np.sign(pd.to_numeric(target, errors="raise").to_numpy(float))
        nonzero = sign[sign != 0.0]
        if not len(nonzero):
            return 1.0
        return 1.0 if float(np.mean(nonzero > 0.0)) >= 0.5 else -1.0


class LongHistoryStressSuite:
    """Runs baseline-compatible and strict-source-quality sensitivity protocols."""

    def __init__(self, evaluator: HistoricalStressEvaluator) -> None:
        self._evaluator = evaluator

    def run(
        self,
        labeled: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        c = self._evaluator._config
        enriched = self._annotate_target_quality(labeled, c)
        primary, primary_folds, primary_oof = self._evaluator.evaluate(
            enriched, feature_names, protocol="BASELINE_COMPATIBLE_ALL_USABLE_H1"
        )
        strict_input = enriched.loc[
            enriched[c.current_quality_name].eq(c.strict_quality_value)
            & enriched[c.target_quality_name].eq(c.strict_quality_value)
        ].copy()
        strict, strict_folds, strict_oof = self._evaluator.evaluate(
            strict_input,
            feature_names,
            protocol="STRICT_CURRENT_AND_TARGET_60_MINUTE_H1",
        )
        final = {
            "methodology": {
                "models_tuned_on_long_history": False,
                "models": list(c.model_names),
                "feature_count": len(feature_names),
                "expanding_calendar_year_walk_forward": True,
                "target_timestamp_purge": True,
                "training_majority_direction_baseline_is_fit_on_each_training_fold_only": True,
                "future_holdout_read": False,
                "historical_2021_2026_rows_are_descriptive_stress_only": True,
            },
            "primary": primary,
            "strict_quality_sensitivity": strict,
            "decision": self._decision(primary, strict),
        }
        folds = pd.concat([primary_folds, strict_folds], ignore_index=True)
        oof = pd.concat(
            [
                primary_oof.assign(protocol=primary["protocol"]),
                strict_oof.assign(protocol=strict["protocol"]),
            ],
            ignore_index=True,
        )
        return final, folds, oof

    @staticmethod
    def _annotate_target_quality(
        frame: pd.DataFrame,
        config: HistoricalStressConfig,
    ) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[config.timestamp_name], utc=True, errors="raise")
        next_ts = ts.shift(-1)
        exact = (next_ts - ts).eq(pd.Timedelta(hours=1))
        out[config.target_quality_name] = out[config.current_quality_name].shift(-1).where(
            exact
        )
        return out

    @staticmethod
    def _decision(primary: dict[str, object], strict: dict[str, object]) -> dict[str, object]:
        def direction_gate(report: dict[str, object], model: str) -> bool:
            uncertainty = report["models"][model]["uncertainty"]["block_120_rows"]
            return (
                uncertainty["directional_accuracy"]["ci95_low"] > 0.5
                and uncertainty["direction_uplift_vs_training_majority"]["ci95_low"] > 0.0
            )

        def mae_gate(report: dict[str, object], model: str) -> bool:
            uncertainty = report["models"][model]["uncertainty"]["block_120_rows"]
            return uncertainty["mae_improvement_vs_zero_return"]["ci95_low"] > 0.0

        details = {}
        for model in ("ridge_alpha_10", "ridge_alpha_100"):
            details[model] = {
                "primary_direction_gate": direction_gate(primary, model),
                "strict_quality_direction_gate": direction_gate(strict, model),
                "primary_mae_gate_vs_zero": mae_gate(primary, model),
                "strict_quality_mae_gate_vs_zero": mae_gate(strict, model),
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
