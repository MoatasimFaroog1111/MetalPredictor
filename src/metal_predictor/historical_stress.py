from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import DefaultModelRegistry, ModelSpec
from metal_predictor.stress_contracts import HistoricalStressConfig, StressFoldMetric
from metal_predictor.stress_reporting import (
    HistoricalStressReportBuilder,
    LongHistoryStressDecisionPolicy,
)
from metal_predictor.stress_split import PurgedCalendarYearSplitter
from metal_predictor.stress_statistics import balanced_direction_accuracy


class HistoricalStressEvaluator:
    """Produce annual OOF predictions for pre-existing fixed model specifications.

    This component owns fold execution only. Statistical aggregation and evidence
    classification are delegated to dedicated report/policy components.
    """

    def __init__(
        self,
        splitter: PurgedCalendarYearSplitter,
        config: HistoricalStressConfig | None = None,
        report_builder: HistoricalStressReportBuilder | None = None,
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
        self._report_builder = report_builder or HistoricalStressReportBuilder(
            self._config, self._metrics
        )

    @property
    def config(self) -> HistoricalStressConfig:
        return self._config

    def evaluate(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        protocol: str,
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        c = self._config
        clean = self._prepare(frame, feature_names)
        fold_metrics: list[StressFoldMetric] = []
        prediction_parts: list[pd.DataFrame] = []

        for fold in self._splitter.split(clean):
            majority_sign = self._training_majority_sign(fold.train[c.target_name])
            predictions = self._predict(fold.train, fold.validation, feature_names)
            predictions["zero_return"] = np.zeros(len(fold.validation), dtype=float)
            predictions["train_majority_sign"] = np.full(
                len(fold.validation), majority_sign * 1e-12, dtype=float
            )
            fold_metrics.extend(
                self._fold_metrics(protocol, fold, predictions)
            )
            prediction_parts.append(self._prediction_frame(fold, predictions))

        folds = pd.DataFrame([item.as_dict() for item in fold_metrics])
        oof = pd.concat(prediction_parts, ignore_index=True).sort_values(
            ["year", c.timestamp_name]
        ).reset_index(drop=True)
        report = self._report_builder.build(protocol, folds, oof)
        return report, folds, oof

    def _prepare(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> pd.DataFrame:
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
        clean[c.timestamp_name] = pd.to_datetime(
            clean[c.timestamp_name], utc=True, errors="raise"
        )
        clean[c.target_timestamp_name] = pd.to_datetime(
            clean[c.target_timestamp_name], utc=True, errors="raise"
        )
        return clean

    def _predict(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        target = self._config.target_name
        predictions: dict[str, np.ndarray] = {}
        for spec in self._specs:
            model = spec.factory()
            model.fit(train.loc[:, feature_names], train[target])
            prediction = np.asarray(
                model.predict(validation.loc[:, feature_names]), dtype=float
            )
            if not np.isfinite(prediction).all():
                raise ValueError(f"{spec.name} produced non-finite stress predictions.")
            predictions[spec.name] = prediction
        return predictions

    def _fold_metrics(
        self,
        protocol: str,
        fold: object,
        predictions: dict[str, np.ndarray],
    ) -> list[StressFoldMetric]:
        c = self._config
        output: list[StressFoldMetric] = []
        actual = fold.validation[c.target_name].to_numpy(float)
        for model_name, prediction in predictions.items():
            metric = self._metrics.calculate(
                actual,
                prediction,
                fold.validation[c.current_close_name],
                fold.validation[c.target_close_name],
            )
            output.append(
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
                        actual, prediction
                    ),
                    strategy_mean_log_return=float(
                        np.mean(np.sign(prediction) * actual)
                    ),
                )
            )
        return output

    def _prediction_frame(
        self,
        fold: object,
        predictions: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        c = self._config
        return pd.DataFrame(
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

    @staticmethod
    def _training_majority_sign(target: pd.Series) -> float:
        sign = np.sign(pd.to_numeric(target, errors="raise").to_numpy(float))
        nonzero = sign[sign != 0.0]
        if not len(nonzero):
            return 1.0
        return 1.0 if float(np.mean(nonzero > 0.0)) >= 0.5 else -1.0


class LongHistoryStressSuite:
    """Run primary and endpoint-quality sensitivity protocols without retuning."""

    def __init__(
        self,
        evaluator: HistoricalStressEvaluator,
        decision_policy: LongHistoryStressDecisionPolicy | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._decision_policy = decision_policy or LongHistoryStressDecisionPolicy()

    def run(
        self,
        labeled: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        c = self._evaluator.config
        enriched = self._annotate_target_quality(labeled, c)
        primary, primary_folds, primary_oof = self._evaluator.evaluate(
            enriched,
            feature_names,
            protocol="BASELINE_COMPATIBLE_ALL_USABLE_H1",
        )
        strict_input = enriched.loc[
            enriched[c.current_quality_name].eq(c.strict_quality_value)
            & enriched[c.target_quality_name].eq(c.strict_quality_value)
        ].copy()
        strict, strict_folds, strict_oof = self._evaluator.evaluate(
            strict_input,
            feature_names,
            protocol="STRICT_CURRENT_AND_TARGET_FULL_60_MINUTE_H1",
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
                "strict_quality_scope": (
                    "current and exact next-hour target bars must each contain 60 source "
                    "minutes; earlier rolling context may still contain explicitly flagged "
                    "partial-source H1 bars"
                ),
            },
            "primary": primary,
            "strict_quality_sensitivity": strict,
            "decision": self._decision_policy.decide(primary, strict),
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
