from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metal_predictor.modeling import ModelSpec
from metal_predictor.oof_stacking import NestedOOFStackingEvaluator, StackingConfig
from metal_predictor.walk_forward import PurgedWalkForwardSplitter


@dataclass(frozen=True)
class StrategyMatrixResult:
    predictions: pd.DataFrame
    returns: pd.DataFrame
    strategy_names: tuple[str, ...]
    excluded_constant_strategies: tuple[str, ...]


class WalkForwardStrategyMatrixBuilder:
    """Builds comparable pre-cost trading-return series from common OOF timestamps.

    Every candidate is evaluated on exactly the same outer validation rows. Strategy
    return is sign(predicted next-hour log return) * realized next-hour log return.
    This is intentionally a statistical research proxy before transaction costs; it
    must not be interpreted as deployable P&L.
    """

    def __init__(
        self,
        splitter: PurgedWalkForwardSplitter,
        feature_names: tuple[str, ...],
        model_specs: tuple[ModelSpec, ...],
        stacking_base_specs: tuple[ModelSpec, ...],
        baseline_spec: ModelSpec,
        target_name: str = "target_log_return_1h",
        timestamp_name: str = "timestamp_utc",
    ) -> None:
        self._splitter = splitter
        self._features = feature_names
        self._specs = model_specs
        self._stack_bases = stacking_base_specs
        self._baseline = baseline_spec
        self._target = target_name
        self._timestamp = timestamp_name

    def build(self, development: pd.DataFrame) -> StrategyMatrixResult:
        folds = self._splitter.split(development)
        prediction_parts: list[pd.DataFrame] = []
        for fold in folds:
            part = pd.DataFrame({
                "timestamp_utc": fold.validation[self._timestamp].to_numpy(),
                "actual": fold.validation[self._target].to_numpy(float),
                "fold": fold.number,
            })
            for spec in self._specs:
                train = self._apply_history_window(fold.train, fold.validation, spec.history_days)
                model = spec.factory()
                model.fit(train.loc[:, self._features], train[self._target])
                prediction = np.asarray(
                    model.predict(fold.validation.loc[:, self._features]), dtype=float
                )
                if not np.isfinite(prediction).all():
                    raise ValueError(f"{spec.name} produced non-finite strategy-matrix predictions.")
                part[spec.name] = prediction
            prediction_parts.append(part)

        predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
            "timestamp_utc"
        ).reset_index(drop=True)

        stack_evaluator = NestedOOFStackingEvaluator(
            config=StackingConfig(),
            outer_splitter=self._splitter,
            feature_names=self._features,
            baseline_spec=self._baseline,
            base_specs=self._stack_bases,
        )
        stack_report = stack_evaluator.evaluate(development)
        stack_oof = stack_report["oof_predictions"].loc[
            :, ["timestamp_utc", "stacked_prediction"]
        ].copy()
        stack_oof["timestamp_utc"] = pd.to_datetime(
            stack_oof["timestamp_utc"], utc=True, errors="raise"
        )
        predictions["timestamp_utc"] = pd.to_datetime(
            predictions["timestamp_utc"], utc=True, errors="raise"
        )
        predictions = predictions.merge(
            stack_oof,
            on="timestamp_utc",
            how="left",
            validate="one_to_one",
        )
        if predictions["stacked_prediction"].isna().any():
            raise ValueError("Stacking OOF timestamps do not align with strategy matrix.")

        returns = pd.DataFrame({
            "timestamp_utc": predictions["timestamp_utc"],
            "actual": predictions["actual"],
        })
        candidate_names = [spec.name for spec in self._specs] + ["oof_stack"]
        excluded: list[str] = []
        active: list[str] = []
        actual = predictions["actual"].to_numpy(float)
        for name in candidate_names:
            prediction_name = "stacked_prediction" if name == "oof_stack" else name
            signal = np.sign(predictions[prediction_name].to_numpy(float))
            strategy_return = signal * actual
            returns[name] = strategy_return
            if np.nanstd(strategy_return, ddof=1) <= 1e-15:
                excluded.append(name)
            else:
                active.append(name)

        if len(active) < 5:
            raise ValueError("Too few non-constant strategies for overfitting analysis.")
        if not np.isfinite(returns.loc[:, active].to_numpy(float)).all():
            raise ValueError("Strategy return matrix contains non-finite values.")
        return StrategyMatrixResult(
            predictions=predictions,
            returns=returns,
            strategy_names=tuple(active),
            excluded_constant_strategies=tuple(excluded),
        )

    def _apply_history_window(
        self,
        train: pd.DataFrame,
        evaluation: pd.DataFrame,
        history_days: int | None,
    ) -> pd.DataFrame:
        if history_days is None:
            return train.copy()
        first_eval_ts = pd.Timestamp(evaluation[self._timestamp].iloc[0])
        cutoff = first_eval_ts - pd.Timedelta(days=history_days)
        selected = train.loc[
            pd.to_datetime(train[self._timestamp], utc=True) >= cutoff
        ].copy()
        if len(selected) < 500:
            raise ValueError(
                f"History window {history_days}d leaves too few rows: {len(selected)}"
            )
        return selected
