from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metal_predictor.metrics import ForecastMetrics, RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec


@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int = 5
    initial_train_fraction: float = 0.50
    min_train_rows: int = 5000
    timestamp_name: str = "timestamp_utc"
    target_timestamp_name: str = "target_timestamp_utc"

    def __post_init__(self) -> None:
        if self.n_splits < 3 or not 0.25 <= self.initial_train_fraction < 0.9 or self.min_train_rows < 100:
            raise ValueError("Invalid walk-forward configuration.")


@dataclass(frozen=True)
class Fold:
    number: int
    train: pd.DataFrame
    validation: pd.DataFrame


class PurgedWalkForwardSplitter:
    """Expanding-window splitter with timestamp-aware label purging."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._config = config

    def split(self, frame: pd.DataFrame) -> tuple[Fold, ...]:
        c = self._config
        ordered = frame.sort_values(c.timestamp_name).reset_index(drop=True)
        n = len(ordered)
        initial = max(c.min_train_rows, int(n * c.initial_train_fraction))
        if initial >= n - c.n_splits:
            raise ValueError("Not enough rows for requested walk-forward configuration.")
        remaining = n - initial
        fold_size = remaining // c.n_splits
        if fold_size <= 0:
            raise ValueError("Walk-forward validation fold would be empty.")
        folds = []
        for i in range(c.n_splits):
            val_start = initial + i * fold_size
            val_end = n if i == c.n_splits - 1 else initial + (i + 1) * fold_size
            validation = ordered.iloc[val_start:val_end].copy()
            first_validation_ts = pd.Timestamp(validation[c.timestamp_name].iloc[0])
            candidates = ordered.iloc[:val_start].copy()
            train = candidates.loc[
                pd.to_datetime(candidates[c.target_timestamp_name], utc=True) < first_validation_ts
            ].copy()
            if len(train) < c.min_train_rows:
                raise ValueError(f"Fold {i + 1} has insufficient purged training rows.")
            if not pd.to_datetime(train[c.target_timestamp_name], utc=True).max() < first_validation_ts:
                raise ValueError(f"Fold {i + 1} label purge failed.")
            folds.append(Fold(i + 1, train.reset_index(drop=True), validation.reset_index(drop=True)))
        return tuple(folds)


@dataclass(frozen=True)
class FoldResult:
    model_name: str
    family: str
    fold: int
    history_days: int | None
    train_rows: int
    validation_rows: int
    train_first_timestamp_utc: str
    train_last_target_timestamp_utc: str
    validation_first_timestamp_utc: str
    validation_last_timestamp_utc: str
    metrics: ForecastMetrics

    def flat_dict(self):
        row = {
            "model_name": self.model_name, "family": self.family, "fold": self.fold,
            "history_days": self.history_days, "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "train_first_timestamp_utc": self.train_first_timestamp_utc,
            "train_last_target_timestamp_utc": self.train_last_target_timestamp_utc,
            "validation_first_timestamp_utc": self.validation_first_timestamp_utc,
            "validation_last_timestamp_utc": self.validation_last_timestamp_utc,
        }
        row.update(self.metrics.as_dict())
        return row


class WalkForwardEvaluator:
    def __init__(self, splitter: PurgedWalkForwardSplitter, metrics: RegressionForecastMetrics,
                 feature_names: tuple[str, ...], target_name: str, timestamp_name: str,
                 target_timestamp_name: str, current_close_name: str, target_close_name: str) -> None:
        self._splitter = splitter
        self._metrics = metrics
        self._features = feature_names
        self._target = target_name
        self._timestamp = timestamp_name
        self._target_timestamp = target_timestamp_name
        self._current_close = current_close_name
        self._target_close = target_close_name

    def evaluate(self, frame: pd.DataFrame, spec: ModelSpec) -> tuple[FoldResult, ...]:
        results = []
        for fold in self._splitter.split(frame):
            train = self._apply_history_window(fold.train, fold.validation, spec.history_days)
            model = spec.factory()
            model.fit(train.loc[:, self._features], train[self._target])
            prediction = np.asarray(model.predict(fold.validation.loc[:, self._features]), dtype=float)
            if not np.isfinite(prediction).all():
                raise ValueError(f"{spec.name} produced non-finite predictions.")
            metrics = self._metrics.calculate(
                fold.validation[self._target], prediction,
                fold.validation[self._current_close], fold.validation[self._target_close])
            results.append(FoldResult(
                model_name=spec.name, family=spec.family, fold=fold.number,
                history_days=spec.history_days, train_rows=len(train), validation_rows=len(fold.validation),
                train_first_timestamp_utc=pd.Timestamp(train[self._timestamp].iloc[0]).isoformat(),
                train_last_target_timestamp_utc=pd.Timestamp(train[self._target_timestamp].iloc[-1]).isoformat(),
                validation_first_timestamp_utc=pd.Timestamp(fold.validation[self._timestamp].iloc[0]).isoformat(),
                validation_last_timestamp_utc=pd.Timestamp(fold.validation[self._timestamp].iloc[-1]).isoformat(),
                metrics=metrics))
        return tuple(results)

    def fit_and_score(self, train: pd.DataFrame, evaluation: pd.DataFrame, spec: ModelSpec):
        effective = self._apply_history_window(train, evaluation, spec.history_days)
        first_eval_ts = pd.Timestamp(evaluation[self._timestamp].iloc[0])
        effective = effective.loc[pd.to_datetime(effective[self._target_timestamp], utc=True) < first_eval_ts].copy()
        if effective.empty:
            raise ValueError(f"No leakage-safe training rows remain for {spec.name}.")
        model = spec.factory()
        model.fit(effective.loc[:, self._features], effective[self._target])
        prediction = np.asarray(model.predict(evaluation.loc[:, self._features]), dtype=float)
        metrics = self._metrics.calculate(evaluation[self._target], prediction,
                                          evaluation[self._current_close], evaluation[self._target_close])
        return model, prediction, metrics

    def fit_final(self, development: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec):
        effective = self._apply_history_window(development, test, spec.history_days)
        first_test_ts = pd.Timestamp(test[self._timestamp].iloc[0])
        effective = effective.loc[pd.to_datetime(effective[self._target_timestamp], utc=True) < first_test_ts].copy()
        model = spec.factory()
        model.fit(effective.loc[:, self._features], effective[self._target])
        return model, effective

    def _apply_history_window(self, train: pd.DataFrame, evaluation: pd.DataFrame, history_days: int | None):
        if history_days is None:
            return train.copy()
        first_eval_ts = pd.Timestamp(evaluation[self._timestamp].iloc[0])
        cutoff = first_eval_ts - pd.Timedelta(days=history_days)
        selected = train.loc[pd.to_datetime(train[self._timestamp], utc=True) >= cutoff].copy()
        if len(selected) < 500:
            raise ValueError(f"History window {history_days}d leaves too few rows: {len(selected)}")
        return selected
