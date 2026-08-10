from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


@dataclass(frozen=True)
class SelectivePredictionConfig:
    coverage_targets: tuple[float, ...] = (0.50, 0.35, 0.25, 0.15, 0.10)
    inner_splits: int = 3
    inner_initial_train_fraction: float = 0.55
    inner_min_train_rows: int = 4000
    min_inner_signals: int = 250
    bootstrap_block_rows: int = 24
    bootstrap_resamples: int = 5000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.coverage_targets:
            raise ValueError("At least one selective coverage target is required.")
        if any(not 0.05 <= value <= 0.80 for value in self.coverage_targets):
            raise ValueError("coverage_targets must be between 5% and 80%.")
        if self.inner_splits < 3 or self.inner_min_train_rows < 1000:
            raise ValueError("Nested calibration requires at least 3 inner folds and sufficient history.")


@dataclass(frozen=True)
class ThresholdCandidate:
    target_coverage: float
    absolute_prediction_threshold: float
    calibration_rows: int
    selected_rows: int
    coverage: float
    directional_accuracy: float
    wilson_lower_95: float
    mean_abs_realized_return: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectiveFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    threshold: float
    calibration_target_coverage: float
    calibration_directional_accuracy: float
    calibration_wilson_lower_95: float
    selected_rows: int
    coverage: float
    directional_accuracy: float | None
    mean_abs_realized_return: float | None
    validation_first_timestamp_utc: str
    validation_last_timestamp_utc: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectiveBootstrapResult:
    directional_accuracy: float
    ci95_low: float
    ci95_high: float
    probability_above_50: float
    coverage: float
    selected_rows: int
    total_rows: int
    block_size_rows: int
    resamples: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class NestedSelectivePredictionEvaluator:
    """Evaluates NO-TRADE abstention without using an outer validation fold to tune its threshold.

    For each outer Walk-Forward fold, the confidence threshold is chosen only from
    nested inner OOF predictions inside the outer training history. The frozen
    estimator is then fitted on the full outer training set, and the chosen threshold
    is applied once to the unseen outer validation fold.
    """

    def __init__(
        self,
        config: SelectivePredictionConfig,
        outer_splitter: PurgedWalkForwardSplitter,
        feature_names: tuple[str, ...],
        target_name: str = "target_log_return_1h",
        timestamp_name: str = "timestamp_utc",
    ) -> None:
        self._config = config
        self._outer_splitter = outer_splitter
        self._features = feature_names
        self._target = target_name
        self._timestamp = timestamp_name

    def evaluate(self, development: pd.DataFrame, model_spec: ModelSpec) -> dict[str, object]:
        fold_results: list[SelectiveFoldResult] = []
        outer_predictions: list[pd.DataFrame] = []
        threshold_audits: list[dict[str, object]] = []

        for outer in self._outer_splitter.split(development):
            calibration, candidates, chosen = self._calibrate_threshold(outer.train, model_spec)
            model = model_spec.factory()
            model.fit(outer.train.loc[:, self._features], outer.train[self._target])
            prediction = np.asarray(
                model.predict(outer.validation.loc[:, self._features]), dtype=float
            )
            if not np.isfinite(prediction).all():
                raise ValueError("Selective baseline model produced non-finite predictions.")

            target = outer.validation[self._target].to_numpy(float)
            selected = np.abs(prediction) >= chosen.absolute_prediction_threshold
            correctness = np.sign(prediction) == np.sign(target)
            selected_rows = int(selected.sum())
            directional_accuracy = (
                float(correctness[selected].mean()) if selected_rows else None
            )
            mean_abs_return = (
                float(np.abs(target[selected]).mean()) if selected_rows else None
            )
            fold_results.append(SelectiveFoldResult(
                fold=outer.number,
                train_rows=len(outer.train),
                validation_rows=len(outer.validation),
                threshold=chosen.absolute_prediction_threshold,
                calibration_target_coverage=chosen.target_coverage,
                calibration_directional_accuracy=chosen.directional_accuracy,
                calibration_wilson_lower_95=chosen.wilson_lower_95,
                selected_rows=selected_rows,
                coverage=selected_rows / len(outer.validation),
                directional_accuracy=directional_accuracy,
                mean_abs_realized_return=mean_abs_return,
                validation_first_timestamp_utc=pd.Timestamp(
                    outer.validation[self._timestamp].iloc[0]
                ).isoformat(),
                validation_last_timestamp_utc=pd.Timestamp(
                    outer.validation[self._timestamp].iloc[-1]
                ).isoformat(),
            ))
            outer_predictions.append(pd.DataFrame({
                "timestamp_utc": outer.validation[self._timestamp].to_numpy(),
                "target_log_return_1h": target,
                "prediction": prediction,
                "absolute_prediction_threshold": chosen.absolute_prediction_threshold,
                "selected": selected,
                "direction_correct": correctness,
                "fold": outer.number,
            }))
            threshold_audits.append({
                "fold": outer.number,
                "chosen": chosen.as_dict(),
                "candidates": [candidate.as_dict() for candidate in candidates],
                "inner_oof_rows": len(calibration),
            })

        oof = pd.concat(outer_predictions, ignore_index=True).sort_values(
            "timestamp_utc"
        ).reset_index(drop=True)
        bootstrap = self._block_bootstrap(oof)
        better_folds = sum(
            result.directional_accuracy is not None
            and result.directional_accuracy > 0.50
            for result in fold_results
        )
        strong = (
            better_folds >= 4
            and bootstrap.ci95_low > 0.50
            and bootstrap.probability_above_50 >= 0.95
            and bootstrap.coverage >= 0.10
        )
        promising = (
            not strong
            and better_folds >= 3
            and bootstrap.directional_accuracy > 0.50
            and bootstrap.probability_above_50 >= 0.80
            and bootstrap.coverage >= 0.10
        )
        evidence = (
            "STRONG_EVIDENCE" if strong
            else "PROMISING_NOT_CONCLUSIVE" if promising
            else "NO_STABLE_EVIDENCE"
        )
        return {
            "status": "PASS",
            "research_policy": {
                "historical_test_read": False,
                "data_used": "original Train + Validation only",
                "estimator": model_spec.name,
                "threshold_tuning": "nested inner purged Walk-Forward only",
                "outer_validation_used_for_threshold_selection": False,
                "abstention_action": "NO_TRADE when |prediction| is below calibrated threshold",
            },
            "outer_walk_forward": {
                "folds": len(fold_results),
                "better_than_50pct_folds": better_folds,
                "fold_results": [result.as_dict() for result in fold_results],
            },
            "pooled_selected_signals": bootstrap.as_dict(),
            "decision": {
                "promote_selective_prediction": strong or promising,
                "evidence_level": evidence,
                "rule_fixed_before_result": True,
                "requires_future_holdout_confirmation": strong or promising,
            },
            "threshold_audit": threshold_audits,
            "oof_predictions": oof,
        }

    def _calibrate_threshold(
        self,
        outer_train: pd.DataFrame,
        model_spec: ModelSpec,
    ) -> tuple[pd.DataFrame, tuple[ThresholdCandidate, ...], ThresholdCandidate]:
        inner_splitter = PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=self._config.inner_splits,
            initial_train_fraction=self._config.inner_initial_train_fraction,
            min_train_rows=min(
                self._config.inner_min_train_rows,
                max(1000, len(outer_train) // 3),
            ),
        ))
        parts: list[pd.DataFrame] = []
        for inner in inner_splitter.split(outer_train):
            model = model_spec.factory()
            model.fit(inner.train.loc[:, self._features], inner.train[self._target])
            prediction = np.asarray(
                model.predict(inner.validation.loc[:, self._features]), dtype=float
            )
            parts.append(pd.DataFrame({
                "target": inner.validation[self._target].to_numpy(float),
                "prediction": prediction,
            }))
        calibration = pd.concat(parts, ignore_index=True)
        if not np.isfinite(calibration[["target", "prediction"]].to_numpy(float)).all():
            raise ValueError("Inner calibration contains non-finite values.")

        candidates: list[ThresholdCandidate] = []
        magnitude = np.abs(calibration["prediction"].to_numpy(float))
        target = calibration["target"].to_numpy(float)
        prediction = calibration["prediction"].to_numpy(float)
        for target_coverage in self._config.coverage_targets:
            threshold = float(np.quantile(magnitude, 1.0 - target_coverage))
            selected = magnitude >= threshold
            selected_rows = int(selected.sum())
            if selected_rows < self._config.min_inner_signals:
                continue
            correct = np.sign(prediction[selected]) == np.sign(target[selected])
            accuracy = float(correct.mean())
            candidates.append(ThresholdCandidate(
                target_coverage=target_coverage,
                absolute_prediction_threshold=threshold,
                calibration_rows=len(calibration),
                selected_rows=selected_rows,
                coverage=selected_rows / len(calibration),
                directional_accuracy=accuracy,
                wilson_lower_95=self._wilson_lower(accuracy, selected_rows),
                mean_abs_realized_return=float(np.abs(target[selected]).mean()),
            ))
        if not candidates:
            raise ValueError("No selective threshold candidate has enough inner signals.")
        # Robust selection: maximize the lower confidence bound, then preserve coverage.
        chosen = max(
            candidates,
            key=lambda item: (
                item.wilson_lower_95,
                item.directional_accuracy,
                item.coverage,
            ),
        )
        return calibration, tuple(candidates), chosen

    def _block_bootstrap(self, oof: pd.DataFrame) -> SelectiveBootstrapResult:
        selected = oof["selected"].to_numpy(bool)
        correct = oof["direction_correct"].to_numpy(bool)
        n = len(oof)
        selected_rows = int(selected.sum())
        if selected_rows == 0:
            raise ValueError("Selective evaluation produced zero signals.")
        observed = float(correct[selected].mean())
        rng = np.random.default_rng(self._config.random_state)
        block = min(self._config.bootstrap_block_rows, n)
        samples: list[float] = []
        for _ in range(self._config.bootstrap_resamples):
            positions: list[int] = []
            while len(positions) < n:
                start = int(rng.integers(0, max(1, n - block + 1)))
                positions.extend(range(start, min(start + block, n)))
            idx = np.asarray(positions[:n], dtype=int)
            mask = selected[idx]
            if not mask.any():
                continue
            samples.append(float(correct[idx][mask].mean()))
        if len(samples) < self._config.bootstrap_resamples * 0.90:
            raise ValueError("Too many bootstrap samples contained no selected signals.")
        arr = np.asarray(samples, dtype=float)
        return SelectiveBootstrapResult(
            directional_accuracy=observed,
            ci95_low=float(np.quantile(arr, 0.025)),
            ci95_high=float(np.quantile(arr, 0.975)),
            probability_above_50=float((arr > 0.50).mean()),
            coverage=selected_rows / n,
            selected_rows=selected_rows,
            total_rows=n,
            block_size_rows=block,
            resamples=len(arr),
        )

    @staticmethod
    def _wilson_lower(accuracy: float, n: int, z: float = 1.959963984540054) -> float:
        if n <= 0:
            return 0.0
        denominator = 1.0 + z * z / n
        centre = accuracy + z * z / (2.0 * n)
        margin = z * math.sqrt(
            accuracy * (1.0 - accuracy) / n + z * z / (4.0 * n * n)
        )
        return (centre - margin) / denominator
