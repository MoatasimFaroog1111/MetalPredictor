from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metal_predictor.metrics import PairedBlockBootstrapComparison, RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


@dataclass(frozen=True)
class StackingConfig:
    inner_splits: int = 3
    inner_initial_train_fraction: float = 0.55
    inner_min_train_rows: int = 3500
    meta_alpha: float = 1.0
    bootstrap_block_rows: int = 24
    bootstrap_resamples: int = 5000

    def __post_init__(self) -> None:
        if self.inner_splits < 3 or self.inner_min_train_rows < 1000:
            raise ValueError("OOF stacking requires at least three inner folds and sufficient history.")
        if self.meta_alpha <= 0:
            raise ValueError("meta_alpha must be positive.")


@dataclass(frozen=True)
class StackFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    meta_oof_rows: int
    baseline_mae: float
    stacked_mae: float
    mae_improvement: float
    baseline_directional_accuracy: float
    stacked_directional_accuracy: float
    meta_coefficients: dict[str, float]
    meta_intercept: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class NestedOOFStackingEvaluator:
    """Leakage-safe stacking: meta learner sees only inner out-of-fold base predictions."""

    def __init__(
        self,
        config: StackingConfig,
        outer_splitter: PurgedWalkForwardSplitter,
        feature_names: tuple[str, ...],
        baseline_spec: ModelSpec,
        base_specs: tuple[ModelSpec, ...],
        metrics: RegressionForecastMetrics | None = None,
        target_name: str = "target_log_return_1h",
        current_close_name: str = "close_usd_per_kg",
        target_close_name: str = "target_close_usd_per_kg",
        timestamp_name: str = "timestamp_utc",
    ) -> None:
        if len(base_specs) < 2:
            raise ValueError("Stacking requires at least two base models.")
        names = [spec.name for spec in base_specs]
        if len(set(names)) != len(names):
            raise ValueError("Base model names must be unique.")
        if any(spec.history_days is not None for spec in base_specs):
            raise ValueError("Stacking base models must share the same outer training history.")
        self._config = config
        self._outer = outer_splitter
        self._features = feature_names
        self._baseline = baseline_spec
        self._bases = base_specs
        self._metrics = metrics or RegressionForecastMetrics()
        self._target = target_name
        self._current = current_close_name
        self._target_close = target_close_name
        self._timestamp = timestamp_name

    def evaluate(self, development: pd.DataFrame) -> dict[str, object]:
        fold_results: list[StackFoldResult] = []
        pooled: list[pd.DataFrame] = []

        for outer in self._outer.split(development):
            meta_X, meta_y = self._inner_oof_matrix(outer.train)
            meta = self._new_meta_model()
            meta.fit(meta_X, meta_y)

            outer_base_predictions = self._fit_base_models(
                outer.train, outer.validation
            )
            stack_matrix = np.column_stack([
                outer_base_predictions[spec.name] for spec in self._bases
            ])
            stacked_prediction = np.asarray(meta.predict(stack_matrix), dtype=float)

            baseline = self._baseline.factory()
            baseline.fit(outer.train.loc[:, self._features], outer.train[self._target])
            baseline_prediction = np.asarray(
                baseline.predict(outer.validation.loc[:, self._features]), dtype=float
            )
            if not np.isfinite(stacked_prediction).all():
                raise ValueError("Stacked model produced non-finite predictions.")

            actual = outer.validation[self._target].to_numpy(float)
            base_metrics = self._metrics.calculate(
                actual,
                baseline_prediction,
                outer.validation[self._current],
                outer.validation[self._target_close],
            )
            stack_metrics = self._metrics.calculate(
                actual,
                stacked_prediction,
                outer.validation[self._current],
                outer.validation[self._target_close],
            )
            ridge = meta.named_steps["ridge"]
            scaler = meta.named_steps["scaler"]
            # Coefficients below are expressed in standardized meta-input space.
            coefficients = {
                spec.name: float(value)
                for spec, value in zip(self._bases, ridge.coef_, strict=True)
            }
            fold_results.append(StackFoldResult(
                fold=outer.number,
                train_rows=len(outer.train),
                validation_rows=len(outer.validation),
                meta_oof_rows=len(meta_X),
                baseline_mae=base_metrics.mae_return,
                stacked_mae=stack_metrics.mae_return,
                mae_improvement=base_metrics.mae_return - stack_metrics.mae_return,
                baseline_directional_accuracy=base_metrics.directional_accuracy,
                stacked_directional_accuracy=stack_metrics.directional_accuracy,
                meta_coefficients=coefficients,
                meta_intercept=float(ridge.intercept_),
            ))
            pooled.append(pd.DataFrame({
                "timestamp_utc": outer.validation[self._timestamp].to_numpy(),
                "actual": actual,
                "baseline_prediction": baseline_prediction,
                "stacked_prediction": stacked_prediction,
                **{
                    f"base_{name}": values
                    for name, values in outer_base_predictions.items()
                },
                "fold": outer.number,
            }))

        oof = pd.concat(pooled, ignore_index=True).sort_values(
            "timestamp_utc"
        ).reset_index(drop=True)
        bootstrap = PairedBlockBootstrapComparison(
            block_size_rows=self._config.bootstrap_block_rows,
            resamples=self._config.bootstrap_resamples,
            random_state=42,
        ).compare(oof["actual"], oof["stacked_prediction"], oof["baseline_prediction"])
        better_folds = sum(row.stacked_mae < row.baseline_mae for row in fold_results)
        strong = (
            better_folds >= 4
            and bootstrap.mae_improvement_vs_baseline > 0
            and bootstrap.improvement_ci95_low > 0
            and bootstrap.probability_selected_better >= 0.95
        )
        promising = (
            not strong
            and better_folds >= 3
            and bootstrap.mae_improvement_vs_baseline > 0
            and bootstrap.probability_selected_better >= 0.80
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
                "base_models": [spec.name for spec in self._bases],
                "meta_model": f"StandardScaler + Ridge(alpha={self._config.meta_alpha})",
                "meta_training_data": "inner purged OOF predictions only",
                "outer_validation_used_for_meta_fit": False,
                "base_models_refit_on_full_outer_train": True,
            },
            "outer_walk_forward": {
                "folds": len(fold_results),
                "stack_better_folds": better_folds,
                "fold_results": [row.as_dict() for row in fold_results],
            },
            "paired_oof_mae": bootstrap.as_dict(),
            "decision": {
                "promote_oof_stack": strong or promising,
                "evidence_level": evidence,
                "rule_fixed_before_result": True,
                "requires_future_holdout_confirmation": strong or promising,
            },
            "oof_predictions": oof,
        }

    def _inner_oof_matrix(self, outer_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        inner = PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=self._config.inner_splits,
            initial_train_fraction=self._config.inner_initial_train_fraction,
            min_train_rows=min(
                self._config.inner_min_train_rows,
                max(1200, len(outer_train) // 3),
            ),
        ))
        matrices: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for fold in inner.split(outer_train):
            predictions = self._fit_base_models(fold.train, fold.validation)
            matrices.append(np.column_stack([
                predictions[spec.name] for spec in self._bases
            ]))
            targets.append(fold.validation[self._target].to_numpy(float))
        X = np.vstack(matrices)
        y = np.concatenate(targets)
        if len(X) < 500 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Insufficient or invalid inner OOF data for stacking.")
        return X, y

    def _fit_base_models(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for spec in self._bases:
            model = spec.factory()
            model.fit(train.loc[:, self._features], train[self._target])
            prediction = np.asarray(
                model.predict(validation.loc[:, self._features]), dtype=float
            )
            if not np.isfinite(prediction).all():
                raise ValueError(f"Base model {spec.name} produced non-finite predictions.")
            result[spec.name] = prediction
        return result

    def _new_meta_model(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self._config.meta_alpha)),
        ])
