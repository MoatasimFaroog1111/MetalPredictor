from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metal_predictor.cpcv import CombinatorialPurgedSplitter
from metal_predictor.metrics import RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import PurgedWalkForwardSplitter, WalkForwardConfig


@dataclass(frozen=True)
class CPCVStackingConfig:
    inner_splits: int = 3
    inner_initial_train_fraction: float = 0.55
    inner_min_train_rows: int = 3000
    meta_alpha: float = 1.0


@dataclass(frozen=True)
class CPCVStackSplitResult:
    split: int
    test_group_ids: tuple[int, ...]
    train_rows: int
    test_rows: int
    purged_train_rows: int
    embargoed_train_rows: int
    meta_oof_rows: int
    baseline_mae: float
    stacked_mae: float
    mae_improvement: float
    baseline_directional_accuracy: float
    stacked_directional_accuracy: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CPCVStackingEvaluator:
    """CPCV robustness diagnostic for the frozen baseline versus the OOF-stacking algorithm."""

    def __init__(
        self,
        config: CPCVStackingConfig,
        splitter: CombinatorialPurgedSplitter,
        feature_names: tuple[str, ...],
        baseline_spec: ModelSpec,
        base_specs: tuple[ModelSpec, ...],
        metrics: RegressionForecastMetrics | None = None,
        target_name: str = "target_log_return_1h",
        current_close_name: str = "close_usd_per_kg",
        target_close_name: str = "target_close_usd_per_kg",
    ) -> None:
        if len(base_specs) < 2:
            raise ValueError("CPCV stacking needs at least two base models.")
        if any(spec.history_days is not None for spec in base_specs):
            raise ValueError("CPCV stack base models must use the full supplied training partition.")
        self._config = config
        self._splitter = splitter
        self._features = feature_names
        self._baseline = baseline_spec
        self._bases = base_specs
        self._metrics = metrics or RegressionForecastMetrics()
        self._target = target_name
        self._current = current_close_name
        self._target_close = target_close_name

    def evaluate(self, development: pd.DataFrame) -> dict[str, object]:
        rows: list[CPCVStackSplitResult] = []
        for split in self._splitter.split(development):
            meta_X, meta_y = self._inner_oof_matrix(split.train)
            meta = self._new_meta_model()
            meta.fit(meta_X, meta_y)

            outer_base = self._fit_base_models(split.train, split.test)
            stack_matrix = np.column_stack([
                outer_base[spec.name] for spec in self._bases
            ])
            stacked_prediction = np.asarray(meta.predict(stack_matrix), dtype=float)

            baseline = self._baseline.factory()
            baseline.fit(split.train.loc[:, self._features], split.train[self._target])
            baseline_prediction = np.asarray(
                baseline.predict(split.test.loc[:, self._features]), dtype=float
            )
            if not np.isfinite(stacked_prediction).all() or not np.isfinite(baseline_prediction).all():
                raise ValueError("CPCV produced non-finite predictions.")

            actual = split.test[self._target].to_numpy(float)
            baseline_metrics = self._metrics.calculate(
                actual,
                baseline_prediction,
                split.test[self._current],
                split.test[self._target_close],
            )
            stack_metrics = self._metrics.calculate(
                actual,
                stacked_prediction,
                split.test[self._current],
                split.test[self._target_close],
            )
            rows.append(CPCVStackSplitResult(
                split=split.number,
                test_group_ids=split.test_group_ids,
                train_rows=len(split.train),
                test_rows=len(split.test),
                purged_train_rows=split.purged_train_rows,
                embargoed_train_rows=split.embargoed_train_rows,
                meta_oof_rows=len(meta_X),
                baseline_mae=baseline_metrics.mae_return,
                stacked_mae=stack_metrics.mae_return,
                mae_improvement=baseline_metrics.mae_return - stack_metrics.mae_return,
                baseline_directional_accuracy=baseline_metrics.directional_accuracy,
                stacked_directional_accuracy=stack_metrics.directional_accuracy,
            ))

        improvements = np.asarray([row.mae_improvement for row in rows], dtype=float)
        better = int((improvements > 0).sum())
        return {
            "status": "PASS",
            "diagnostic_scope": (
                "Combinatorial purged robustness diagnostic. Training complements can include "
                "data after an earlier held-out group; live deployment chronology remains the "
                "responsibility of PurgedWalkForwardSplitter."
            ),
            "split_count": len(rows),
            "stack_better_splits": better,
            "stack_better_fraction": better / len(rows),
            "mean_mae_improvement": float(improvements.mean()),
            "median_mae_improvement": float(np.median(improvements)),
            "improvement_quantile_05": float(np.quantile(improvements, 0.05)),
            "improvement_quantile_95": float(np.quantile(improvements, 0.95)),
            "split_results": [row.as_dict() for row in rows],
        }

    def _inner_oof_matrix(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        inner = PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=self._config.inner_splits,
            initial_train_fraction=self._config.inner_initial_train_fraction,
            min_train_rows=min(
                self._config.inner_min_train_rows,
                max(1200, len(train) // 3),
            ),
        ))
        matrices: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for fold in inner.split(train):
            predictions = self._fit_base_models(fold.train, fold.validation)
            matrices.append(np.column_stack([
                predictions[spec.name] for spec in self._bases
            ]))
            targets.append(fold.validation[self._target].to_numpy(float))
        X = np.vstack(matrices)
        y = np.concatenate(targets)
        if len(X) < 500 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("CPCV stack inner OOF matrix is insufficient or invalid.")
        return X, y

    def _fit_base_models(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for spec in self._bases:
            model = spec.factory()
            model.fit(train.loc[:, self._features], train[self._target])
            prediction = np.asarray(model.predict(test.loc[:, self._features]), dtype=float)
            if not np.isfinite(prediction).all():
                raise ValueError(f"CPCV base model {spec.name} produced invalid predictions.")
            output[spec.name] = prediction
        return output

    def _new_meta_model(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self._config.meta_alpha)),
        ])
