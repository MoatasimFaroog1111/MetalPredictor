from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from metal_predictor.metrics import PairedBlockBootstrapComparison, RegressionForecastMetrics
from metal_predictor.modeling import ModelSpec
from metal_predictor.walk_forward import Fold, PurgedWalkForwardSplitter, WalkForwardConfig


REGIMES = ("HIGH_VOL", "TREND_UP", "TREND_DOWN", "RANGE")


@dataclass(frozen=True)
class RegimeThresholds:
    high_volatility: float
    strong_trend_abs: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class TrainOnlyRegimeDetector:
    """Deterministic Silver-only regime detector whose thresholds are fit on training history only."""

    def __init__(
        self,
        volatility_feature: str = "realized_vol_24h",
        trend_feature: str = "close_vs_sma_72h",
        high_vol_quantile: float = 0.67,
        trend_quantile: float = 0.67,
    ) -> None:
        if not 0.5 <= high_vol_quantile <= 0.9 or not 0.5 <= trend_quantile <= 0.9:
            raise ValueError("Regime quantiles must be between 0.5 and 0.9.")
        self._vol = volatility_feature
        self._trend = trend_feature
        self._vol_q = high_vol_quantile
        self._trend_q = trend_quantile
        self._thresholds: RegimeThresholds | None = None

    @property
    def required_features(self) -> tuple[str, str]:
        return self._vol, self._trend

    @property
    def thresholds(self) -> RegimeThresholds:
        if self._thresholds is None:
            raise RuntimeError("Regime detector has not been fit.")
        return self._thresholds

    def fit(self, train: pd.DataFrame) -> "TrainOnlyRegimeDetector":
        missing = set(self.required_features).difference(train.columns)
        if missing:
            raise ValueError(f"Regime detector missing features: {sorted(missing)}")
        vol = pd.to_numeric(train[self._vol], errors="coerce").dropna()
        trend = pd.to_numeric(train[self._trend], errors="coerce").abs().dropna()
        if len(vol) < 500 or len(trend) < 500:
            raise ValueError("Insufficient finite training history to fit regime thresholds.")
        high_volatility = float(vol.quantile(self._vol_q))
        strong_trend_abs = float(trend.quantile(self._trend_q))
        if not np.isfinite([high_volatility, strong_trend_abs]).all():
            raise ValueError("Regime thresholds are not finite.")
        self._thresholds = RegimeThresholds(high_volatility, strong_trend_abs)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        t = self.thresholds
        vol = pd.to_numeric(frame[self._vol], errors="coerce")
        trend = pd.to_numeric(frame[self._trend], errors="coerce")
        result = pd.Series("RANGE", index=frame.index, dtype="string")
        high_vol = vol.notna() & vol.ge(t.high_volatility)
        result.loc[high_vol] = "HIGH_VOL"
        remaining = ~high_vol & trend.notna()
        result.loc[remaining & trend.ge(t.strong_trend_abs)] = "TREND_UP"
        result.loc[remaining & trend.le(-t.strong_trend_abs)] = "TREND_DOWN"
        return result


@dataclass(frozen=True)
class RegimeModelChoice:
    regime: str
    model_name: str
    inner_rows: int
    inner_mae: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeFoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    baseline_mae: float
    adaptive_mae: float
    mae_improvement: float
    baseline_directional_accuracy: float
    adaptive_directional_accuracy: float
    thresholds: dict[str, float]
    train_regime_counts: dict[str, int]
    validation_regime_counts: dict[str, int]
    model_choices: tuple[RegimeModelChoice, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["model_choices"] = [choice.as_dict() for choice in self.model_choices]
        return result


@dataclass(frozen=True)
class RegimeAdaptiveConfig:
    inner_splits: int = 3
    inner_initial_train_fraction: float = 0.55
    inner_min_train_rows: int = 3500
    min_regime_train_rows: int = 600
    min_regime_inner_validation_rows: int = 150
    bootstrap_block_rows: int = 24
    bootstrap_resamples: int = 5000


class RegimeAdaptiveEvaluator:
    """Nested model selection for specialized regime models with honest outer evaluation."""

    def __init__(
        self,
        config: RegimeAdaptiveConfig,
        outer_splitter: PurgedWalkForwardSplitter,
        feature_names: tuple[str, ...],
        baseline_spec: ModelSpec,
        specialist_specs: tuple[ModelSpec, ...],
        metrics: RegressionForecastMetrics | None = None,
        target_name: str = "target_log_return_1h",
        current_close_name: str = "close_usd_per_kg",
        target_close_name: str = "target_close_usd_per_kg",
        timestamp_name: str = "timestamp_utc",
    ) -> None:
        if not specialist_specs:
            raise ValueError("At least one specialist model candidate is required.")
        if any(spec.history_days is not None for spec in specialist_specs):
            raise ValueError("Regime specialist candidates must use the same full training history.")
        self._config = config
        self._outer = outer_splitter
        self._features = feature_names
        self._baseline = baseline_spec
        self._specialists = specialist_specs
        self._metrics = metrics or RegressionForecastMetrics()
        self._target = target_name
        self._current_close = current_close_name
        self._target_close = target_close_name
        self._timestamp = timestamp_name

    def evaluate(self, development: pd.DataFrame) -> dict[str, object]:
        folds: list[RegimeFoldResult] = []
        pooled: list[pd.DataFrame] = []
        for outer in self._outer.split(development):
            choices = self._select_specialists(outer.train)
            detector = TrainOnlyRegimeDetector().fit(outer.train)
            train_regime = detector.transform(outer.train)
            validation_regime = detector.transform(outer.validation)

            baseline = self._baseline.factory()
            baseline.fit(outer.train.loc[:, self._features], outer.train[self._target])
            baseline_prediction = np.asarray(
                baseline.predict(outer.validation.loc[:, self._features]), dtype=float
            )
            adaptive_prediction = self._fit_predict_specialists(
                outer,
                train_regime,
                validation_regime,
                choices,
            )

            actual = outer.validation[self._target].to_numpy(float)
            base_metrics = self._metrics.calculate(
                actual,
                baseline_prediction,
                outer.validation[self._current_close],
                outer.validation[self._target_close],
            )
            adaptive_metrics = self._metrics.calculate(
                actual,
                adaptive_prediction,
                outer.validation[self._current_close],
                outer.validation[self._target_close],
            )
            folds.append(RegimeFoldResult(
                fold=outer.number,
                train_rows=len(outer.train),
                validation_rows=len(outer.validation),
                baseline_mae=base_metrics.mae_return,
                adaptive_mae=adaptive_metrics.mae_return,
                mae_improvement=base_metrics.mae_return - adaptive_metrics.mae_return,
                baseline_directional_accuracy=base_metrics.directional_accuracy,
                adaptive_directional_accuracy=adaptive_metrics.directional_accuracy,
                thresholds=detector.thresholds.as_dict(),
                train_regime_counts=self._counts(train_regime),
                validation_regime_counts=self._counts(validation_regime),
                model_choices=tuple(choices[regime] for regime in REGIMES),
            ))
            pooled.append(pd.DataFrame({
                "timestamp_utc": outer.validation[self._timestamp].to_numpy(),
                "actual": actual,
                "baseline_prediction": baseline_prediction,
                "adaptive_prediction": adaptive_prediction,
                "regime": validation_regime.to_numpy(),
                "fold": outer.number,
            }))

        oof = pd.concat(pooled, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
        bootstrap = PairedBlockBootstrapComparison(
            block_size_rows=self._config.bootstrap_block_rows,
            resamples=self._config.bootstrap_resamples,
            random_state=42,
        ).compare(oof["actual"], oof["adaptive_prediction"], oof["baseline_prediction"])
        better_folds = sum(result.adaptive_mae < result.baseline_mae for result in folds)
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
                "regime_threshold_fit_scope": "outer/inner training history only",
                "specialist_model_selection_scope": "nested inner purged Walk-Forward only",
                "outer_validation_used_for_model_selection": False,
                "baseline_model": self._baseline.name,
                "candidate_models": [spec.name for spec in self._specialists],
                "regimes": list(REGIMES),
            },
            "outer_walk_forward": {
                "folds": len(folds),
                "adaptive_better_folds": better_folds,
                "fold_results": [result.as_dict() for result in folds],
            },
            "paired_oof_mae": bootstrap.as_dict(),
            "decision": {
                "promote_regime_adaptive_model": strong or promising,
                "evidence_level": evidence,
                "rule_fixed_before_result": True,
                "requires_future_holdout_confirmation": strong or promising,
            },
            "oof_predictions": oof,
        }

    def _select_specialists(self, outer_train: pd.DataFrame) -> dict[str, RegimeModelChoice]:
        inner = PurgedWalkForwardSplitter(WalkForwardConfig(
            n_splits=self._config.inner_splits,
            initial_train_fraction=self._config.inner_initial_train_fraction,
            min_train_rows=min(
                self._config.inner_min_train_rows,
                max(1200, len(outer_train) // 3),
            ),
        ))
        errors: dict[tuple[str, str], list[np.ndarray]] = {
            (regime, spec.name): []
            for regime in REGIMES
            for spec in self._specialists
        }
        for fold in inner.split(outer_train):
            detector = TrainOnlyRegimeDetector().fit(fold.train)
            train_regime = detector.transform(fold.train)
            val_regime = detector.transform(fold.validation)
            for regime in REGIMES:
                train_mask = train_regime.eq(regime).to_numpy()
                val_mask = val_regime.eq(regime).to_numpy()
                if train_mask.sum() < self._config.min_regime_train_rows or not val_mask.any():
                    continue
                X_train = fold.train.loc[train_mask, self._features]
                y_train = fold.train.loc[train_mask, self._target]
                X_val = fold.validation.loc[val_mask, self._features]
                y_val = fold.validation.loc[val_mask, self._target].to_numpy(float)
                for spec in self._specialists:
                    model = spec.factory()
                    model.fit(X_train, y_train)
                    prediction = np.asarray(model.predict(X_val), dtype=float)
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"{spec.name} produced non-finite regime predictions.")
                    errors[(regime, spec.name)].append(np.abs(y_val - prediction))

        result: dict[str, RegimeModelChoice] = {}
        for regime in REGIMES:
            candidates: list[RegimeModelChoice] = []
            for spec in self._specialists:
                pieces = errors[(regime, spec.name)]
                if not pieces:
                    continue
                combined = np.concatenate(pieces)
                if len(combined) < self._config.min_regime_inner_validation_rows:
                    continue
                candidates.append(RegimeModelChoice(
                    regime=regime,
                    model_name=spec.name,
                    inner_rows=len(combined),
                    inner_mae=float(combined.mean()),
                ))
            if candidates:
                result[regime] = min(candidates, key=lambda choice: (choice.inner_mae, choice.model_name))
            else:
                result[regime] = RegimeModelChoice(
                    regime=regime,
                    model_name=self._baseline.name,
                    inner_rows=0,
                    inner_mae=None,
                )
        return result

    def _fit_predict_specialists(
        self,
        outer: Fold,
        train_regime: pd.Series,
        validation_regime: pd.Series,
        choices: dict[str, RegimeModelChoice],
    ) -> np.ndarray:
        prediction = np.full(len(outer.validation), np.nan, dtype=float)
        spec_by_name = {spec.name: spec for spec in (*self._specialists, self._baseline)}
        global_fallback = self._baseline.factory()
        global_fallback.fit(outer.train.loc[:, self._features], outer.train[self._target])

        for regime in REGIMES:
            val_mask = validation_regime.eq(regime).to_numpy()
            if not val_mask.any():
                continue
            train_mask = train_regime.eq(regime).to_numpy()
            choice = choices[regime]
            if train_mask.sum() < self._config.min_regime_train_rows:
                model = global_fallback
            else:
                spec = spec_by_name[choice.model_name]
                model = spec.factory()
                model.fit(
                    outer.train.loc[train_mask, self._features],
                    outer.train.loc[train_mask, self._target],
                )
            prediction[val_mask] = np.asarray(
                model.predict(outer.validation.loc[val_mask, self._features]), dtype=float
            )
        if not np.isfinite(prediction).all():
            raise ValueError("Regime-adaptive prediction has missing or infinite rows.")
        return prediction

    @staticmethod
    def _counts(regime: pd.Series) -> dict[str, int]:
        counts = regime.value_counts().to_dict()
        return {name: int(counts.get(name, 0)) for name in REGIMES}
