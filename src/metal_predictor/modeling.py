from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series): ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelSpec:
    """Immutable recipe for one model configuration."""

    name: str
    family: str
    factory: Callable[[], Regressor]
    history_days: int | None = None
    description: str = ""


class ZeroReturnRegressor(BaseEstimator, RegressorMixin):
    """Strong financial baseline: forecast no next-hour price change."""

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class PreviousReturnRegressor(BaseEstimator, RegressorMixin):
    """Momentum baseline: repeat the latest exact one-hour log return."""

    def __init__(self, feature_name: str = "log_return_1h") -> None:
        self.feature_name = feature_name

    def fit(self, X, y):
        if self.feature_name not in X.columns:
            raise ValueError(f"Required baseline feature missing: {self.feature_name}")
        return self

    def predict(self, X):
        values = pd.to_numeric(X[self.feature_name], errors="coerce")
        return values.fillna(0.0).to_numpy(dtype=float)


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def _extra_trees(*, min_samples_leaf: int, max_features: float, max_depth: int | None, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("regressor", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=2,
        )),
    ])


def _hist_gb(*, loss: str, learning_rate: float, max_iter: int, max_leaf_nodes: int,
             min_samples_leaf: int, l2_regularization: float, random_state: int):
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=random_state,
    )


def _xgboost(**kwargs):
    from xgboost import XGBRegressor
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="mae",
        random_state=42,
        n_jobs=2,
        verbosity=0,
        **kwargs,
    )


def _lightgbm(**kwargs):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=42, n_jobs=2, verbosity=-1, **kwargs)


class DefaultModelRegistry:
    """Constructs deterministic candidates only; no splitting or selection logic."""

    def __init__(self, random_state: int = 42) -> None:
        self._seed = random_state

    def candidates(self) -> tuple[ModelSpec, ...]:
        seed = self._seed
        return (
            ModelSpec("zero_return", "baseline_zero", lambda: ZeroReturnRegressor(), description="No-change baseline."),
            ModelSpec("previous_return", "baseline_previous_return", lambda: PreviousReturnRegressor(), description="Repeat latest 1h return."),
            ModelSpec("ridge_alpha_0_1", "ridge", lambda: _ridge(0.1)),
            ModelSpec("ridge_alpha_10", "ridge", lambda: _ridge(10.0)),
            ModelSpec("ridge_alpha_100", "ridge", lambda: _ridge(100.0)),
            ModelSpec("histgb_squared_small", "hist_gradient_boosting", lambda: _hist_gb(
                loss="squared_error", learning_rate=0.035, max_iter=500, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=0.01, random_state=seed)),
            ModelSpec("histgb_absolute_regularized", "hist_gradient_boosting", lambda: _hist_gb(
                loss="absolute_error", learning_rate=0.04, max_iter=450, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=0.05, random_state=seed)),
            ModelSpec("histgb_squared_recent_730d", "hist_gradient_boosting", lambda: _hist_gb(
                loss="squared_error", learning_rate=0.03, max_iter=650, max_leaf_nodes=31,
                min_samples_leaf=60, l2_regularization=0.1, random_state=seed), history_days=730),
            ModelSpec("extra_trees_regularized", "extra_trees", lambda: _extra_trees(
                min_samples_leaf=12, max_features=0.75, max_depth=None, random_state=seed)),
            ModelSpec("extra_trees_shallow", "extra_trees", lambda: _extra_trees(
                min_samples_leaf=24, max_features=0.65, max_depth=16, random_state=seed)),
            ModelSpec("xgb_depth2", "xgboost", lambda: _xgboost(
                n_estimators=600, learning_rate=0.035, max_depth=2, min_child_weight=12,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.02, reg_lambda=6.0)),
            ModelSpec("xgb_depth3_regularized", "xgboost", lambda: _xgboost(
                n_estimators=750, learning_rate=0.025, max_depth=3, min_child_weight=20,
                subsample=0.9, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=10.0)),
            ModelSpec("xgb_depth4_recent_730d", "xgboost", lambda: _xgboost(
                n_estimators=800, learning_rate=0.025, max_depth=4, min_child_weight=25,
                subsample=0.9, colsample_bytree=0.8, reg_alpha=0.08, reg_lambda=12.0), history_days=730),
            ModelSpec("lgbm_l1_small", "lightgbm", lambda: _lightgbm(
                objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=15,
                max_depth=5, min_child_samples=80, colsample_bytree=0.85, reg_alpha=0.02, reg_lambda=5.0)),
            ModelSpec("lgbm_l2_regularized", "lightgbm", lambda: _lightgbm(
                objective="regression", n_estimators=700, learning_rate=0.025, num_leaves=15,
                max_depth=6, min_child_samples=100, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=8.0)),
            ModelSpec("lgbm_l1_recent_730d", "lightgbm", lambda: _lightgbm(
                objective="regression_l1", n_estimators=750, learning_rate=0.025, num_leaves=31,
                max_depth=6, min_child_samples=100, colsample_bytree=0.8, reg_alpha=0.08, reg_lambda=10.0), history_days=730),
        )
