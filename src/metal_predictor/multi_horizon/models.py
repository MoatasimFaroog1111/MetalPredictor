from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metal_predictor.multi_horizon.preregistration import CandidateModelSpec


class ReturnRegressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ReturnRegressor": ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DevelopmentModelFactory:
    """Build only the estimators locked by the Stage-2 preregistration."""

    def create(self, spec: CandidateModelSpec) -> Pipeline:
        estimator: RegressorMixin
        params = dict(spec.parameters)
        if spec.estimator == "sklearn.linear_model.Ridge":
            estimator = Ridge(**params)
        elif spec.estimator == "sklearn.linear_model.HuberRegressor":
            estimator = HuberRegressor(**params)
        elif spec.estimator == "sklearn.linear_model.ElasticNet":
            estimator = ElasticNet(**params)
        else:
            raise ValueError(f"Unregistered Stage-3 estimator: {spec.estimator!r}.")

        if spec.preprocessing != ("StandardScaler(train_only)",):
            raise ValueError(
                f"Unexpected preprocessing contract for {spec.candidate_id!r}: "
                f"{spec.preprocessing!r}."
            )
        return Pipeline(
            steps=(
                ("scale", StandardScaler()),
                ("regressor", estimator),
            )
        )


def random_walk_zero_return(row_count: int) -> np.ndarray:
    if row_count < 0:
        raise ValueError("row_count must be non-negative.")
    return np.zeros(int(row_count), dtype=float)
