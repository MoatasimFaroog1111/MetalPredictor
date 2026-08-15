from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metal_predictor.multi_horizon.stage6_preregistration import Stage6CandidateSpec


class TrainMedianReturnRegressor:
    """Train-only absolute-error-optimal constant return estimator."""

    def __init__(self) -> None:
        self._median: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TrainMedianReturnRegressor":
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if x_array.ndim != 2 or x_array.shape[0] != y_array.size:
            raise ValueError("TrainMedianReturnRegressor requires aligned 2-D X and 1-D y.")
        if y_array.size == 0 or not np.isfinite(y_array).all():
            raise ValueError("TrainMedianReturnRegressor requires finite non-empty targets.")
        self._median = float(np.median(y_array))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._median is None:
            raise RuntimeError("TrainMedianReturnRegressor must be fitted before predict().")
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 2:
            raise ValueError("TrainMedianReturnRegressor requires a 2-D feature matrix.")
        return np.full(x_array.shape[0], self._median, dtype=float)


@dataclass(frozen=True)
class Stage6DevelopmentModelFactory:
    """Construct only estimators frozen by the Stage-6 preregistration."""

    def create(self, spec: Stage6CandidateSpec):
        params = dict(spec.parameters)
        if spec.estimator.endswith("TrainMedianReturnRegressor"):
            if params or spec.preprocessing:
                raise ValueError("Train median candidate must not have parameters or preprocessing.")
            return TrainMedianReturnRegressor()

        if spec.estimator == "sklearn.linear_model.Ridge":
            if spec.preprocessing != ("StandardScaler(train_only)",):
                raise ValueError("Stage-6 Ridge must use train-only StandardScaler.")
            return Pipeline(
                steps=(
                    ("scale", StandardScaler()),
                    ("regressor", Ridge(**params)),
                )
            )

        if spec.preprocessing:
            raise ValueError(
                f"Unexpected preprocessing for {spec.candidate_id!r}: {spec.preprocessing!r}."
            )
        if spec.estimator == "sklearn.ensemble.RandomForestRegressor":
            return RandomForestRegressor(**params)
        if spec.estimator == "sklearn.ensemble.HistGradientBoostingRegressor":
            return HistGradientBoostingRegressor(**params)
        raise ValueError(f"Unregistered Stage-6 estimator: {spec.estimator!r}.")
