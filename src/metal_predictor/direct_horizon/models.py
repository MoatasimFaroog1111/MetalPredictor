from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metal_predictor.direct_horizon.preregistration import Stage7CandidateSpec


@dataclass(frozen=True)
class Stage7ModelFactory:
    """Construct only the estimator recipes frozen by Stage-7 preregistration."""

    def create(self, spec: Stage7CandidateSpec):
        params = dict(spec.parameters)
        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )
        if spec.estimator == "sklearn.linear_model.Ridge":
            return Pipeline(
                steps=(
                    ("imputer", imputer),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(**params)),
                )
            )
        if spec.estimator == "sklearn.ensemble.HistGradientBoostingRegressor":
            return Pipeline(
                steps=(
                    ("imputer", imputer),
                    ("regressor", HistGradientBoostingRegressor(**params)),
                )
            )
        if spec.estimator == "sklearn.ensemble.ExtraTreesRegressor":
            return Pipeline(
                steps=(
                    ("imputer", imputer),
                    ("regressor", ExtraTreesRegressor(**params)),
                )
            )
        raise ValueError(f"Unregistered Stage-7 estimator: {spec.estimator!r}.")
