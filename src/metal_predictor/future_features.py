from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.core import PipelineConfig
from metal_predictor.data import SilverDatasetValidator
from metal_predictor.features import (
    MomentumFeatures,
    PriceActionFeatures,
    QualityFeatures,
    TemporalFeatures,
    TrendFeatures,
    VolatilityFeatures,
)


class SilverFeatureAssembler:
    """Replays the exact frozen Silver-only feature graph without constructing labels."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        c = self._config.columns
        f = self._config.features
        self._components = (
            PriceActionFeatures(c),
            MomentumFeatures(c, f),
            VolatilityFeatures(c, f),
            TrendFeatures(c, f),
            TemporalFeatures(c),
            QualityFeatures(c),
        )
        self._validator = SilverDatasetValidator(c)
        self.feature_names = tuple(
            name
            for component in self._components
            for name in component.feature_names
        )

    def transform(self, hourly: pd.DataFrame) -> pd.DataFrame:
        c = self._config.columns
        required = {c.timestamp, c.open, c.high, c.low, c.close}
        missing = required.difference(hourly.columns)
        if missing:
            raise ValueError(f"Future feature input missing columns: {sorted(missing)}")
        out = hourly.copy(deep=True)
        out[c.timestamp] = pd.to_datetime(out[c.timestamp], utc=True, errors="raise")
        out = out.sort_values(c.timestamp).reset_index(drop=True)
        for name in (c.open, c.high, c.low, c.close):
            out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
        self._validator.validate(out)
        for component in self._components:
            out = component.transform(out)
        values = out.loc[:, self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("Future feature graph produced infinite values.")
        return out
