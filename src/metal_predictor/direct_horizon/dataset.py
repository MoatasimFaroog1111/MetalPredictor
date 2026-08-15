from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metal_predictor.core import ColumnConfig, FeatureConfig
from metal_predictor.data import ParquetDataLoader, SilverDatasetValidator
from metal_predictor.features import (
    MomentumFeatures,
    PriceActionFeatures,
    QualityFeatures,
    TemporalFeatures,
    TrendFeatures,
    VolatilityFeatures,
)
from metal_predictor.direct_horizon.preregistration import (
    STAGE7_FEATURE_GRAPH_VERSION,
    STAGE7_SOURCE_PATH,
    Stage7HorizonSpec,
)


@dataclass(frozen=True)
class Stage7HorizonDataset:
    horizon: Stage7HorizonSpec
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_name: str
    target_close_name: str
    target_timestamp_name: str = "target_timestamp_utc"
    timestamp_name: str = "timestamp_utc"
    current_close_name: str = "close_usd_per_kg"
    feature_graph_version: str = STAGE7_FEATURE_GRAPH_VERSION


class Stage7ExactClockTargetBuilder:
    """Build a direct future target only when t+h exists exactly on the H1 clock."""

    def build(self, frame: pd.DataFrame, horizon: Stage7HorizonSpec) -> tuple[pd.DataFrame, str, str]:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        close = pd.to_numeric(out["close_usd_per_kg"], errors="raise").astype(float)
        idx = pd.DatetimeIndex(ts)
        keyed_close = pd.Series(close.to_numpy(float), index=idx)
        wanted = idx + pd.Timedelta(hours=horizon.hours)
        future_close = keyed_close.reindex(wanted).to_numpy(dtype=float)
        exact = np.isfinite(future_close)

        target_name = f"target_log_return_{horizon.hours}h"
        target_close_name = f"target_close_{horizon.hours}h_usd_per_kg"
        out["target_timestamp_utc"] = pd.Series(
            pd.DatetimeIndex(wanted).where(exact), index=out.index
        )
        out[target_close_name] = np.where(exact, future_close, np.nan)
        out[target_name] = np.where(exact, np.log(future_close / close.to_numpy(float)), np.nan)
        return out, target_name, target_close_name


class Stage7CausalFeatureBuilder:
    """Reuse the canonical 52 causal H1 feature components without mutating them."""

    def __init__(self, columns: ColumnConfig | None = None, config: FeatureConfig | None = None) -> None:
        self._columns = columns or ColumnConfig()
        self._config = config or FeatureConfig()
        c, f = self._columns, self._config
        self._components = (
            PriceActionFeatures(c),
            MomentumFeatures(c, f),
            VolatilityFeatures(c, f),
            TrendFeatures(c, f),
            TemporalFeatures(c),
            QualityFeatures(c),
        )

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
        out = frame.copy(deep=True)
        names: list[str] = []
        for component in self._components:
            out = component.transform(out)
            names.extend(component.feature_names)
        feature_names = tuple(names)
        if len(feature_names) != 52 or len(set(feature_names)) != 52:
            raise ValueError("Stage-7 canonical feature graph must contain exactly 52 unique features.")
        return out, feature_names


class Stage7DatasetBuilder:
    def __init__(
        self,
        *,
        repo_root: Path = Path("."),
        loader: ParquetDataLoader | None = None,
        validator: SilverDatasetValidator | None = None,
        features: Stage7CausalFeatureBuilder | None = None,
        targets: Stage7ExactClockTargetBuilder | None = None,
    ) -> None:
        self._root = Path(repo_root)
        self._columns = ColumnConfig()
        self._loader = loader or ParquetDataLoader()
        self._validator = validator or SilverDatasetValidator(self._columns)
        self._features = features or Stage7CausalFeatureBuilder(self._columns, FeatureConfig())
        self._targets = targets or Stage7ExactClockTargetBuilder()

    def build(self, horizon: Stage7HorizonSpec) -> Stage7HorizonDataset:
        raw = self._loader.load(self._root / STAGE7_SOURCE_PATH)
        raw = self._normalize(raw)
        self._validator.validate(raw)
        featured, feature_names = self._features.transform(raw)
        targeted, target_name, target_close_name = self._targets.build(featured, horizon)
        usable = targeted.dropna(subset=[target_name, target_close_name, "target_timestamp_utc"]).copy()
        matrix = usable.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        inf_rows = np.isinf(matrix.to_numpy(dtype=float)).any(axis=1)
        usable = usable.loc[~inf_rows].copy()
        usable = usable.sort_values("timestamp_utc").reset_index(drop=True)
        if usable.empty:
            raise ValueError(f"No exact-clock rows available for horizon {horizon.key}.")
        delta = pd.to_datetime(usable["target_timestamp_utc"], utc=True) - pd.to_datetime(
            usable["timestamp_utc"], utc=True
        )
        if not delta.eq(pd.Timedelta(hours=horizon.hours)).all():
            raise ValueError(f"Stage-7 {horizon.key} target alignment failed.")
        return Stage7HorizonDataset(
            horizon=horizon,
            frame=usable,
            feature_names=feature_names,
            target_name=target_name,
            target_close_name=target_close_name,
        )

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        c = self._columns
        out = frame.copy(deep=True)
        out[c.timestamp] = pd.to_datetime(out[c.timestamp], utc=True, errors="raise")
        out = out.sort_values(c.timestamp).reset_index(drop=True)
        for name in (c.open, c.high, c.low, c.close):
            out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
        return out
