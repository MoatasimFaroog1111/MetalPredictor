from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig


def _exact_lag(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    keyed = pd.Series(series.to_numpy(dtype=float), index=idx)
    wanted = idx - pd.Timedelta(hours=hours)
    return pd.Series(keyed.reindex(wanted).to_numpy(), index=series.index, dtype=float)


@dataclass(frozen=True)
class PreciousMetalSpec:
    key: str
    label: str


class PlatinumPalladiumCrossAssetFeatures:
    """Causal XPT/XPD features aligned to Silver only at identical UTC H1 timestamps.

    This component is research-only. It does not mutate the frozen Silver feature graph,
    model artifacts, or holdout. Missing exact timestamps remain missing; no forward fill
    or interpolation is performed.
    """

    _METALS = (
        PreciousMetalSpec("platinum", "XPT"),
        PreciousMetalSpec("palladium", "XPD"),
    )

    def __init__(
        self,
        platinum_frame: pd.DataFrame,
        palladium_frame: pd.DataFrame,
        aligner: ExactTimestampAligner,
        silver_columns: ColumnConfig,
        lags: tuple[int, ...] = (1, 3, 6, 12, 24),
        windows: tuple[int, ...] = (24, 72),
    ) -> None:
        self._frames = {
            "platinum": self._validate_frame(platinum_frame, "Platinum"),
            "palladium": self._validate_frame(palladium_frame, "Palladium"),
        }
        self._aligner = aligner
        self._c = silver_columns
        self._lags = lags
        self._windows = windows
        self._names = self._build_feature_names()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        silver_close = pd.to_numeric(out[self._c.close], errors="coerce").astype(float)
        silver_log = np.log(silver_close)

        metal_returns: dict[tuple[str, int], pd.Series] = {}
        metal_log_closes: dict[str, pd.Series] = {}

        for spec in self._METALS:
            table = self._build_metal_table(spec)
            aligned_columns = tuple(column for column in table.columns if column != "timestamp_utc")
            aligned = self._aligner.align(ts, table, aligned_columns)
            for column in aligned.columns:
                out[column] = aligned[column]

            metal_close = pd.to_numeric(
                out.pop(f"{spec.key}_close_usd_per_kg_internal"), errors="coerce"
            )
            metal_log = np.log(metal_close)
            metal_log_closes[spec.key] = metal_log
            out[f"{spec.key}_has_exact_current"] = metal_close.notna().astype("int8")
            out[f"log_{spec.key}_silver_ratio"] = metal_log - silver_log

            for lag in self._lags:
                availability = f"{spec.key}_has_exact_{lag}h"
                out[availability] = pd.to_numeric(
                    out[availability], errors="coerce"
                ).fillna(0).astype("int8")

                silver_prior = _exact_lag(silver_log, ts, lag)
                silver_return = silver_log - silver_prior
                metal_return = pd.to_numeric(out[f"{spec.key}_log_return_{lag}h"], errors="coerce")
                metal_returns[(spec.key, lag)] = metal_return
                out[f"{spec.key}_silver_relative_return_{lag}h"] = silver_return - metal_return

                ratio = out[f"log_{spec.key}_silver_ratio"]
                prior_ratio = _exact_lag(ratio, ts, lag)
                ratio_available = ratio.notna() & prior_ratio.notna()
                out[f"{spec.key}_silver_ratio_has_exact_{lag}h"] = ratio_available.astype("int8")
                out[f"{spec.key}_silver_log_ratio_change_{lag}h"] = ratio - prior_ratio

            silver_prior_1h = _exact_lag(silver_log, ts, 1)
            silver_return_1h = silver_log - silver_prior_1h
            metal_return_1h = pd.to_numeric(out[f"{spec.key}_log_return_1h"], errors="coerce")
            idx = pd.DatetimeIndex(ts)
            silver_keyed = pd.Series(silver_return_1h.to_numpy(float), index=idx)
            metal_keyed = pd.Series(metal_return_1h.to_numpy(float), index=idx)
            for window in self._windows:
                min_periods = max(4, int(np.ceil(window * 0.5)))
                corr = silver_keyed.rolling(f"{window}h", min_periods=min_periods).corr(metal_keyed)
                out[f"silver_{spec.key}_corr_{window}h"] = corr.to_numpy()

        platinum_log = metal_log_closes["platinum"]
        palladium_log = metal_log_closes["palladium"]
        pt_pd_ratio = platinum_log - palladium_log
        out["log_platinum_palladium_ratio"] = pt_pd_ratio

        for lag in self._lags:
            silver_prior = _exact_lag(silver_log, ts, lag)
            silver_return = silver_log - silver_prior
            pt_return = metal_returns[("platinum", lag)]
            pd_return = metal_returns[("palladium", lag)]
            returns = pd.concat([silver_return, pt_return, pd_return], axis=1)
            out[f"precious_metals_return_dispersion_{lag}h"] = returns.std(axis=1, ddof=0)
            out[f"precious_metals_return_breadth_{lag}h"] = (
                returns.gt(0).sum(axis=1) / returns.notna().sum(axis=1).replace(0, np.nan)
            )
            prior_ratio = _exact_lag(pt_pd_ratio, ts, lag)
            out[f"platinum_palladium_log_ratio_change_{lag}h"] = pt_pd_ratio - prior_ratio

        return out

    def _build_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for spec in self._METALS:
            names.extend([
                f"{spec.key}_has_exact_current",
                f"{spec.key}_is_partial_source_hour",
                f"{spec.key}_candle_range_pct",
                f"{spec.key}_candle_body_pct",
                f"log_{spec.key}_silver_ratio",
            ])
            for lag in self._lags:
                names.extend([
                    f"{spec.key}_has_exact_{lag}h",
                    f"{spec.key}_log_return_{lag}h",
                    f"{spec.key}_silver_relative_return_{lag}h",
                    f"{spec.key}_silver_ratio_has_exact_{lag}h",
                    f"{spec.key}_silver_log_ratio_change_{lag}h",
                ])
            for window in self._windows:
                names.extend([
                    f"{spec.key}_realized_vol_{window}h",
                    f"silver_{spec.key}_corr_{window}h",
                ])
        names.append("log_platinum_palladium_ratio")
        for lag in self._lags:
            names.extend([
                f"precious_metals_return_dispersion_{lag}h",
                f"precious_metals_return_breadth_{lag}h",
                f"platinum_palladium_log_ratio_change_{lag}h",
            ])
        return tuple(names)

    def _build_metal_table(self, spec: PreciousMetalSpec) -> pd.DataFrame:
        metal = self._frames[spec.key].copy(deep=True)
        ts = pd.to_datetime(metal["timestamp_utc"], utc=True)
        close = metal["close_usd_per_kg"].astype(float)
        log_close = np.log(close)
        open_price = metal["open_usd_per_kg"].astype(float)
        high = metal["high_usd_per_kg"].astype(float)
        low = metal["low_usd_per_kg"].astype(float)
        safe_open = open_price.replace(0.0, np.nan)

        result = pd.DataFrame({
            "timestamp_utc": ts,
            f"{spec.key}_close_usd_per_kg_internal": close,
            f"{spec.key}_is_partial_source_hour": metal["quality_flag"].eq(
                "PARTIAL_SOURCE_HOUR"
            ).astype("int8"),
            f"{spec.key}_candle_range_pct": (high - low) / safe_open,
            f"{spec.key}_candle_body_pct": (close - open_price) / safe_open,
        })

        for lag in self._lags:
            prior = _exact_lag(log_close, ts, lag)
            result[f"{spec.key}_has_exact_{lag}h"] = prior.notna().astype("int8")
            result[f"{spec.key}_log_return_{lag}h"] = log_close - prior

        prior_1h = _exact_lag(log_close, ts, 1)
        return_1h = log_close - prior_1h
        keyed = pd.Series(return_1h.to_numpy(float), index=pd.DatetimeIndex(ts))
        for window in self._windows:
            min_periods = max(4, int(np.ceil(window * 0.5)))
            result[f"{spec.key}_realized_vol_{window}h"] = (
                keyed.rolling(f"{window}h", min_periods=min_periods).std(ddof=0).to_numpy()
            )
        return result

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        required = {
            "timestamp_utc",
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
            "quality_flag",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():
            raise ValueError(f"{label} timestamps must be unique.")
        price_columns = [
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
        ]
        prices = out[price_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
            raise ValueError(f"{label} frame contains invalid prices.")
        out[price_columns] = prices
        return out
