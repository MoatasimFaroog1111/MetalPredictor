from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig


def _exact_lag(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    keyed = pd.Series(series.to_numpy(dtype=float), index=idx)
    wanted = idx - pd.Timedelta(hours=hours)
    return pd.Series(keyed.reindex(wanted).to_numpy(), index=series.index, dtype=float)


class GoldSilverCrossAssetFeatures:
    """Causal XAU/XAG features. Gold is aligned only at identical UTC bar timestamps."""

    def __init__(
        self,
        gold_frame: pd.DataFrame,
        aligner: ExactTimestampAligner,
        silver_columns: ColumnConfig,
        lags: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168),
        windows: tuple[int, ...] = (24, 72, 168),
    ) -> None:
        self._gold = self._validate_gold(gold_frame)
        self._aligner = aligner
        self._c = silver_columns
        self._lags = lags
        self._windows = windows

        names: list[str] = [
            "gold_has_exact_current",
            "gold_is_partial_source_hour",
            "gold_candle_range_pct",
            "gold_candle_body_pct",
            "log_gold_silver_ratio",
        ]
        for lag in lags:
            names.extend([
                f"gold_has_exact_{lag}h",
                f"gold_log_return_{lag}h",
                f"gold_silver_relative_return_{lag}h",
                f"gold_silver_ratio_has_exact_{lag}h",
                f"gold_silver_log_ratio_change_{lag}h",
            ])
        for window in windows:
            names.extend([
                f"gold_realized_vol_{window}h",
                f"gold_silver_corr_{window}h",
            ])
        self._names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        silver_close = pd.to_numeric(out[self._c.close], errors="coerce").astype(float)

        gold_features = self._build_gold_feature_table()
        aligned_columns = tuple(column for column in gold_features.columns if column != "timestamp_utc")
        aligned = self._aligner.align(ts, gold_features, aligned_columns)
        for column in aligned.columns:
            out[column] = aligned[column]

        gold_close = pd.to_numeric(out.pop("gold_close_usd_per_kg_internal"), errors="coerce")
        out["gold_has_exact_current"] = gold_close.notna().astype("int8")
        out["gold_is_partial_source_hour"] = (
            pd.to_numeric(out["gold_is_partial_source_hour"], errors="coerce").fillna(0).astype("int8")
        )
        for lag in self._lags:
            availability = f"gold_has_exact_{lag}h"
            out[availability] = pd.to_numeric(out[availability], errors="coerce").fillna(0).astype("int8")

        log_ratio = np.log(gold_close / silver_close)
        out["log_gold_silver_ratio"] = log_ratio

        silver_log = np.log(silver_close)
        for lag in self._lags:
            silver_prior = _exact_lag(silver_log, ts, lag)
            silver_return = silver_log - silver_prior
            gold_return = pd.to_numeric(out[f"gold_log_return_{lag}h"], errors="coerce")
            out[f"gold_silver_relative_return_{lag}h"] = silver_return - gold_return

            prior_ratio = _exact_lag(log_ratio, ts, lag)
            ratio_available = log_ratio.notna() & prior_ratio.notna()
            out[f"gold_silver_ratio_has_exact_{lag}h"] = ratio_available.astype("int8")
            out[f"gold_silver_log_ratio_change_{lag}h"] = log_ratio - prior_ratio

        silver_prior_1h = _exact_lag(silver_log, ts, 1)
        silver_return_1h = silver_log - silver_prior_1h
        gold_return_1h = pd.to_numeric(out["gold_log_return_1h"], errors="coerce")
        idx = pd.DatetimeIndex(ts)
        silver_keyed = pd.Series(silver_return_1h.to_numpy(float), index=idx)
        gold_keyed = pd.Series(gold_return_1h.to_numpy(float), index=idx)
        for window in self._windows:
            min_periods = max(4, int(np.ceil(window * 0.5)))
            corr = silver_keyed.rolling(f"{window}h", min_periods=min_periods).corr(gold_keyed)
            out[f"gold_silver_corr_{window}h"] = corr.to_numpy()

        return out

    def _build_gold_feature_table(self) -> pd.DataFrame:
        gold = self._gold.copy(deep=True)
        ts = pd.to_datetime(gold["timestamp_utc"], utc=True)
        close = gold["close_usd_per_kg"].astype(float)
        log_close = np.log(close)
        open_price = gold["open_usd_per_kg"].astype(float)
        high = gold["high_usd_per_kg"].astype(float)
        low = gold["low_usd_per_kg"].astype(float)
        safe_open = open_price.replace(0.0, np.nan)

        result = pd.DataFrame({
            "timestamp_utc": ts,
            "gold_close_usd_per_kg_internal": close,
            "gold_is_partial_source_hour": gold["quality_flag"].eq("PARTIAL_SOURCE_HOUR").astype("int8"),
            "gold_candle_range_pct": (high - low) / safe_open,
            "gold_candle_body_pct": (close - open_price) / safe_open,
        })

        for lag in self._lags:
            prior = _exact_lag(log_close, ts, lag)
            result[f"gold_has_exact_{lag}h"] = prior.notna().astype("int8")
            result[f"gold_log_return_{lag}h"] = log_close - prior

        prior_1h = _exact_lag(log_close, ts, 1)
        return_1h = log_close - prior_1h
        keyed = pd.Series(return_1h.to_numpy(float), index=pd.DatetimeIndex(ts))
        for window in self._windows:
            min_periods = max(4, int(np.ceil(window * 0.5)))
            result[f"gold_realized_vol_{window}h"] = (
                keyed.rolling(f"{window}h", min_periods=min_periods).std(ddof=0).to_numpy()
            )
        return result

    @staticmethod
    def _validate_gold(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "timestamp_utc", "open_usd_per_kg", "high_usd_per_kg",
            "low_usd_per_kg", "close_usd_per_kg", "quality_flag",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Gold frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():
            raise ValueError("Gold timestamps must be unique.")
        price_columns = ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]
        prices = out[price_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
            raise ValueError("Gold frame contains invalid prices.")
        out[price_columns] = prices
        return out
