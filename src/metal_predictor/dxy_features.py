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


class DollarIndexCrossAssetFeatures:
    """Causal UDX/DXY features exactly aligned to completed XAG hourly bars."""

    def __init__(
        self,
        dxy_frame: pd.DataFrame,
        aligner: ExactTimestampAligner,
        silver_columns: ColumnConfig,
        lags: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168),
        windows: tuple[int, ...] = (24, 72, 168),
    ) -> None:
        self._dxy = self._validate_dxy(dxy_frame)
        self._aligner = aligner
        self._c = silver_columns
        self._lags = lags
        self._windows = windows

        names: list[str] = [
            "dxy_has_exact_current",
            "dxy_is_partial_source_hour",
            "dxy_candle_range_pct",
            "dxy_candle_body_pct",
        ]
        for lag in lags:
            names.extend([f"dxy_has_exact_{lag}h", f"dxy_log_return_{lag}h"])
        for window in windows:
            names.extend([
                f"dxy_realized_vol_{window}h",
                f"dxy_close_vs_sma_{window}h",
                f"dxy_range_position_{window}h",
                f"silver_dxy_corr_{window}h",
            ])
        self._names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        dxy_features = self._build_dxy_feature_table()
        aligned_columns = tuple(column for column in dxy_features.columns if column != "timestamp_utc")
        aligned = self._aligner.align(ts, dxy_features, aligned_columns)
        for column in aligned.columns:
            out[column] = aligned[column]

        dxy_close = pd.to_numeric(out.pop("dxy_close_value_internal"), errors="coerce")
        out["dxy_has_exact_current"] = dxy_close.notna().astype("int8")
        out["dxy_is_partial_source_hour"] = (
            pd.to_numeric(out["dxy_is_partial_source_hour"], errors="coerce").fillna(0).astype("int8")
        )
        for lag in self._lags:
            name = f"dxy_has_exact_{lag}h"
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0).astype("int8")

        silver_close = pd.to_numeric(out[self._c.close], errors="coerce").astype(float)
        silver_log = np.log(silver_close)
        silver_prior_1h = _exact_lag(silver_log, ts, 1)
        silver_return_1h = silver_log - silver_prior_1h
        dxy_return_1h = pd.to_numeric(out["dxy_log_return_1h"], errors="coerce")
        idx = pd.DatetimeIndex(ts)
        silver_keyed = pd.Series(silver_return_1h.to_numpy(float), index=idx)
        dxy_keyed = pd.Series(dxy_return_1h.to_numpy(float), index=idx)
        for window in self._windows:
            min_periods = max(4, int(np.ceil(window * 0.5)))
            corr = silver_keyed.rolling(f"{window}h", min_periods=min_periods).corr(dxy_keyed)
            out[f"silver_dxy_corr_{window}h"] = corr.to_numpy()
        return out

    def _build_dxy_feature_table(self) -> pd.DataFrame:
        dxy = self._dxy.copy(deep=True)
        ts = pd.to_datetime(dxy["timestamp_utc"], utc=True)
        close = dxy["close_value"].astype(float)
        log_close = np.log(close)
        open_value = dxy["open_value"].astype(float)
        high = dxy["high_value"].astype(float)
        low = dxy["low_value"].astype(float)
        safe_open = open_value.replace(0.0, np.nan)

        result = pd.DataFrame({
            "timestamp_utc": ts,
            "dxy_close_value_internal": close,
            "dxy_is_partial_source_hour": dxy["quality_flag"].eq("PARTIAL_SOURCE_HOUR").astype("int8"),
            "dxy_candle_range_pct": (high - low) / safe_open,
            "dxy_candle_body_pct": (close - open_value) / safe_open,
        })

        for lag in self._lags:
            prior = _exact_lag(log_close, ts, lag)
            result[f"dxy_has_exact_{lag}h"] = prior.notna().astype("int8")
            result[f"dxy_log_return_{lag}h"] = log_close - prior

        prior_1h = _exact_lag(log_close, ts, 1)
        return_1h = log_close - prior_1h
        idx = pd.DatetimeIndex(ts)
        return_keyed = pd.Series(return_1h.to_numpy(float), index=idx)
        close_keyed = pd.Series(close.to_numpy(float), index=idx)
        for window in self._windows:
            min_periods = max(4, int(np.ceil(window * 0.5)))
            rolling_close = close_keyed.rolling(f"{window}h", min_periods=min_periods)
            sma = rolling_close.mean()
            rolling_low = rolling_close.min()
            rolling_high = rolling_close.max()
            span = (rolling_high - rolling_low).replace(0.0, np.nan)
            result[f"dxy_realized_vol_{window}h"] = (
                return_keyed.rolling(f"{window}h", min_periods=min_periods).std(ddof=0).to_numpy()
            )
            result[f"dxy_close_vs_sma_{window}h"] = close.to_numpy() / sma.to_numpy() - 1.0
            result[f"dxy_range_position_{window}h"] = (
                ((close_keyed - rolling_low) / span).fillna(0.5).to_numpy()
            )
        return result

    @staticmethod
    def _validate_dxy(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "timestamp_utc", "open_value", "high_value", "low_value", "close_value", "quality_flag"
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DXY frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():
            raise ValueError("DXY timestamps must be unique.")
        value_columns = ["open_value", "high_value", "low_value", "close_value"]
        values = out[value_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all() or (values <= 0).any().any():
            raise ValueError("DXY frame contains invalid index values.")
        out[value_columns] = values
        return out
