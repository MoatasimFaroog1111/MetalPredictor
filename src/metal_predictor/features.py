from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig, FeatureConfig

class PriceActionFeatures:
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns
        self._names = (
            "candle_range_pct","candle_body_pct","upper_wick_pct",
            "lower_wick_pct","close_location_value","log_hl_range",
        )
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        o, h, l, c = out[self._c.open], out[self._c.high], out[self._c.low], out[self._c.close]
        safe_o = o.replace(0, np.nan)
        span = h - l
        safe_span = span.replace(0, np.nan)
        out["candle_range_pct"] = span / safe_o
        out["candle_body_pct"] = (c - o) / safe_o
        out["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / safe_o
        out["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / safe_o
        out["close_location_value"] = ((c - l) / safe_span).fillna(0.5)
        out["log_hl_range"] = np.log(h / l)
        return out

class MomentumFeatures:
    def __init__(self, columns: ColumnConfig, config: FeatureConfig) -> None:
        self._c, self._cfg = columns, config
        names = [f"log_return_{lag}h" for lag in config.return_lags]
        names += [f"momentum_{lag}h" for lag in config.return_lags if lag > 1]
        names += [f"rsi_{config.rsi_window}"]
        self._names = tuple(names)
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        close = out[self._c.close].astype(float)
        log_close = np.log(close)
        for lag in self._cfg.return_lags:
            out[f"log_return_{lag}h"] = log_close - log_close.shift(lag)
            if lag > 1:
                out[f"momentum_{lag}h"] = close / close.shift(lag) - 1.0
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        n = self._cfg.rsi_window
        avg_gain = gains.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        avg_loss = losses.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        both_zero = avg_gain.eq(0.0) & avg_loss.eq(0.0)
        only_loss_zero = avg_loss.eq(0.0) & avg_gain.gt(0.0)
        out[f"rsi_{n}"] = rsi.mask(both_zero, 50.0).mask(only_loss_zero, 100.0)
        return out

class VolatilityFeatures:
    def __init__(self, columns: ColumnConfig, config: FeatureConfig) -> None:
        self._c, self._cfg = columns, config
        self._names = tuple([f"realized_vol_{w}h" for w in config.volatility_windows] + [f"atr_pct_{config.atr_window}"])
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        close = out[self._c.close].astype(float)
        log_return_1h = np.log(close).diff()
        for window in self._cfg.volatility_windows:
            out[f"realized_vol_{window}h"] = log_return_1h.rolling(window=window, min_periods=window).std(ddof=0)
        high = out[self._c.high].astype(float)
        low = out[self._c.low].astype(float)
        prev_close = close.shift(1)
        true_range = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        n = self._cfg.atr_window
        atr = true_range.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        out[f"atr_pct_{n}"] = atr / close
        return out

class TrendFeatures:
    def __init__(self, columns: ColumnConfig, config: FeatureConfig) -> None:
        self._c, self._cfg = columns, config
        names = []
        for window in config.trend_windows:
            names.extend([f"close_vs_sma_{window}h", f"close_vs_ema_{window}h", f"ema_slope_{window}h"])
        self._names = tuple(names)
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        close = out[self._c.close].astype(float)
        for window in self._cfg.trend_windows:
            sma = close.rolling(window=window, min_periods=window).mean()
            ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
            out[f"close_vs_sma_{window}h"] = close / sma - 1.0
            out[f"close_vs_ema_{window}h"] = close / ema - 1.0
            out[f"ema_slope_{window}h"] = ema / ema.shift(1) - 1.0
        return out

class TemporalFeatures:
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns
        self._names = ("hour_sin","hour_cos","weekday_sin","weekday_cos","gap_from_previous_hours","is_contiguous_from_previous")
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        ts = pd.to_datetime(out[self._c.timestamp], utc=True)
        hour = ts.dt.hour.astype(float)
        weekday = ts.dt.dayofweek.astype(float)
        out["hour_sin"] = np.sin(2*np.pi*hour/24.0)
        out["hour_cos"] = np.cos(2*np.pi*hour/24.0)
        out["weekday_sin"] = np.sin(2*np.pi*weekday/7.0)
        out["weekday_cos"] = np.cos(2*np.pi*weekday/7.0)
        gap = ts.diff().dt.total_seconds().div(3600.0)
        out["gap_from_previous_hours"] = gap
        out["is_contiguous_from_previous"] = gap.eq(1.0).astype("int8")
        return out

class QualityFeatures:
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns
        self._names = ("is_partial_source_hour",)
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if self._c.quality in out.columns:
            q = out[self._c.quality].astype("string").fillna("")
            out["is_partial_source_hour"] = q.eq("PARTIAL_SOURCE_HOUR").astype("int8")
        else:
            out["is_partial_source_hour"] = 0
        return out
