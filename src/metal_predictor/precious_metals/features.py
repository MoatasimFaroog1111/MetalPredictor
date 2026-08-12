from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig


FEATURE_VERSION = "precious-metals-cross-asset-v1"


def _exact_lag(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    keyed = pd.Series(series.to_numpy(dtype=float), index=idx)
    wanted = idx - pd.Timedelta(hours=hours)
    return pd.Series(keyed.reindex(wanted).to_numpy(), index=series.index, dtype=float)


class PlatinumPalladiumCrossAssetFeatures:
    """Pre-registered causal XPT/XPD feature family for Silver research.

    Auxiliary candles are joined only at identical UTC bar-start timestamps. Missing
    hours remain missing; no forward-fill, interpolation, backward-fill or nearest-time
    matching is permitted. The feature family is intentionally fixed before any model
    comparison so research results cannot silently change feature definitions.
    """

    def __init__(
        self,
        platinum_frame: pd.DataFrame,
        palladium_frame: pd.DataFrame,
        aligner: ExactTimestampAligner,
        silver_columns: ColumnConfig,
        return_lags: tuple[int, ...] = (1, 6, 24),
        ratio_lags: tuple[int, ...] = (6, 24),
        corr_windows: tuple[int, ...] = (24, 72),
        volatility_window: int = 24,
    ) -> None:
        if return_lags != (1, 6, 24) or ratio_lags != (6, 24) or corr_windows != (24, 72):
            raise ValueError("Precious-metals v1 feature windows are pre-registered and immutable.")
        if volatility_window != 24:
            raise ValueError("Precious-metals v1 volatility window is fixed at 24 hours.")
        self._frames = {
            "xpt": self._validate_market(platinum_frame, "XPT"),
            "xpd": self._validate_market(palladium_frame, "XPD"),
        }
        self._aligner = aligner
        self._c = silver_columns
        self._return_lags = return_lags
        self._ratio_lags = ratio_lags
        self._corr_windows = corr_windows
        self._vol_window = volatility_window
        self._names = self._feature_names()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def feature_version(self) -> str:
        return FEATURE_VERSION

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        silver_close = pd.to_numeric(out[self._c.close], errors="coerce").astype(float)
        silver_log = np.log(silver_close)

        for prefix, market in self._frames.items():
            aux = self._build_auxiliary_table(market, prefix)
            aligned_columns = tuple(column for column in aux.columns if column != "timestamp_utc")
            aligned = self._aligner.align(ts, aux, aligned_columns)
            for column in aligned.columns:
                out[column] = aligned[column]

            metal_close = pd.to_numeric(out.pop(f"{prefix}_close_usd_per_kg_internal"), errors="coerce")
            out[f"{prefix}_has_exact_current"] = metal_close.notna().astype("int8")
            ratio = np.log(metal_close / silver_close)
            out[f"log_{prefix}_silver_ratio"] = ratio

            for lag in self._return_lags:
                availability = f"{prefix}_has_exact_{lag}h"
                out[availability] = pd.to_numeric(out[availability], errors="coerce").fillna(0).astype("int8")
                silver_prior = _exact_lag(silver_log, ts, lag)
                silver_return = silver_log - silver_prior
                metal_return = pd.to_numeric(out[f"{prefix}_log_return_{lag}h"], errors="coerce")
                out[f"{prefix}_silver_relative_return_{lag}h"] = metal_return - silver_return

            for lag in self._ratio_lags:
                prior_ratio = _exact_lag(ratio, ts, lag)
                out[f"{prefix}_silver_log_ratio_change_{lag}h"] = ratio - prior_ratio

            silver_prior_1h = _exact_lag(silver_log, ts, 1)
            silver_return_1h = silver_log - silver_prior_1h
            metal_return_1h = pd.to_numeric(out[f"{prefix}_log_return_1h"], errors="coerce")
            idx = pd.DatetimeIndex(ts)
            silver_keyed = pd.Series(silver_return_1h.to_numpy(float), index=idx)
            metal_keyed = pd.Series(metal_return_1h.to_numpy(float), index=idx)
            for window in self._corr_windows:
                min_periods = max(4, int(np.ceil(window * 0.5)))
                corr = silver_keyed.rolling(f"{window}h", min_periods=min_periods).corr(metal_keyed)
                out[f"{prefix}_silver_corr_{window}h"] = corr.to_numpy()

        self._add_joint_features(out, ts)
        return out

    def _build_auxiliary_table(self, market: pd.DataFrame, prefix: str) -> pd.DataFrame:
        ts = pd.to_datetime(market["timestamp_utc"], utc=True)
        close = market["close_usd_per_kg"].astype(float)
        log_close = np.log(close)
        open_price = market["open_usd_per_kg"].astype(float)
        high = market["high_usd_per_kg"].astype(float)
        low = market["low_usd_per_kg"].astype(float)
        safe_open = open_price.replace(0.0, np.nan)

        result = pd.DataFrame({
            "timestamp_utc": ts,
            f"{prefix}_close_usd_per_kg_internal": close,
            f"{prefix}_candle_range_pct": (high - low) / safe_open,
            f"{prefix}_candle_body_pct": (close - open_price) / safe_open,
        })
        for lag in self._return_lags:
            prior = _exact_lag(log_close, ts, lag)
            result[f"{prefix}_has_exact_{lag}h"] = prior.notna().astype("int8")
            result[f"{prefix}_log_return_{lag}h"] = log_close - prior

        prior_1h = _exact_lag(log_close, ts, 1)
        return_1h = log_close - prior_1h
        keyed = pd.Series(return_1h.to_numpy(float), index=pd.DatetimeIndex(ts))
        min_periods = max(4, int(np.ceil(self._vol_window * 0.5)))
        result[f"{prefix}_realized_vol_{self._vol_window}h"] = (
            keyed.rolling(f"{self._vol_window}h", min_periods=min_periods).std(ddof=0).to_numpy()
        )
        return result

    @staticmethod
    def _add_joint_features(out: pd.DataFrame, ts: pd.Series) -> None:
        xpt_close = pd.to_numeric(out.get("xpt_close_usd_per_kg_internal"), errors="coerce") if "xpt_close_usd_per_kg_internal" in out else None
        xpd_close = pd.to_numeric(out.get("xpd_close_usd_per_kg_internal"), errors="coerce") if "xpd_close_usd_per_kg_internal" in out else None
        # Internal close columns are popped before joint features, so reconstruct the
        # cross-metal ratio from Silver ratios. Their difference is log(XPT/XPD).
        xpt_silver = pd.to_numeric(out["log_xpt_silver_ratio"], errors="coerce")
        xpd_silver = pd.to_numeric(out["log_xpd_silver_ratio"], errors="coerce")
        xpt_xpd_ratio = xpt_silver - xpd_silver
        out["both_metals_have_exact_current"] = (
            out["xpt_has_exact_current"].eq(1) & out["xpd_has_exact_current"].eq(1)
        ).astype("int8")
        out["log_xpt_xpd_ratio"] = xpt_xpd_ratio
        out["xpt_xpd_log_ratio_change_1h"] = xpt_xpd_ratio - _exact_lag(xpt_xpd_ratio, ts, 1)

        xpt_ret = pd.to_numeric(out["xpt_log_return_1h"], errors="coerce")
        xpd_ret = pd.to_numeric(out["xpd_log_return_1h"], errors="coerce")
        both = xpt_ret.notna() & xpd_ret.notna()
        mean_return = (xpt_ret + xpd_ret) / 2.0
        spread = xpt_ret - xpd_ret
        breadth = (np.sign(xpt_ret) + np.sign(xpd_ret)) / 2.0
        out["metal_complex_mean_return_1h"] = mean_return.where(both)
        out["metal_complex_return_dispersion_1h"] = spread.abs().where(both)
        out["metal_complex_breadth_1h"] = breadth.where(both)
        out["xpt_xpd_return_spread_1h"] = spread.where(both)

    def _feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for prefix in ("xpt", "xpd"):
            names.extend([
                f"{prefix}_has_exact_current",
                f"{prefix}_candle_range_pct",
                f"{prefix}_candle_body_pct",
                f"log_{prefix}_silver_ratio",
            ])
            for lag in self._return_lags:
                names.extend([
                    f"{prefix}_has_exact_{lag}h",
                    f"{prefix}_log_return_{lag}h",
                    f"{prefix}_silver_relative_return_{lag}h",
                ])
            for lag in self._ratio_lags:
                names.append(f"{prefix}_silver_log_ratio_change_{lag}h")
            names.extend([
                f"{prefix}_realized_vol_{self._vol_window}h",
                *[f"{prefix}_silver_corr_{window}h" for window in self._corr_windows],
            ])
        names.extend([
            "both_metals_have_exact_current",
            "log_xpt_xpd_ratio",
            "xpt_xpd_log_ratio_change_1h",
            "metal_complex_mean_return_1h",
            "metal_complex_return_dispersion_1h",
            "metal_complex_breadth_1h",
            "xpt_xpd_return_spread_1h",
        ])
        return tuple(names)

    @staticmethod
    def _validate_market(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
        required = {
            "timestamp_utc", "open_usd_per_kg", "high_usd_per_kg",
            "low_usd_per_kg", "close_usd_per_kg", "quality_flag",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{asset} frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():
            raise ValueError(f"{asset} timestamps must be unique.")
        if not out["timestamp_utc"].dt.minute.eq(0).all() or not out["timestamp_utc"].dt.second.eq(0).all():
            raise ValueError(f"{asset} timestamps must align to exact UTC hours.")
        price_columns = ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]
        prices = out[price_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
            raise ValueError(f"{asset} frame contains invalid prices.")
        invalid_ohlc = (
            prices["high_usd_per_kg"].lt(prices["low_usd_per_kg"])
            | prices["high_usd_per_kg"].lt(prices[["open_usd_per_kg", "close_usd_per_kg"]].max(axis=1))
            | prices["low_usd_per_kg"].gt(prices[["open_usd_per_kg", "close_usd_per_kg"]].min(axis=1))
        )
        if invalid_ohlc.any():
            raise ValueError(f"{asset} frame violates OHLC invariants.")
        out[price_columns] = prices
        return out
