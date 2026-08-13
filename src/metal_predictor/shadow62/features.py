from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.precious_metals.confirmation import CANDIDATE_FEATURES


SHADOW_FEATURE_VERSION = "xpt-xpd-confirmed-10-v1"


def _exact_lag(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    keyed = pd.Series(series.to_numpy(dtype=float), index=index)
    wanted = index - pd.Timedelta(hours=hours)
    return pd.Series(keyed.reindex(wanted).to_numpy(), index=series.index, dtype=float)


class ConfirmedPreciousMetalsShadowFeatures:
    """Pure exact-clock assembler for the 10 historically confirmed XPT/XPD features.

    This component intentionally excludes every rejected/redundant family from the
    43-feature research registry. Auxiliary values are aligned only at identical UTC
    bar-start timestamps; gaps remain NaN and are handled only by the frozen model's
    training-time imputer contract.
    """

    def __init__(self, aligner: ExactTimestampAligner | None = None) -> None:
        self._aligner = aligner or ExactTimestampAligner()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return CANDIDATE_FEATURES

    @property
    def feature_version(self) -> str:
        return SHADOW_FEATURE_VERSION

    def transform(
        self,
        silver_frame: pd.DataFrame,
        platinum_frame: pd.DataFrame,
        palladium_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if "timestamp_utc" not in silver_frame.columns:
            raise ValueError("Silver frame requires timestamp_utc.")
        out = silver_frame.copy(deep=True)
        timestamps = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")

        for prefix, market, asset in (
            ("xpt", platinum_frame, "XPT"),
            ("xpd", palladium_frame, "XPD"),
        ):
            validated = self._validate_market(market, asset)
            auxiliary = self._build_auxiliary(validated, prefix)
            aligned = self._aligner.align(
                timestamps,
                auxiliary,
                tuple(column for column in auxiliary.columns if column != "timestamp_utc"),
            )
            for column in aligned.columns:
                out[column] = aligned[column]

        missing = set(CANDIDATE_FEATURES).difference(out.columns)
        if missing:
            raise ValueError(f"Shadow candidate features missing after assembly: {sorted(missing)}")
        return out

    @staticmethod
    def exact_current_available(market: pd.DataFrame, timestamp_utc: pd.Timestamp) -> bool:
        if market.empty or "timestamp_utc" not in market.columns:
            return False
        timestamps = pd.to_datetime(market["timestamp_utc"], utc=True, errors="coerce")
        return bool(timestamps.eq(pd.Timestamp(timestamp_utc)).any())

    @staticmethod
    def _build_auxiliary(market: pd.DataFrame, prefix: str) -> pd.DataFrame:
        timestamps = pd.to_datetime(market["timestamp_utc"], utc=True, errors="raise")
        open_price = pd.to_numeric(market["open_usd_per_kg"], errors="raise").astype(float)
        high = pd.to_numeric(market["high_usd_per_kg"], errors="raise").astype(float)
        low = pd.to_numeric(market["low_usd_per_kg"], errors="raise").astype(float)
        close = pd.to_numeric(market["close_usd_per_kg"], errors="raise").astype(float)
        log_close = np.log(close)

        result = pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                f"{prefix}_candle_range_pct": (high - low) / open_price,
                f"{prefix}_candle_body_pct": (close - open_price) / open_price,
            }
        )
        for lag in (1, 6, 24):
            result[f"{prefix}_log_return_{lag}h"] = log_close - _exact_lag(
                log_close, timestamps, lag
            )
        return result

    @staticmethod
    def _validate_market(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
        required = {
            "timestamp_utc",
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{asset} shadow source missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="raise")
        out = out.sort_values("timestamp_utc").reset_index(drop=True)
        if out["timestamp_utc"].duplicated().any():
            raise ValueError(f"{asset} shadow timestamps must be unique.")
        exact = (
            out["timestamp_utc"].dt.minute.eq(0)
            & out["timestamp_utc"].dt.second.eq(0)
            & out["timestamp_utc"].dt.microsecond.eq(0)
        )
        if not exact.all():
            raise ValueError(f"{asset} shadow timestamps must align to exact UTC hours.")
        price_columns = [
            "open_usd_per_kg",
            "high_usd_per_kg",
            "low_usd_per_kg",
            "close_usd_per_kg",
        ]
        prices = out[price_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
            raise ValueError(f"{asset} shadow source contains invalid prices.")
        invalid = (
            prices["high_usd_per_kg"].lt(prices["low_usd_per_kg"])
            | prices["high_usd_per_kg"].lt(
                prices[["open_usd_per_kg", "close_usd_per_kg"]].max(axis=1)
            )
            | prices["low_usd_per_kg"].gt(
                prices[["open_usd_per_kg", "close_usd_per_kg"]].min(axis=1)
            )
        )
        if invalid.any():
            raise ValueError(f"{asset} shadow source violates OHLC invariants.")
        out[price_columns] = prices
        return out
