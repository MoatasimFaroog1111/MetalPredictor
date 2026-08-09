from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.core import ColumnConfig
from metal_predictor.decision_time import CompletedHourlyBarDecisionClock, DecisionClock
from metal_predictor.published_state import PublishedStateAligner


class VixDailyFeatures:
    """Causal daily VIX state features exposed only after the official daily close."""

    _RETURN_HORIZONS = (1, 5, 20)
    _WINDOWS = (5, 20, 60)

    def __init__(
        self,
        vix: pd.DataFrame,
        aligner: PublishedStateAligner,
        silver_columns: ColumnConfig,
        decision_clock: DecisionClock | None = None,
    ) -> None:
        self._vix = self._validate_vix(vix)
        self._aligner = aligner
        self._c = silver_columns
        self._decision_clock = decision_clock or CompletedHourlyBarDecisionClock()

        names = [
            "vix_has_published_state",
            "vix_publication_age_hours",
            "vix_close",
            "vix_daily_range_pct",
            "vix_daily_body_pct",
            "vix_close_location",
            "vix_new_close_within_1h",
        ]
        names.extend(f"vix_log_return_{horizon}release" for horizon in self._RETURN_HORIZONS)
        for window in self._WINDOWS:
            names.extend([
                f"vix_return_vol_{window}release",
                f"vix_close_vs_sma_{window}release",
                f"vix_zscore_{window}release",
                f"vix_range_position_{window}release",
            ])
        self._names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        bar_start = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        decision_time = self._decision_clock.available_at(bar_start)
        published = self._feature_table()
        numeric_columns = tuple(
            column for column in published.columns
            if column not in {"available_from_utc", "observation_date"}
        )
        aligned = self._aligner.align(
            decision_time,
            published,
            value_columns=("observation_date", *numeric_columns),
        )
        available = pd.to_datetime(
            aligned.pop("available_from_utc"), utc=True, errors="coerce"
        )
        aligned.pop("observation_date")
        age = pd.to_numeric(aligned.pop("published_state_age_hours"), errors="coerce")
        out["vix_has_published_state"] = available.notna().astype("int8")
        out["vix_publication_age_hours"] = age
        out["vix_new_close_within_1h"] = (
            age.between(0.0, 1.0, inclusive="both").fillna(False).astype("int8")
        )
        for column in numeric_columns:
            out[column] = pd.to_numeric(aligned[column], errors="coerce")
        return out

    def _feature_table(self) -> pd.DataFrame:
        vix = self._vix.copy(deep=True)
        close = vix["vix_close"].astype(float)
        open_value = vix["vix_open"].astype(float)
        high = vix["vix_high"].astype(float)
        low = vix["vix_low"].astype(float)
        safe_open = open_value.replace(0.0, np.nan)
        span = (high - low).replace(0.0, np.nan)
        log_close = np.log(close)
        one_return = log_close.diff()

        result = pd.DataFrame({
            "observation_date": vix["observation_date"],
            "available_from_utc": vix["available_from_utc"],
            "vix_close": close,
            "vix_daily_range_pct": (high - low) / safe_open,
            "vix_daily_body_pct": (close - open_value) / safe_open,
            "vix_close_location": ((close - low) / span).fillna(0.5),
        })
        for horizon in self._RETURN_HORIZONS:
            result[f"vix_log_return_{horizon}release"] = log_close - log_close.shift(horizon)

        for window in self._WINDOWS:
            min_periods = window
            rolling_close = close.rolling(window, min_periods=min_periods)
            mean = rolling_close.mean()
            std = rolling_close.std(ddof=0).replace(0.0, np.nan)
            rolling_low = rolling_close.min()
            rolling_high = rolling_close.max()
            rolling_span = (rolling_high - rolling_low).replace(0.0, np.nan)
            result[f"vix_return_vol_{window}release"] = one_return.rolling(
                window, min_periods=min_periods
            ).std(ddof=0)
            result[f"vix_close_vs_sma_{window}release"] = close / mean - 1.0
            result[f"vix_zscore_{window}release"] = (close - mean) / std
            result[f"vix_range_position_{window}release"] = (
                (close - rolling_low) / rolling_span
            ).fillna(0.5)
        return result

    @staticmethod
    def _validate_vix(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "observation_date", "available_from_utc",
            "vix_open", "vix_high", "vix_low", "vix_close",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"VIX frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["observation_date"] = pd.to_datetime(
            out["observation_date"], errors="raise"
        ).dt.normalize()
        out["available_from_utc"] = pd.to_datetime(
            out["available_from_utc"], utc=True, errors="raise"
        ).astype("datetime64[ns, UTC]")
        out = out.sort_values("available_from_utc").reset_index(drop=True)
        if out["available_from_utc"].duplicated().any():
            raise ValueError("VIX publication timestamps must be unique.")
        price_columns = ["vix_open", "vix_high", "vix_low", "vix_close"]
        values = out[price_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all() or (values <= 0).any().any():
            raise ValueError("VIX frame contains invalid values.")
        out[price_columns] = values
        return out
