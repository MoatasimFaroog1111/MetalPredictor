from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.core import ColumnConfig
from metal_predictor.decision_time import CompletedHourlyBarDecisionClock, DecisionClock
from metal_predictor.published_state import PublishedStateAligner


class CftcSilverCotFeatures:
    """Causal COMEX Silver COT/Open-Interest features.

    Weekly COT states are transformed at their release cadence, then exposed to an
    hourly Silver row only when the report is already public at the completed-bar
    decision time. No report-date value is backdated into the Tuesday observation.
    """

    _CHANGE_HORIZONS = (1, 4, 13)
    _ZSCORE_WINDOWS = (13, 26, 52)

    def __init__(
        self,
        cot: pd.DataFrame,
        aligner: PublishedStateAligner,
        silver_columns: ColumnConfig,
        decision_clock: DecisionClock | None = None,
    ) -> None:
        self._cot = self._validate(cot)
        self._aligner = aligner
        self._c = silver_columns
        self._decision_clock = decision_clock or CompletedHourlyBarDecisionClock()

        names: list[str] = [
            "cot_has_published_state",
            "cot_publication_age_hours",
            "cot_new_report_within_1h",
            "cot_open_interest",
            "cot_managed_net_contracts",
            "cot_managed_net_oi",
            "cot_managed_gross_oi",
            "cot_managed_spread_oi",
            "cot_producer_net_contracts",
            "cot_producer_net_oi",
            "cot_producer_gross_oi",
            "cot_swap_net_contracts",
            "cot_swap_net_oi",
            "cot_swap_gross_oi",
            "cot_swap_spread_oi",
            "cot_other_net_oi",
            "cot_nonreportable_net_oi",
            "cot_managed_vs_producer_divergence",
        ]
        for horizon in self._CHANGE_HORIZONS:
            names.extend([
                f"cot_open_interest_change_{horizon}release_pct",
                f"cot_managed_net_oi_change_{horizon}release",
                f"cot_producer_net_oi_change_{horizon}release",
                f"cot_swap_net_oi_change_{horizon}release",
                f"cot_divergence_change_{horizon}release",
            ])
        for window in self._ZSCORE_WINDOWS:
            names.extend([
                f"cot_open_interest_zscore_{window}release",
                f"cot_managed_net_oi_zscore_{window}release",
                f"cot_producer_net_oi_zscore_{window}release",
                f"cot_divergence_zscore_{window}release",
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
            if column not in {"available_from_utc", "report_date"}
        )
        aligned = self._aligner.align(
            decision_time,
            published,
            value_columns=("report_date", *numeric_columns),
        )
        available = pd.to_datetime(
            aligned.pop("available_from_utc"), utc=True, errors="coerce"
        )
        aligned.pop("report_date")
        age = pd.to_numeric(aligned.pop("published_state_age_hours"), errors="coerce")
        out["cot_has_published_state"] = available.notna().astype("int8")
        out["cot_publication_age_hours"] = age
        out["cot_new_report_within_1h"] = (
            age.between(0.0, 1.0, inclusive="both").fillna(False).astype("int8")
        )
        for column in numeric_columns:
            out[column] = pd.to_numeric(aligned[column], errors="coerce")
        return out

    def _feature_table(self) -> pd.DataFrame:
        cot = self._cot.copy(deep=True)
        oi = cot["open_interest"].astype(float)
        safe_oi = oi.replace(0.0, np.nan)

        managed_net = cot["managed_long"] - cot["managed_short"]
        producer_net = cot["producer_long"] - cot["producer_short"]
        swap_net = cot["swap_long"] - cot["swap_short"]
        other_net = cot["other_long"] - cot["other_short"]
        nonreportable_net = cot["nonreportable_long"] - cot["nonreportable_short"]

        managed_net_oi = managed_net / safe_oi
        producer_net_oi = producer_net / safe_oi
        swap_net_oi = swap_net / safe_oi
        other_net_oi = other_net / safe_oi
        nonreportable_net_oi = nonreportable_net / safe_oi
        divergence = managed_net_oi - producer_net_oi

        result = pd.DataFrame({
            "report_date": cot["report_date"],
            "available_from_utc": cot["available_from_utc"],
            "cot_open_interest": oi,
            "cot_managed_net_contracts": managed_net,
            "cot_managed_net_oi": managed_net_oi,
            "cot_managed_gross_oi": (cot["managed_long"] + cot["managed_short"]) / safe_oi,
            "cot_managed_spread_oi": cot["managed_spread"] / safe_oi,
            "cot_producer_net_contracts": producer_net,
            "cot_producer_net_oi": producer_net_oi,
            "cot_producer_gross_oi": (cot["producer_long"] + cot["producer_short"]) / safe_oi,
            "cot_swap_net_contracts": swap_net,
            "cot_swap_net_oi": swap_net_oi,
            "cot_swap_gross_oi": (cot["swap_long"] + cot["swap_short"]) / safe_oi,
            "cot_swap_spread_oi": cot["swap_spread"] / safe_oi,
            "cot_other_net_oi": other_net_oi,
            "cot_nonreportable_net_oi": nonreportable_net_oi,
            "cot_managed_vs_producer_divergence": divergence,
        })

        change_series = {
            "cot_managed_net_oi": managed_net_oi,
            "cot_producer_net_oi": producer_net_oi,
            "cot_swap_net_oi": swap_net_oi,
            "cot_divergence": divergence,
        }
        for horizon in self._CHANGE_HORIZONS:
            result[f"cot_open_interest_change_{horizon}release_pct"] = oi / oi.shift(horizon) - 1.0
            for name, values in change_series.items():
                result[f"{name}_change_{horizon}release"] = values - values.shift(horizon)

        zscore_series = {
            "cot_open_interest": oi,
            "cot_managed_net_oi": managed_net_oi,
            "cot_producer_net_oi": producer_net_oi,
            "cot_divergence": divergence,
        }
        for window in self._ZSCORE_WINDOWS:
            for name, values in zscore_series.items():
                rolling = values.rolling(window, min_periods=window)
                mean = rolling.mean()
                std = rolling.std(ddof=0).replace(0.0, np.nan)
                result[f"{name}_zscore_{window}release"] = (values - mean) / std
        return result

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "report_date", "available_from_utc", "open_interest",
            "producer_long", "producer_short", "swap_long", "swap_short", "swap_spread",
            "managed_long", "managed_short", "managed_spread",
            "other_long", "other_short", "other_spread",
            "nonreportable_long", "nonreportable_short",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CFTC COT frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["report_date"] = pd.to_datetime(out["report_date"], errors="raise").dt.normalize()
        out["available_from_utc"] = pd.to_datetime(
            out["available_from_utc"], utc=True, errors="raise"
        ).astype("datetime64[ns, UTC]")
        out = out.sort_values("available_from_utc").reset_index(drop=True)
        if out["available_from_utc"].duplicated().any():
            raise ValueError("CFTC COT publication timestamps must be unique.")
        numeric = sorted(required.difference({"report_date", "available_from_utc"}))
        values = out[numeric].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError("CFTC COT frame contains invalid numeric values.")
        if (values["open_interest"] <= 0).any():
            raise ValueError("CFTC COT Open Interest must be positive.")
        out[numeric] = values
        return out
