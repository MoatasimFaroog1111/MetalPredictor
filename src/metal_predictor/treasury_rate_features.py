from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.core import ColumnConfig
from metal_predictor.published_state import PublishedStateAligner


class TreasuryRateFeatures:
    """Causal Treasury-curve features based only on H.15-published states."""

    _HORIZONS = (1, 5, 20)
    _VOL_WINDOWS = (5, 20)

    def __init__(
        self,
        rates: pd.DataFrame,
        aligner: PublishedStateAligner,
        silver_columns: ColumnConfig,
    ) -> None:
        self._rates = self._validate_rates(rates)
        self._aligner = aligner
        self._c = silver_columns
        names = [
            "rates_has_published_state",
            "rates_publication_age_hours",
            "treasury_2y_percent",
            "treasury_10y_percent",
            "treasury_curve_10y_2y_pctpt",
            "rates_new_release_within_1h",
        ]
        for horizon in self._HORIZONS:
            for stem in ("treasury_2y", "treasury_10y", "treasury_curve_10y_2y"):
                names.append(f"{stem}_change_{horizon}release_pctpt")
        for window in self._VOL_WINDOWS:
            names.extend([
                f"treasury_2y_change_vol_{window}release",
                f"treasury_10y_change_vol_{window}release",
                f"treasury_curve_change_vol_{window}release",
            ])
        self._names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy(deep=True)
        ts = pd.to_datetime(out[self._c.timestamp], utc=True, errors="raise")
        published = self._feature_table()
        numeric_columns = tuple(
            column for column in published.columns
            if column not in {"available_from_utc", "observation_date"}
        )
        aligned = self._aligner.align(
            ts,
            published,
            value_columns=("observation_date", *numeric_columns),
        )

        available = pd.to_datetime(aligned.pop("available_from_utc"), utc=True, errors="coerce")
        aligned.pop("observation_date")
        age = pd.to_numeric(aligned.pop("published_state_age_hours"), errors="coerce")
        out["rates_has_published_state"] = available.notna().astype("int8")
        out["rates_publication_age_hours"] = age
        out["rates_new_release_within_1h"] = age.between(0.0, 1.0, inclusive="both").fillna(False).astype("int8")
        for column in numeric_columns:
            out[column] = pd.to_numeric(aligned[column], errors="coerce")
        return out

    def _feature_table(self) -> pd.DataFrame:
        rates = self._rates.copy(deep=True)
        r2 = rates["rate_2y_percent"].astype(float)
        r10 = rates["rate_10y_percent"].astype(float)
        curve = r10 - r2
        result = pd.DataFrame({
            "observation_date": rates["observation_date"],
            "available_from_utc": rates["available_from_utc"],
            "treasury_2y_percent": r2,
            "treasury_10y_percent": r10,
            "treasury_curve_10y_2y_pctpt": curve,
        })
        series = {
            "treasury_2y": r2,
            "treasury_10y": r10,
            "treasury_curve_10y_2y": curve,
        }
        for horizon in self._HORIZONS:
            for stem, values in series.items():
                result[f"{stem}_change_{horizon}release_pctpt"] = values - values.shift(horizon)

        one_release_changes = {
            "treasury_2y": r2.diff(),
            "treasury_10y": r10.diff(),
            "treasury_curve": curve.diff(),
        }
        for window in self._VOL_WINDOWS:
            for stem, values in one_release_changes.items():
                result[f"{stem}_change_vol_{window}release"] = (
                    values.rolling(window, min_periods=window).std(ddof=0)
                )
        return result

    @staticmethod
    def _validate_rates(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "observation_date", "available_from_utc", "rate_2y_percent", "rate_10y_percent"
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Treasury rates frame missing columns: {sorted(missing)}")
        out = frame.copy(deep=True)
        out["observation_date"] = pd.to_datetime(out["observation_date"], errors="raise").dt.normalize()
        out["available_from_utc"] = pd.to_datetime(out["available_from_utc"], utc=True, errors="raise")
        out = out.sort_values("available_from_utc").reset_index(drop=True)
        if out["available_from_utc"].duplicated().any():
            raise ValueError("Treasury publication timestamps must be unique.")
        if not out["available_from_utc"].is_monotonic_increasing:
            raise ValueError("Treasury publication timestamps must be chronological.")
        for column in ("rate_2y_percent", "rate_10y_percent"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
            finite = out[column].dropna().to_numpy(float)
            if not np.isfinite(finite).all():
                raise ValueError(f"{column} contains infinite values.")
        return out
