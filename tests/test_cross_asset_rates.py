from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from metal_predictor.core import ColumnConfig
from metal_predictor.publication_time import H15PublicationPolicy
from metal_predictor.published_state import PublishedStateAligner
from metal_predictor.treasury_rate_features import TreasuryRateFeatures


C = ColumnConfig()


def test_h15_publication_policy_is_dst_aware_and_next_business_day() -> None:
    policy = H15PublicationPolicy()
    observations = pd.Series(pd.to_datetime(["2026-07-27", "2026-01-02"]))
    available = policy.available_from_utc(observations)
    assert available.iloc[0] == pd.Timestamp("2026-07-28 20:15:00", tz="UTC")
    assert available.iloc[1] == pd.Timestamp("2026-01-05 21:15:00", tz="UTC")


def test_h15_policy_handles_board_closure_and_documented_treasury_omissions() -> None:
    policy = H15PublicationPolicy()
    observations = pd.Series(pd.to_datetime([
        "2025-01-08",
        "2023-08-01",
        "2023-09-12",
    ]))
    available = policy.available_from_utc(observations)
    assert available.iloc[0] == pd.Timestamp("2025-01-10 21:15:00", tz="UTC")
    assert available.iloc[1] == pd.Timestamp("2023-08-03 20:15:00", tz="UTC")
    assert available.iloc[2] == pd.Timestamp("2023-09-14 20:15:00", tz="UTC")
    assert date(2023, 8, 1) in policy.delayed_observation_dates


def test_published_state_is_not_visible_before_release_time() -> None:
    feature_ts = pd.Series(pd.to_datetime([
        "2026-07-28 20:00:00+00:00",
        "2026-07-28 21:00:00+00:00",
        "2026-07-29 12:00:00+00:00",
    ]))
    published = pd.DataFrame({
        "available_from_utc": [pd.Timestamp("2026-07-28 20:15:00", tz="UTC")],
        "value": [4.25],
    })
    out = PublishedStateAligner().align(feature_ts, published, ("value",))
    assert np.isnan(out.iloc[0]["value"])
    assert out.iloc[1]["value"] == 4.25
    assert out.iloc[2]["value"] == 4.25
    assert np.isnan(out.iloc[0]["published_state_age_hours"])
    assert out.iloc[1]["published_state_age_hours"] == 0.75


def _silver(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.03 + np.sin(np.arange(rows) / 17.0)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.1,
        C.high: close + 0.3,
        C.low: close - 0.3,
        C.close: close,
        C.quality: "OK",
    })


def _rates(days: int = 60) -> pd.DataFrame:
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    observation = pd.date_range("2024-12-02", periods=days, freq=business_day)
    policy = H15PublicationPolicy()
    available = policy.available_from_utc(pd.Series(observation))
    x = np.arange(days, dtype=float)
    return pd.DataFrame({
        "observation_date": observation,
        "available_from_utc": available,
        "rate_2y_percent": 4.0 + np.sin(x / 7.0) * 0.1,
        "rate_10y_percent": 4.3 + np.cos(x / 9.0) * 0.1,
    })


def test_treasury_features_are_causal_under_later_release_perturbation() -> None:
    silver = _silver()
    rates = _rates(80)
    cutoff = pd.Timestamp("2025-01-12 12:00:00", tz="UTC")
    component = TreasuryRateFeatures(rates, PublishedStateAligner(), C)
    baseline = component.transform(silver)

    changed_rates = rates.copy()
    later = pd.to_datetime(changed_rates["available_from_utc"], utc=True) > cutoff
    changed_rates.loc[later, ["rate_2y_percent", "rate_10y_percent"]] += 10.0
    changed_component = TreasuryRateFeatures(changed_rates, PublishedStateAligner(), C)
    changed = changed_component.transform(silver)

    rows = pd.to_datetime(silver[C.timestamp], utc=True) <= cutoff
    pd.testing.assert_frame_equal(
        baseline.loc[rows, component.feature_names].reset_index(drop=True),
        changed.loc[rows, changed_component.feature_names].reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=1e-14,
    )


def test_rate_feature_names_contain_no_future_or_target_tokens() -> None:
    component = TreasuryRateFeatures(_rates(), PublishedStateAligner(), C)
    names = [name.lower() for name in component.feature_names]
    assert not any("future" in name or "target" in name or "next_" in name for name in names)


def test_new_release_flag_only_turns_on_after_publication() -> None:
    rates = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-07-27"]),
        "available_from_utc": [pd.Timestamp("2026-07-28 20:15:00", tz="UTC")],
        "rate_2y_percent": [4.0],
        "rate_10y_percent": [4.3],
    })
    silver = pd.DataFrame({
        C.timestamp: pd.to_datetime([
            "2026-07-28 20:00:00+00:00",
            "2026-07-28 21:00:00+00:00",
            "2026-07-28 22:00:00+00:00",
        ]),
        C.open: [1000.0] * 3,
        C.high: [1001.0] * 3,
        C.low: [999.0] * 3,
        C.close: [1000.0] * 3,
        C.quality: ["OK"] * 3,
    })
    out = TreasuryRateFeatures(rates, PublishedStateAligner(), C).transform(silver)
    assert out.loc[0, "rates_has_published_state"] == 0
    assert out.loc[1, "rates_new_release_within_1h"] == 1
    assert out.loc[2, "rates_new_release_within_1h"] == 0
