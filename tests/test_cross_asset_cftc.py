from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.cftc_cot_features import CftcSilverCotFeatures
from metal_predictor.cftc_cot_publication import CftcCotPublicationPolicy
from metal_predictor.core import ColumnConfig
from metal_predictor.decision_time import CompletedHourlyBarDecisionClock
from metal_predictor.published_state import PublishedStateAligner


C = ColumnConfig()


def test_cftc_normal_release_is_friday_1530_et_with_dst() -> None:
    policy = CftcCotPublicationPolicy()
    report_dates = pd.Series(pd.to_datetime(["2026-07-14", "2026-02-10"]))
    available = policy.available_from_utc(report_dates)
    assert available.iloc[0] == pd.Timestamp("2026-07-17 19:30:00", tz="UTC")
    assert available.iloc[1] == pd.Timestamp("2026-02-13 20:30:00", tz="UTC")


def test_cftc_documented_2023_and_2025_delays_are_exact() -> None:
    policy = CftcCotPublicationPolicy()
    report_dates = pd.Series(pd.to_datetime([
        "2023-01-31",
        "2023-02-14",
        "2025-09-30",
        "2025-11-25",
    ]))
    available = policy.available_from_utc(report_dates)
    assert available.iloc[0] == pd.Timestamp("2023-02-24 20:30:00", tz="UTC")
    assert available.iloc[1] == pd.Timestamp("2023-03-08 20:30:00", tz="UTC")
    assert available.iloc[2] == pd.Timestamp("2025-11-19 20:30:00", tz="UTC")
    assert available.iloc[3] == pd.Timestamp("2025-12-15 20:30:00", tz="UTC")


def _silver(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.02 + np.sin(np.arange(rows) / 17.0)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.1,
        C.high: close + 0.3,
        C.low: close - 0.3,
        C.close: close,
        C.quality: "OK",
    })


def _cot(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2023-07-04", periods=rows, freq="7D")
    available = CftcCotPublicationPolicy().available_from_utc(pd.Series(dates))
    x = np.arange(rows, dtype=float)
    oi = 120_000.0 + x * 80.0 + np.sin(x / 5.0) * 2_000.0
    return pd.DataFrame({
        "report_date": dates,
        "available_from_utc": available,
        "open_interest": oi,
        "producer_long": 30_000 + x * 10,
        "producer_short": 70_000 + x * 12,
        "swap_long": 25_000 + x * 8,
        "swap_short": 20_000 + x * 7,
        "swap_spread": 6_000 + x * 2,
        "managed_long": 45_000 + x * 15,
        "managed_short": 18_000 + x * 9,
        "managed_spread": 12_000 + x * 3,
        "other_long": 14_000 + x * 4,
        "other_short": 10_000 + x * 3,
        "other_spread": 4_000 + x,
        "nonreportable_long": 20_000 + x * 2,
        "nonreportable_short": 15_000 + x * 2,
    })


def test_cot_report_is_not_visible_before_release_during_current_h1_bar() -> None:
    report_date = pd.Timestamp("2026-07-14")
    available = CftcCotPublicationPolicy().available_from_utc(pd.Series([report_date])).iloc[0]
    cot = _cot(60).iloc[[-1]].copy()
    cot.loc[:, "report_date"] = report_date
    cot.loc[:, "available_from_utc"] = available
    silver = pd.DataFrame({
        C.timestamp: pd.to_datetime([
            "2026-07-17 18:00:00+00:00",
            "2026-07-17 19:00:00+00:00",
            "2026-07-17 20:00:00+00:00",
        ]),
        C.open: [1000.0] * 3,
        C.high: [1001.0] * 3,
        C.low: [999.0] * 3,
        C.close: [1000.0] * 3,
        C.quality: ["OK"] * 3,
    })
    component = CftcSilverCotFeatures(cot, PublishedStateAligner(), C)
    out = component.transform(silver)
    assert out.loc[0, "cot_has_published_state"] == 0
    assert out.loc[1, "cot_has_published_state"] == 1
    assert out.loc[1, "cot_new_report_within_1h"] == 1
    assert np.isclose(out.loc[1, "cot_publication_age_hours"], 0.5)
    assert out.loc[2, "cot_new_report_within_1h"] == 0


def test_cot_features_are_causal_under_later_report_perturbation() -> None:
    silver = _silver()
    cot = _cot()
    decision_cutoff = pd.Timestamp("2025-01-25 12:00:00", tz="UTC")
    component = CftcSilverCotFeatures(cot, PublishedStateAligner(), C)
    baseline = component.transform(silver)

    changed_cot = cot.copy()
    later = pd.to_datetime(changed_cot["available_from_utc"], utc=True) > decision_cutoff
    numeric = [
        "open_interest", "producer_long", "producer_short", "swap_long", "swap_short",
        "swap_spread", "managed_long", "managed_short", "managed_spread", "other_long",
        "other_short", "other_spread", "nonreportable_long", "nonreportable_short",
    ]
    changed_cot.loc[later, numeric] *= 4.0
    changed_component = CftcSilverCotFeatures(changed_cot, PublishedStateAligner(), C)
    changed = changed_component.transform(silver)

    decisions = CompletedHourlyBarDecisionClock().available_at(silver[C.timestamp])
    rows = decisions <= decision_cutoff
    pd.testing.assert_frame_equal(
        baseline.loc[rows, component.feature_names].reset_index(drop=True),
        changed.loc[rows, changed_component.feature_names].reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=1e-14,
    )


def test_cot_feature_names_have_no_future_or_target_tokens() -> None:
    component = CftcSilverCotFeatures(_cot(), PublishedStateAligner(), C)
    lowered = [name.lower() for name in component.feature_names]
    assert not any("future" in name or "target" in name or "next_" in name for name in lowered)
