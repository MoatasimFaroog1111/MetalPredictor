from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.core import ColumnConfig
from metal_predictor.decision_time import CompletedHourlyBarDecisionClock
from metal_predictor.published_state import PublishedStateAligner
from metal_predictor.vix_features import VixDailyFeatures
from metal_predictor.vix_publication import VixDailyClosePublicationPolicy


C = ColumnConfig()


def test_vix_daily_close_publication_is_dst_aware() -> None:
    policy = VixDailyClosePublicationPolicy()
    dates = pd.Series(pd.to_datetime(["2026-07-27", "2026-01-05"]))
    available = policy.available_from_utc(dates)
    assert available.iloc[0] == pd.Timestamp("2026-07-27 20:15:00", tz="UTC")
    assert available.iloc[1] == pd.Timestamp("2026-01-05 21:15:00", tz="UTC")


def _silver(rows: int = 1000) -> pd.DataFrame:
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


def _vix(days: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2024-10-01", periods=days)
    x = np.arange(days, dtype=float)
    close = 18.0 + np.sin(x / 8.0) * 2.0 + np.cos(x / 19.0)
    open_value = close + np.sin(x / 5.0) * 0.3
    high = np.maximum(open_value, close) + 0.8
    low = np.minimum(open_value, close) - 0.8
    return pd.DataFrame({
        "observation_date": dates,
        "available_from_utc": VixDailyClosePublicationPolicy().available_from_utc(
            pd.Series(dates)
        ),
        "vix_open": open_value,
        "vix_high": high,
        "vix_low": low,
        "vix_close": close,
    })


def test_vix_close_released_during_current_bar_is_only_visible_at_bar_completion() -> None:
    vix = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-07-28"]),
        "available_from_utc": [pd.Timestamp("2026-07-28 20:15:00", tz="UTC")],
        "vix_open": [17.0],
        "vix_high": [18.5],
        "vix_low": [16.5],
        "vix_close": [18.0],
    })
    silver = pd.DataFrame({
        C.timestamp: pd.to_datetime([
            "2026-07-28 19:00:00+00:00",
            "2026-07-28 20:00:00+00:00",
            "2026-07-28 21:00:00+00:00",
        ]),
        C.open: [1000.0] * 3,
        C.high: [1001.0] * 3,
        C.low: [999.0] * 3,
        C.close: [1000.0] * 3,
        C.quality: ["OK"] * 3,
    })
    out = VixDailyFeatures(vix, PublishedStateAligner(), C).transform(silver)
    assert out.loc[0, "vix_has_published_state"] == 0
    assert out.loc[1, "vix_has_published_state"] == 1
    assert out.loc[1, "vix_new_close_within_1h"] == 1
    assert np.isclose(out.loc[1, "vix_publication_age_hours"], 0.75)
    assert out.loc[1, "vix_close"] == 18.0
    assert out.loc[2, "vix_new_close_within_1h"] == 0


def test_vix_features_are_causal_under_later_release_perturbation() -> None:
    silver = _silver()
    vix = _vix(120)
    decision_cutoff = pd.Timestamp("2025-01-20 18:00:00", tz="UTC")
    component = VixDailyFeatures(vix, PublishedStateAligner(), C)
    baseline = component.transform(silver)

    changed_vix = vix.copy()
    later = pd.to_datetime(changed_vix["available_from_utc"], utc=True) > decision_cutoff
    price_cols = ["vix_open", "vix_high", "vix_low", "vix_close"]
    changed_vix.loc[later, price_cols] *= 4.0
    changed_component = VixDailyFeatures(changed_vix, PublishedStateAligner(), C)
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


def test_vix_state_persists_after_release_without_preclose_backfill() -> None:
    vix = _vix(80)
    release = pd.Timestamp(vix.loc[70, "available_from_utc"])
    before = release - pd.Timedelta(minutes=30)
    after = release + pd.Timedelta(hours=20)
    aligned = PublishedStateAligner().align(
        pd.Series([before, release, after]),
        vix,
        ("vix_close",),
    )
    assert aligned.loc[0, "vix_close"] != vix.loc[70, "vix_close"]
    assert aligned.loc[1, "vix_close"] == vix.loc[70, "vix_close"]
    assert aligned.loc[2, "vix_close"] == vix.loc[70, "vix_close"]


def test_vix_feature_names_contain_no_future_or_target_tokens() -> None:
    component = VixDailyFeatures(_vix(), PublishedStateAligner(), C)
    names = [name.lower() for name in component.feature_names]
    assert not any("future" in name or "target" in name or "next_" in name for name in names)
