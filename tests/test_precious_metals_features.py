from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.precious_metals_features import PlatinumPalladiumCrossAssetFeatures


def _frame(start: str, closes: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open_usd_per_kg": close * 0.999,
        "high_usd_per_kg": close * 1.002,
        "low_usd_per_kg": close * 0.998,
        "close_usd_per_kg": close,
        "quality_flag": ["COMPLETE_SOURCE_HOUR"] * len(close),
    })


def _silver(start: str, closes: list[float]) -> pd.DataFrame:
    frame = _frame(start, closes)
    frame["minute_count"] = 60
    frame["source_provider"] = "test"
    frame["source_symbol"] = "XAGUSD"
    frame["market_type"] = "test"
    return frame


def test_precious_metals_features_are_exact_clock_and_causal() -> None:
    silver = _silver("2026-01-01", [100, 101, 103, 102, 104, 106])
    platinum = _frame("2026-01-01", [200, 202, 204, 205, 207, 210])
    palladium = _frame("2026-01-01", [300, 299, 301, 303, 302, 304])

    features = PlatinumPalladiumCrossAssetFeatures(
        platinum,
        palladium,
        ExactTimestampAligner(),
        ColumnConfig(),
        lags=(1, 3),
        windows=(3,),
    )
    out = features.transform(silver)

    assert "platinum_log_return_1h" in features.feature_names
    assert "palladium_silver_relative_return_3h" in features.feature_names
    assert "precious_metals_return_dispersion_1h" in features.feature_names
    assert "log_platinum_palladium_ratio" in features.feature_names

    i = 4
    expected_pt_ret = np.log(207.0 / 205.0)
    expected_pd_ret = np.log(302.0 / 303.0)
    expected_silver_ret = np.log(104.0 / 102.0)
    assert out.loc[i, "platinum_log_return_1h"] == pytest.approx(expected_pt_ret)
    assert out.loc[i, "palladium_log_return_1h"] == pytest.approx(expected_pd_ret)
    assert out.loc[i, "platinum_silver_relative_return_1h"] == pytest.approx(
        expected_silver_ret - expected_pt_ret
    )
    assert out.loc[i, "palladium_silver_relative_return_1h"] == pytest.approx(
        expected_silver_ret - expected_pd_ret
    )

    expected_dispersion = np.std([expected_silver_ret, expected_pt_ret, expected_pd_ret], ddof=0)
    assert out.loc[i, "precious_metals_return_dispersion_1h"] == pytest.approx(expected_dispersion)
    assert out.loc[i, "precious_metals_return_breadth_1h"] == pytest.approx(2.0 / 3.0)


def test_missing_exact_timestamp_is_not_forward_filled() -> None:
    silver = _silver("2026-01-01", [100, 101, 102, 103, 104])
    platinum = _frame("2026-01-01", [200, 201, 202, 203, 204]).drop(index=2).reset_index(drop=True)
    palladium = _frame("2026-01-01", [300, 301, 302, 303, 304])

    out = PlatinumPalladiumCrossAssetFeatures(
        platinum,
        palladium,
        ExactTimestampAligner(),
        ColumnConfig(),
        lags=(1,),
        windows=(3,),
    ).transform(silver)

    assert out.loc[2, "platinum_has_exact_current"] == 0
    assert pd.isna(out.loc[2, "log_platinum_silver_ratio"])
    assert out.loc[3, "platinum_has_exact_1h"] == 0
    assert pd.isna(out.loc[3, "platinum_log_return_1h"])
    assert out.loc[2, "palladium_has_exact_current"] == 1


def test_future_metal_rows_do_not_change_past_features() -> None:
    silver = _silver("2026-01-01", [100, 101, 102, 103, 104, 105])
    platinum = _frame("2026-01-01", [200, 201, 202, 203, 204, 205])
    palladium = _frame("2026-01-01", [300, 301, 302, 303, 304, 305])

    component = lambda pt, pd_: PlatinumPalladiumCrossAssetFeatures(
        pt,
        pd_,
        ExactTimestampAligner(),
        ColumnConfig(),
        lags=(1, 3),
        windows=(3,),
    )

    full = component(platinum, palladium).transform(silver)
    truncated = component(platinum.iloc[:4], palladium.iloc[:4]).transform(silver.iloc[:4].copy())

    for name in component(platinum, palladium).feature_names:
        left = full.loc[:3, name].reset_index(drop=True)
        right = truncated[name].reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False)


def test_invalid_duplicate_metal_timestamps_are_rejected() -> None:
    silver = _silver("2026-01-01", [100, 101, 102])
    platinum = _frame("2026-01-01", [200, 201, 202])
    palladium = _frame("2026-01-01", [300, 301, 302])
    platinum.loc[2, "timestamp_utc"] = platinum.loc[1, "timestamp_utc"]

    with pytest.raises(ValueError, match="Platinum timestamps must be unique"):
        PlatinumPalladiumCrossAssetFeatures(
            platinum,
            palladium,
            ExactTimestampAligner(),
            ColumnConfig(),
        )
