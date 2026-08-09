from __future__ import annotations

import numpy as np
import pandas as pd

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.dxy_features import DollarIndexCrossAssetFeatures
from metal_predictor.price_normalization import IdentityIndexNormalizer, PreciousMetalUsdKgNormalizer, TROY_OZ_PER_KG


C = ColumnConfig()


def test_price_normalization_strategy_keeps_index_points_and_converts_metals() -> None:
    source = pd.DataFrame({
        "open_source": [100.0], "high_source": [101.0],
        "low_source": [99.0], "close_source": [100.5],
    })
    index = IdentityIndexNormalizer().normalize(source)
    metal = PreciousMetalUsdKgNormalizer().normalize(source)
    assert index.loc[0, "close_value"] == 100.5
    assert "close_usd_per_kg" not in index.columns
    assert np.isclose(metal.loc[0, "close_usd_per_kg"], 100.5 * TROY_OZ_PER_KG)
    assert metal.loc[0, "close_value"] == metal.loc[0, "close_usd_per_kg"]


def _silver(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 1000.0 + np.arange(rows) * 0.04 + np.sin(np.arange(rows) / 17.0)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.1,
        C.high: close + 0.3,
        C.low: close - 0.3,
        C.close: close,
        C.quality: "OK",
    })


def _dxy(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 103.0 + np.arange(rows) * 0.0005 + np.sin(np.arange(rows) / 14.0) * 0.08
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open_value": close - 0.01,
        "high_value": close + 0.03,
        "low_value": close - 0.03,
        "close_value": close,
        "quality_flag": "OK",
    })


def test_dxy_cross_asset_features_are_causal_under_future_perturbation() -> None:
    silver = _silver()
    dxy = _dxy()
    cutoff = 300
    base_component = DollarIndexCrossAssetFeatures(dxy, ExactTimestampAligner(), C)
    base = base_component.transform(silver)

    changed_dxy = dxy.copy()
    value_cols = ["open_value", "high_value", "low_value", "close_value"]
    changed_dxy.loc[cutoff + 1 :, value_cols] *= 1.5
    changed_component = DollarIndexCrossAssetFeatures(changed_dxy, ExactTimestampAligner(), C)
    changed = changed_component.transform(silver)

    pd.testing.assert_frame_equal(
        base.loc[:cutoff, base_component.feature_names],
        changed.loc[:cutoff, changed_component.feature_names],
        check_dtype=False,
        rtol=0.0,
        atol=1e-14,
    )


def test_missing_dxy_bar_remains_missing_without_asof_fill() -> None:
    silver = _silver(80)
    dxy = _dxy(80).drop(index=[30]).reset_index(drop=True)
    component = DollarIndexCrossAssetFeatures(
        dxy, ExactTimestampAligner(), C, lags=(1, 3), windows=(24,)
    )
    out = component.transform(silver)
    assert out.loc[30, "dxy_has_exact_current"] == 0
    assert np.isnan(out.loc[30, "dxy_log_return_1h"])
    assert np.isnan(out.loc[30, "dxy_candle_body_pct"])


def test_dxy_feature_names_do_not_contain_future_or_target_tokens() -> None:
    component = DollarIndexCrossAssetFeatures(_dxy(100), ExactTimestampAligner(), C, lags=(1, 3), windows=(24,))
    lowered = [name.lower() for name in component.feature_names]
    assert not any("future" in name or "target" in name or "next_" in name for name in lowered)
