from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.precious_metals.contracts import PALLADIUM, PLATINUM
from metal_predictor.precious_metals.dukascopy_source import DukascopyHistoricalMetalSource
from metal_predictor.precious_metals.features import (
    FEATURE_VERSION,
    PlatinumPalladiumCrossAssetFeatures,
)
from metal_predictor.price_normalization import TROY_OZ_PER_KG


C = ColumnConfig()


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_json(self, params):
        call = dict(params)
        self.calls.append(call)
        if call["path"] == "api/instrumentList":
            return [
                {"id": 101, "name": "XPT.CMD/USD", "nameLong": "Platinum", "pipValue": 0.01},
                {"id": 102, "name": "XPD.CMD/USD", "nameLong": "Palladium", "pipValue": 0.01},
            ]
        assert call["path"] == "api/historicalPrices"
        start_ms = int(call["start"])
        return [
            {"timestamp": start_ms, "open": 1000.0, "high": 1010.0, "low": 995.0, "close": 1005.0},
            {"timestamp": start_ms + 3_600_000, "open": 1005.0, "high": 1012.0, "low": 1001.0, "close": 1008.0},
        ]


def test_dukascopy_source_resolves_instrument_and_normalizes_usd_per_kg() -> None:
    transport = FakeTransport()
    source = DukascopyHistoricalMetalSource("not-a-real-secret", transport=transport)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)

    frame = source.fetch_hourly(PLATINUM, start, end)

    assert len(frame) == 2
    assert frame.loc[0, "timestamp_utc"] == start
    assert frame.loc[0, "open_usd_per_kg"] == pytest.approx(1000.0 * TROY_OZ_PER_KG)
    assert frame.loc[0, "source_provider"] == "Dukascopy"
    assert frame.loc[0, "source_symbol"] == "XPT.CMD/USD"
    historical = [call for call in transport.calls if call["path"] == "api/historicalPrices"]
    assert len(historical) == 1
    assert historical[0]["instrument"] == 101
    assert historical[0]["timeFrame"] == "1hour"
    assert historical[0]["dayStartTime"] == "UTC"
    assert historical[0]["offerSide"] == "B"
    assert historical[0]["count"] == 5000


def test_source_rejects_non_exact_requested_hours() -> None:
    source = DukascopyHistoricalMetalSource("research-key", transport=FakeTransport())
    with pytest.raises(ValueError, match="exact UTC hour"):
        source.fetch_hourly(
            PALLADIUM,
            datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        )


def _silver_frame(rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 2100.0 + np.arange(rows) * 0.02 + np.sin(np.arange(rows) / 13.0)
    return pd.DataFrame({
        C.timestamp: ts,
        C.open: close - 0.2,
        C.high: close + 0.5,
        C.low: close - 0.5,
        C.close: close,
        C.quality: "OK",
    })


def _metal_frame(asset: str, rows: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    if asset == "XPT":
        close = 31000.0 + np.arange(rows) * 0.8 + np.sin(np.arange(rows) / 9.0) * 8.0
    else:
        close = 33000.0 + np.arange(rows) * 0.6 + np.cos(np.arange(rows) / 10.0) * 12.0
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open_usd_per_kg": close - 2.0,
        "high_usd_per_kg": close + 4.0,
        "low_usd_per_kg": close - 4.0,
        "close_usd_per_kg": close,
        "quality_flag": "PROVIDER_H1_BID",
        "source_provider": "Dukascopy",
        "source_symbol": "XPT.CMD/USD" if asset == "XPT" else "XPD.CMD/USD",
    })


def test_feature_family_is_pre_registered_and_fixed() -> None:
    component = PlatinumPalladiumCrossAssetFeatures(
        _metal_frame("XPT"), _metal_frame("XPD"), ExactTimestampAligner(), C
    )
    assert component.feature_version == FEATURE_VERSION
    assert len(component.feature_names) == 43
    assert len(set(component.feature_names)) == 43
    assert "xpt_log_return_1h" in component.feature_names
    assert "xpd_silver_corr_72h" in component.feature_names
    assert "metal_complex_breadth_1h" in component.feature_names

    with pytest.raises(ValueError, match="pre-registered"):
        PlatinumPalladiumCrossAssetFeatures(
            _metal_frame("XPT"),
            _metal_frame("XPD"),
            ExactTimestampAligner(),
            C,
            return_lags=(1, 3, 6),
        )


def test_precious_metals_features_are_causal_under_future_perturbation() -> None:
    silver = _silver_frame()
    platinum = _metal_frame("XPT")
    palladium = _metal_frame("XPD")
    cutoff = 300

    baseline_component = PlatinumPalladiumCrossAssetFeatures(
        platinum, palladium, ExactTimestampAligner(), C
    )
    baseline = baseline_component.transform(silver)

    changed_platinum = platinum.copy()
    changed_palladium = palladium.copy()
    price_cols = ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]
    changed_platinum.loc[cutoff + 1 :, price_cols] *= 3.0
    changed_palladium.loc[cutoff + 1 :, price_cols] *= 4.0
    changed_component = PlatinumPalladiumCrossAssetFeatures(
        changed_platinum, changed_palladium, ExactTimestampAligner(), C
    )
    changed = changed_component.transform(silver)

    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff, baseline_component.feature_names],
        changed.loc[:cutoff, changed_component.feature_names],
        check_dtype=False,
        rtol=0.0,
        atol=1e-14,
    )


def test_missing_auxiliary_hour_remains_explicitly_missing() -> None:
    silver = _silver_frame(100)
    platinum = _metal_frame("XPT", 100).drop(index=[40]).reset_index(drop=True)
    palladium = _metal_frame("XPD", 100)
    component = PlatinumPalladiumCrossAssetFeatures(
        platinum, palladium, ExactTimestampAligner(), C
    )
    out = component.transform(silver)

    assert out.loc[40, "xpt_has_exact_current"] == 0
    assert np.isnan(out.loc[40, "log_xpt_silver_ratio"])
    assert np.isnan(out.loc[40, "xpt_log_return_1h"])
    assert out.loc[40, "xpd_has_exact_current"] == 1
    assert out.loc[40, "both_metals_have_exact_current"] == 0


def test_no_execution_or_holdout_concepts_exist_in_research_components() -> None:
    source = DukascopyHistoricalMetalSource("research-key", transport=FakeTransport())
    assert not hasattr(source, "place_order")
    assert not hasattr(source, "cancel_order")
    assert not hasattr(source, "submit_order")
