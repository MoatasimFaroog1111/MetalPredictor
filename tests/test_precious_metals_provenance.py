from __future__ import annotations

import pytest

from metal_predictor.precious_metals.provenance import (
    HistoricalBootstrapManifest,
    HistoricalBootstrapProvenanceGate,
)


def _fixed_manifest(asset: str) -> HistoricalBootstrapManifest:
    symbol = "PL=F" if asset == "XPT" else "PA=F"
    path = "data/PTXLN_hourly.pkl" if asset == "XPT" else "data/PDXLN_hourly.pkl"
    return HistoricalBootstrapManifest(
        asset=asset,
        provider="Yahoo Finance via yfinance",
        source_symbol=symbol,
        instrument_semantics="FUTURES",
        interval="1h",
        requested_history_days=729,
        has_open=False,
        has_high=True,
        has_low=True,
        has_close=True,
        exact_utc_hour_guaranteed=False,
        forward_fill_used=False,
        interpolation_used=False,
        provenance_manifest_embedded=False,
        source_repository="MoatasimFaroog1111/fixed",
        source_commit="672845b854cbfae2284f68df28369dd6f3de94bb",
        source_path=path,
    )


def test_fixed_hourly_platinum_and_palladium_fail_closed() -> None:
    gate = HistoricalBootstrapProvenanceGate()

    for asset in ("XPT", "XPD"):
        result = gate.assess(_fixed_manifest(asset))
        assert result.status == "REJECT"
        assert result.accepted is False
        assert set(result.failures) == {
            "INSTRUMENT_SEMANTICS_NOT_ALLOWED",
            "INSUFFICIENT_HOURLY_HISTORY",
            "INCOMPLETE_OHLC_SCHEMA",
            "EXACT_UTC_HOUR_NOT_GUARANTEED",
            "PROVENANCE_MANIFEST_MISSING",
        }


def test_gate_accepts_a_five_year_exact_hour_cross_feed_manifest() -> None:
    manifest = HistoricalBootstrapManifest(
        asset="XPT",
        provider="research-provider",
        source_symbol="XPT/USD",
        instrument_semantics="COMMODITY_CFD_CROSS_FEED",
        interval="1h",
        requested_history_days=1826,
        has_open=True,
        has_high=True,
        has_low=True,
        has_close=True,
        exact_utc_hour_guaranteed=True,
        forward_fill_used=False,
        interpolation_used=False,
        provenance_manifest_embedded=True,
    )

    result = HistoricalBootstrapProvenanceGate().assess(manifest)
    assert result.status == "PASS"
    assert result.accepted is True
    assert result.failures == ()


def test_validate_many_rejects_before_any_model_fit() -> None:
    gate = HistoricalBootstrapProvenanceGate()
    with pytest.raises(ValueError, match="Historical bootstrap provenance gate rejected"):
        gate.validate_many((_fixed_manifest("XPT"), _fixed_manifest("XPD")))
