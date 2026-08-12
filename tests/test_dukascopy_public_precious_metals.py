from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from metal_predictor.precious_metals.contracts import PALLADIUM, PLATINUM
from metal_predictor.precious_metals.dukascopy_public_source import (
    DukascopyCompressedH1Decoder,
    DukascopyPublicH1UrlPlanner,
    DukascopyPublicHistoricalMetalSource,
)
from metal_predictor.price_normalization import TROY_OZ_PER_KG


UTC = timezone.utc


class FakeTransport:
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get_json(self, url: str) -> dict[str, object]:
        self.calls.append(url)
        try:
            return self.payloads[url]
        except KeyError:
            raise AssertionError(f"Unexpected URL: {url}") from None


def payload(
    *,
    timestamp_ms: int,
    times: list[int],
    opens: list[int] | None = None,
    highs: list[int] | None = None,
    lows: list[int] | None = None,
    closes: list[int] | None = None,
) -> dict[str, object]:
    length = len(times)
    return {
        "timestamp": timestamp_ms,
        "multiplier": 0.01,
        "open": 1000.00,
        "high": 1010.00,
        "low": 990.00,
        "close": 1005.00,
        "shift": 3_600_000,
        "times": times,
        "opens": opens if opens is not None else [0] * length,
        "highs": highs if highs is not None else [0] * length,
        "lows": lows if lows is not None else [0] * length,
        "closes": closes if closes is not None else [0] * length,
        "volumes": [1.0] * length,
    }


def test_planner_uses_completed_month_and_active_month_without_credentials() -> None:
    planner = DukascopyPublicH1UrlPlanner()
    buckets = planner.plan(
        PLATINUM,
        datetime(2026, 7, 1, tzinfo=UTC),
        datetime(2026, 8, 10, tzinfo=UTC),
        now_utc=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
    )

    assert len(buckets) == 2
    assert buckets[0].url == (
        "https://jetta.dukascopy.com/v1/candles/hour/XPT.CMD-USD/BID/2026/7"
    )
    assert buckets[0].active_bucket is False
    assert buckets[1].url == (
        "https://jetta.dukascopy.com/v1/candles/hour/XPT.CMD-USD/BID"
        "?from=1785542400000"
    )
    assert buckets[1].active_bucket is True
    assert "key=" not in buckets[0].url.lower()
    assert "key=" not in buckets[1].url.lower()


def test_planner_clamps_to_provider_h1_start_dates() -> None:
    planner = DukascopyPublicH1UrlPlanner()
    pt = planner.plan(
        PLATINUM,
        datetime(2021, 8, 1, tzinfo=UTC),
        datetime(2021, 12, 1, tzinfo=UTC),
        now_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    pdm = planner.plan(
        PALLADIUM,
        datetime(2021, 6, 1, tzinfo=UTC),
        datetime(2021, 8, 1, tzinfo=UTC),
        now_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert pt[0].first_hour_utc == datetime(2021, 11, 1, 0, 0, tzinfo=UTC)
    assert pdm[0].first_hour_utc == datetime(2021, 7, 4, 22, 0, tzinfo=UTC)
    assert "/2021/11" in pt[0].url
    assert "/2021/7" in pdm[0].url


def test_decoder_preserves_real_gaps_instead_of_creating_flat_candles() -> None:
    decoder = DukascopyCompressedH1Decoder()
    base = datetime(2025, 1, 1, tzinfo=UTC)
    rows = decoder.decode(
        payload(
            timestamp_ms=int(base.timestamp() * 1000),
            times=[0, 1, 2],
            opens=[0, 100, 100],
            highs=[0, 100, 100],
            lows=[0, 100, 100],
            closes=[0, 100, 100],
        )
    )

    assert [row[0] for row in rows] == [
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 1, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 3, 0, tzinfo=UTC),
    ]
    assert len(rows) == 3


def test_decoder_rejects_non_h1_shift_and_bad_column_lengths() -> None:
    decoder = DukascopyCompressedH1Decoder()
    bad_shift = payload(timestamp_ms=0, times=[0])
    bad_shift["shift"] = 60_000
    with pytest.raises(RuntimeError, match="expected exact H1 shift"):
        decoder.decode(bad_shift)

    bad_lengths = payload(timestamp_ms=0, times=[0, 1])
    bad_lengths["opens"] = [0]
    with pytest.raises(RuntimeError, match="mismatched lengths"):
        decoder.decode(bad_lengths)


def test_source_decodes_bid_ohlc_filters_window_and_normalizes_to_usd_per_kg() -> None:
    planner = DukascopyPublicH1UrlPlanner()
    start = datetime(2025, 1, 1, 1, 0, tzinfo=UTC)
    end = datetime(2025, 1, 1, 3, 0, tzinfo=UTC)
    buckets = planner.plan(
        PLATINUM,
        start,
        end,
        now_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(buckets) == 1
    url = buckets[0].url
    month_start = datetime(2025, 1, 1, tzinfo=UTC)
    transport = FakeTransport(
        {
            url: payload(
                timestamp_ms=int(month_start.timestamp() * 1000),
                times=[0, 1, 1, 1, 1],
                opens=[0, 100, 100, 100, 100],
                highs=[0, 100, 100, 100, 100],
                lows=[0, 100, 100, 100, 100],
                closes=[0, 100, 100, 100, 100],
            )
        }
    )
    source = DukascopyPublicHistoricalMetalSource(transport=transport, planner=planner)

    frame = source.fetch_hourly(PLATINUM, start, end)

    assert transport.calls == [url]
    assert list(pd.to_datetime(frame["timestamp_utc"], utc=True)) == list(
        pd.date_range(start, end, freq="h", tz="UTC")
    )
    assert frame.loc[0, "open_usd_per_oz"] == pytest.approx(1001.0)
    assert frame.loc[0, "open_usd_per_kg"] == pytest.approx(1001.0 * TROY_OZ_PER_KG)
    assert frame["source_provider"].eq("Dukascopy Public Historical Feed").all()
    assert frame["source_symbol"].eq("XPT.CMD/USD").all()
    assert frame["market_type"].eq("commodity_cfd_cross_feed").all()
    assert frame["quality_flag"].eq("PROVIDER_H1_BID").all()


def test_source_surface_has_no_execution_methods() -> None:
    source = DukascopyPublicHistoricalMetalSource(transport=FakeTransport({}))
    for name in ("place_order", "cancel_order", "submit_order", "execute", "buy", "sell"):
        assert not hasattr(source, name)
