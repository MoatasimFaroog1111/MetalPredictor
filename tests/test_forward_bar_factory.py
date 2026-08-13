from datetime import datetime, timedelta, timezone

import pytest

from metal_predictor.forward_bars.alignment import bucket_start_for
from metal_predictor.forward_bars.contracts import QuoteSample
from metal_predictor.forward_bars.factory import ForwardBarFactory
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository

UTC = timezone.utc

class Source:
    def __init__(self, samples): self.samples = samples
    def first_sample_at(self, **_): return self.samples[0].captured_at_utc
    def samples_between(self, start_utc, end_utc, **_):
        return [x for x in self.samples if start_utc <= x.captured_at_utc < end_utc]

def q(at, bid, ask):
    return QuoteSample("BullionVault", "AGXLN", "USD", at, "AUTHENTICATED_READ_ONLY", "CURRENT_MARKET_VIEW", bid, ask)

def test_alignment_and_observed_only_bar(tmp_path):
    start = datetime(2026, 8, 13, 8, tzinfo=UTC)
    assert bucket_start_for(datetime(2026, 8, 13, 15, 33, tzinfo=UTC), 14_400) == datetime(2026, 8, 13, 12, tzinfo=UTC)
    samples = [q(start + timedelta(minutes=1), 2100, 2102), q(start + timedelta(hours=1), 2110, 2113), q(start + timedelta(hours=3, minutes=59), 2105, 2107)]
    repo = SQLiteForwardBarRepository(tmp_path / "bars.sqlite3")
    factory = ForwardBarFactory(Source(samples), repo, security_id="AGXLN", currency="USD", source_cadence_seconds=60, close_delay_seconds=120)
    assert factory.materialize_horizon("4h", now_utc=start + timedelta(hours=4, minutes=3))["materialized_bars"] == 1
    bar = repo.latest_bar("4h")
    assert bar is not None
    assert bar.snapshot_count == 3 and bar.expected_snapshot_count == 240
    assert bar.coverage_ratio == pytest.approx(3 / 240)
    assert bar.open_mid_usd_per_kg == pytest.approx(2101)
    assert bar.high_mid_usd_per_kg == pytest.approx(2111.5)
    assert bar.close_mid_usd_per_kg == pytest.approx(2106)
    assert bar.quality_status == "SPARSE_COVERAGE_AUTHENTICATED"
