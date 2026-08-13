from datetime import datetime, timedelta, timezone

from metal_predictor.forward_bars.contracts import ForwardBar
from metal_predictor.forward_bars.forecasting import MultiHorizonBaselineForecastService
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository


def test_later_gap_blocks_older_admitted_bar(tmp_path):
    utc = timezone.utc
    start = datetime(2026, 8, 13, 0, tzinfo=utc)
    end = start + timedelta(hours=4)
    bar = ForwardBar(
        horizon_key="4h",
        interval_seconds=14_400,
        bucket_start_utc=start,
        bucket_end_utc=end,
        source_provider="BullionVault",
        security_id="AGXLN",
        currency="USD",
        open_mid_usd_per_kg=2100.0,
        high_mid_usd_per_kg=2110.0,
        low_mid_usd_per_kg=2090.0,
        close_mid_usd_per_kg=2105.0,
        open_bid_usd_per_kg=2099.0,
        close_bid_usd_per_kg=2104.0,
        open_ask_usd_per_kg=2101.0,
        close_ask_usd_per_kg=2106.0,
        mean_spread_usd_per_kg=2.0,
        max_spread_usd_per_kg=3.0,
        close_spread_usd_per_kg=2.0,
        snapshot_count=238,
        expected_snapshot_count=240,
        coverage_ratio=238 / 240,
        first_sample_at_utc=start + timedelta(minutes=1),
        last_sample_at_utc=end - timedelta(minutes=1),
        access_mode_counts={"AUTHENTICATED_READ_ONLY": 238},
        freshness_status_counts={"CURRENT_GUI_SOURCE": 238},
        quality_status="HIGH_COVERAGE_AUTHENTICATED",
    )
    repository = SQLiteForwardBarRepository(tmp_path / "forward.sqlite3")
    assert repository.append_bar(bar) is True
    assert repository.append_gap(
        horizon_key="4h",
        interval_seconds=14_400,
        bucket_start_utc=end,
        bucket_end_utc=end + timedelta(hours=4),
        reason="NO_OBSERVED_SNAPSHOTS",
        snapshot_count=0,
    ) is True

    payload = MultiHorizonBaselineForecastService(repository).forecast("4h")
    assert payload["forecast_available"] is False
    assert payload["reason"] == "LATEST_ASSESSED_BUCKET_IS_EXPLICIT_GAP"
    assert "forecast" not in payload
