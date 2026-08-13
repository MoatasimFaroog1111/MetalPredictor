from __future__ import annotations

from datetime import datetime

from metal_predictor.forward_bars.contracts import ForwardBar


def bar_from_payload(payload: dict[str, object]) -> ForwardBar:
    access_raw = payload["access_mode_counts"]
    freshness_raw = payload["freshness_status_counts"]
    if not isinstance(access_raw, dict) or not isinstance(freshness_raw, dict):
        raise ValueError("Stored forward-bar provenance maps are malformed.")
    return ForwardBar(
        horizon_key=str(payload["horizon_key"]),
        interval_seconds=int(payload["interval_seconds"]),
        bucket_start_utc=datetime.fromisoformat(str(payload["bucket_start_utc"])),
        bucket_end_utc=datetime.fromisoformat(str(payload["bucket_end_utc"])),
        source_provider=str(payload["source_provider"]),
        security_id=str(payload["security_id"]),
        currency=str(payload["currency"]),
        open_mid_usd_per_kg=float(payload["open_mid_usd_per_kg"]),
        high_mid_usd_per_kg=float(payload["high_mid_usd_per_kg"]),
        low_mid_usd_per_kg=float(payload["low_mid_usd_per_kg"]),
        close_mid_usd_per_kg=float(payload["close_mid_usd_per_kg"]),
        open_bid_usd_per_kg=float(payload["open_bid_usd_per_kg"]),
        close_bid_usd_per_kg=float(payload["close_bid_usd_per_kg"]),
        open_ask_usd_per_kg=float(payload["open_ask_usd_per_kg"]),
        close_ask_usd_per_kg=float(payload["close_ask_usd_per_kg"]),
        mean_spread_usd_per_kg=float(payload["mean_spread_usd_per_kg"]),
        max_spread_usd_per_kg=float(payload["max_spread_usd_per_kg"]),
        close_spread_usd_per_kg=float(payload["close_spread_usd_per_kg"]),
        snapshot_count=int(payload["snapshot_count"]),
        expected_snapshot_count=int(payload["expected_snapshot_count"]),
        coverage_ratio=float(payload["coverage_ratio"]),
        first_sample_at_utc=datetime.fromisoformat(str(payload["first_sample_at_utc"])),
        last_sample_at_utc=datetime.fromisoformat(str(payload["last_sample_at_utc"])),
        access_mode_counts={str(k): int(v) for k, v in access_raw.items()},
        freshness_status_counts={str(k): int(v) for k, v in freshness_raw.items()},
        quality_status=str(payload["quality_status"]),
        source_stream=str(payload["source_stream"]),
        bar_version=str(payload["bar_version"]),
    )
