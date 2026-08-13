from __future__ import annotations

from datetime import datetime, timedelta, timezone


EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return value.astimezone(timezone.utc)


def bucket_start_for(timestamp_utc: datetime, interval_seconds: int) -> datetime:
    timestamp = ensure_utc(timestamp_utc)
    interval = int(interval_seconds)
    if interval <= 0:
        raise ValueError("interval_seconds must be positive.")
    elapsed = int((timestamp - EPOCH_UTC).total_seconds())
    bucket_offset = (elapsed // interval) * interval
    return EPOCH_UTC + timedelta(seconds=bucket_offset)


def latest_eligible_bucket_end(
    now_utc: datetime,
    *,
    interval_seconds: int,
    close_delay_seconds: int,
) -> datetime:
    now = ensure_utc(now_utc)
    delayed = now - timedelta(seconds=int(close_delay_seconds))
    return bucket_start_for(delayed, int(interval_seconds))
