from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import math

from metal_predictor.forward_bars.alignment import bucket_start_for, latest_eligible_bucket_end
from metal_predictor.forward_bars.contracts import (
    FORWARD_HORIZON_SECONDS,
    ForwardBar,
    ForwardBarRepository,
    QuoteSample,
    QuoteSampleSource,
)


class ForwardBarFactory:
    """Materialize immutable UTC-aligned bars from observed BullionVault snapshots only."""

    def __init__(
        self,
        sample_source: QuoteSampleSource,
        repository: ForwardBarRepository,
        *,
        security_id: str,
        currency: str,
        source_cadence_seconds: int,
        close_delay_seconds: int = 120,
        max_buckets_per_cycle: int = 512,
    ) -> None:
        cadence = int(source_cadence_seconds)
        delay = int(close_delay_seconds)
        max_buckets = int(max_buckets_per_cycle)
        if cadence <= 0:
            raise ValueError("source_cadence_seconds must be positive.")
        if not 30 <= delay <= 3600:
            raise ValueError("close_delay_seconds must be between 30 and 3600.")
        if not 1 <= max_buckets <= 10_000:
            raise ValueError("max_buckets_per_cycle must be between 1 and 10000.")
        self._source = sample_source
        self._repository = repository
        self._security_id = security_id.strip().upper()
        self._currency = currency.strip().upper()
        self._source_cadence_seconds = cadence
        self._close_delay_seconds = delay
        self._max_buckets_per_cycle = max_buckets

    @property
    def close_delay_seconds(self) -> int:
        return self._close_delay_seconds

    @property
    def source_cadence_seconds(self) -> int:
        return self._source_cadence_seconds

    def materialize_all(self, now_utc: datetime | None = None) -> dict[str, object]:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        results = {
            key: self.materialize_horizon(key, now_utc=now)
            for key in FORWARD_HORIZON_SECONDS
        }
        return {
            "component": "bullionvault-forward-multi-horizon-data-factory",
            "bar_version": "bullionvault-forward-multi-horizon-v1",
            "source_stream": "BULLIONVAULT_READ_ONLY_MICROSTRUCTURE_SNAPSHOTS",
            "run_at_utc": now.isoformat(),
            "horizons": results,
            "safety": {
                "edge_status": "NOT_PROVEN",
                "research_only": True,
                "buy_sell_enabled": False,
                "execution_enabled": False,
                "live_model_mutated": False,
                "frozen_52_feature_graph_mutated": False,
                "shadow62_mutated": False,
            },
        }

    def materialize_horizon(
        self,
        horizon_key: str,
        *,
        now_utc: datetime | None = None,
    ) -> dict[str, object]:
        if horizon_key not in FORWARD_HORIZON_SECONDS:
            raise ValueError(f"Unsupported forward-bar horizon {horizon_key!r}.")
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        interval = FORWARD_HORIZON_SECONDS[horizon_key]
        first_sample = self._source.first_sample_at(
            security_id=self._security_id,
            currency=self._currency,
        )
        if first_sample is None:
            return {
                "horizon_key": horizon_key,
                "state": "WAITING_FOR_FIRST_OBSERVED_SNAPSHOT",
                "materialized_bars": 0,
                "recorded_gaps": 0,
            }

        latest_end = self._repository.latest_assessed_end(horizon_key)
        start = latest_end or bucket_start_for(first_sample, interval)
        eligible_end = latest_eligible_bucket_end(
            now,
            interval_seconds=interval,
            close_delay_seconds=self._close_delay_seconds,
        )

        materialized = 0
        gaps = 0
        assessed = 0
        while start + timedelta(seconds=interval) <= eligible_end:
            if assessed >= self._max_buckets_per_cycle:
                break
            end = start + timedelta(seconds=interval)
            if self._repository.has_assessment(horizon_key, start):
                start = end
                assessed += 1
                continue
            samples = list(
                self._source.samples_between(
                    start,
                    end,
                    security_id=self._security_id,
                    currency=self._currency,
                )
            )
            if len(samples) < 2:
                created = self._repository.append_gap(
                    horizon_key=horizon_key,
                    interval_seconds=interval,
                    bucket_start_utc=start,
                    bucket_end_utc=end,
                    reason=(
                        "NO_OBSERVED_SNAPSHOTS"
                        if not samples
                        else "INSUFFICIENT_OBSERVED_SNAPSHOTS"
                    ),
                    snapshot_count=len(samples),
                )
                if created:
                    gaps += 1
            else:
                bar = self._build_bar(
                    horizon_key=horizon_key,
                    interval_seconds=interval,
                    bucket_start_utc=start,
                    bucket_end_utc=end,
                    samples=samples,
                )
                if self._repository.append_bar(bar):
                    materialized += 1
            assessed += 1
            start = end

        return {
            "horizon_key": horizon_key,
            "state": "ACTIVE",
            "materialized_bars": materialized,
            "recorded_gaps": gaps,
            "assessed_buckets_this_cycle": assessed,
            "latest_eligible_bucket_end_utc": eligible_end.isoformat(),
            "catch_up_limited": assessed >= self._max_buckets_per_cycle,
        }

    def _build_bar(
        self,
        *,
        horizon_key: str,
        interval_seconds: int,
        bucket_start_utc: datetime,
        bucket_end_utc: datetime,
        samples: list[QuoteSample],
    ) -> ForwardBar:
        ordered = sorted(samples, key=lambda item: item.captured_at_utc)
        for sample in ordered:
            if sample.security_id.strip().upper() != self._security_id:
                raise ValueError("Forward-bar source security changed inside a bucket.")
            if sample.currency.strip().upper() != self._currency:
                raise ValueError("Forward-bar source currency changed inside a bucket.")
            if not (
                bucket_start_utc <= sample.captured_at_utc.astimezone(timezone.utc)
                < bucket_end_utc
            ):
                raise ValueError("Forward-bar source returned a sample outside the requested bucket.")

        providers = {sample.source_provider for sample in ordered}
        if len(providers) != 1:
            raise ValueError("Forward-bar source provider changed inside a bucket.")

        mids = [sample.mid_usd_per_kg for sample in ordered]
        spreads = [sample.spread_usd_per_kg for sample in ordered]
        expected = max(1, int(math.ceil(interval_seconds / self._source_cadence_seconds)))
        coverage = min(1.0, len(ordered) / expected)
        access_counts = Counter(sample.access_mode for sample in ordered)
        freshness_counts = Counter(sample.freshness_status for sample in ordered)
        quality = self._quality_status(coverage, access_counts)

        return ForwardBar(
            horizon_key=horizon_key,
            interval_seconds=interval_seconds,
            bucket_start_utc=bucket_start_utc,
            bucket_end_utc=bucket_end_utc,
            source_provider=ordered[0].source_provider,
            security_id=self._security_id,
            currency=self._currency,
            open_mid_usd_per_kg=mids[0],
            high_mid_usd_per_kg=max(mids),
            low_mid_usd_per_kg=min(mids),
            close_mid_usd_per_kg=mids[-1],
            open_bid_usd_per_kg=ordered[0].best_bid_usd_per_kg,
            close_bid_usd_per_kg=ordered[-1].best_bid_usd_per_kg,
            open_ask_usd_per_kg=ordered[0].best_ask_usd_per_kg,
            close_ask_usd_per_kg=ordered[-1].best_ask_usd_per_kg,
            mean_spread_usd_per_kg=sum(spreads) / len(spreads),
            max_spread_usd_per_kg=max(spreads),
            close_spread_usd_per_kg=spreads[-1],
            snapshot_count=len(ordered),
            expected_snapshot_count=expected,
            coverage_ratio=coverage,
            first_sample_at_utc=ordered[0].captured_at_utc.astimezone(timezone.utc),
            last_sample_at_utc=ordered[-1].captured_at_utc.astimezone(timezone.utc),
            access_mode_counts=dict(access_counts),
            freshness_status_counts=dict(freshness_counts),
            quality_status=quality,
        )

    @staticmethod
    def _quality_status(
        coverage_ratio: float,
        access_mode_counts: Counter[str],
    ) -> str:
        authenticated_only = bool(access_mode_counts) and all(
            "AUTHENTICATED" in key.upper() for key in access_mode_counts
        )
        if coverage_ratio >= 0.90:
            base = "HIGH_COVERAGE"
        elif coverage_ratio >= 0.50:
            base = "PARTIAL_COVERAGE"
        else:
            base = "SPARSE_COVERAGE"
        return f"{base}_{'AUTHENTICATED' if authenticated_only else 'PROVENANCE_MIXED_OR_PUBLIC'}"
