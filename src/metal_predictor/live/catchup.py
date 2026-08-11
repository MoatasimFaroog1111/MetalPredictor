from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from metal_predictor.live.contracts import ForecastRepository, MarketBarBackfillSource
from metal_predictor.live.inference import LiveForecastOrchestrator, LivePredictionEngine


@dataclass(frozen=True)
class LiveCatchUpResult:
    requested_start_utc: datetime | None
    requested_end_utc: datetime
    fetched_bars: int
    created_bars: int
    forecast_created: bool
    forecast_timestamp_utc: datetime | None
    status: str

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("requested_start_utc", "requested_end_utc", "forecast_timestamp_utc"):
            value = data[key]
            data[key] = value.isoformat() if isinstance(value, datetime) else None
        return data


class LiveMarketCatchUpService:
    """Fill missing live H1 context without fabricating retroactive live forecasts.

    Missing bars are fetched and persisted in chronological order. Only the requested
    latest completed hour may produce a forecast. Earlier catch-up hours become causal
    feature context only, so the audit trail never presents backfilled history as if it
    had been forecast live at the time.
    """

    def __init__(
        self,
        source: MarketBarBackfillSource,
        repository: ForecastRepository,
        engine: LivePredictionEngine,
        orchestrator: LiveForecastOrchestrator,
    ) -> None:
        self._source = source
        self._repository = repository
        self._engine = engine
        self._orchestrator = orchestrator

    def catch_up(self, through_hour_utc: datetime) -> LiveCatchUpResult:
        through = self._hour_start(through_hour_utc)
        recent = self._repository.recent_bars(limit=1)
        last = (
            recent[-1].timestamp_utc.astimezone(timezone.utc)
            if recent
            else self._engine.historical_last_datetime_utc.astimezone(timezone.utc)
        )
        start = last + timedelta(hours=1)

        if start > through:
            return LiveCatchUpResult(
                requested_start_utc=None,
                requested_end_utc=through,
                fetched_bars=0,
                created_bars=0,
                forecast_created=False,
                forecast_timestamp_utc=None,
                status="ALREADY_CAUGHT_UP",
            )

        bars = self._source.fetch_completed_range(start, through)
        self._validate_range(bars, start, through)
        created = 0
        for bar in bars:
            created += int(self._orchestrator.ingest_bar(bar))

        if not bars or bars[-1].timestamp_utc.astimezone(timezone.utc) != through:
            return LiveCatchUpResult(
                requested_start_utc=start,
                requested_end_utc=through,
                fetched_bars=len(bars),
                created_bars=created,
                forecast_created=False,
                forecast_timestamp_utc=None,
                status="LATEST_HOUR_NOT_AVAILABLE",
            )

        snapshot, forecast_created = self._orchestrator.materialize_latest_forecast()

        return LiveCatchUpResult(
            requested_start_utc=start,
            requested_end_utc=through,
            fetched_bars=len(bars),
            created_bars=created,
            forecast_created=forecast_created,
            forecast_timestamp_utc=snapshot.feature_timestamp_utc,
            status="FORECAST_MATERIALIZED" if forecast_created else "FORECAST_ALREADY_EXISTS",
        )

    @staticmethod
    def _validate_range(bars, start: datetime, end: datetime) -> None:
        previous: datetime | None = None
        for bar in bars:
            if bar.timestamp_utc.tzinfo is None:
                raise ValueError("Catch-up source returned a timezone-naive H1 bar.")
            timestamp = bar.timestamp_utc.astimezone(timezone.utc)
            if timestamp < start or timestamp > end:
                raise ValueError("Catch-up source returned an H1 bar outside the requested range.")
            if previous is not None and timestamp <= previous:
                raise ValueError("Catch-up source returned duplicate or non-chronological H1 bars.")
            previous = timestamp

    @staticmethod
    def _hour_start(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("through_hour_utc must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError("through_hour_utc must align to an exact UTC hour.")
        return utc
