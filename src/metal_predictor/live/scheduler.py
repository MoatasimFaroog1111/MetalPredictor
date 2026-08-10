from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging

from metal_predictor.live.contracts import MarketBarSource
from metal_predictor.live.inference import LiveForecastOrchestrator


logger = logging.getLogger(__name__)


class HourlyCollectionScheduler:
    """Single-process hourly collector with deterministic UTC scheduling.

    The scheduler is deliberately independent of FastAPI. Duplicate executions are safe
    because the repository/orchestrator enforce idempotency and revision rejection.
    """

    def __init__(
        self,
        source: MarketBarSource,
        orchestrator: LiveForecastOrchestrator,
        delay_minutes: int = 5,
    ) -> None:
        if not 1 <= int(delay_minutes) <= 30:
            raise ValueError("delay_minutes must be between 1 and 30.")
        self._source = source
        self._orchestrator = orchestrator
        self._delay_minutes = int(delay_minutes)

    async def run_forever(self) -> None:
        now = datetime.now(timezone.utc)
        if now.minute >= self._delay_minutes:
            await self.collect_previous_completed_hour(now)
        while True:
            now = datetime.now(timezone.utc)
            due = self.next_due_utc(now, self._delay_minutes)
            await asyncio.sleep(max(0.0, (due - now).total_seconds()))
            await self.collect_previous_completed_hour(due)

    async def collect_previous_completed_hour(
        self,
        now_utc: datetime | None = None,
    ) -> None:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        target = self._orchestrator.previous_completed_hour(now)
        try:
            bar = await asyncio.to_thread(self._source.fetch_completed_hour, target)
            await asyncio.to_thread(self._orchestrator.ingest_and_forecast, bar)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Hourly live collection failed for %s", target.isoformat())

    @staticmethod
    def next_due_utc(now_utc: datetime, delay_minutes: int = 5) -> datetime:
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware.")
        if not 1 <= int(delay_minutes) <= 30:
            raise ValueError("delay_minutes must be between 1 and 30.")
        now = now_utc.astimezone(timezone.utc)
        due = now.replace(minute=int(delay_minutes), second=0, microsecond=0)
        if due <= now:
            due += timedelta(hours=1)
        return due
