from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Protocol


logger = logging.getLogger(__name__)


class CatchUpCollector(Protocol):
    def catch_up(self, through_hour_utc: datetime) -> object: ...


class HourlyCollectionScheduler:
    """Single-process UTC scheduler for the independently testable catch-up service."""

    def __init__(
        self,
        collector: CatchUpCollector,
        delay_minutes: int = 5,
    ) -> None:
        if not 1 <= int(delay_minutes) <= 30:
            raise ValueError("delay_minutes must be between 1 and 30.")
        self._collector = collector
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
        target = self.previous_completed_hour(now)
        try:
            await asyncio.to_thread(self._collector.catch_up, target)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Hourly live catch-up failed through %s", target.isoformat())

    @staticmethod
    def previous_completed_hour(now_utc: datetime) -> datetime:
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware.")
        now = now_utc.astimezone(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        return current_hour - timedelta(hours=1)

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
