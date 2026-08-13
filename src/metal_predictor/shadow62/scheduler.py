from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging

from metal_predictor.shadow62.service import Shadow62Service


logger = logging.getLogger(__name__)


class Shadow62Scheduler:
    """Runs the isolated shadow service after the normal Silver collection window."""

    def __init__(self, service: Shadow62Service, delay_minutes: int = 8) -> None:
        if not 1 <= int(delay_minutes) <= 30:
            raise ValueError("Shadow62 delay_minutes must be between 1 and 30.")
        self._service = service
        self._delay_minutes = int(delay_minutes)

    async def run_forever(self) -> None:
        while True:
            delay = self.seconds_until_next_run()
            await asyncio.sleep(delay)
            try:
                await asyncio.to_thread(self._service.run_once)
            except Exception:
                logger.exception("Shadow62 research collection failed; live forecast remains unaffected.")

    def seconds_until_next_run(self, now_utc: datetime | None = None) -> float:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        this_hour = now.replace(minute=0, second=0, microsecond=0)
        candidate = this_hour + timedelta(minutes=self._delay_minutes)
        if candidate <= now:
            candidate = candidate + timedelta(hours=1)
        return max((candidate - now).total_seconds(), 0.0)
